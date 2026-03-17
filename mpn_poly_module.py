"""
Cubic MPN Module for Reinforcement Learning

Variant of the MPN layer where the M-matrix update uses a learnable cubic
polynomial with stable fixed points:

    M_t = -λ * M_{t-1} ⊙ (M_{t-1} - |a|) ⊙ (M_{t-1} - |b|) + η * h_t ⊗ x_t^T

where λ, a, b, η are trainable scalar parameters and ⊙ is the Hadamard product.
|a| and |b| are used in the computation so the non-zero roots are always positive.

This cubic has roots at 0, |a|, and |b|, so M=0 is always a fixed point and the
network can learn non-trivial attractors at |a| and |b|. When |a| ≠ |b| and λ > 0
the map can exhibit bistable working-memory dynamics (two non-zero attractors).

Reference: eLife-83035 - Multi-plasticity networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class MPNPolyLayer(nn.Module):
    """
    MPN Layer with cubic M-matrix update.

    The M-matrix update is:
        M_t = -λ * M_{t-1} ⊙ (M_{t-1} - |a|) ⊙ (M_{t-1} - |b|) + η * h_t ⊗ x_t^T

    where λ, a, b, η are trainable scalar parameters.  |a| and |b| are used so
    the non-zero roots of the cubic are always positive.  M=0 is always a fixed
    point, and the network can learn additional attractors by adjusting a and b.
    The forward computation (modulated weights, pre-activation, activation) is
    identical to MPNLayer.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer (output of this layer)
        activation: Activation function ('relu', 'tanh', 'sigmoid', or 'linear')
        bias: Whether to use bias term
        freeze_plasticity: Disable Hebbian updates

    Shape:
        - Input: (batch_size, input_dim)
        - State: (batch_size, hidden_dim, input_dim)  # M matrix
        - Output: (batch_size, hidden_dim)
        - New State: (batch_size, hidden_dim, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'tanh',
        bias: bool = True,
        freeze_plasticity: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freeze_plasticity = freeze_plasticity

        # Cubic update parameters (learnable via backprop)
        # lam scales the cubic term overall.
        self.lam = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        # a and b are the non-zero roots of the cubic M*(M-a)*(M-b).
        # Initialising both to 1 so training starts with a degenerate (double)
        # root at 1 and a single root at 0; they are free to separate.
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        # eta scales the Hebbian outer product.
        self.eta = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

        # Long-term synaptic weights
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_buffer('bias', torch.zeros(hidden_dim))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize the synaptic modulation matrix M to zeros."""
        if device is None:
            device = self.W.device
        return torch.zeros(batch_size, self.hidden_dim, self.input_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with cubic M-update.

        1. Pre-activation: y = bias + (W * (M + 1)) @ x
        2. Activation:     h = activation(y)
        3. Cubic update:   M_new = -λ*(M⊙(M-|a|)⊙(M-|b|)) + η*h⊗x^T

        Args:
            x: (batch_size, input_dim)
            state: (batch_size, hidden_dim, input_dim) or None

        Returns:
            h:       (batch_size, hidden_dim)
            M_new:   (batch_size, hidden_dim, input_dim)
        """
        batch_size = x.shape[0]

        if state is None:
            M = self.init_state(batch_size, device=x.device)
        else:
            M = state

        if self.freeze_plasticity:
            y = torch.nn.functional.linear(x, self.W, self.bias)
            h = self.activation(y)
            M_new = torch.zeros_like(M).detach()
        else:
            # Modulated weights: W * (M + 1)
            W_mod = self.W.unsqueeze(0) * (M + 1.0)  # [B, H, I]

            # Pre-activation
            y = self.bias.unsqueeze(0) + torch.bmm(W_mod, x.unsqueeze(2)).squeeze(2)

            # Activation
            h = self.activation(y)

            # Cubic M update: -λ * M*(M-|a|)*(M-|b|) + η * h ⊗ x^T
            # abs enforces a,b > 0 so the non-zero roots stay on one side of 0.
            # h: [B, H, 1]  x: [B, 1, I]  -> outer product [B, H, I]
            hebbian = self.eta * torch.bmm(h.unsqueeze(2), x.unsqueeze(1))
            a_pos = torch.abs(self.a)
            b_pos = torch.abs(self.b)
            M_new = -self.lam * M * (M - a_pos) * (M - b_pos) + hebbian

        return h, M_new


class MPNPoly(nn.Module):
    """
    Complete Cubic MPN for RL.

    Wraps MPNPolyLayer (cubic update) with the same interface as MPN.

    Args:
        input_dim: Dimension of observations
        hidden_dim: Dimension of hidden layer
        activation: Activation function for hidden layer
        freeze_plasticity: Disable Hebbian updates
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'tanh',
        freeze_plasticity: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freeze_plasticity = freeze_plasticity

        self.mpn_layer = MPNPolyLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            bias=True,
            freeze_plasticity=freeze_plasticity,
        )

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.mpn_layer.init_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mpn_layer(x, state)


if __name__ == "__main__":
    print("Testing MPNPoly module (cubic update)...")

    mpn = MPNPoly(input_dim=4, hidden_dim=8)
    print(f"lam (init): {mpn.mpn_layer.lam.item():.4f}")
    print(f"a   (init): {mpn.mpn_layer.a.item():.4f}")
    print(f"b   (init): {mpn.mpn_layer.b.item():.4f}")
    print(f"eta (init): {mpn.mpn_layer.eta.item():.4f}")

    batch_size = 2
    x = torch.randn(batch_size, 4)
    state = mpn.init_state(batch_size)

    hidden, new_state = mpn(x, state)
    print(f"Input shape:     {x.shape}")
    print(f"State shape:     {state.shape}")
    print(f"Hidden shape:    {hidden.shape}")
    print(f"New state shape: {new_state.shape}")
    print(f"State changed:   {not torch.allclose(state, new_state)}")

    print("\nTesting sequence of 5 steps:")
    state = mpn.init_state(batch_size)
    for t in range(5):
        x = torch.randn(batch_size, 4)
        hidden, state = mpn(x, state)
        print(f"  Step {t}: M mean={state.mean().item():.4f}, std={state.std().item():.4f}")

    print("\nMPNPoly module test completed!")
