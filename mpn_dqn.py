"""
Deep Q-Network with Multi-Plasticity Network as Recurrent Layer

This module implements a Deep Recurrent Q-Network (DRQN) using MPN as the recurrent
layer instead of LSTM/GRU. The MPN's synaptic modulation matrix serves as the
recurrent state, providing a biologically-inspired alternative to traditional RNNs.

Architecture:
    Observation → MPN (recurrent) → Linear → Q-values

The MPN layer maintains a synaptic modulation matrix M that captures temporal
dependencies within an episode, while the Q-values are learned via standard DQN
training across episodes.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from mpn_module import MPN


class MPNDQN(nn.Module):
    """
    Deep Q-Network with MPN as the recurrent layer.

    This network processes observations through an MPN layer and outputs Q-values
    for each action. The MPN's internal state (M matrix) acts as recurrent memory,
    allowing the network to integrate information over time within an episode.

    Args:
        obs_dim: Dimension of observations
        hidden_dim: Dimension of MPN hidden layer
        action_dim: Number of discrete actions
        eta: Hebbian learning rate for MPN
        lambda_decay: Decay factor for M matrix
        activation: Activation function for MPN layer

    Shape:
        - Input: (batch_size, obs_dim)
        - State: (batch_size, hidden_dim, obs_dim)  # M matrix
        - Output: (batch_size, action_dim)  # Q-values
        - New State: (batch_size, hidden_dim, obs_dim)

    Example:
        >>> dqn = MPNDQN(obs_dim=4, hidden_dim=64, action_dim=2)
        >>> obs = torch.randn(32, 4)
        >>> state = dqn.init_state(batch_size=32)
        >>> q_values, new_state = dqn(obs, state)
        >>> action = q_values.argmax(dim=1)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        eta: float = 0.01,
        lambda_decay: float = 0.95,
        activation: str = 'tanh',
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # MPN recurrent layer
        self.mpn = MPN(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation
        )

        # Q-value head
        self.q_head = nn.Linear(hidden_dim, action_dim)

        # Initialize Q-head with small weights for stability
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)
        nn.init.zeros_(self.q_head.bias)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize the MPN state (M matrix) for a new episode.

        Args:
            batch_size: Number of parallel episodes
            device: Device to create state on

        Returns:
            Initial M matrix of shape (batch_size, hidden_dim, obs_dim)
        """
        return self.mpn.init_state(batch_size, device)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Q-values.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            state: Current M matrix. If None, initializes to zeros.

        Returns:
            q_values: Q-values for each action, shape (batch_size, action_dim)
            new_state: Updated M matrix, shape (batch_size, hidden_dim, obs_dim)
        """
        # Process through MPN
        hidden, new_state = self.mpn(obs, state)

        # Compute Q-values
        q_values = self.q_head(hidden)

        return q_values, new_state

    def select_action(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using epsilon-greedy policy.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            state: Current M matrix
            epsilon: Exploration rate (0 = greedy, 1 = random)

        Returns:
            actions: Selected actions, shape (batch_size,)
            new_state: Updated M matrix
        """
        with torch.no_grad():
            q_values, new_state = self.forward(obs, state)

            batch_size = obs.shape[0]

            # Epsilon-greedy action selection
            if epsilon > 0:
                # Random actions
                random_actions = torch.randint(0, self.action_dim, (batch_size,), device=obs.device)
                # Greedy actions
                greedy_actions = q_values.argmax(dim=1)
                # Mix based on epsilon
                explore_mask = torch.rand(batch_size, device=obs.device) < epsilon
                actions = torch.where(explore_mask, random_actions, greedy_actions)
            else:
                # Greedy only
                actions = q_values.argmax(dim=1)

        return actions, new_state


class DoubleMPNDQN(nn.Module):
    """
    Double DQN with MPN recurrent layers.

    Implements Double DQN to reduce overestimation bias. Maintains both an online
    network and a target network, where the target network is periodically updated
    from the online network.

    Args:
        obs_dim: Dimension of observations
        hidden_dim: Dimension of MPN hidden layer
        action_dim: Number of discrete actions
        eta: Hebbian learning rate
        lambda_decay: Decay factor for M matrix
        activation: Activation function

    Example:
        >>> double_dqn = DoubleMPNDQN(obs_dim=4, hidden_dim=64, action_dim=2)
        >>>
        >>> # Training step
        >>> q_values, state = double_dqn.online(obs, state)
        >>> target_q, target_state = double_dqn.target(next_obs, next_state)
        >>>
        >>> # Update target network
        >>> double_dqn.update_target(tau=0.005)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        eta: float = 0.01,
        lambda_decay: float = 0.95,
        activation: str = 'tanh',
    ):
        super().__init__()

        # Online network (updated via gradient descent)
        self.online = MPNDQN(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation
        )

        # Target network (updated via soft/hard updates from online)
        self.target = MPNDQN(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation
        )

        # Initialize target network with same weights as online
        self.hard_update()

        # Target network is not trained
        for param in self.target.parameters():
            param.requires_grad = False

    def hard_update(self):
        """Copy online network weights to target network."""
        self.target.load_state_dict(self.online.state_dict())

    def soft_update(self, tau: float = 0.005):
        """
        Soft update target network using Polyak averaging.

        θ_target = τ * θ_online + (1 - τ) * θ_target

        Args:
            tau: Interpolation parameter (typically small, e.g., 0.005)
        """
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    def forward(self, obs: torch.Tensor, state: Optional[torch.Tensor] = None):
        """Forward pass through online network."""
        return self.online(obs, state)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None):
        """Initialize state for both networks."""
        return self.online.init_state(batch_size, device)


if __name__ == "__main__":
    print("Testing MPN-DQN...")

    # Test basic MPNDQN
    dqn = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2)

    batch_size = 3
    obs = torch.randn(batch_size, 4)
    state = dqn.init_state(batch_size)

    print(f"\nInput observation shape: {obs.shape}")
    print(f"Initial state shape: {state.shape}")

    # Forward pass
    q_values, new_state = dqn(obs, state)
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values}")
    print(f"New state shape: {new_state.shape}")

    # Action selection
    actions, new_state = dqn.select_action(obs, state, epsilon=0.1)
    print(f"\nSelected actions (ε=0.1): {actions}")

    # Test sequence
    print("\nTesting sequence of 5 steps:")
    state = dqn.init_state(batch_size)
    for t in range(5):
        obs = torch.randn(batch_size, 4)
        q_values, state = dqn(obs, state)
        actions = q_values.argmax(dim=1)
        print(f"Step {t}: Q-values mean = {q_values.mean().item():.4f}, Actions = {actions.tolist()}")

    # Test Double DQN
    print("\n" + "="*50)
    print("Testing Double MPN-DQN...")

    double_dqn = DoubleMPNDQN(obs_dim=4, hidden_dim=8, action_dim=2)

    obs = torch.randn(batch_size, 4)
    state = double_dqn.init_state(batch_size)

    # Online network
    q_online, state_online = double_dqn.online(obs, state)
    print(f"Online Q-values: {q_online[0].detach()}")

    # Target network
    q_target, state_target = double_dqn.target(obs, state)
    print(f"Target Q-values (before update): {q_target[0].detach()}")

    # Verify they're the same initially
    print(f"Online and target match: {torch.allclose(q_online, q_target)}")

    # Modify online network
    optimizer = torch.optim.Adam(double_dqn.online.parameters(), lr=0.01)
    loss = q_online.mean()
    loss.backward()
    optimizer.step()

    # Check they're different now
    q_online_new, _ = double_dqn.online(obs, state)
    q_target_new, _ = double_dqn.target(obs, state)
    print(f"After online update, Q-values differ: {not torch.allclose(q_online_new, q_target_new)}")

    # Soft update
    double_dqn.soft_update(tau=0.1)
    q_target_updated, _ = double_dqn.target(obs, state)
    print(f"After soft update, target moved toward online: {not torch.allclose(q_target_new, q_target_updated)}")

    print("\nMPN-DQN test completed!")
