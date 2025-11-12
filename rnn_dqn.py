"""
Deep Q-Network with Vanilla RNN as Recurrent Layer

This module implements a Deep Recurrent Q-Network (DRQN) using a standard RNN
as the recurrent layer. This serves as a baseline for comparison with MPN-DQN.

Architecture:
    Observation → RNN (recurrent) → Linear → Q-values

The RNN layer maintains a hidden state that captures temporal dependencies
within an episode, similar to how MPN uses its M matrix.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RNNDQN(nn.Module):
    """
    Deep Q-Network with vanilla RNN as the recurrent layer.

    This network processes observations through an RNN layer and outputs Q-values
    for each action. The RNN's hidden state acts as recurrent memory, allowing
    the network to integrate information over time within an episode.

    This is a baseline model for comparison with MPNDQN.

    Args:
        obs_dim: Dimension of observations
        hidden_dim: Dimension of RNN hidden layer
        action_dim: Number of discrete actions
        activation: Activation function for RNN layer ('tanh', 'relu')
        eta: Ignored (for compatibility with MPNDQN interface)
        lambda_decay: Ignored (for compatibility with MPNDQN interface)
        freeze_plasticity: Ignored (for compatibility with MPNDQN interface)

    Shape:
        - Input: (batch_size, obs_dim)
        - State: (batch_size, hidden_dim)  # RNN hidden state
        - Output: (batch_size, action_dim)  # Q-values
        - New State: (batch_size, hidden_dim)

    Example:
        >>> dqn = RNNDQN(obs_dim=4, hidden_dim=64, action_dim=2)
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
        activation: str = 'tanh',
        eta: float = 0.0,  # Ignored, for compatibility
        lambda_decay: float = 0.0,  # Ignored, for compatibility
        freeze_plasticity: bool = False,  # Ignored, for compatibility
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.activation = activation

        # RNN recurrent layer
        # Use RNNCell for step-by-step processing (same interface as MPN)
        self.rnn = nn.RNNCell(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            nonlinearity=activation if activation in ['tanh', 'relu'] else 'tanh'
        )

        # Q-value head (same as MPNDQN)
        self.q_head = nn.Linear(hidden_dim, action_dim)

        # Initialize Q-head with small weights for stability (same as MPNDQN)
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)
        nn.init.zeros_(self.q_head.bias)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize the RNN hidden state for a new episode.

        Args:
            batch_size: Number of parallel episodes
            device: Device to create state on

        Returns:
            Initial hidden state of shape (batch_size, hidden_dim)
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(
        self,
        obs: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing Q-values.

        Args:
            obs: Observations of shape (batch_size, obs_dim)
            state: Current RNN hidden state. If None, initializes to zeros.

        Returns:
            q_values: Q-values for each action, shape (batch_size, action_dim)
            new_state: Updated hidden state, shape (batch_size, hidden_dim)
        """
        # Initialize state if not provided
        if state is None:
            state = self.init_state(obs.shape[0], obs.device)

        # Process through RNN
        new_state = self.rnn(obs, state)

        # Compute Q-values
        q_values = self.q_head(new_state)

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
            state: Current RNN hidden state
            epsilon: Exploration rate (0 = greedy, 1 = random)

        Returns:
            actions: Selected actions, shape (batch_size,)
            new_state: Updated hidden state
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


if __name__ == "__main__":
    print("Testing RNN-DQN...")

    # Test basic RNNDQN
    dqn = RNNDQN(obs_dim=4, hidden_dim=8, action_dim=2)

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

    print("\nRNN-DQN test completed!")
1