"""
TorchRL wrapper for MPN-DQN

This module provides TorchRL-compatible wrappers for MPN-DQN, making it easy to use
with TorchRL's training pipelines, environments, and DQN loss functions.

Example:
    >>> from mpn_torchrl import MPNDQNTorchRL
    >>> from torchrl.envs import GymEnv
    >>>
    >>> # Create environment and DQN
    >>> env = GymEnv("CartPole-v1")
    >>> dqn = MPNDQNTorchRL(obs_dim=4, hidden_dim=64, action_dim=2)
    >>>
    >>> # Use with TorchRL
    >>> tensordict = env.reset()
    >>> tensordict = dqn(tensordict)
"""

import torch
import torch.nn as nn
from typing import Optional, List
from mpn_dqn import MPNDQN, DoubleMPNDQN

from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase


class MPNDQNTorchRL(nn.Module):
    """
    TorchRL-compatible wrapper for MPN-DQN.

    This wrapper allows MPN-DQN to work seamlessly with TorchRL's TensorDict-based
    infrastructure, including environments, data collectors, and loss functions.

    Args:
        obs_dim: Dimension of observations
        hidden_dim: Dimension of MPN hidden layer
        action_dim: Number of discrete actions
        eta: Hebbian learning rate
        lambda_decay: Decay factor for M matrix
        activation: Activation function
        in_keys: Keys to read from input TensorDict (default: ["observation"])
        out_keys: Keys to write to output TensorDict (default: ["action_value"])
        state_key: Key for storing M matrix state (default: "mpn_state")

    Example:
        >>> dqn = MPNDQNTorchRL(obs_dim=4, hidden_dim=64, action_dim=2)
        >>> td = TensorDict({"observation": torch.randn(32, 4)}, batch_size=[32])
        >>> td = dqn(td)  # Adds "action_value" and "mpn_state"
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        action_dim: int,
        eta: float = 0.01,
        lambda_decay: float = 0.95,
        activation: str = 'tanh',
        in_keys: Optional[List[str]] = None,
        out_keys: Optional[List[str]] = None,
        state_key: str = "mpn_state",
    ):
        super().__init__()

        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = ["action_value"]

        self.in_keys = in_keys
        self.out_keys = out_keys
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_key = state_key

        # Create MPN-DQN
        self.dqn = MPNDQN(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation
        )

    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Forward pass computing Q-values using TensorDict.

        Args:
            tensordict: Input TensorDict containing observation(s)

        Returns:
            Updated TensorDict with Q-values and new M matrix state
        """
        # Get input
        obs = tensordict[self.in_keys[0]]

        # Get or initialize state
        if self.state_key in tensordict.keys():
            state = tensordict[self.state_key]
        else:
            # Infer batch size from observation
            batch_size = obs.shape[0] if obs.dim() > 1 else 1
            state = self.dqn.init_state(batch_size, device=obs.device)

        # Forward pass
        q_values, new_state = self.dqn(obs, state)

        # Update tensordict
        tensordict[self.out_keys[0]] = q_values
        tensordict[self.state_key] = new_state

        return tensordict

    def reset_state(self, tensordict: TensorDict) -> TensorDict:
        """
        Reset the M matrix state in the tensordict.

        Args:
            tensordict: TensorDict to reset state in

        Returns:
            TensorDict with reset state
        """
        batch_size = tensordict.batch_size[0]
        device = tensordict.device
        tensordict[self.state_key] = self.dqn.init_state(batch_size, device)
        return tensordict

    def select_action(
        self,
        tensordict: TensorDict,
        epsilon: float = 0.0
    ) -> TensorDict:
        """
        Select action using epsilon-greedy policy.

        Args:
            tensordict: Input TensorDict with observations
            epsilon: Exploration rate

        Returns:
            TensorDict with selected actions and updated state
        """
        obs = tensordict[self.in_keys[0]]

        # Get or initialize state
        if self.state_key in tensordict.keys():
            state = tensordict[self.state_key]
        else:
            batch_size = obs.shape[0] if obs.dim() > 1 else 1
            state = self.dqn.init_state(batch_size, device=obs.device)

        # Select action
        actions, new_state = self.dqn.select_action(obs, state, epsilon)

        # Update tensordict
        tensordict["action"] = actions
        tensordict[self.state_key] = new_state

        return tensordict


class DoubleMPNDQNTorchRL(nn.Module):
    """
    Double DQN with MPN for TorchRL.

    Wraps DoubleMPNDQN for use with TorchRL's training infrastructure.
    Provides both online and target networks with TensorDict interface.

    Args:
        obs_dim: Dimension of observations
        hidden_dim: Dimension of MPN hidden layer
        action_dim: Number of discrete actions
        eta: Hebbian learning rate
        lambda_decay: Decay factor
        activation: Activation function

    Example:
        >>> double_dqn = DoubleMPNDQNTorchRL(
        ...     obs_dim=4, hidden_dim=64, action_dim=2
        ... )
        >>> td = TensorDict({"observation": torch.randn(32, 4)}, batch_size=[32])
        >>>
        >>> # Online network
        >>> td_online = double_dqn.online_module(td.clone())
        >>>
        >>> # Target network
        >>> td_target = double_dqn.target_module(td.clone())
        >>>
        >>> # Update target
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

        # Double DQN backend
        self.double_dqn = DoubleMPNDQN(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation
        )

        # TorchRL wrappers
        self.online_module = MPNDQNTorchRL(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation,
            state_key="online_mpn_state"
        )
        self.online_module.dqn = self.double_dqn.online

        self.target_module = MPNDQNTorchRL(
            obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation,
            state_key="target_mpn_state"
        )
        self.target_module.dqn = self.double_dqn.target

    def hard_update(self):
        """Copy online network weights to target network."""
        self.double_dqn.hard_update()

    def soft_update(self, tau: float = 0.005):
        """Soft update target network."""
        self.double_dqn.soft_update(tau)



if __name__ == "__main__":
    print("Testing TorchRL MPN-DQN wrapper...")

    # Test MPNDQNTorchRL
    dqn_torchrl = MPNDQNTorchRL(obs_dim=4, hidden_dim=8, action_dim=2)

    # Create sample tensordict
    td = TensorDict({
        "observation": torch.randn(3, 4)
    }, batch_size=[3])

    print(f"\nInput TensorDict keys: {list(td.keys())}")

    # Forward pass (initializes state automatically)
    td = dqn_torchrl(td)
    print(f"After forward: {list(td.keys())}")
    print(f"Q-values shape: {td['action_value'].shape}")
    print(f"Q-values:\n{td['action_value']}")
    print(f"State shape: {td['mpn_state'].shape}")

    # Second forward pass (uses existing state)
    td = dqn_torchrl(td)
    print(f"\nAfter second forward: Q-values changed (state updated)")

    # Action selection
    td_action = td.clone()
    td_action = dqn_torchrl.select_action(td_action, epsilon=0.1)
    print(f"Selected actions (ε=0.1): {td_action['action']}")

    # Reset state
    td = dqn_torchrl.reset_state(td)
    print(f"After reset: state mean = {td['mpn_state'].mean().item():.4f}")

    # Test Double DQN
    print("\n" + "="*50)
    print("Testing Double MPN-DQN TorchRL...")
    double_dqn_torchrl = DoubleMPNDQNTorchRL(
        obs_dim=4,
        hidden_dim=8,
        action_dim=2
    )

    td = TensorDict({
        "observation": torch.randn(3, 4)
    }, batch_size=[3])

    # Online network
    td_online = double_dqn_torchrl.online_module(td.clone())
    print(f"Online Q-values: {td_online['action_value'][0]}")

    # Target network
    td_target = double_dqn_torchrl.target_module(td.clone())
    print(f"Target Q-values: {td_target['action_value'][0]}")

    print(f"Match initially: {torch.allclose(td_online['action_value'], td_target['action_value'])}")

    # Soft update
    # Modify online
    optimizer = torch.optim.Adam(double_dqn_torchrl.double_dqn.online.parameters(), lr=0.01)
    loss = td_online['action_value'].mean()
    loss.backward()
    optimizer.step()

    # Check difference
    td_online_new = double_dqn_torchrl.online_module(td.clone())
    td_target_new = double_dqn_torchrl.target_module(td.clone())
    print(f"After online update, Q-values differ: {not torch.allclose(td_online_new['action_value'], td_target_new['action_value'])}")

    # Soft update target
    double_dqn_torchrl.soft_update(tau=0.1)
    td_target_updated = double_dqn_torchrl.target_module(td.clone())
    print(f"After soft update, target moved toward online: {not torch.allclose(td_target_new['action_value'], td_target_updated['action_value'])}")

    print("\nTorchRL MPN-DQN wrapper test completed!")

