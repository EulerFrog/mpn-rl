"""
Model save/load utilities and experiment management for MPN-DQN

Handles:
- Saving/loading model weights and optimizer states
- Experiment directory structure
- Configuration management
- Training history tracking
- Random experiment name generation
- Replay buffer and TD loss computation
"""

import os
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Any
import random
from datetime import datetime
from collections import deque, namedtuple


# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'next_obs', 'done', 'state', 'next_state'])


class ReplayBuffer:
    """Simple replay buffer for DQN."""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        """Add an experience to the buffer."""
        self.buffer.append(Experience(*args))

    def sample(self, batch_size):
        """Sample a batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)

        # Stack into tensors
        obs = torch.stack([e.obs for e in experiences])
        actions = torch.tensor([e.action for e in experiences], dtype=torch.long)
        rewards = torch.tensor([e.reward for e in experiences], dtype=torch.float32)
        next_obs = torch.stack([e.next_obs for e in experiences])
        dones = torch.tensor([e.done for e in experiences], dtype=torch.float32)
        states = torch.stack([e.state for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])

        return obs, actions, rewards, next_obs, dones, states, next_states

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(dqn, target_dqn, batch, gamma=0.99):
    """
    Compute TD loss for DQN.

    Uses Double DQN update: Q_target = r + γ * Q_target(s', argmax_a Q_online(s', a))

    Args:
        dqn: Online DQN network
        target_dqn: Target DQN network
        batch: Tuple of (obs, actions, rewards, next_obs, dones, states, next_states)
        gamma: Discount factor

    Returns:
        loss: Smooth L1 loss between current Q-values and target Q-values
    """
    obs, actions, rewards, next_obs, dones, states, next_states = batch

    # Current Q-values: Q(s, a)
    current_q, _ = dqn(obs, states)
    current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Next Q-values from target network (Double DQN)
    with torch.no_grad():
        # Get best actions from online network
        next_q_online, _ = dqn(next_obs, next_states)
        next_actions = next_q_online.argmax(dim=1)

        # Evaluate those actions with target network
        next_q_target, _ = target_dqn(next_obs, next_states)
        next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

        # TD target: r + γ * Q_target(s', a*)
        target_q = rewards + gamma * next_q * (1 - dones)

    # Compute loss
    loss = F.smooth_l1_loss(current_q, target_q)

    return loss


# Word lists for random experiment names
ADJECTIVES = [
    'swift', 'brave', 'bright', 'calm', 'clever', 'bold', 'eager', 'fair',
    'gentle', 'happy', 'keen', 'lively', 'merry', 'noble', 'polite', 'proud',
    'quiet', 'rapid', 'sincere', 'tender', 'vivid', 'wise', 'zealous', 'agile',
    'cosmic', 'digital', 'electric', 'frozen', 'golden', 'lunar', 'mystic',
    'neural', 'quantum', 'radiant', 'silver', 'stellar', 'turbo', 'ultra'
]

NOUNS = [
    'tiger', 'eagle', 'falcon', 'dragon', 'phoenix', 'wolf', 'bear', 'lion',
    'hawk', 'raven', 'shark', 'panther', 'cobra', 'viper', 'mantis', 'spider',
    'scorpion', 'leopard', 'cheetah', 'jaguar', 'orca', 'dolphin', 'owl',
    'condor', 'lynx', 'puma', 'fox', 'badger', 'otter', 'weasel', 'mink',
    'neuron', 'synapse', 'cortex', 'network', 'circuit', 'matrix', 'tensor'
]


def generate_experiment_name() -> str:
    """Generate a random experiment name like 'brave-tiger' or 'swift-eagle'."""
    adj = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    return f"{adj}-{noun}"


class ExperimentManager:
    """
    Manages experiment directory structure and file I/O.

    Directory structure:
        experiments/{experiment_name}/
        ├── config.json
        ├── training_history.json
        ├── checkpoints/
        │   ├── best_model.pt
        │   ├── checkpoint_100.pt
        │   └── final_model.pt
        ├── videos/
        │   └── episode_*.gif
        └── plots/
            └── training_curves.png
    """

    def __init__(self, experiment_name: Optional[str] = None, base_dir: str = "experiments"):
        """
        Args:
            experiment_name: Name of experiment (generates random if None)
            base_dir: Base directory for all experiments
        """
        if experiment_name is None:
            experiment_name = generate_experiment_name()

        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name

        # Create directory structure
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.video_dir = self.exp_dir / "videos"
        self.plot_dir = self.exp_dir / "plots"

        for dir_path in [self.checkpoint_dir, self.video_dir, self.plot_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.config_path = self.exp_dir / "config.json"
        self.history_path = self.exp_dir / "training_history.json"

    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {self.config_path}")

    def load_config(self) -> Dict[str, Any]:
        """Load experiment configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return json.load(f)

    def save_model(
        self,
        model,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_name: str = "model.pt",
        metadata: Optional[Dict] = None
    ):
        """
        Save model checkpoint.

        Args:
            model: Model to save (nn.Module)
            optimizer: Optimizer to save (optional, for resuming training)
            checkpoint_name: Name of checkpoint file
            metadata: Additional metadata (episode, reward, etc.)
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_model(
        self,
        model,
        checkpoint_name: str = "best_model.pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = 'cpu'
    ) -> Dict:
        """
        Load model checkpoint.

        Args:
            model: Model to load weights into
            checkpoint_name: Name of checkpoint file
            optimizer: Optimizer to load state into (if resuming training)
            device: Device to load model onto

        Returns:
            metadata: Dictionary with episode, reward, etc.
        """
        checkpoint_path = self.checkpoint_dir / checkpoint_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint.get('metadata', {})

    def save_training_history(self, history: Dict[str, list]):
        """Save training history (rewards, losses, etc.)."""
        with open(self.history_path, 'w') as f:
            json.dump(history, f, indent=2)

    def load_training_history(self) -> Dict[str, list]:
        """Load training history."""
        if not self.history_path.exists():
            return {
                'episodes': [],
                'rewards': [],
                'lengths': [],
                'losses': [],
                'epsilons': []
            }
        with open(self.history_path, 'r') as f:
            return json.load(f)

    def append_training_history(self, episode: int, reward: float,
                                length: int, loss: float, epsilon: float):
        """Append new data to training history."""
        history = self.load_training_history()
        history['episodes'].append(episode)
        history['rewards'].append(reward)
        history['lengths'].append(length)
        history['losses'].append(loss)
        history['epsilons'].append(epsilon)
        self.save_training_history(history)

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best model checkpoint."""
        best_path = self.checkpoint_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to most recent checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints:
            return None
        # Sort by episode number
        checkpoints.sort(key=lambda p: int(p.stem.split('_')[1]))
        return str(checkpoints[-1])

    def __repr__(self):
        return f"ExperimentManager('{self.experiment_name}', dir='{self.exp_dir}')"


def save_checkpoint(
    experiment_manager: ExperimentManager,
    model,
    optimizer,
    episode: int,
    avg_reward: float,
    is_best: bool = False,
    is_final: bool = False
):
    """
    Convenience function to save a checkpoint.

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to save
        optimizer: Optimizer to save
        episode: Current episode number
        avg_reward: Average reward (for tracking best)
        is_best: Whether this is the best model so far
        is_final: Whether this is the final checkpoint
    """
    metadata = {
        'episode': episode,
        'avg_reward': avg_reward,
        'timestamp': datetime.now().isoformat()
    }

    # Save periodic checkpoint
    checkpoint_name = f"checkpoint_{episode}.pt"
    experiment_manager.save_model(model, optimizer, checkpoint_name, metadata)

    # Save as best if applicable
    if is_best:
        experiment_manager.save_model(model, optimizer, "best_model.pt", metadata)
        print(f"New best model! Avg reward: {avg_reward:.2f}")

    # Save as final if applicable
    if is_final:
        experiment_manager.save_model(model, optimizer, "final_model.pt", metadata)


def load_checkpoint_for_eval(
    experiment_manager: ExperimentManager,
    model,
    checkpoint_name: str = "best_model.pt",
    device: str = 'cpu'
) -> Dict:
    """
    Load checkpoint for evaluation (no optimizer).

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to load weights into
        checkpoint_name: Which checkpoint to load
        device: Device to load onto

    Returns:
        metadata: Checkpoint metadata
    """
    return experiment_manager.load_model(model, checkpoint_name, optimizer=None, device=device)


def load_checkpoint_for_resume(
    experiment_manager: ExperimentManager,
    model,
    optimizer,
    device: str = 'cpu'
) -> Dict:
    """
    Load latest checkpoint to resume training.

    Args:
        experiment_manager: ExperimentManager instance
        model: Model to load weights into
        optimizer: Optimizer to load state into
        device: Device to load onto

    Returns:
        metadata: Checkpoint metadata with episode number
    """
    # Try to get latest checkpoint first
    latest_checkpoint = experiment_manager.get_latest_checkpoint()

    if latest_checkpoint:
        checkpoint_name = Path(latest_checkpoint).name
    else:
        # Fall back to best or final model
        if (experiment_manager.checkpoint_dir / "best_model.pt").exists():
            checkpoint_name = "best_model.pt"
        elif (experiment_manager.checkpoint_dir / "final_model.pt").exists():
            checkpoint_name = "final_model.pt"
        else:
            raise FileNotFoundError("No checkpoints found to resume from")

    metadata = experiment_manager.load_model(model, checkpoint_name, optimizer, device)
    print(f"Resuming from episode {metadata.get('episode', 0)}")
    return metadata


if __name__ == "__main__":
    print("Testing ExperimentManager...")

    # Test random name generation
    print("\nRandom experiment names:")
    for _ in range(5):
        print(f"  - {generate_experiment_name()}")

    # Test experiment manager
    print("\nCreating experiment...")
    exp = ExperimentManager("test-experiment")
    print(f"Created: {exp}")
    print(f"Experiment dir: {exp.exp_dir}")

    # Test config save/load
    print("\nTesting config save/load...")
    config = {
        'env_name': 'CartPole-v1',
        'hidden_dim': 64,
        'eta': 0.05,
        'lambda_decay': 0.9,
        'num_episodes': 500
    }
    exp.save_config(config)
    loaded_config = exp.load_config()
    print(f"Loaded config: {loaded_config}")

    # Test history
    print("\nTesting history...")
    for ep in range(5):
        exp.append_training_history(ep, ep*10, ep*5, 0.1, 0.9)
    history = exp.load_training_history()
    print(f"History episodes: {history['episodes']}")
    print(f"History rewards: {history['rewards']}")

    # Test model save (dummy model)
    print("\nTesting model save...")
    from mpn_dqn import MPNDQN
    model = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2)
    optimizer = torch.optim.Adam(model.parameters())

    save_checkpoint(exp, model, optimizer, episode=100, avg_reward=150.0, is_best=True)

    print("\nExperimentManager test completed!")
    print(f"Check the '{exp.exp_dir}' directory to see generated files")
