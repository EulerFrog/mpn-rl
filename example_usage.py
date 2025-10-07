"""
Example usage of MPN-DQN for Reinforcement Learning

This script demonstrates how to train an MPN-DQN agent on a Gym environment.
It includes:
- Basic DQN training loop
- Experience replay buffer
- Epsilon-greedy exploration
- Target network updates
- Training with and without TorchRL
- Visualization of training progress and M matrix evolution

Run with: python3 example_usage.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
import random
from mpn_dqn import MPNDQN, DoubleMPNDQN
from visualize import TrainingVisualizer, MMatrixVisualizer, visualize_episode
import gymnasium as gym


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


def train_mpn_dqn(
    env_name='CartPole-v1',
    num_episodes=500,
    hidden_dim=64,
    eta=0.01,
    lambda_decay=0.95,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    buffer_size=10000,
    learning_rate=1e-3,
    target_update_freq=10,
    print_freq=10,
    plot_training=True,
    save_plot=None
):
    """
    Train MPN-DQN on a Gym environment.

    Args:
        env_name: Gym environment name
        num_episodes: Number of episodes to train
        hidden_dim: Hidden dimension of MPN
        eta: Hebbian learning rate
        lambda_decay: Decay factor for M matrix
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay per episode
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        learning_rate: Learning rate for optimizer
        target_update_freq: Episodes between target network updates
        print_freq: Episodes between printing
    """
    # Create environment
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Training MPN-DQN on {env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"MPN hidden dim: {hidden_dim}, eta: {eta}, λ: {lambda_decay}\n")

    # Create networks
    online_dqn = MPNDQN(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        eta=eta,
        lambda_decay=lambda_decay
    )

    target_dqn = MPNDQN(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        eta=eta,
        lambda_decay=lambda_decay
    )
    target_dqn.load_state_dict(online_dqn.state_dict())

    # Optimizer
    optimizer = torch.optim.Adam(online_dqn.parameters(), lr=learning_rate)

    # Replay buffer
    replay_buffer = ReplayBuffer(capacity=buffer_size)

    # Visualization
    viz = TrainingVisualizer() if plot_training else None

    # Training loop
    epsilon = epsilon_start
    episode_rewards = []
    episode_lengths = []
    episode_losses = []

    for episode in range(num_episodes):
        # Reset environment and MPN state
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        obs = torch.FloatTensor(obs)
        state = online_dqn.init_state(batch_size=1)

        episode_reward = 0
        episode_length = 0
        episode_loss = []

        while True:
            # Select action
            with torch.no_grad():
                q_values, next_state = online_dqn(obs.unsqueeze(0), state)

                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()

            # Take action
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result

            next_obs = torch.FloatTensor(next_obs)

            # Store experience
            replay_buffer.push(
                obs, action, reward, next_obs, done,
                state.squeeze(0), next_state.squeeze(0)
            )

            # Update state
            obs = next_obs
            state = next_state
            episode_reward += reward
            episode_length += 1

            # Train if enough samples
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = compute_td_loss(online_dqn, target_dqn, batch, gamma)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_dqn.parameters(), 10.0)
                optimizer.step()

                episode_loss.append(loss.item())

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Update target network
        if episode % target_update_freq == 0:
            target_dqn.load_state_dict(online_dqn.state_dict())

        # Track metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        avg_loss = np.mean(episode_loss) if len(episode_loss) > 0 else 0.0
        episode_losses.append(avg_loss)

        # Update visualization
        if viz is not None:
            viz.update(episode=episode, episode_reward=episode_reward,
                      episode_length=episode_length, loss=avg_loss, epsilon=epsilon)

        # Print progress
        if episode % print_freq == 0:
            avg_reward = np.mean(episode_rewards[-print_freq:])
            avg_length = np.mean(episode_lengths[-print_freq:])
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Avg Length: {avg_length:.1f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Buffer: {len(replay_buffer)}")

    env.close()

    print(f"\nTraining completed!")
    print(f"Final avg reward (last 10 episodes): {np.mean(episode_rewards[-10:]):.2f}")

    # Plot training curves
    if viz is not None:
        viz.plot(save_path=save_plot)

    return online_dqn, episode_rewards


def compare_mpn_vs_standard_dqn(env_name='CartPole-v1', num_episodes=300):
    """
    Compare MPN-DQN with a standard (non-recurrent) DQN.
    """
    print("="*60)
    print("Comparing MPN-DQN vs Standard DQN")
    print("="*60)

    # Train MPN-DQN
    print("\n[1/2] Training MPN-DQN...")
    mpn_dqn, mpn_rewards = train_mpn_dqn(
        env_name=env_name,
        num_episodes=num_episodes,
        hidden_dim=32,
        eta=0.05,
        lambda_decay=0.9,
        print_freq=50
    )

    # Create standard DQN for comparison
    print("\n[2/2] Training Standard DQN...")
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # Simple feedforward network
    class StandardDQN(nn.Module):
        def __init__(self, obs_dim, hidden_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim)
            )

        def forward(self, obs):
            return self.net(obs)

    standard_dqn = StandardDQN(obs_dim, 32, action_dim)
    target_standard = StandardDQN(obs_dim, 32, action_dim)
    target_standard.load_state_dict(standard_dqn.state_dict())

    optimizer = torch.optim.Adam(standard_dqn.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = 1.0
    standard_rewards = []

    env = gym.make(env_name)

    for episode in range(num_episodes):
        obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
        obs = torch.FloatTensor(obs)
        episode_reward = 0

        while True:
            with torch.no_grad():
                q_values = standard_dqn(obs)
                action = q_values.argmax().item() if random.random() > epsilon else env.action_space.sample()

            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result

            next_obs = torch.FloatTensor(next_obs)

            # Store with dummy states for compatibility
            replay_buffer.push(
                obs, action, reward, next_obs, done,
                torch.zeros(1, 1), torch.zeros(1, 1)
            )

            obs = next_obs
            episode_reward += reward

            if len(replay_buffer) >= 64:
                batch = replay_buffer.sample(64)
                obs_b, actions_b, rewards_b, next_obs_b, dones_b, _, _ = batch

                current_q = standard_dqn(obs_b).gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = standard_dqn(next_obs_b).argmax(dim=1)
                    next_q = target_standard(next_obs_b).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_b + 0.99 * next_q * (1 - dones_b)

                loss = F.smooth_l1_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

        epsilon = max(0.01, epsilon * 0.995)

        if episode % 10 == 0:
            target_standard.load_state_dict(standard_dqn.state_dict())

        standard_rewards.append(episode_reward)

        if episode % 50 == 0:
            print(f"Episode {episode}/{num_episodes} | "
                  f"Avg Reward: {np.mean(standard_rewards[-50:]):.2f}")

    env.close()

    # Compare results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"MPN-DQN final avg reward: {np.mean(mpn_rewards[-50:]):.2f}")
    print(f"Standard DQN final avg reward: {np.mean(standard_rewards[-50:]):.2f}")
    print(f"Improvement: {(np.mean(mpn_rewards[-50:]) - np.mean(standard_rewards[-50:])):.2f}")


def demonstrate_M_matrix_visualization(dqn, env_name='CartPole-v1'):
    """
    Demonstrate M matrix visualization by running an episode with a trained agent.
    """
    print("\n" + "="*60)
    print("Demonstrating M Matrix Visualization")
    print("="*60)

    env = gym.make(env_name)
    total_reward, viz = visualize_episode(dqn, env, max_steps=500,
                                          save_M_plot='m_matrix_evolution.png')

    print(f"Episode reward: {total_reward:.2f}")
    print(f"Saved M matrix visualization to: m_matrix_evolution.png")

    # Also show M matrix snapshot
    viz.plot_M_snapshot(timestep=-1, save_path='m_matrix_snapshot.png')
    print(f"Saved M matrix snapshot to: m_matrix_snapshot.png")

    env.close()


if __name__ == "__main__":
    # Basic training
    print("Example 1: Basic MPN-DQN training on CartPole")
    print("="*60)
    mpn_dqn, rewards = train_mpn_dqn(
        env_name='CartPole-v1',
        num_episodes=200,
        hidden_dim=32,
        eta=0.05,
        lambda_decay=0.9,
        plot_training=True
    )

    # Visualize M matrix evolution
    demonstrate_M_matrix_visualization(mpn_dqn, env_name='CartPole-v1')

    print("\n\n")

    # Comparison (commented out to keep example quick)
    # Uncomment to compare MPN-DQN with standard DQN:
    # compare_mpn_vs_standard_dqn(env_name='CartPole-v1', num_episodes=300)

