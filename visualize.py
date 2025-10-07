"""
Visualization tools for MPN-DQN training and analysis

This module provides tools for:
- Plotting training metrics (rewards, losses, etc.)
- Visualizing M matrix evolution during episodes
- Recording agent behavior in environments
- Real-time monitoring during training

Dependencies: matplotlib, numpy, imageio
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import imageio
from typing import Optional, List, Dict, Tuple


class TrainingVisualizer:
    """
    Visualize training metrics in real-time or post-training.

    Example:
        >>> viz = TrainingVisualizer()
        >>> for episode in range(100):
        >>>     reward = train_episode()
        >>>     viz.update(episode_reward=reward)
        >>>     if episode % 10 == 0:
        >>>         viz.plot()
    """

    def __init__(self, metrics=None):
        """
        Args:
            metrics: List of metric names to track (default: ['episode_reward', 'episode_length', 'loss'])
        """
        if metrics is None:
            metrics = ['episode_reward', 'episode_length', 'loss', 'epsilon']

        self.metrics = metrics
        self.data = {metric: [] for metric in metrics}
        self.episodes = []

    def update(self, episode: Optional[int] = None, **kwargs):
        """
        Update metrics for current episode.

        Args:
            episode: Episode number (auto-incremented if None)
            **kwargs: Metric values (e.g., episode_reward=100, loss=0.5)
        """
        if episode is None:
            episode = len(self.episodes)

        self.episodes.append(episode)

        for key, value in kwargs.items():
            if key in self.data:
                self.data[key].append(value)
            else:
                self.data[key] = [value]

    def plot(self, save_path: Optional[str] = None, window_size: int = 10):
        """
        Plot training metrics.

        Args:
            save_path: Path to save figure (displays if None)
            window_size: Window size for moving average
        """
        n_metrics = len(self.data)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 3*n_metrics))

        if n_metrics == 1:
            axes = [axes]

        for ax, (metric_name, values) in zip(axes, self.data.items()):
            if len(values) == 0:
                continue

            ax.plot(self.episodes[:len(values)], values, alpha=0.3, label='Raw')

            # Moving average
            if len(values) >= window_size:
                moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                ax.plot(self.episodes[window_size-1:len(moving_avg)+window_size-1],
                       moving_avg, label=f'MA({window_size})', linewidth=2)

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_name.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved training plot to {save_path}")
        else:
            plt.show()

    def get_summary(self, last_n: int = 10) -> Dict:
        """Get summary statistics for last N episodes."""
        summary = {}
        for metric_name, values in self.data.items():
            if len(values) >= last_n:
                recent_values = values[-last_n:]
                summary[metric_name] = {
                    'mean': np.mean(recent_values),
                    'std': np.std(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values)
                }
        return summary


class MMatrixVisualizer:
    """
    Visualize the evolution of the M matrix (synaptic modulation) during an episode.

    Example:
        >>> viz = MMatrixVisualizer()
        >>> state = dqn.init_state(1)
        >>> for t in range(episode_length):
        >>>     q, state = dqn(obs, state)
        >>>     viz.record_step(state, obs, q)
        >>> viz.plot_evolution()
    """

    def __init__(self):
        self.M_history = []
        self.obs_history = []
        self.q_history = []
        self.action_history = []

    def record_step(self, M: torch.Tensor, obs: torch.Tensor,
                   q_values: torch.Tensor, action: Optional[int] = None):
        """
        Record state at current timestep.

        Args:
            M: M matrix state (batch_size, hidden_dim, obs_dim)
            obs: Observation (batch_size, obs_dim)
            q_values: Q-values (batch_size, action_dim)
            action: Selected action
        """
        self.M_history.append(M.detach().cpu().squeeze(0).numpy())
        self.obs_history.append(obs.detach().cpu().squeeze(0).numpy())
        self.q_history.append(q_values.detach().cpu().squeeze(0).numpy())
        if action is not None:
            self.action_history.append(action)

    def reset(self):
        """Clear recorded history."""
        self.M_history = []
        self.obs_history = []
        self.q_history = []
        self.action_history = []

    def plot_evolution(self, save_path: Optional[str] = None, max_timesteps: int = 50):
        """
        Plot M matrix evolution over time.

        Args:
            save_path: Path to save figure
            max_timesteps: Maximum timesteps to display
        """
        if len(self.M_history) == 0:
            print("No data recorded. Call record_step() during episode.")
            return

        M_array = np.array(self.M_history[:max_timesteps])  # [T, hidden_dim, obs_dim]
        T = M_array.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. M matrix heatmap over time (average across hidden dims)
        ax = axes[0, 0]
        M_mean = M_array.mean(axis=1)  # [T, obs_dim]
        im = ax.imshow(M_mean.T, aspect='auto', cmap='RdBu_r',
                      vmin=-np.abs(M_mean).max(), vmax=np.abs(M_mean).max())
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Input Dimension')
        ax.set_title('M Matrix Evolution (avg over hidden dims)')
        plt.colorbar(im, ax=ax, label='Modulation Strength')

        # 2. M matrix norm over time
        ax = axes[0, 1]
        M_norm = np.linalg.norm(M_array, axis=(1, 2))  # [T]
        ax.plot(M_norm, linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('||M||_F')
        ax.set_title('M Matrix Frobenius Norm')
        ax.grid(True, alpha=0.3)

        # 3. Q-values over time
        ax = axes[1, 0]
        q_array = np.array(self.q_history[:max_timesteps])  # [T, action_dim]
        for i in range(q_array.shape[1]):
            ax.plot(q_array[:, i], label=f'Action {i}', linewidth=2)
        if len(self.action_history) > 0:
            actions = self.action_history[:max_timesteps]
            ax.scatter(range(len(actions)),
                      [q_array[t, a] for t, a in enumerate(actions)],
                      color='red', s=50, zorder=10, label='Selected')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Q-value')
        ax.set_title('Q-values Over Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Observations over time
        ax = axes[1, 1]
        obs_array = np.array(self.obs_history[:max_timesteps])  # [T, obs_dim]
        for i in range(min(obs_array.shape[1], 4)):  # Show max 4 obs dims
            ax.plot(obs_array[:, i], label=f'Obs dim {i}', linewidth=2)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Observation Value')
        ax.set_title('Observations Over Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved M matrix evolution to {save_path}")
        else:
            plt.show()

    def plot_M_snapshot(self, timestep: int = -1, save_path: Optional[str] = None):
        """
        Plot M matrix at a specific timestep.

        Args:
            timestep: Timestep to visualize (-1 for last)
            save_path: Path to save figure
        """
        if len(self.M_history) == 0:
            print("No data recorded.")
            return

        M = self.M_history[timestep]  # [hidden_dim, obs_dim]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(M, aspect='auto', cmap='RdBu_r',
                      vmin=-np.abs(M).max(), vmax=np.abs(M).max())
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Hidden Dimension')
        ax.set_title(f'M Matrix at Timestep {timestep if timestep >= 0 else len(self.M_history)+timestep}')
        plt.colorbar(im, ax=ax, label='Modulation Strength')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved M matrix snapshot to {save_path}")
        else:
            plt.show()


def visualize_episode(dqn, env, max_steps: int = 500,
                     save_animation: Optional[str] = None,
                     save_M_plot: Optional[str] = None):
    """
    Run an episode and visualize both the environment and M matrix evolution.

    Args:
        dqn: Trained MPNDQN model
        env: Gym environment
        max_steps: Maximum steps per episode
        save_animation: Path to save environment animation (requires imageio)
        save_M_plot: Path to save M matrix evolution plot

    Returns:
        total_reward: Total episode reward
        M_viz: MMatrixVisualizer with recorded data
    """
    viz = MMatrixVisualizer()

    obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    frames = []
    total_reward = 0

    for step in range(max_steps):
        # Record for visualization
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)
            action = q_values.argmax(dim=1).item()

        viz.record_step(new_state, obs.unsqueeze(0), q_values, action)

        # Take step
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result

        # Record frame if saving animation
        if save_animation:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except:
                pass

        obs = torch.FloatTensor(next_obs)
        state = new_state
        total_reward += reward

        if done:
            break

    # Save animation if requested
    if save_animation and len(frames) > 0:
        imageio.mimsave(save_animation, frames, fps=30)
        print(f"Saved animation to {save_animation}")

    # Plot M matrix evolution
    if save_M_plot:
        viz.plot_evolution(save_path=save_M_plot)

    return total_reward, viz


def compare_models_visualization(results: Dict[str, List[float]],
                                 save_path: Optional[str] = None,
                                 window_size: int = 10):
    """
    Compare multiple models' training curves.

    Args:
        results: Dict mapping model names to reward lists
        save_path: Path to save figure
        window_size: Window for moving average

    Example:
        >>> results = {
        >>>     'MPN-DQN': mpn_rewards,
        >>>     'Standard DQN': standard_rewards,
        >>>     'LSTM-DQN': lstm_rewards
        >>> }
        >>> compare_models_visualization(results, 'comparison.png')
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    for model_name, rewards in results.items():
        episodes = np.arange(len(rewards))

        # Raw rewards (transparent)
        ax.plot(episodes, rewards, alpha=0.2)

        # Moving average
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            ax.plot(episodes[window_size-1:len(moving_avg)+window_size-1],
                   moving_avg, label=model_name, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Model Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved comparison plot to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("Testing visualization tools...")

    # Test TrainingVisualizer
    print("\n1. Testing TrainingVisualizer...")
    viz = TrainingVisualizer()

    # Simulate training
    for ep in range(100):
        reward = 50 + 30 * np.sin(ep/10) + np.random.randn()*10
        loss = 1.0 / (1 + ep/20) + np.random.randn()*0.1
        epsilon = max(0.01, 1.0 * 0.99**ep)

        viz.update(episode=ep, episode_reward=reward, loss=loss,
                  epsilon=epsilon, episode_length=ep+10)

    viz.plot()

    print("\nSummary (last 10 episodes):")
    summary = viz.get_summary(last_n=10)
    for metric, stats in summary.items():
        print(f"{metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    # Test MMatrixVisualizer
    print("\n2. Testing MMatrixVisualizer...")
    from mpn_dqn import MPNDQN

    dqn = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2)
    m_viz = MMatrixVisualizer()

    # Simulate episode
    state = dqn.init_state(1)
    for t in range(30):
        obs = torch.randn(1, 4)
        q_values, state = dqn(obs, state)
        action = q_values.argmax(dim=1).item()
        m_viz.record_step(state, obs, q_values, action)

    m_viz.plot_evolution()

    # Test comparison
    print("\n3. Testing model comparison...")
    results = {
        'Model A': 50 + 30*np.sin(np.arange(100)/10) + np.random.randn(100)*5,
        'Model B': 40 + 40*np.sin(np.arange(100)/10 + 1) + np.random.randn(100)*5,
    }
    compare_models_visualization(results)

    print("\nVisualization tests completed!")
