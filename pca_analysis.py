"""
PCA Analysis for MPN-DQN Hidden States and M Matrices

This module provides tools to analyze the dynamics of MPN-DQN agents by:
- Collecting hidden states and M matrices during episode rollouts
- Performing PCA on hidden states and flattened M matrices
- Visualizing trajectories in PC space
- Computing participation ratios to quantify dimensionality

Inspired by the analysis in the MPN paper (eLife-83035).
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, List, Dict, Tuple
import gymnasium as gym
from pathlib import Path


def participation_ratio(variances: np.ndarray) -> float:
    """
    Compute participation ratio from PCA explained variances.

    PR = (sum of variances)^2 / (sum of variances^2)

    This measures the effective dimensionality of the data.

    Args:
        variances: Array of PCA explained variances

    Returns:
        Participation ratio
    """
    return np.sum(variances) ** 2 / np.sum(variances ** 2)


class HiddenStateCollector:
    """
    Collects hidden states, M matrices, and observations during episode rollouts.

    Example:
        >>> collector = HiddenStateCollector()
        >>> for episode in range(n_episodes):
        >>>     obs, _ = env.reset()
        >>>     state = dqn.init_state(1)
        >>>     collector.start_episode()
        >>>     while not done:
        >>>         q, state = dqn(obs, state)
        >>>         collector.record_step(hidden, state, obs, action, reward)
        >>>         obs, reward, done = env.step(action)
        >>>     collector.end_episode()
        >>> data = collector.get_data()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all collected data."""
        # Episode-level data (list of episodes)
        self.episodes_hidden = []      # List of [T, hidden_dim]
        self.episodes_M = []            # List of [T, hidden_dim, obs_dim]
        self.episodes_obs = []          # List of [T, obs_dim]
        self.episodes_actions = []      # List of [T]
        self.episodes_rewards = []      # List of [T]
        self.episodes_lengths = []      # List of episode lengths

        # Current episode buffers
        self.current_hidden = []
        self.current_M = []
        self.current_obs = []
        self.current_actions = []
        self.current_rewards = []

    def start_episode(self):
        """Start collecting a new episode."""
        self.current_hidden = []
        self.current_M = []
        self.current_obs = []
        self.current_actions = []
        self.current_rewards = []

    def record_step(self, hidden: torch.Tensor, M_state: torch.Tensor,
                   obs: torch.Tensor, action: int, reward: float):
        """
        Record data from a single timestep.

        Args:
            hidden: Hidden state from MPN, shape [1, hidden_dim] or [hidden_dim]
            M_state: M matrix, shape [1, hidden_dim, obs_dim] or [hidden_dim, obs_dim]
            obs: Observation, shape [obs_dim]
            action: Action taken
            reward: Reward received
        """
        # Convert to numpy and squeeze batch dimension
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.detach().cpu().squeeze(0).numpy()
        if isinstance(M_state, torch.Tensor):
            M_state = M_state.detach().cpu().squeeze(0).numpy()
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        self.current_hidden.append(hidden)
        self.current_M.append(M_state)
        self.current_obs.append(obs)
        self.current_actions.append(action)
        self.current_rewards.append(reward)

    def end_episode(self):
        """Finalize the current episode and add to collection."""
        if len(self.current_hidden) > 0:
            self.episodes_hidden.append(np.array(self.current_hidden))
            self.episodes_M.append(np.array(self.current_M))
            self.episodes_obs.append(np.array(self.current_obs))
            self.episodes_actions.append(np.array(self.current_actions))
            self.episodes_rewards.append(np.array(self.current_rewards))
            self.episodes_lengths.append(len(self.current_hidden))

    def get_data(self) -> Dict:
        """
        Get all collected data as numpy arrays.

        Returns:
            Dictionary with keys:
                - 'hidden': [n_episodes, max_T, hidden_dim]
                - 'M': [n_episodes, max_T, hidden_dim, obs_dim]
                - 'obs': [n_episodes, max_T, obs_dim]
                - 'actions': [n_episodes, max_T]
                - 'rewards': [n_episodes, max_T]
                - 'lengths': [n_episodes]
        """
        return {
            'hidden': self.episodes_hidden,
            'M': self.episodes_M,
            'obs': self.episodes_obs,
            'actions': self.episodes_actions,
            'rewards': self.episodes_rewards,
            'lengths': np.array(self.episodes_lengths)
        }

    def get_pooled_data(self) -> Dict:
        """
        Get data pooled across all episodes (concatenated along time dimension).

        Returns:
            Dictionary with keys:
                - 'hidden': [total_timesteps, hidden_dim]
                - 'M_flat': [total_timesteps, hidden_dim * obs_dim]
                - 'obs': [total_timesteps, obs_dim]
                - 'episode_ids': [total_timesteps] - which episode each timestep belongs to
        """
        # Concatenate all episodes
        hidden_pooled = np.concatenate(self.episodes_hidden, axis=0)
        obs_pooled = np.concatenate(self.episodes_obs, axis=0)

        # Flatten M matrices: [T, hidden_dim, obs_dim] -> [T, hidden_dim * obs_dim]
        M_pooled = []
        for M_ep in self.episodes_M:
            M_flat = M_ep.reshape(M_ep.shape[0], -1)  # [T, hidden_dim * obs_dim]
            M_pooled.append(M_flat)
        M_pooled = np.concatenate(M_pooled, axis=0)

        # Create episode IDs for each timestep
        episode_ids = []
        for ep_idx, length in enumerate(self.episodes_lengths):
            episode_ids.extend([ep_idx] * length)
        episode_ids = np.array(episode_ids)

        return {
            'hidden': hidden_pooled,
            'M_flat': M_pooled,
            'obs': obs_pooled,
            'episode_ids': episode_ids
        }


class MPNPCAAnalyzer:
    """
    Performs PCA analysis on MPN hidden states and M matrices.

    Example:
        >>> analyzer = MPNPCAAnalyzer()
        >>> analyzer.fit_hidden_pca(hidden_states, n_components=10)
        >>> hidden_pcs = analyzer.transform_hidden(hidden_states)
        >>> analyzer.plot_trajectories(hidden_pcs, colors, save_path='plot.png')
    """

    def __init__(self):
        self.hidden_pca = None
        self.M_pca = None

    def fit_hidden_pca(self, hidden_states: np.ndarray, n_components: int = 100):
        """
        Fit PCA on hidden states.

        Args:
            hidden_states: Array of shape [n_samples, hidden_dim]
            n_components: Number of PCA components to keep
        """
        self.hidden_pca = PCA(n_components=n_components)
        self.hidden_pca.fit(hidden_states)

        print(f"Hidden State PCA fitted:")
        print(f"  Shape: {hidden_states.shape}")
        print(f"  Components: {n_components}")
        print(f"  Explained variance (top 10): {self.hidden_pca.explained_variance_[:10]}")
        print(f"  Participation ratio: {participation_ratio(self.hidden_pca.explained_variance_):.2f}")

    def fit_M_pca(self, M_flat: np.ndarray, n_components: int = 100):
        """
        Fit PCA on flattened M matrices.

        Args:
            M_flat: Array of shape [n_samples, hidden_dim * obs_dim]
            n_components: Number of PCA components to keep
        """
        self.M_pca = PCA(n_components=n_components)
        self.M_pca.fit(M_flat)

        print(f"\nM Matrix PCA fitted:")
        print(f"  Shape: {M_flat.shape}")
        print(f"  Components: {n_components}")
        print(f"  Explained variance (top 10): {self.M_pca.explained_variance_[:10]}")
        print(f"  Participation ratio: {participation_ratio(self.M_pca.explained_variance_):.2f}")

    def transform_hidden(self, hidden_states: np.ndarray) -> np.ndarray:
        """Transform hidden states to PC space."""
        if self.hidden_pca is None:
            raise ValueError("Must call fit_hidden_pca first")
        return self.hidden_pca.transform(hidden_states)

    def transform_M(self, M_flat: np.ndarray) -> np.ndarray:
        """Transform M matrices to PC space."""
        if self.M_pca is None:
            raise ValueError("Must call fit_M_pca first")
        return self.M_pca.transform(M_flat)

    def plot_variance_explained(self, save_path: Optional[str] = None, max_components: int = 50):
        """
        Plot explained variance for hidden states and M matrices.

        Args:
            save_path: Path to save figure
            max_components: Maximum number of components to show
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Hidden state variance
        if self.hidden_pca is not None:
            ax = axes[0]
            n_comp = min(max_components, len(self.hidden_pca.explained_variance_))
            ax.bar(range(n_comp), self.hidden_pca.explained_variance_[:n_comp],
                   color='steelblue', alpha=0.7)
            ax.set_xlabel('Principal Component', fontsize=12)
            ax.set_ylabel('Explained Variance', fontsize=12)
            ax.set_title(f'Hidden State PCA\nPR = {participation_ratio(self.hidden_pca.explained_variance_):.2f}',
                        fontsize=12)
            ax.grid(True, alpha=0.3)

        # M matrix variance
        if self.M_pca is not None:
            ax = axes[1]
            n_comp = min(max_components, len(self.M_pca.explained_variance_))
            ax.bar(range(n_comp), self.M_pca.explained_variance_[:n_comp],
                   color='coral', alpha=0.7)
            ax.set_xlabel('Principal Component', fontsize=12)
            ax.set_ylabel('Explained Variance', fontsize=12)
            ax.set_title(f'M Matrix PCA\nPR = {participation_ratio(self.M_pca.explained_variance_):.2f}',
                        fontsize=12)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved variance plot to {save_path}")
        else:
            plt.show()

        plt.close()


def collect_episodes(dqn, env, n_episodes: int, device: torch.device,
                     epsilon: float = 0.0, max_steps: int = 500) -> HiddenStateCollector:
    """
    Run episodes and collect hidden states, M matrices, and observations.

    Args:
        dqn: Trained MPNDQN model
        env: Gym environment
        n_episodes: Number of episodes to collect
        device: Device to run model on
        epsilon: Exploration rate (0 = greedy)
        max_steps: Maximum steps per episode

    Returns:
        HiddenStateCollector with collected data
    """
    collector = HiddenStateCollector()

    dqn.eval()

    for ep in range(n_episodes):
        # Reset environment
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs = torch.FloatTensor(obs).to(device)

        # Initialize state
        state = dqn.init_state(batch_size=1, device=device)

        collector.start_episode()

        episode_reward = 0
        for step in range(max_steps):
            with torch.no_grad():
                # Get Q-values and new state
                q_values, new_state = dqn(obs.unsqueeze(0), state)

                # Select action (epsilon-greedy)
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = q_values.argmax(dim=1).item()

                # Get hidden state from MPN layer
                hidden, _ = dqn.mpn(obs.unsqueeze(0), state)

            # Record before taking action
            collector.record_step(hidden, new_state, obs, action, 0)

            # Take step
            step_result = env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_obs, reward, done, _ = step_result

            # Update for next iteration
            obs = torch.FloatTensor(next_obs).to(device)
            state = new_state
            episode_reward += reward

            if done:
                break

        collector.end_episode()

        if (ep + 1) % 10 == 0:
            print(f"Collected {ep + 1}/{n_episodes} episodes, last reward: {episode_reward:.2f}")

    return collector


def plot_trajectories_2d(trajectories_pcs: List[np.ndarray],
                         colors: List[np.ndarray],
                         pc_pairs: List[Tuple[int, int]] = [(0, 1), (0, 2), (1, 2)],
                         title: str = "Hidden State PC Trajectories",
                         save_path: Optional[str] = None,
                         readout_vectors: Optional[np.ndarray] = None,
                         readout_pcs: Optional[np.ndarray] = None,
                         color_label: str = "State Feature"):
    """
    Plot 2D trajectories in principal component space.

    Args:
        trajectories_pcs: List of PC trajectories, each [T, n_components]
        colors: List of color values for each trajectory, each [T]
        pc_pairs: List of (pcx, pcy) tuples to plot
        title: Overall title
        save_path: Path to save figure
        readout_vectors: Readout weight matrix [n_actions, hidden_dim] (optional)
        readout_pcs: Readout vectors projected to PC space [n_actions, n_components] (optional)
        color_label: Label for colorbar
    """
    n_plots = len(pc_pairs)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 5))

    if n_plots == 1:
        axes = [axes]

    # Determine colormap limits across all episodes
    all_colors = np.concatenate(colors)
    vmin, vmax = np.percentile(all_colors, [5, 95])

    for ax, (pcx, pcy) in zip(axes, pc_pairs):
        # Plot all trajectories
        for traj_pc, traj_colors in zip(trajectories_pcs, colors):
            # Plot trajectory as line
            ax.plot(traj_pc[:, pcx], traj_pc[:, pcy],
                   alpha=0.3, linewidth=0.5, color='gray', zorder=1)

            # Scatter points colored by state feature
            scatter = ax.scatter(traj_pc[:, pcx], traj_pc[:, pcy],
                               c=traj_colors, cmap='viridis',
                               s=10, alpha=0.6, vmin=vmin, vmax=vmax, zorder=2)

            # Mark final point
            ax.scatter(traj_pc[-1, pcx], traj_pc[-1, pcy],
                      marker='o', s=50, color='red',
                      edgecolors='black', linewidths=1, zorder=3)

        # Plot readout vectors if provided
        if readout_pcs is not None:
            zero_point = np.zeros(readout_pcs.shape[1])
            for i, ro_pc in enumerate(readout_pcs):
                ax.arrow(0, 0, ro_pc[pcx], ro_pc[pcy],
                        head_width=0.3, head_length=0.3,
                        fc=f'C{i}', ec=f'C{i}', linewidth=2,
                        alpha=0.8, zorder=10, length_includes_head=True)
                ax.text(ro_pc[pcx]*1.1, ro_pc[pcy]*1.1, f'A{i}',
                       fontsize=10, fontweight='bold')

        ax.set_xlabel(f'PC{pcx}', fontsize=12)
        ax.set_ylabel(f'PC{pcy}', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, orientation='horizontal',
                       pad=0.1, fraction=0.05, aspect=40)
    cbar.set_label(color_label, fontsize=11)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory plot to {save_path}")
    else:
        plt.show()

    plt.close()


def get_cartpole_colors(observations: List[np.ndarray],
                       feature_idx: int = 0) -> List[np.ndarray]:
    """
    Extract color values from CartPole observations.

    CartPole observation space:
        0: Cart Position
        1: Cart Velocity
        2: Pole Angle
        3: Pole Angular Velocity

    Args:
        observations: List of observation arrays, each [T, 4]
        feature_idx: Which feature to use for coloring (0-3)

    Returns:
        List of color arrays, each [T]
    """
    return [obs[:, feature_idx] for obs in observations]


if __name__ == "__main__":
    print("Testing PCA analysis tools...")

    # Test participation ratio
    variances = np.array([10, 5, 3, 2, 1, 0.5, 0.3, 0.1])
    pr = participation_ratio(variances)
    print(f"Participation ratio: {pr:.2f}")

    # Test collector
    print("\nTesting HiddenStateCollector...")
    collector = HiddenStateCollector()

    # Simulate 3 episodes
    for ep in range(3):
        collector.start_episode()
        for t in range(10 + ep * 5):
            hidden = np.random.randn(8)
            M = np.random.randn(8, 4)
            obs = np.random.randn(4)
            collector.record_step(hidden, M, obs, action=0, reward=1.0)
        collector.end_episode()

    data = collector.get_data()
    print(f"Collected {len(data['hidden'])} episodes")
    print(f"Episode lengths: {data['lengths']}")

    pooled = collector.get_pooled_data()
    print(f"Pooled hidden shape: {pooled['hidden'].shape}")
    print(f"Pooled M_flat shape: {pooled['M_flat'].shape}")

    # Test PCA analyzer
    print("\nTesting MPNPCAAnalyzer...")
    analyzer = MPNPCAAnalyzer()
    analyzer.fit_hidden_pca(pooled['hidden'], n_components=8)
    analyzer.fit_M_pca(pooled['M_flat'], n_components=16)

    hidden_pcs = analyzer.transform_hidden(pooled['hidden'])
    print(f"Transformed hidden PCs shape: {hidden_pcs.shape}")

    print("\nPCA analysis test completed!")
