"""
Plot rewards and trajectories for random agents on the GoNogo task with stimulus visualization.

This script:
1. Runs multiple random agents on the GoNogo task
2. Plots the observations (stimuli) over time
3. Plots the rewards over time
4. Plots the cumulative reward trajectories
"""

import os
import sys
import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np
import torch
from torchrl.envs import Compose, StepCounter, InitTracker, TransformedEnv
from torchrl.envs.libs.gym import GymEnv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment with wrapper to capture ground truth
gymenv = GymEnv("GoNogo-v0", device=device)
gymenv._env = NeuroGymInfoWrapper(gymenv._env)

env = TransformedEnv(
    gymenv,
    Compose(
        NeuroGymInfoTransform(),
        StepCounter(),
        InitTracker(),
    )
)

print(f"Action space: {env.action_spec}")
print(f"Observation space: {env.observation_spec}")

# Test parameters
num_episodes = 5  # Number of episodes to visualize
episode_length = 200  # Steps per episode


def run_random_episode(env, episode_length, seed=None):
    """
    Run a single episode with random actions and collect all data.

    Returns:
        dict with observations, rewards, actions, ground_truth, new_trial
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    td = env.reset()

    observations = []
    rewards = []
    actions = []
    ground_truths = []
    new_trials = []

    for step in range(episode_length):
        # Random action
        action = torch.zeros(env.action_spec.shape, device=device)
        action_choice = np.random.randint(0, env.action_spec.shape[0])
        action[action_choice] = 1.0

        # Store pre-step data
        observations.append(td["observation"].cpu().numpy())
        if "ground_truth" in td.keys():
            ground_truths.append(td["ground_truth"].cpu().numpy())
        if "new_trial" in td.keys():
            new_trials.append(td["new_trial"].cpu().numpy())

        # Take step
        td["action"] = action
        td = env.step(td)

        # Get reward
        reward = td["next", "reward"].item()
        rewards.append(reward)
        actions.append(action_choice)

        # Check if done
        done = td["next", "done"].item() if "done" in td["next"].keys() else False
        terminated = td["next", "terminated"].item() if "terminated" in td["next"].keys() else False

        if done or terminated:
            td = env.reset()
        else:
            td = env.step_mdp(td)

    return {
        'observations': np.array(observations),
        'rewards': np.array(rewards),
        'actions': np.array(actions),
        'ground_truths': np.array(ground_truths) if ground_truths else None,
        'new_trials': np.array(new_trials) if new_trials else None,
    }


# Run multiple random episodes
print(f"\nRunning {num_episodes} random episodes...")
episodes_data = []

for ep_idx in range(num_episodes):
    print(f"  Episode {ep_idx + 1}/{num_episodes}...")
    episode_data = run_random_episode(env, episode_length, seed=ep_idx)
    total_reward = episode_data['rewards'].sum()
    print(f"    Total reward: {total_reward:.2f}")
    episodes_data.append(episode_data)

# Create visualization
print("\nCreating visualization...")

# Create a figure with subplots for each episode
fig = plt.figure(figsize=(16, 4 * num_episodes))

for ep_idx, episode_data in enumerate(episodes_data):
    obs = episode_data['observations']
    rewards = episode_data['rewards']
    actions = episode_data['actions']
    ground_truths = episode_data['ground_truths']
    new_trials = episode_data['new_trials']

    timesteps = np.arange(len(obs))
    n_obs_channels = obs.shape[1]

    # Create 3 subplots for this episode: observations+stimuli, rewards, cumulative rewards
    ax_obs = plt.subplot(num_episodes, 3, ep_idx * 3 + 1)
    ax_rew = plt.subplot(num_episodes, 3, ep_idx * 3 + 2)
    ax_cum = plt.subplot(num_episodes, 3, ep_idx * 3 + 3)

    # Plot 1: Observations (stimuli) with vertical offsets
    offset = 2.0
    colors = ['blue', 'red', 'green', 'purple']
    labels = ['Fixation', 'NoGo Stimulus', 'Go Stimulus']

    for i in range(min(n_obs_channels, 3)):  # Only plot first 3 channels (fixation, nogo, go)
        ax_obs.plot(timesteps, obs[:, i] + i * offset,
                   label=labels[i], linewidth=1.5, alpha=0.8, color=colors[i])

    # Highlight trial boundaries if available
    if new_trials is not None:
        trial_starts = np.where(new_trials > 0.5)[0]
        for trial_start in trial_starts:
            ax_obs.axvline(x=trial_start, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax_rew.axvline(x=trial_start, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax_cum.axvline(x=trial_start, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    ax_obs.set_xlabel('Timestep', fontsize=10)
    ax_obs.set_ylabel('Stimulus Value (offset)', fontsize=10)
    ax_obs.set_title(f'Episode {ep_idx + 1}: Stimuli', fontsize=11, fontweight='bold')
    ax_obs.legend(loc='upper right', fontsize=8)
    ax_obs.grid(True, alpha=0.3)

    # Plot 2: Rewards over time
    # Color-code rewards: positive (green), negative (red), zero (gray)
    reward_colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in rewards]
    ax_rew.bar(timesteps, rewards, color=reward_colors, alpha=0.6, width=1.0)
    ax_rew.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax_rew.set_xlabel('Timestep', fontsize=10)
    ax_rew.set_ylabel('Reward', fontsize=10)
    ax_rew.set_title(f'Episode {ep_idx + 1}: Rewards (Total: {rewards.sum():.2f})',
                     fontsize=11, fontweight='bold')
    ax_rew.grid(True, alpha=0.3, axis='y')

    # Plot 3: Cumulative reward trajectory
    cumulative_rewards = np.cumsum(rewards)
    ax_cum.plot(timesteps, cumulative_rewards, linewidth=2, color='darkblue')
    ax_cum.fill_between(timesteps, 0, cumulative_rewards, alpha=0.3, color='darkblue')
    ax_cum.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax_cum.set_xlabel('Timestep', fontsize=10)
    ax_cum.set_ylabel('Cumulative Reward', fontsize=10)
    ax_cum.set_title(f'Episode {ep_idx + 1}: Cumulative Trajectory', fontsize=11, fontweight='bold')
    ax_cum.grid(True, alpha=0.3)

plt.tight_layout()

# Save the figure
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gonogo_random_agents_trajectories.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to {output_path}")

# Also create a summary plot across all episodes
print("\nCreating summary plot...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: All cumulative reward trajectories
for ep_idx, episode_data in enumerate(episodes_data):
    rewards = episode_data['rewards']
    cumulative_rewards = np.cumsum(rewards)
    timesteps = np.arange(len(rewards))
    total_reward = rewards.sum()

    ax1.plot(timesteps, cumulative_rewards, linewidth=2, alpha=0.6,
            label=f'Episode {ep_idx + 1} (Total: {total_reward:.2f})')

# Compute mean trajectory
min_length = min(len(ep['rewards']) for ep in episodes_data)
aligned_rewards = np.array([ep['rewards'][:min_length] for ep in episodes_data])
mean_rewards = np.mean(aligned_rewards, axis=0)
mean_cumulative = np.cumsum(mean_rewards)

ax1.plot(np.arange(min_length), mean_cumulative, linewidth=3, color='black',
        linestyle='--', label=f'Mean (Total: {mean_cumulative[-1]:.2f})')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

ax1.set_xlabel('Timestep', fontsize=12)
ax1.set_ylabel('Cumulative Reward', fontsize=12)
ax1.set_title(f'Random Agent Cumulative Reward Trajectories ({num_episodes} episodes)',
             fontsize=14, fontweight='bold')
ax1.legend(fontsize=10, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of total rewards
total_rewards = [ep['rewards'].sum() for ep in episodes_data]
ax2.hist(total_rewards, bins=10, alpha=0.7, color='darkblue', edgecolor='black')
ax2.axvline(x=np.mean(total_rewards), color='red', linestyle='--', linewidth=2,
           label=f'Mean: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}')
ax2.set_xlabel('Total Episode Reward', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Total Rewards', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save summary plot
summary_output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'gonogo_random_agents_summary.png')
plt.savefig(summary_output_path, dpi=150, bbox_inches='tight')
print(f"Saved summary plot to {summary_output_path}")

# Print statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS:")
print("="*80)
print(f"Number of episodes: {num_episodes}")
print(f"Episode length: {episode_length} steps")
print(f"Total rewards: {total_rewards}")
print(f"Mean total reward: {np.mean(total_rewards):.3f} ± {np.std(total_rewards):.3f}")
print(f"Min total reward: {np.min(total_rewards):.3f}")
print(f"Max total reward: {np.max(total_rewards):.3f}")
print(f"Mean reward per step: {np.mean(total_rewards) / episode_length:.4f}")

print("\nVisualization complete!")
plt.show()
