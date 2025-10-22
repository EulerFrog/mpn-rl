"""
Evaluate multiple models across different seeds for fair comparison.

Usage:
    python evaluate_multiple_seeds.py --experiments longer-test-mpn longer-test-frozen longer-test-rnn --num-seeds 5
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from mpn_dqn import MPNDQN
from rnn_dqn import RNNDQN
from model_utils import ExperimentManager
from neurogym_wrapper import make_neurogym_env


def evaluate_episode(dqn, env, seed, max_steps=500):
    """
    Evaluate a single episode with a given seed.

    Returns:
        total_reward: Episode reward
    """
    # Seed everything
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    # Reset
    obs, _ = env.reset()
    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    total_reward = 0

    for step in range(max_steps):
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)
            action = q_values.argmax(dim=1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs = torch.FloatTensor(next_obs)
        state = new_state
        total_reward += reward

        if done:
            break

    return total_reward


def load_model(experiment_name, device='cpu'):
    """Load model from experiment."""
    exp_manager = ExperimentManager(experiment_name)
    config = exp_manager.load_config()

    # Detect model type
    model_type = config.get('model_type', 'mpn')
    use_rnn = (model_type == 'rnn')
    freeze_plasticity = (model_type == 'mpn-frozen')

    # Create environment to get dimensions
    env_name = config['env_name']
    env = make_neurogym_env(env_name, max_episode_steps=10)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    # Create model
    if use_rnn:
        dqn = RNNDQN(
            obs_dim=obs_dim,
            hidden_dim=config['hidden_dim'],
            action_dim=action_dim,
            activation=config.get('activation', 'tanh')
        ).to(device)
    else:
        dqn = MPNDQN(
            obs_dim=obs_dim,
            hidden_dim=config['hidden_dim'],
            action_dim=action_dim,
            eta=config.get('eta', 0.1),
            lambda_decay=config.get('lambda_decay', 0.95),
            activation=config.get('activation', 'tanh'),
            freeze_plasticity=freeze_plasticity
        ).to(device)

    # Load checkpoint
    checkpoint_path = exp_manager.checkpoint_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dqn.load_state_dict(checkpoint['model_state_dict'])
    dqn.eval()

    return dqn, config, model_type


def evaluate_multiple_seeds(experiments, seeds, max_steps=500, device='cpu'):
    """
    Evaluate multiple experiments across multiple seeds.

    Returns:
        results: Dict mapping experiment_name to list of rewards
    """
    results = {}

    for exp_name in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp_name}")
        print(f"{'='*60}")

        # Load model
        dqn, config, model_type = load_model(exp_name, device)
        print(f"Model type: {model_type.upper()}")

        # Create environment
        env = make_neurogym_env(config['env_name'], max_episode_steps=max_steps)

        # Evaluate across seeds
        rewards = []
        for seed in seeds:
            reward = evaluate_episode(dqn, env, seed, max_steps)
            rewards.append(reward)
            print(f"  Seed {seed:3d}: Reward = {reward:6.2f}")

        env.close()

        # Store results
        results[exp_name] = rewards
        print(f"  Mean: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    return results


def plot_comparison(results, experiment_names, labels, output_path='seed_comparison.png'):
    """
    Plot bar chart comparing models across seeds.

    Args:
        results: Dict mapping experiment_name to list of rewards
        experiment_names: List of experiment names (for ordering)
        labels: List of labels for each experiment
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute statistics
    means = []
    stds = []
    for exp_name in experiment_names:
        rewards = results[exp_name]
        means.append(np.mean(rewards))
        stds.append(np.std(rewards))

    # Create bar plot
    x = np.arange(len(labels))
    width = 0.6

    bars = ax.bar(x, means, width, yerr=stds, capsize=10,
                   alpha=0.8, color=['#2ca02c', '#ff7f0e', '#1f77b4'])

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Episode Reward', fontsize=13)
    ax.set_title('Model Comparison Across Multiple Seeds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Model':<30s} {'Mean':<10s} {'Std':<10s} {'Min':<10s} {'Max':<10s}")
    print("-"*60)
    for exp_name, label in zip(experiment_names, labels):
        rewards = results[exp_name]
        print(f"{label:<30s} {np.mean(rewards):>9.2f} {np.std(rewards):>9.2f} "
              f"{np.min(rewards):>9.2f} {np.max(rewards):>9.2f}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate models across multiple seeds')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                       help='List of experiment names to evaluate')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                       help='Custom labels for experiments (optional)')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of seeds to test (default: 5)')
    parser.add_argument('--seed-start', type=int, default=42,
                       help='Starting seed value (default: 42)')
    parser.add_argument('--max-steps', type=int, default=500,
                       help='Maximum steps per episode (default: 500)')
    parser.add_argument('--output', type=str, default='seed_comparison.png',
                       help='Output plot path')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (default: cpu)')

    args = parser.parse_args()

    # Generate seeds
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    print(f"Testing seeds: {seeds}")

    # Default labels
    if args.labels is None:
        labels = args.experiments
    else:
        if len(args.labels) != len(args.experiments):
            raise ValueError("Number of labels must match number of experiments")
        labels = args.labels

    # Evaluate
    results = evaluate_multiple_seeds(
        args.experiments,
        seeds,
        max_steps=args.max_steps,
        device=args.device
    )

    # Plot
    plot_comparison(results, args.experiments, labels, args.output)


if __name__ == '__main__':
    main()
