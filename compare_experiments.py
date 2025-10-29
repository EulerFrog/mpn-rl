"""
Compare training performance across multiple experiments.

Usage:
    python compare_experiments.py --experiments exp1 exp2 exp3 --output comparison.png
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_history(experiment_name):
    """Load training history from experiment directory."""
    exp_dir = Path('experiments') / experiment_name
    history_path = exp_dir / 'training_history.json'

    if not history_path.exists():
        raise FileNotFoundError(f"Training history not found: {history_path}")

    with open(history_path, 'r') as f:
        data = json.load(f)

    return data


def load_config(experiment_name):
    """Load config from experiment directory."""
    exp_dir = Path('experiments') / experiment_name
    config_path = exp_dir / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def smooth_curve(values, window_size=10):
    """Apply moving average smoothing to values."""
    if len(values) < window_size:
        return values

    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return smoothed


def plot_comparison(experiments, labels=None, output_path='comparison.png',
                   window_size=10, show_raw=False, color_by='learning_rate'):
    """
    Plot training comparison across experiments with multiple metrics.

    Args:
        experiments: List of experiment names
        labels: List of labels for each experiment (defaults to experiment names)
        output_path: Path to save the plot
        window_size: Window size for moving average smoothing
        show_raw: Whether to show raw (unsmoothed) data as well
        color_by: What to color by ('learning_rate' or 'sequence_length')
    """
    if labels is None:
        labels = experiments

    # Load configs and extract parameters
    param_values = []
    for exp_name in experiments:
        config = load_config(exp_name)
        if color_by == 'sequence_length':
            param = config.get('sequence_length', 10)
        elif color_by == 'hidden_dim':
            param = config.get('hidden_dim', 128)
        else:  # learning_rate
            param = config.get('learning_rate', 0.001)
        param_values.append(param)

    # Create color mapping based on parameter
    if color_by == 'sequence_length':
        param_color_map = {
            25: '#1f77b4',   # Blue
            30: '#2ca02c',   # Green
            35: '#ff7f0e',   # Orange
            40: '#d62728',   # Red
        }
        param_name = "seq_len"
    elif color_by == 'hidden_dim':
        param_color_map = {
            64: '#1f77b4',    # Blue
            128: '#2ca02c',   # Green
            256: '#ff7f0e',   # Orange
            512: '#d62728',   # Red
        }
        param_name = "hidden"
    else:  # learning_rate
        param_color_map = {
            0.1: '#d62728',    # Red
            0.01: '#ff7f0e',   # Orange
            0.005: '#2ca02c',  # Green
            0.001: '#1f77b4',  # Blue
        }
        param_name = "lr"

    # Fallback colors if needed
    default_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    unique_params = sorted(set(param_values))
    for i, param in enumerate(unique_params):
        if param not in param_color_map:
            param_color_map[param] = default_colors[i % len(default_colors)]

    # Create 4-panel figure (rewards, lengths, losses, trials)
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    for i, (exp_name, label) in enumerate(zip(experiments, labels)):
        print(f"Loading {exp_name}...")
        data = load_training_history(exp_name)

        episodes = np.array(data['episodes'])
        rewards = np.array(data['rewards'])
        lengths = np.array(data['lengths'])
        losses = np.array(data['losses'])
        trials = np.array(data.get('trials', [0] * len(episodes)))  # Default to 0 if not present

        # Calculate reward/trials efficiency ratio (avoid division by zero)
        efficiency = np.where(trials > 0, rewards / trials, 0)

        # Get color based on parameter
        param = param_values[i]
        color = param_color_map[param]

        # Update label to include parameter
        label_with_param = f"{label} ({param_name}={param})"

        # Panel 1: Rewards
        if show_raw:
            axes[0].plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.5)

        if len(rewards) >= window_size:
            smoothed_rewards = smooth_curve(rewards, window_size)
            smoothed_episodes = episodes[window_size-1:]
            axes[0].plot(smoothed_episodes, smoothed_rewards, color=color,
                        label=label_with_param, linewidth=2)
        else:
            axes[0].plot(episodes, rewards, color=color, label=label_with_param, linewidth=2)

        # Panel 2: Episode Lengths (number of steps)
        if show_raw:
            axes[1].plot(episodes, lengths, color=color, alpha=0.15, linewidth=0.5)

        if len(lengths) >= window_size:
            smoothed_lengths = smooth_curve(lengths, window_size)
            axes[1].plot(smoothed_episodes, smoothed_lengths, color=color,
                        label=label_with_param, linewidth=2)
        else:
            axes[1].plot(episodes, lengths, color=color, label=label_with_param, linewidth=2)

        # Panel 3: Losses
        if show_raw:
            axes[2].plot(episodes, losses, color=color, alpha=0.15, linewidth=0.5)

        if len(losses) >= window_size:
            smoothed_losses = smooth_curve(losses, window_size)
            axes[2].plot(smoothed_episodes, smoothed_losses, color=color,
                        label=label_with_param, linewidth=2)
        else:
            axes[2].plot(episodes, losses, color=color, label=label_with_param, linewidth=2)

        # Panel 4: Reward/Trials Efficiency
        if show_raw:
            axes[3].plot(episodes, efficiency, color=color, alpha=0.15, linewidth=0.5)

        if len(efficiency) >= window_size:
            smoothed_efficiency = smooth_curve(efficiency, window_size)
            axes[3].plot(smoothed_episodes, smoothed_efficiency, color=color,
                        label=label_with_param, linewidth=2)
        else:
            axes[3].plot(episodes, efficiency, color=color, label=label_with_param, linewidth=2)

    # Create color legend text
    if color_by == 'sequence_length':
        param_legend_display = "Colors: Blue=25, Green=30, Orange=35, Red=40"
    elif color_by == 'hidden_dim':
        param_legend_display = "Colors: Blue=64, Green=128, Orange=256, Red=512"
    else:
        param_legend_display = "Colors: Blue=0.001, Green=0.005, Orange=0.01, Red=0.1"

    # Configure panel 1: Rewards
    axes[0].set_xlabel('Episode', fontsize=11)
    axes[0].set_ylabel('Reward', fontsize=11)
    axes[0].set_title(f'Training Rewards (Smoothed, window={window_size}) - {param_legend_display}', fontsize=12)
    axes[0].legend(fontsize=8, loc='best', ncol=2)
    axes[0].grid(True, alpha=0.3)

    # Configure panel 2: Lengths
    axes[1].set_xlabel('Episode', fontsize=11)
    axes[1].set_ylabel('Episode Length (steps)', fontsize=11)
    axes[1].set_title(f'Episode Lengths (Smoothed, window={window_size})', fontsize=13)
    axes[1].legend(fontsize=8, loc='best', ncol=2)
    axes[1].grid(True, alpha=0.3)

    # Configure panel 3: Losses
    axes[2].set_xlabel('Episode', fontsize=11)
    axes[2].set_ylabel('TD Loss', fontsize=11)
    axes[2].set_title(f'Training Loss (Smoothed, window={window_size})', fontsize=13)
    axes[2].legend(fontsize=8, loc='best', ncol=2)
    axes[2].grid(True, alpha=0.3)

    # Configure panel 4: Reward/Trials Efficiency
    axes[3].set_xlabel('Episode', fontsize=11)
    axes[3].set_ylabel('Reward/Trials Ratio', fontsize=11)
    axes[3].set_title(f'Task Efficiency: Reward/Trials (Smoothed, window={window_size})', fontsize=13)
    axes[3].legend(fontsize=8, loc='best', ncol=2)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print statistics
    if color_by == 'sequence_length':
        param_header = 'SeqLen'
    elif color_by == 'hidden_dim':
        param_header = 'Hidden'
    else:
        param_header = 'LR'

    print("\nFinal Performance (last 20 episodes):")
    print("-" * 120)
    print(f"{'Experiment':<30s} {param_header:>8s} {'Reward':>12s} {'Length':>12s} {'Loss':>12s} {'Trials':>12s} {'R/T Ratio':>12s}")
    print("-" * 120)
    for exp_name, label, param in zip(experiments, labels, param_values):
        data = load_training_history(exp_name)
        rewards = np.array(data['rewards'])
        lengths = np.array(data['lengths'])
        losses = np.array(data['losses'])
        trials = np.array(data.get('trials', [0] * len(rewards)))

        reward_mean = np.mean(rewards[-20:])
        reward_std = np.std(rewards[-20:])
        length_mean = np.mean(lengths[-20:])
        loss_mean = np.mean(losses[-20:])
        trials_mean = np.mean(trials[-20:]) if len(trials) > 0 else 0.0

        # Calculate efficiency (reward/trials ratio)
        efficiency_mean = reward_mean / trials_mean if trials_mean > 0 else 0.0

        if color_by in ['sequence_length', 'hidden_dim']:
            print(f"{label:<30s} {param:8d} {reward_mean:7.2f}±{reward_std:5.2f} "
                  f"{length_mean:9.1f} {loss_mean:11.4f} {trials_mean:11.1f} {efficiency_mean:11.3f}")
        else:
            print(f"{label:<30s} {param:8.4f} {reward_mean:7.2f}±{reward_std:5.2f} "
                  f"{length_mean:9.1f} {loss_mean:11.4f} {trials_mean:11.1f} {efficiency_mean:11.3f}")
    print("-" * 120)


def main():
    parser = argparse.ArgumentParser(description='Compare training performance across experiments')
    parser.add_argument('--experiments', type=str, nargs='+', required=True,
                       help='List of experiment names to compare')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                       help='Custom labels for experiments (optional)')
    parser.add_argument('--output', type=str, default='comparison.png',
                       help='Output plot filename')
    parser.add_argument('--window', type=int, default=10,
                       help='Smoothing window size (default: 10)')
    parser.add_argument('--show-raw', action='store_true',
                       help='Show raw (unsmoothed) data in background')
    parser.add_argument('--color-by', type=str, default='learning_rate',
                       choices=['learning_rate', 'sequence_length', 'hidden_dim'],
                       help='What parameter to use for coloring (default: learning_rate)')

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.experiments):
        raise ValueError("Number of labels must match number of experiments")

    plot_comparison(
        experiments=args.experiments,
        labels=args.labels,
        output_path=args.output,
        window_size=args.window,
        show_raw=args.show_raw,
        color_by=args.color_by
    )


if __name__ == '__main__':
    main()
