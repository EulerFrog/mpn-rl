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


def smooth_curve(values, window_size=10):
    """Apply moving average smoothing to values."""
    if len(values) < window_size:
        return values

    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    return smoothed


def plot_comparison(experiments, labels=None, output_path='comparison.png',
                   window_size=10, show_raw=False):
    """
    Plot training rewards comparison across experiments.

    Args:
        experiments: List of experiment names
        labels: List of labels for each experiment (defaults to experiment names)
        output_path: Path to save the plot
        window_size: Window size for moving average smoothing
        show_raw: Whether to show raw (unsmoothed) data as well
    """
    if labels is None:
        labels = experiments

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, (exp_name, label) in enumerate(zip(experiments, labels)):
        print(f"Loading {exp_name}...")
        data = load_training_history(exp_name)

        episodes = np.array(data['episodes'])
        rewards = np.array(data['rewards'])

        color = colors[i % len(colors)]

        # Plot raw data with transparency
        if show_raw:
            ax.plot(episodes, rewards, color=color, alpha=0.2, linewidth=0.5)

        # Plot smoothed data
        if len(rewards) >= window_size:
            smoothed_rewards = smooth_curve(rewards, window_size)
            smoothed_episodes = episodes[window_size-1:]
            ax.plot(smoothed_episodes, smoothed_rewards, color=color,
                   label=label, linewidth=2)
        else:
            ax.plot(episodes, rewards, color=color, label=label, linewidth=2)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Reward', fontsize=12)
    ax.set_title(f'Training Comparison (Smoothed with window={window_size})', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print statistics
    print("\nFinal Performance (last 20 episodes):")
    print("-" * 60)
    for exp_name, label in zip(experiments, labels):
        data = load_training_history(exp_name)
        rewards = np.array(data['rewards'])
        final_mean = np.mean(rewards[-20:])
        final_std = np.std(rewards[-20:])
        print(f"{label:20s}: {final_mean:7.2f} ± {final_std:.2f}")

    plt.show()


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

    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.experiments):
        raise ValueError("Number of labels must match number of experiments")

    plot_comparison(
        experiments=args.experiments,
        labels=args.labels,
        output_path=args.output,
        window_size=args.window,
        show_raw=args.show_raw
    )


if __name__ == '__main__':
    main()
