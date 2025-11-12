"""
Plot training metrics from saved JSON file.
"""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(metrics_path, output_path=None):
    """
    Plot training metrics including accuracy, misses, false alarms, and total trials.

    Args:
        metrics_path: Path to training_metrics.json
        output_path: Path to save plot (optional)
    """
    # Load metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Create figure with subplots
    fig, axs = plt.subplots(4, 2, figsize=(14, 14))
    fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

    # Subplot 1: Accuracy
    if metrics.get('eval_accuracies'):
        axs[0, 0].plot(metrics['eval_accuracies'], 'b-', linewidth=2)
        axs[0, 0].set_title('Evaluation Accuracy')
        axs[0, 0].set_xlabel('Evaluation Step')
        axs[0, 0].set_ylabel('Accuracy')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].set_ylim([0, 1])

    # Subplot 2: Total Trials
    if metrics.get('eval_total_trials'):
        axs[0, 1].plot(metrics['eval_total_trials'], 'g-', linewidth=2)
        axs[0, 1].set_title('Total Trials per Evaluation')
        axs[0, 1].set_xlabel('Evaluation Step')
        axs[0, 1].set_ylabel('Number of Trials')
        axs[0, 1].grid(True, alpha=0.3)

    # Subplot 3: True Positives
    if metrics.get('eval_true_positives'):
        axs[1, 0].plot(metrics['eval_true_positives'], 'green', linewidth=2)
        axs[1, 0].set_title('True Positives (Correct Go Responses)')
        axs[1, 0].set_xlabel('Evaluation Step')
        axs[1, 0].set_ylabel('Number of True Positives')
        axs[1, 0].grid(True, alpha=0.3)

    # Subplot 4: True Negatives
    if metrics.get('eval_true_negatives'):
        axs[1, 1].plot(metrics['eval_true_negatives'], 'blue', linewidth=2)
        axs[1, 1].set_title('True Negatives (Correct No-go Rejections)')
        axs[1, 1].set_xlabel('Evaluation Step')
        axs[1, 1].set_ylabel('Number of True Negatives')
        axs[1, 1].grid(True, alpha=0.3)

    # Subplot 5: False Negatives
    if metrics.get('eval_false_negatives'):
        axs[2, 0].plot(metrics['eval_false_negatives'], 'r-', linewidth=2)
        axs[2, 0].set_title('False Negatives (Missed Go Trials)')
        axs[2, 0].set_xlabel('Evaluation Step')
        axs[2, 0].set_ylabel('Number of False Negatives')
        axs[2, 0].grid(True, alpha=0.3)

    # Subplot 6: False Positives
    if metrics.get('eval_false_positives'):
        axs[2, 1].plot(metrics['eval_false_positives'], 'orange', linewidth=2)
        axs[2, 1].set_title('False Positives (Incorrect No-go Responses)')
        axs[2, 1].set_xlabel('Evaluation Step')
        axs[2, 1].set_ylabel('Number of False Positives')
        axs[2, 1].grid(True, alpha=0.3)

    # Subplot 7: Batch Rewards (moving average)
    if metrics.get('batch_rewards'):
        rewards = metrics['batch_rewards']
        # Calculate moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            axs[3, 0].plot(moving_avg, 'purple', linewidth=2)
        else:
            axs[3, 0].plot(rewards, 'purple', linewidth=2)
        axs[3, 0].set_title(f'Batch Rewards (Moving Avg, window={window})')
        axs[3, 0].set_xlabel('Batch')
        axs[3, 0].set_ylabel('Average Reward')
        axs[3, 0].grid(True, alpha=0.3)

    # Subplot 8: Batch Losses (moving average)
    if metrics.get('batch_losses'):
        losses = metrics['batch_losses']
        # Calculate moving average
        window = min(100, len(losses) // 10)
        if window > 1:
            moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
            axs[3, 1].plot(moving_avg, 'brown', linewidth=2)
        else:
            axs[3, 1].plot(losses, 'brown', linewidth=2)
        axs[3, 1].set_title(f'Batch Losses (Moving Avg, window={window})')
        axs[3, 1].set_xlabel('Batch')
        axs[3, 1].set_ylabel('Loss')
        axs[3, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument('--metrics-path', type=str, required=True,
                        help='Path to training_metrics.json')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: show plot)')

    args = parser.parse_args()

    plot_metrics(args.metrics_path, args.output)
