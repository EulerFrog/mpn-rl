# Experiment Guide for MPN-DQN

This guide provides practical examples for conducting experiments with Multi-Plasticity Networks in reinforcement learning.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Experiment Workflow](#basic-experiment-workflow)
- [Experiment Scenarios](#experiment-scenarios)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Comparing Models](#comparing-models)
- [Analyzing Results](#analyzing-results)
- [Reproducibility](#reproducibility)
- [Best Practices](#best-practices)

---

## Quick Start

### Your First Experiment

Train an MPN-DQN agent on CartPole with default settings:

```bash
# Start training
python main.py train --experiment-name first-experiment \
    --num-episodes 500 \
    --plot-training

# Monitor the experiment directory
ls experiments/first-experiment/
```

After training completes:
```bash
# Evaluate the trained agent
python main.py eval --experiment-name first-experiment --num-eval-episodes 20

# Create a visualization
python main.py render --experiment-name first-experiment --output first_demo.gif
```

### GPU Acceleration

MPN-DQN automatically uses GPU if available. Check GPU status:

```bash
# Test GPU setup
python test_gpu.py

# Train with GPU (automatic detection)
python main.py train --experiment-name gpu-experiment

# Force CPU (useful for debugging)
python main.py train --experiment-name cpu-experiment --device cpu
```

**Speed comparison (CartPole, 500 episodes):**
- CPU: ~5-10 minutes
- GPU (NVIDIA RTX): ~1-2 minutes

For large hyperparameter sweeps, GPU can provide 5-10x speedup.

---

## Basic Experiment Workflow

### 1. Design Your Experiment

Before running experiments, define:
- **Research question**: What are you testing? (e.g., "Does higher eta improve learning speed?")
- **Metrics**: How will you measure success? (avg reward, convergence time, final performance)
- **Baseline**: What are you comparing against? (standard DQN, different hyperparameters)
- **Variables**: Which parameters will you vary?

### 2. Run Training

```bash
# Train with specific configuration
python main.py train \
    --experiment-name exp-eta-0.05 \
    --num-episodes 1000 \
    --eta 0.05 \
    --lambda-decay 0.9 \
    --hidden-dim 64 \
    --checkpoint-freq 100 \
    --plot-training
```

### 3. Monitor Progress

Check training progress in real-time:
```bash
# View training history
cat experiments/exp-eta-0.05/training_history.json | python -m json.tool | head -30

# Check latest checkpoints
ls -lh experiments/exp-eta-0.05/checkpoints/
```

### 4. Evaluate Performance

```bash
# Evaluate best model
python main.py eval \
    --experiment-name exp-eta-0.05 \
    --num-eval-episodes 50 \
    --checkpoint best_model.pt

# Evaluate at specific checkpoint (e.g., mid-training)
python main.py eval \
    --experiment-name exp-eta-0.05 \
    --checkpoint checkpoint_500.pt \
    --num-eval-episodes 20
```

### 5. Visualize and Analyze

```bash
# Create GIF of agent behavior
python main.py render \
    --experiment-name exp-eta-0.05 \
    --output visualizations/eta_0.05_demo.gif

# Training curves are auto-saved at:
# experiments/exp-eta-0.05/plots/training_curves.png
```

---

## Experiment Scenarios

### Scenario 1: Hyperparameter Sweep

**Goal**: Find optimal MPN parameters (eta and lambda)

```bash
#!/bin/bash
# sweep_mpn_params.sh

# Test different eta values
for eta in 0.01 0.05 0.1 0.2; do
    python main.py train \
        --experiment-name sweep-eta-${eta} \
        --eta ${eta} \
        --lambda-decay 0.9 \
        --num-episodes 500 \
        --checkpoint-freq 50
done

# Test different lambda values
for lambda in 0.8 0.9 0.95 0.99; do
    python main.py train \
        --experiment-name sweep-lambda-${lambda} \
        --eta 0.05 \
        --lambda-decay ${lambda} \
        --num-episodes 500 \
        --checkpoint-freq 50
done
```

After running:
```python
# analyze_sweep.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

experiments = [
    'sweep-eta-0.01',
    'sweep-eta-0.05',
    'sweep-eta-0.1',
    'sweep-eta-0.2'
]

plt.figure(figsize=(12, 6))

for exp_name in experiments:
    history_path = Path(f'experiments/{exp_name}/training_history.json')
    with open(history_path) as f:
        history = json.load(f)

    rewards = history['rewards']
    # Moving average
    window = 10
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')

    plt.plot(smoothed, label=exp_name, linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Reward (smoothed)')
plt.title('Eta Hyperparameter Sweep')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('eta_sweep_comparison.png', dpi=150)
print("Saved comparison to eta_sweep_comparison.png")
```

### Scenario 2: MPN vs Standard DQN

**Goal**: Compare MPN-DQN with non-recurrent DQN

```bash
# Train MPN-DQN
python main.py train \
    --experiment-name mpn-vs-standard-mpn \
    --eta 0.05 \
    --lambda-decay 0.9 \
    --hidden-dim 64 \
    --num-episodes 1000

# Train baseline (MPN with eta=0, lambda=0 acts like feedforward)
python main.py train \
    --experiment-name mpn-vs-standard-baseline \
    --eta 0.0 \
    --lambda-decay 0.0 \
    --hidden-dim 64 \
    --num-episodes 1000
```

Compare using the programmatic API:
```python
# compare_mpn_baseline.py
from visualize import compare_models_visualization
import json

# Load results
results = {}
for name in ['mpn-vs-standard-mpn', 'mpn-vs-standard-baseline']:
    with open(f'experiments/{name}/training_history.json') as f:
        history = json.load(f)
        results[name] = history['rewards']

# Visualize
compare_models_visualization(results, save_path='mpn_vs_baseline.png')
```

### Scenario 3: Testing Different Network Sizes

**Goal**: Determine optimal hidden layer size

```bash
#!/bin/bash
# sweep_hidden_dim.sh

for hidden in 16 32 64 128 256; do
    python main.py train \
        --experiment-name hidden-dim-${hidden} \
        --hidden-dim ${hidden} \
        --eta 0.05 \
        --lambda-decay 0.9 \
        --num-episodes 500
done
```

### Scenario 4: Long Training Run with Monitoring

**Goal**: Train for extended period with periodic GIF renders

```bash
# Long training with GIF renders every 10 minutes
python main.py train \
    --experiment-name long-run-cartpole \
    --num-episodes 5000 \
    --render-freq-mins 10 \
    --checkpoint-freq 100 \
    --plot-training \
    --eta 0.05 \
    --lambda-decay 0.9
```

Check GIFs during training:
```bash
# View generated GIFs
ls -lh experiments/long-run-cartpole/videos/

# Count frames in latest GIF (longer episodes = better performance)
# (requires imageio)
python -c "import imageio; print(len(imageio.imread('experiments/long-run-cartpole/videos/episode_00500.gif')))"
```

### Scenario 5: Resume Interrupted Training

**Goal**: Continue training after interruption or to extend training

```bash
# Initial training
python main.py train \
    --experiment-name resumable-experiment \
    --num-episodes 500

# Training interrupted or you want more episodes
# Resume and train for 500 more episodes
python main.py resume \
    --experiment-name resumable-experiment \
    --num-episodes 500

# Training history is automatically appended
# Episode numbers continue from where they left off
```

---

## Hyperparameter Tuning

### Key MPN Hyperparameters

| Parameter | Typical Range | Effect | Tuning Strategy |
|-----------|---------------|--------|-----------------|
| `eta` | 0.01 - 0.2 | Hebbian learning rate | Start at 0.05, increase if learning too slow |
| `lambda_decay` | 0.8 - 0.99 | M matrix memory | Higher = longer memory, start at 0.9 |
| `hidden_dim` | 32 - 256 | Network capacity | Match task complexity |
| `activation` | tanh, relu, sigmoid | Nonlinearity | tanh generally good for MPNs |

### Standard RL Hyperparameters

| Parameter | Typical Range | Effect | Tuning Strategy |
|-----------|---------------|--------|-----------------|
| `learning_rate` | 1e-4 - 1e-2 | Optimization speed | 1e-3 is good default |
| `gamma` | 0.95 - 0.999 | Discount factor | Higher for long-horizon tasks |
| `epsilon_decay` | 0.99 - 0.999 | Exploration schedule | Slower decay for harder tasks |
| `batch_size` | 32 - 128 | Training stability | Larger = more stable but slower |
| `buffer_size` | 10k - 100k | Experience diversity | Larger for complex tasks |

### Recommended Starting Points

**For CartPole:**
```bash
python main.py train \
    --eta 0.05 \
    --lambda-decay 0.9 \
    --hidden-dim 64 \
    --learning-rate 0.001 \
    --gamma 0.99 \
    --epsilon-decay 0.995
```

**For more complex environments:**
```bash
python main.py train \
    --eta 0.02 \
    --lambda-decay 0.95 \
    --hidden-dim 128 \
    --learning-rate 0.0005 \
    --gamma 0.99 \
    --buffer-size 50000
```

---

## Comparing Models

### Using Python API

Create a comparison script:

```python
# compare_experiments.py
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_experiment(exp_name):
    """Load experiment data."""
    exp_dir = Path('experiments') / exp_name

    with open(exp_dir / 'config.json') as f:
        config = json.load(f)

    with open(exp_dir / 'training_history.json') as f:
        history = json.load(f)

    return config, history

def compare_experiments(exp_names, metric='rewards', window=10):
    """Compare multiple experiments."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for exp_name in exp_names:
        config, history = load_experiment(exp_name)

        data = history[metric]
        episodes = history['episodes']

        # Moving average
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        episodes_smooth = episodes[window-1:len(smoothed)+window-1]

        # Extract key params for label
        eta = config.get('eta', 'N/A')
        lam = config.get('lambda_decay', 'N/A')
        label = f"{exp_name} (η={eta}, λ={lam})"

        ax.plot(episodes_smooth, smoothed, label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Experiment Comparison', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('experiment_comparison.png', dpi=150)
    print("Saved comparison to experiment_comparison.png")

    # Print summary statistics
    print("\nSummary Statistics (last 50 episodes):")
    print("-" * 80)
    for exp_name in exp_names:
        _, history = load_experiment(exp_name)
        recent = history[metric][-50:]
        print(f"{exp_name:40s} | Mean: {np.mean(recent):7.2f} | Std: {np.std(recent):6.2f}")

# Example usage
if __name__ == "__main__":
    experiments = [
        'sweep-eta-0.01',
        'sweep-eta-0.05',
        'sweep-eta-0.1',
        'sweep-eta-0.2'
    ]

    compare_experiments(experiments, metric='rewards', window=20)
```

Run it:
```bash
python compare_experiments.py
```

---

## Analyzing Results

### Extracting Metrics

```python
# analyze_results.py
import json
import numpy as np
from pathlib import Path

def analyze_experiment(exp_name):
    """Analyze a single experiment."""
    exp_dir = Path('experiments') / exp_name

    with open(exp_dir / 'training_history.json') as f:
        history = json.load(f)

    rewards = np.array(history['rewards'])
    lengths = np.array(history['episodes_lengths']) if 'episode_lengths' in history else None

    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")

    # Learning curve analysis
    print("\nLearning Curve:")
    for interval in [50, 100, -50]:
        if interval > 0:
            data = rewards[:interval]
            label = f"First {interval} episodes"
        else:
            data = rewards[interval:]
            label = f"Last {-interval} episodes"

        print(f"  {label:30s}: {np.mean(data):7.2f} ± {np.std(data):6.2f}")

    # Convergence
    window = 10
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    max_reward = np.max(smoothed)
    max_episode = np.argmax(smoothed) + window - 1

    print(f"\n  Peak performance: {max_reward:.2f} at episode {max_episode}")

    # Stability (last 100 episodes)
    if len(rewards) >= 100:
        recent_std = np.std(rewards[-100:])
        print(f"  Recent stability (std): {recent_std:.2f}")

    # Sample efficiency (episodes to reach threshold)
    threshold = 195 if 'cartpole' in exp_name.lower() else max_reward * 0.9
    above_threshold = smoothed >= threshold
    if np.any(above_threshold):
        first_success = np.argmax(above_threshold) + window - 1
        print(f"  First reached {threshold:.0f} reward: episode {first_success}")

    return {
        'mean_final': np.mean(rewards[-50:]),
        'max_smoothed': max_reward,
        'convergence_episode': max_episode,
    }

# Analyze all experiments
experiments = [
    'sweep-eta-0.01',
    'sweep-eta-0.05',
    'sweep-eta-0.1',
]

results = {}
for exp in experiments:
    if Path(f'experiments/{exp}').exists():
        results[exp] = analyze_experiment(exp)

# Rank by performance
print(f"\n{'='*60}")
print("Ranking by Final Performance:")
print(f"{'='*60}")
ranked = sorted(results.items(), key=lambda x: x[1]['mean_final'], reverse=True)
for i, (name, metrics) in enumerate(ranked, 1):
    print(f"{i}. {name:40s} | Final: {metrics['mean_final']:7.2f}")
```

### Visualizing M Matrix Evolution

```python
# analyze_m_matrix.py
from visualize import visualize_episode
import gymnasium as gym
from mpn_dqn import MPNDQN
from model_utils import ExperimentManager, load_checkpoint_for_eval

def analyze_m_matrix_dynamics(exp_name):
    """Analyze M matrix evolution for an experiment."""
    # Load experiment
    exp_manager = ExperimentManager(exp_name)
    config = exp_manager.load_config()

    # Create environment and model
    env = gym.make(config['env_name'], render_mode='rgb_array')
    dqn = MPNDQN(
        obs_dim=env.observation_space.shape[0],
        hidden_dim=config['hidden_dim'],
        action_dim=env.action_space.n,
        eta=config['eta'],
        lambda_decay=config['lambda_decay']
    )

    # Load best model
    load_checkpoint_for_eval(exp_manager, dqn, 'best_model.pt')

    # Visualize episode
    save_path = f"{exp_name}_m_matrix_analysis.png"
    reward, viz = visualize_episode(dqn, env, max_steps=500, save_M_plot=save_path)

    env.close()

    print(f"M matrix analysis saved to: {save_path}")
    print(f"Episode reward: {reward:.2f}")

# Example
analyze_m_matrix_dynamics('sweep-eta-0.05')
```

---

## Reproducibility

### Ensuring Reproducible Experiments

1. **Save configuration**:
   - Configs are automatically saved to `experiments/{name}/config.json`
   - This captures all hyperparameters

2. **Set random seeds** (add to main.py if needed):
```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
```

3. **Document environment**:
```bash
# Save package versions
pip list > experiments/{exp-name}/requirements.txt

# Save git commit
git rev-parse HEAD > experiments/{exp-name}/git_commit.txt
```

4. **Run multiple seeds**:
```bash
#!/bin/bash
# run_with_seeds.sh

for seed in 42 123 456 789 1011; do
    python main.py train \
        --experiment-name eta-0.05-seed-${seed} \
        --eta 0.05 \
        --num-episodes 500
done
```

---

## Best Practices

### 1. Start Simple
- Begin with default hyperparameters
- Use CartPole-v1 for initial testing
- Run short experiments first (100-200 episodes)

### 2. Monitor During Training
- Check training curves regularly
- Watch for divergence or plateau
- Use GIF renders to verify agent is learning

### 3. Keep Organized
- Use descriptive experiment names
- Document what each experiment tests
- Keep a lab notebook or README in experiments/

### 4. Compare Fairly
- Run same number of episodes
- Use multiple random seeds
- Compare on same environment

### 5. Save Everything
- Checkpoint frequently
- Save training curves
- Keep GIF visualizations

### 6. Analyze Thoroughly
- Look at learning curves
- Check final performance
- Examine M matrix dynamics
- Test generalization

### 7. Version Control
```bash
# Initialize git in experiments directory
cd experiments
git init
git add */config.json */training_history.json
git commit -m "Experiment results for eta sweep"
```

---

## Example Research Questions

Here are research questions you can investigate:

1. **Does MPN improve sample efficiency?**
   - Compare MPN vs no plasticity (eta=0)
   - Measure episodes to reach threshold

2. **How does eta affect learning speed?**
   - Sweep eta from 0.01 to 0.2
   - Plot convergence rate

3. **What is the optimal memory length (lambda)?**
   - Sweep lambda from 0.8 to 0.99
   - Analyze on tasks with different time horizons

4. **Does MPN help in partially observable environments?**
   - Test on POMDPs
   - Compare with LSTM-DQN

5. **How does M matrix evolve during episodes?**
   - Visualize M matrix dynamics
   - Correlate with task phases

---

## Quick Reference

### Training
```bash
python main.py train --experiment-name NAME --num-episodes N [options]
```

### Evaluation
```bash
python main.py eval --experiment-name NAME --num-eval-episodes N
```

### Resuming
```bash
python main.py resume --experiment-name NAME --num-episodes N
```

### Rendering
```bash
python main.py render --experiment-name NAME --output FILE.gif
```

### Common Options
- `--eta 0.05` - Hebbian learning rate
- `--lambda-decay 0.9` - M matrix decay
- `--hidden-dim 64` - Hidden layer size
- `--checkpoint-freq 50` - Save every N episodes
- `--render-freq-mins 5` - Render GIF every N minutes
- `--plot-training` - Auto-generate training plots

---

## Need Help?

- Check `README.md` for API documentation
- Run `python main.py --help` for CLI options
- Look at `example_usage.py` for programmatic examples
- Review existing experiment configs in `experiments/*/config.json`


