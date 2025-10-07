# PCA Analysis Usage Examples

## Basic Commands

### Default Analysis
```bash
python3 main.py analyze --experiment-name long-run-5k
```
- Uses 100 episodes
- Colors by cart position (feature 0)
- Loads best_model.pt
- Analyzes both hidden states and M matrices

### Specify Number of Episodes
```bash
python3 main.py analyze --experiment-name long-run-5k --num-episodes 200
```

### Specify Checkpoint
```bash
python3 main.py analyze \
    --experiment-name long-run-5k \
    --checkpoint checkpoint_500.pt
```

### Color by Different State Features

```bash
# Cart position (default)
python3 main.py analyze --experiment-name long-run-5k --color-feature 0

# Cart velocity
python3 main.py analyze --experiment-name long-run-5k --color-feature 1

# Pole angle
python3 main.py analyze --experiment-name long-run-5k --color-feature 2

# Pole angular velocity
python3 main.py analyze --experiment-name long-run-5k --color-feature 3
```

### Specify Number of Components
```bash
python3 main.py analyze \
    --experiment-name long-run-5k \
    --n-components 50
```

### Quick Test (Fewer Episodes)
```bash
python3 main.py analyze \
    --experiment-name long-run-5k \
    --num-episodes 20 \
    --n-components 30
```

## Comparing Across Training

### Analyze Multiple Checkpoints Sequentially

```bash
# Early training
python3 main.py analyze \
    --experiment-name long-run-5k \
    --checkpoint checkpoint_500.pt \
    --num-episodes 100

# Mid training
python3 main.py analyze \
    --experiment-name long-run-5k \
    --checkpoint checkpoint_1000.pt \
    --num-episodes 100

# Late training
python3 main.py analyze \
    --experiment-name long-run-5k \
    --checkpoint checkpoint_1500.pt \
    --num-episodes 100

# Best model
python3 main.py analyze \
    --experiment-name long-run-5k \
    --checkpoint best_model.pt \
    --num-episodes 100
```

Note: Outputs are saved with same filenames, so rename or move output directories between runs to preserve comparisons.

## Analyzing Different Experiments

```bash
# Experiment 1
python3 main.py analyze --experiment-name happy-cobra

# Experiment 2
python3 main.py analyze --experiment-name long-run-5k

# Experiment 3
python3 main.py analyze --experiment-name test-experiment
```

## Device Selection

```bash
# Use GPU
python3 main.py analyze --experiment-name long-run-5k --device cuda

# Use CPU
python3 main.py analyze --experiment-name long-run-5k --device cpu

# Auto-detect (default)
python3 main.py analyze --experiment-name long-run-5k --device auto
```

## Batch Analysis Script

Create `analyze_all_features.sh`:
```bash
#!/bin/bash

EXPERIMENT="long-run-5k"
NUM_EP=100

for feature in 0 1 2 3; do
    echo "Analyzing feature $feature..."
    python3 main.py analyze \
        --experiment-name $EXPERIMENT \
        --num-episodes $NUM_EP \
        --color-feature $feature

    # Move results to preserve them
    mv experiments/$EXPERIMENT/analysis experiments/$EXPERIMENT/analysis_feature_$feature
done

echo "Complete! Results in experiments/$EXPERIMENT/analysis_feature_*/"
```

Run: `bash analyze_all_features.sh`

## Batch Analysis Across Checkpoints

Create `analyze_training_progression.sh`:
```bash
#!/bin/bash

EXPERIMENT="long-run-5k"
NUM_EP=100

for checkpoint in checkpoint_500.pt checkpoint_1000.pt checkpoint_1500.pt best_model.pt; do
    echo "Analyzing $checkpoint..."
    python3 main.py analyze \
        --experiment-name $EXPERIMENT \
        --checkpoint $checkpoint \
        --num-episodes $NUM_EP

    # Move results to preserve them
    checkpoint_name="${checkpoint%.pt}"  # Remove .pt extension
    mv experiments/$EXPERIMENT/analysis experiments/$EXPERIMENT/analysis_$checkpoint_name
done

echo "Complete! Results in experiments/$EXPERIMENT/analysis_*/"
```

Run: `bash analyze_training_progression.sh`

## Output Locations

All outputs saved to: `experiments/<experiment-name>/analysis/`

Generated files:
- `pca_variance.png`
- `trajectories_hidden.png`
- `trajectories_M.png`

## Viewing Results

```bash
# List generated files
ls experiments/long-run-5k/analysis/

# Open with image viewer (Ubuntu/Linux)
xdg-open experiments/long-run-5k/analysis/pca_variance.png

# Or use your preferred image viewer
eog experiments/long-run-5k/analysis/*.png
```

## Programmatic Usage

Can also use the module directly in Python scripts:

```python
import torch
from pca_analysis import (
    collect_episodes,
    MPNPCAAnalyzer,
    plot_trajectories_2d,
    participation_ratio
)
from mpn_dqn import MPNDQN
import gymnasium as gym

# Load model
dqn = MPNDQN(obs_dim=4, hidden_dim=64, action_dim=2)
dqn.load_state_dict(torch.load('model.pt'))
dqn.eval()

# Create environment
env = gym.make('CartPole-v1')

# Collect data
collector = collect_episodes(
    dqn, env,
    n_episodes=100,
    device='cuda',
    epsilon=0.0,
    max_steps=500
)

# Get pooled data
pooled = collector.get_pooled_data()

# Perform PCA
analyzer = MPNPCAAnalyzer()
analyzer.fit_hidden_pca(pooled['hidden'], n_components=50)

# Get participation ratio
pr = participation_ratio(analyzer.hidden_pca.explained_variance_)
print(f"Participation Ratio: {pr:.2f}")

# Transform and plot
data = collector.get_data()
hidden_pcs = [analyzer.transform_hidden(h) for h in data['hidden']]
colors = [obs[:, 0] for obs in data['obs']]  # Color by cart position

plot_trajectories_2d(
    hidden_pcs, colors,
    save_path='my_analysis.png',
    title='Custom Analysis'
)
```

## Testing the Implementation

```bash
# Test the module
python3 pca_analysis.py

# View CLI help
python3 main.py analyze --help

# Quick test with minimal episodes
python3 main.py analyze \
    --experiment-name long-run-5k \
    --num-episodes 10 \
    --n-components 20
```

## Troubleshooting

### Check Available Experiments
```bash
ls experiments/
```

### Check Available Checkpoints
```bash
ls experiments/long-run-5k/checkpoints/
```

### Check Experiment Config
```bash
cat experiments/long-run-5k/config.json
```

### Memory Issues
If running out of memory with long episodes:
```bash
python3 main.py analyze \
    --experiment-name long-run-5k \
    --num-episodes 50 \
    --max-steps 200
```

### Device Issues
Force CPU if GPU memory issues:
```bash
python3 main.py analyze \
    --experiment-name long-run-5k \
    --device cpu
```
