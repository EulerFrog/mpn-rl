# Multi-Plasticity Networks for Reinforcement Learning

This directory contains a standalone implementation of Multi-Plasticity Networks (MPNs) adapted for Deep Reinforcement Learning, specifically for use as a recurrent layer in Deep Q-Networks (DQN).

**📘 New to experiments? Check out [EXPERIMENTS.md](EXPERIMENTS.md) for a complete guide on conducting research with MPN-DQN!**

## Overview

Multi-Plasticity Networks combine:
- **Long-term learning** via backpropagation (weight matrix W)
- **Short-term plasticity** via Hebbian updates (modulation matrix M)

This implementation uses MPNs as the recurrent layer in a Deep Q-Network, providing an alternative to LSTM/GRU that is biologically inspired and may offer advantages for temporal credit assignment.

## Architecture

```
Observation → MPN (recurrent) → Linear → Q-values
                ↓
            M matrix (Hebbian state)
```

### MPN Update Rules

**Hebbian Plasticity:**
```
M_t = λM_{t-1} + ηh_t x_t^T
```

**Multiplicative Modulation:**
```
h = activation(b + W*(M + 1)*x)
```

Where:
- `M`: Synaptic modulation matrix (recurrent state)
- `W`: Long-term weights (learned via backprop)
- `η` (eta): Hebbian learning rate (fixed)
- `λ` (lambda): Decay factor (fixed)
- `h`: Hidden activations
- `x`: Input observations

## Files

### Core Modules

- **`mpn_module.py`**: Standalone MPN implementation
  - `MPNLayer`: Single MPN layer with Hebbian plasticity
  - `MPN`: Complete network with RNN-like interface
  - Dependencies: PyTorch only
  - Interface: `forward(x, state) -> (output, new_state)`

- **`mpn_dqn.py`**: Deep Q-Network with MPN recurrent layer
  - `MPNDQN`: DQN with MPN as recurrent layer
  - `DoubleMPNDQN`: Double DQN variant with target network
  - Epsilon-greedy action selection
  - Compatible with standard DQN training

- **`mpn_torchrl.py`**: TorchRL wrappers
  - `MPNDQNTorchRL`: TensorDict-compatible wrapper
  - `DoubleMPNDQNTorchRL`: Double DQN for TorchRL
  - Integrates with TorchRL's training infrastructure
  - Requires: `pip install torchrl tensordict`

- **`model_utils.py`**: Training utilities and experiment management
  - `ReplayBuffer`: Experience replay buffer for DQN
  - `compute_td_loss`: TD loss computation for Double DQN
  - `ExperimentManager`: Experiment directory and checkpoint management
  - Configuration and training history tracking

- **`visualize.py`**: Visualization tools
  - `TrainingVisualizer`: Plot training metrics (rewards, losses, epsilon)
  - `MMatrixVisualizer`: Visualize M matrix evolution during episodes
  - `visualize_episode()`: Record and visualize agent behavior
  - `compare_models_visualization()`: Compare multiple models

- **`pca_analysis.py`**: PCA analysis tools
  - `HiddenStateCollector`: Collect hidden states and M matrices during episodes
  - `MPNPCAAnalyzer`: Perform PCA on collected data
  - `plot_trajectories_2d()`: Visualize trajectories in PC space
  - `participation_ratio()`: Compute effective dimensionality
  - Requires: `pip install scikit-learn matplotlib`

## Installation

This project uses `pyproject.toml` for dependency management. All dependencies are required for full functionality.

### Using uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd mpn_rl

# Install with uv
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd mpn_rl

# Install dependencies
pip install -e .
```

### Dependencies

All required packages are installed automatically:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Gymnasium >= 0.29.0
- scikit-learn >= 1.3.0 (for PCA analysis)
- imageio >= 2.31.0
- Pillow >= 10.0.0
- tensordict >= 0.2.0

### GPU Acceleration

MPN-DQN automatically detects and uses GPU if available:

```bash
# Check GPU availability
python test_gpu.py

# Train with automatic GPU detection (default)
python main.py train

# Force CPU usage
python main.py train --device cpu

# Use specific GPU
python main.py train --device cuda:0
```

**Installing PyTorch with CUDA:**
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Check installation
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Performance benefits:**
- 5-10x speedup for training on GPU vs CPU
- Larger batch sizes possible with GPU memory
- Faster hyperparameter sweeps

**Note:** GIF rendering always uses CPU to avoid GPU-CPU transfer overhead.

## Usage

### Command-Line Interface (Recommended)

The easiest way to train, evaluate, and render MPN-DQN agents is via the CLI:

#### Train a new agent

```bash
# Train with default settings (generates random experiment name)
python main.py train

# Train with custom name and hyperparameters
python main.py train --experiment-name my-agent \
    --num-episodes 1000 \
    --hidden-dim 64 \
    --eta 0.05 \
    --lambda-decay 0.9

# Train with GIF rendering every 5 minutes
python main.py train --experiment-name cartpole-demo \
    --render-freq-mins 5 \
    --plot-training
```

#### Resume training

```bash
# Continue training for 500 more episodes
python main.py resume --experiment-name my-agent --num-episodes 500
```

#### Evaluate trained agent

```bash
# Evaluate for 10 episodes
python main.py eval --experiment-name my-agent --num-eval-episodes 10

# Evaluate specific checkpoint
python main.py eval --experiment-name my-agent \
    --checkpoint checkpoint_500.pt \
    --num-eval-episodes 20
```

#### Render episode to GIF

```bash
# Render single episode showing CartPole + M matrix
python main.py render --experiment-name my-agent --output demo.gif

# Render with custom settings
python main.py render --experiment-name my-agent \
    --output render.gif \
    --max-steps 500 \
    --fps 30
```

#### Analyze with PCA

```bash
# Analyze hidden states and M matrices with PCA
python main.py analyze --experiment-name my-agent

# Analyze with pole angle coloring
python main.py analyze --experiment-name my-agent --color-feature 2

# Analyze with more episodes for better statistics
python main.py analyze --experiment-name my-agent \
    --num-episodes 200 \
    --n-components 50
```

#### Experiment directory structure

Experiments are automatically organized:
```
experiments/{experiment-name}/
├── config.json              # Hyperparameters
├── training_history.json    # Episode rewards, losses, etc.
├── checkpoints/
│   ├── best_model.pt       # Best model (highest avg reward)
│   ├── checkpoint_50.pt    # Periodic checkpoints
│   ├── checkpoint_100.pt
│   └── final_model.pt      # Final checkpoint
├── videos/
│   ├── episode_00000.gif   # Periodic GIF renders
│   └── episode_00100.gif
├── plots/
│   └── training_curves.png
└── analysis/               # PCA analysis results
    ├── pca_variance.png   # Explained variance plots
    ├── trajectories_hidden.png  # Hidden state trajectories
    └── trajectories_M.png       # M matrix trajectories
```

### Programmatic API

### Basic MPN

```python
from mpn_module import MPN
import torch

# Create MPN
mpn = MPN(input_dim=4, hidden_dim=64, eta=0.01, lambda_decay=0.95)

# Initialize state for episode
state = mpn.init_state(batch_size=1)

# Step through episode
obs = torch.randn(1, 4)
hidden, new_state = mpn(obs, state)
```

### MPN-DQN

```python
from mpn_dqn import MPNDQN
import torch

# Create DQN
dqn = MPNDQN(obs_dim=4, hidden_dim=64, action_dim=2)

# Initialize state
state = dqn.init_state(batch_size=1)

# Get Q-values
obs = torch.randn(1, 4)
q_values, new_state = dqn(obs, state)

# Select action (epsilon-greedy)
action, new_state = dqn.select_action(obs, state, epsilon=0.1)
```

### Training on Gym

Use the CLI for training (see Command-Line Interface section above):

```bash
# Train with custom hyperparameters
python main.py train --experiment-name my-agent \
    --num-episodes 200 \
    --hidden-dim 32 \
    --eta 0.05 \
    --lambda-decay 0.9
```

For programmatic training, you can import the training components:

```python
from model_utils import ReplayBuffer, compute_td_loss
from mpn_dqn import MPNDQN
import gymnasium as gym

# See main.py train() function for full training loop example
```

### TorchRL Integration

```python
from mpn_torchrl import MPNDQNTorchRL
from tensordict import TensorDict
import torch

# Create TorchRL-compatible DQN
dqn = MPNDQNTorchRL(obs_dim=4, hidden_dim=64, action_dim=2)

# Use with TensorDict
td = TensorDict({"observation": torch.randn(32, 4)}, batch_size=[32])
td = dqn(td)  # Adds "action_value" and "mpn_state" keys

# Action selection
td = dqn.select_action(td, epsilon=0.1)  # Adds "action" key
```

### Visualization

```python
from visualize import TrainingVisualizer, visualize_episode

# Visualize training progress
viz = TrainingVisualizer()
for episode in range(100):
    reward, loss = train_episode()  # your training function
    viz.update(episode_reward=reward, loss=loss)
viz.plot(save_path='training_curves.png')

# Visualize M matrix evolution during an episode
import gymnasium as gym
env = gym.make('CartPole-v1')
total_reward, m_viz = visualize_episode(
    dqn, env,
    save_M_plot='m_matrix_evolution.png'
)

# The visualization shows:
# - M matrix heatmap over time
# - M matrix norm evolution
# - Q-values across timesteps
# - Observations over time
```

## Key Parameters

### MPN Parameters
- `eta`: Hebbian learning rate (controls how fast M updates)
  - Typical values: 0.01 - 0.1
  - Higher = faster adaptation within episode

- `lambda_decay`: Decay factor for M matrix
  - Range: 0 < λ ≤ 1
  - Typical values: 0.9 - 0.99
  - Higher = longer memory

- `activation`: Activation function
  - Options: 'relu', 'tanh', 'sigmoid', 'linear'
  - Default: 'tanh'

### DQN Training Parameters
- `gamma`: Discount factor (0.99)
- `epsilon_start/end/decay`: Exploration schedule
- `batch_size`: Mini-batch size (32-128)
- `buffer_size`: Replay buffer capacity (10000+)
- `target_update_freq`: Episodes between target updates (10-100)

## Testing

Each module includes self-tests:

```bash
# Test MPN module
python3 mpn_module.py

# Test DQN
python3 mpn_dqn.py

# Test TorchRL wrapper
python3 mpn_torchrl.py

# Run training via CLI (requires gymnasium)
python3 main.py train --num-episodes 100
```

## Comparison: MPN-DQN vs Standard DQN

**Advantages of MPN:**
- Built-in temporal integration via M matrix
- Fast within-episode adaptation
- Biologically plausible learning
- May improve sample efficiency on temporal tasks

**When to use MPN:**
- Tasks requiring temporal integration
- Partially observable environments (POMDP)
- When credit assignment is delayed
- When exploring biologically-inspired RL

**When to use standard DQN:**
- Fully observable Markov environments
- When simplicity is preferred
- When computational efficiency is critical

## Directory Structure

```
mpn_rl/
├── README.md                 # This file
├── EXPERIMENTS.md           # Complete guide for conducting experiments
├── pyproject.toml           # Dependency management (uv/pip)
├── .gitignore               # Git ignore patterns
├── main.py                  # CLI for train/eval/render/analyze
├── mpn_module.py            # Core MPN implementation
├── mpn_dqn.py               # DQN with MPN recurrent layer
├── mpn_torchrl.py           # TorchRL wrappers
├── pca_analysis.py          # PCA analysis tools
├── model_utils.py           # Training utils and experiment management
├── render_utils.py          # GIF rendering with CartPole + M matrix
├── visualize.py             # Visualization tools
└── ai_docs/                 # AI-generated documentation
    ├── YYYY-MM-DD/         # Session-specific logs
    ├── IMPLEMENTATION_REFERENCE.md
    ├── USAGE_EXAMPLES.md
    └── RESEARCH_QUESTIONS.md
```

## Reference

This implementation is based on the Multi-Plasticity Networks described in:
- **eLife-83035**: Multi-plasticity networks for temporal integration

## Related Work

The parent directory `/home/eulerfrog/KAM/mpn/` contains the original paper implementation with supervised learning tasks.

## License

See parent directory for licensing information.

## Next Steps

### Getting Started with Experiments
1. **Read [EXPERIMENTS.md](EXPERIMENTS.md)** - Complete guide for conducting research
2. **Run your first experiment**: `python main.py train --plot-training`
3. **Try hyperparameter sweeps** - See EXPERIMENTS.md for examples
4. **Compare MPN vs baseline** - Quantify the benefits of plasticity

### Potential Research Extensions
1. **More RL algorithms**: Extend to A2C, PPO, SAC
2. **Recurrent tasks**: Test on POMDPs (e.g., T-Maze, Memory tasks)
3. **Different environments**: Try MuJoCo, Atari with frame stacking
4. **Presynaptic-only updates**: Implement STSP-like plasticity variant
5. **Multi-task learning**: Test transfer and continual learning scenarios
