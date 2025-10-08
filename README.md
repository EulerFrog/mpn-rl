# Multi-Plasticity Networks for Reinforcement Learning

Multi-Plasticity Networks (MPNs) for Deep Reinforcement Learning. MPNs combine long-term learning via backpropagation with short-term Hebbian plasticity as a recurrent layer in Deep Q-Networks.

## Architecture

```
Observation → MPN (recurrent) → Linear → Q-values
                ↓
            M matrix (Hebbian state)
```

**MPN Update Rules:**

Hebbian Plasticity:
```
M_t = λM_{t-1} + ηh_t x_t^T
```

Multiplicative Modulation:
```
h = activation(b + W*(M + 1)*x)
```

Where:
- `M`: Synaptic modulation matrix (recurrent state)
- `W`: Long-term weights (learned via backprop)
- `η` (eta): Hebbian learning rate
- `λ` (lambda): Decay factor
- `h`: Hidden activations
- `x`: Input observations

## Installation

### Using uv (recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd mpn-rl

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Mac/Linux
uv pip install -e .
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd mpn-rl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -e .
```

### Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Gymnasium >= 0.29.0
- scikit-learn >= 1.3.0
- imageio >= 2.31.0
- Pillow >= 10.0.0
- tensordict >= 0.2.0

## Quick Start

### Train an agent

```bash
# Train with default settings
python main.py train

# Train with custom hyperparameters
python main.py train --experiment-name my-agent \
    --num-episodes 1000 \
    --hidden-dim 64 \
    --eta 0.05 \
    --lambda-decay 0.9
```

### Evaluate trained agent

```bash
python main.py eval --experiment-name my-agent --num-eval-episodes 10
```

### Render episode to GIF

```bash
python main.py render --experiment-name my-agent --output demo.gif
```

### Analyze with PCA

```bash
python main.py analyze --experiment-name my-agent
```

## Key Files

- **`mpn_module.py`**: Core MPN implementation with Hebbian plasticity
- **`mpn_dqn.py`**: DQN with MPN recurrent layer
- **`main.py`**: CLI for train/eval/render/analyze
- **`model_utils.py`**: Training utilities and experiment management
- **`visualize.py`**: Training and M matrix visualization tools
- **`pca_analysis.py`**: PCA analysis of hidden states and M matrices
- **`mpn_torchrl.py`**: TorchRL integration (optional)

## Experiment Directory Structure

```
experiments/{experiment-name}/
├── config.json              # Hyperparameters
├── training_history.json    # Episode rewards, losses
├── checkpoints/
│   ├── best_model.pt
│   └── final_model.pt
├── videos/
│   └── episode_*.gif
└── analysis/
    ├── pca_variance.png
    └── trajectories_*.png
```

## Key Parameters

### MPN Parameters
- `eta`: Hebbian learning rate (0.01 - 0.1)
- `lambda_decay`: Decay factor for M matrix (0.9 - 0.99)
- `hidden_dim`: Hidden layer size (32, 64, 128)

### DQN Parameters
- `gamma`: Discount factor (default: 0.99)
- `epsilon_start/end/decay`: Exploration schedule
- `batch_size`: Mini-batch size (32-128)
- `buffer_size`: Replay buffer capacity (10000+)

## Programmatic Usage

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

## Testing

```bash
# Test individual modules
python mpn_module.py
python mpn_dqn.py

# Run training
python main.py train --num-episodes 100
```

## Reference

Based on Multi-Plasticity Networks described in eLife-83035: Multi-plasticity networks for temporal integration.

## License

See parent directory for licensing information.
