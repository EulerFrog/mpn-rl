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
# Collect episode data for analysis
python main.py analyze collect --experiment-name my-agent --num-episodes 100

# Generate PCA plots from collected data
python main.py analyze plot --experiment-name my-agent
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

## Running Jobs on HTCondor

HTCondor allows you to run large-scale parameter sweeps and experiments across multiple machines. This is useful for hyperparameter searches and running multiple seeds in parallel.

### Prerequisites

1. Access to an HTCondor cluster
2. Virtual environment set up with all dependencies installed
3. Condor logs directory created:
   ```bash
   mkdir -p .condor_logs
   ```

### Basic Job Structure

A Condor job consists of three components:

1. **Submit file** (`*.job`) - Defines resource requirements and job parameters
2. **Executable script** (`*.sh`) - The script that runs your experiment
3. **Arguments file** (`*_args.txt`) - Parameter combinations for batch jobs

### Example: Simple Parameter Sweep

The included `dummy.job` demonstrates a basic parameter sweep:

```bash
# Submit the example job
condor_submit dummy.job
```

This will queue 10 jobs with different parameter combinations from `dummy_args.txt`.

### Creating Your Own Training Jobs

1. **Create an executable script** (e.g., `train_sweep.sh`):
   ```bash
   #!/bin/bash

   # Parse arguments
   SEED=$1
   ETA=$2
   LAMBDA=$3
   HIDDEN_DIM=$4
   EXP_NAME=$5

   # Activate virtual environment
   source .venv/bin/activate

   # Run training
   python main.py train \
       --experiment-name ${EXP_NAME} \
       --seed ${SEED} \
       --eta ${ETA} \
       --lambda-decay ${LAMBDA} \
       --hidden-dim ${HIDDEN_DIM} \
       --num-episodes 1000
   ```

2. **Make it executable**:
   ```bash
   chmod +x train_sweep.sh
   ```

3. **Create arguments file** (`train_args.txt`):
   ```
   42 0.05 0.9 64 mpn_seed42_eta0.05
   43 0.05 0.9 64 mpn_seed43_eta0.05
   44 0.05 0.9 64 mpn_seed44_eta0.05
   42 0.1 0.9 64 mpn_seed42_eta0.1
   43 0.1 0.9 64 mpn_seed43_eta0.1
   ```

4. **Create submit file** (`train.job`):
   ```
   universe = vanilla
   executable = ./train_sweep.sh

   # Resource requests
   request_cpus = 1
   request_gpus = 0
   request_memory = 4GB

   # Output files
   output = .condor_logs/train_$(Process).out
   error  = .condor_logs/train_$(Process).err
   log    = .condor_logs/train_$(Process).log

   # Desktop group (if needed for your cluster)
   +CSCI_GrpDesktop = true

   # Limit concurrent jobs (optional)
   max_materialize = 4

   # Queue jobs from arguments file
   arguments = $(args)
   Queue arguments from train_args.txt
   ```

### Submitting and Managing Jobs

```bash
# Submit jobs
condor_submit train.job

# Check job status
condor_q

# Check detailed status for your jobs
condor_q -nobatch

# Monitor a specific job
condor_q <job_id>

# View job history
condor_history

# Remove all your jobs
condor_rm <username>

# Remove specific job
condor_rm <job_id>

# Check why a job is held
condor_q -hold

# Release held jobs
condor_release <job_id>
```

### Monitoring Progress

Check log files in `.condor_logs/`:
```bash
# View stdout from job 0
cat .condor_logs/train_0.out

# Watch logs in real-time
tail -f .condor_logs/train_0.out

# Check for errors
cat .condor_logs/train_0.err
```

### Resource Guidelines

- **CPU-only training**: `request_cpus = 1`, `request_gpus = 0`, `request_memory = 4GB`
- **GPU training**: `request_cpus = 2`, `request_gpus = 1`, `request_memory = 8GB`
- **Evaluation**: `request_cpus = 1`, `request_gpus = 0`, `request_memory = 2GB`

### Tips

- Use `max_materialize` to limit concurrent jobs and avoid overwhelming the cluster
- Always test your script locally before submitting to Condor
- Check `.err` files if jobs fail
- Use descriptive experiment names to organize results
- The `experiments/` directory will contain all training outputs
- Add `.condor_logs/` to `.gitignore` to avoid committing log files

