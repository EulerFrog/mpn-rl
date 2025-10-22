# Training MPN-DQN on NeuroGym Environments

This guide shows how to train MPN-DQN agents on NeuroGym cognitive neuroscience tasks using **trial-based replay with Truncated BPTT**.

**Recent Updates (2025-10-20):**
- ✅ MPN state now persists across trials (only resets between episodes)
- ✅ Truncated BPTT support for efficient training on long sequences
- ✅ Configurable chunk size for memory management

## Quick Start

```bash
# Train on ContextDecisionMaking (default)
python main.py train-neurogym

# Train with Truncated BPTT (recommended for long sequences)
python main.py train-neurogym --bptt-chunk-size 30

# Train on specific environment
python main.py train-neurogym --env-name DelayMatchSample-v0

# Custom experiment with more episodes
python main.py train-neurogym \
    --experiment-name my-neurogym-run \
    --num-episodes 2000 \
    --hidden-dim 256 \
    --bptt-chunk-size 30 \
    --plot-training
```

## Available Environments

### Recommended (Memory-based tasks)
- **ContextDecisionMaking-v0** - Context-dependent integration (similar to MPN paper)
- **DelayMatchSample-v0** - Working memory across delay periods
- **PerceptualDecisionMaking-v0** - Evidence integration over time

### All NeuroGym Environments
See: https://neurogym.github.io/neurogym/latest/

## Key Differences from Standard Training

### MPN Architecture (Not a Standard RNN!)
- **W (weights)**: Learned via backpropagation
- **M (modulation matrix)**: Updated via Hebbian plasticity during forward pass
- **State persistence**: M matrix persists across trials within an episode
- **State reset**: Only resets at episode boundaries (not trial boundaries)

### Trial-Based Replay with Truncated BPTT
- Stores **complete trial sequences** (not individual transitions)
- Replays trials from scratch (regenerates fresh MPN states)
- Uses Truncated BPTT to manage memory for long sequences (100-800 steps)
- Configurable chunk size for gradient truncation

### Hyperparameter Defaults
Different defaults optimized for NeuroGym tasks:

| Parameter | Standard DQN | NeuroGym |
|-----------|--------------|----------|
| `hidden_dim` | 64 | 128 |
| `eta` | 0.05 | 0.1 |
| `lambda_decay` | 0.9 | 0.95 |
| `epsilon_start` | 1.0 | 0.3 |
| `trial_batch_size` | N/A | 4 |
| `buffer_size` | 10000 (transitions) | 500 (trials) |
| `bptt_chunk_size` | N/A | None (full) |

## Command Line Options

```bash
python main.py train-neurogym --help
```

### Key Arguments
- `--env-name` - NeuroGym environment (default: ContextDecisionMaking-v0)
- `--num-episodes` - Training episodes (default: 1000)
- `--hidden-dim` - MPN hidden dimension (default: 128)
- `--eta` - Hebbian learning rate (default: 0.1)
- `--lambda-decay` - M matrix decay (default: 0.95)
- `--trial-batch-size` - Trials per batch (default: 4)
- `--buffer-size` - Trial buffer capacity (default: 500 trials)
- `--bptt-chunk-size` - Chunk size for Truncated BPTT (default: None = full BPTT)
- `--device` - Device: auto/cuda/cpu (default: auto)

**Note on `--bptt-chunk-size`:**
- `None` (default): Full BPTT through entire trial (memory intensive for long trials)
- `20-50`: Recommended for long sequences (100-800 steps), balances memory and learning
- Smaller values: Less memory, more gradient truncation
- Larger values: More memory, better gradient flow

## Examples

### Basic Training
```bash
# Train with default settings
python main.py train-neurogym

# Results saved to: experiments/[random-name]/
```

### GPU Training
```bash
python main.py train-neurogym \
    --device cuda \
    --num-episodes 2000 \
    --experiment-name gpu-context-task
```

### Hyperparameter Tuning
```bash
# Larger network, stronger memory, with Truncated BPTT
python3 main.py train-neurogym \
    --hidden-dim 256 \
    --eta 0.15 \
    --lambda-decay 0.98 \
    --trial-batch-size 8 \
    --buffer-size 1000 \
    --bptt-chunk-size 40
```

### Multiple Environments
```bash
# Train on different tasks
python3 main.py train-neurogym --env-name ContextDecisionMaking-v0 --experiment-name context-task
python3 main.py train-neurogym --env-name DelayMatchSample-v0 --experiment-name delay-task
python3 main.py train-neurogym --env-name PerceptualDecisionMaking-v0 --experiment-name perceptual-task
```

## Evaluation

Use standard evaluation command:
```bash
python3 main.py eval \
    --experiment-name my-neurogym-run \
    --num-eval-episodes 20
```

## Rendering
```bash
python3 main.py render \
    --experiment-name my-neurogym-run \
    --output neurogym_episode.gif
```

## Monitoring Training

### Training Curves
Add `--plot-training` flag:
```bash
python3 main.py train-neurogym --plot-training
# Saves plots to: experiments/[name]/plots/training_curves.png
```

### Progress Logs
```bash
# More frequent printing
python3 main.py train-neurogym --print-freq 5

# Example output:
# Ep   100/1000 | Reward:    0.45 | Len:  287 | Loss: 0.0234 | ε: 0.152 | Trials:   45
```

## Understanding MPN Training (Updated 2025-10-20)

### MPN vs Standard RNN
**MPN is NOT a standard RNN!** Key differences:
- **Standard RNN**: All parameters learned via backpropagation
- **MPN**:
  - W (weights) learned via backpropagation
  - M (modulation) updated via Hebbian plasticity (during forward pass)
  - M provides fast within-episode adaptation
  - W provides slow across-episode learning

### State Management
1. **Collection**: MPN state **persists across trials** within an episode
2. **Storage**: Buffer stores complete trial sequences
3. **Sampling**: Samples random trials for training
4. **Replay**: Replays from zero state (simpler, still learns dynamics)
5. **State Reset**: Only resets at **episode boundaries** (not trial boundaries!)

### Truncated BPTT
For long sequences (100-800 timesteps):
- Breaks trial into chunks (e.g., 30 timesteps)
- Gradients truncated between chunks
- State (M matrix) continues across chunks
- Reduces memory usage, enables longer training

### Why Trial-Based?
Standard DQN replay breaks for recurrent networks because:
- ❌ Random sampling uses "stale" states (computed by old network)
- ❌ Temporal coherence broken
- ❌ Difficult with sparse rewards

Trial-based replay solves this:
- ✅ Fresh states regenerated with current network
- ✅ Temporal coherence preserved
- ✅ Natural for NeuroGym's trial structure
- ✅ Handles sparse rewards naturally

**For detailed architecture explanation, see:** `ai_docs/MPN_TRAINING_ARCHITECTURE.md`

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size or hidden dim
python main.py train-neurogym \
    --trial-batch-size 2 \
    --hidden-dim 64
```

### Slow Training
Trial-based training is slower than transition-based (expected):
- Forward passes through entire trials
- ~4x slower for trial_batch_size=4
- Trade-off for temporal coherence

### Low Rewards
- Increase exploration: `--epsilon-start 0.5`
- Stronger memory: `--eta 0.15 --lambda-decay 0.98`
- Larger network: `--hidden-dim 256`
- More episodes: `--num-episodes 2000`

## Implementation Details

See documentation:
- `ai_docs/2025-10-13/implementation_summary.md` - Complete implementation details
- `ai_docs/2025-10-13/architecture_diagrams.md` - Architecture diagrams
- `ai_docs/replay_buffer_insight.md` - Why standard replay breaks

## Architecture

```
NeuroGym Env → NeuroGymWrapper → MPN-DQN
                                    ↓
                           TrialReplayBuffer
                                    ↓
                         compute_td_loss_trial
```

**Key Components**:
- `NeuroGymWrapper` - Flattens observations, tracks trial boundaries
- `TrialReplayBuffer` - Stores complete trial sequences
- `compute_td_loss_trial()` - TD loss over full sequences

## References

- **NeuroGym**: https://neurogym.github.io/neurogym/latest/
- **MPN Paper**: eLife 2023, Multi-Plasticity Networks
- **Design Docs**: `ai_docs/2025-10-13/`

---

**Ready to train!** Start with:
```bash
python main.py train-neurogym --num-episodes 100 --print-freq 10
```
