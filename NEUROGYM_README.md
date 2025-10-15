# Training MPN-DQN on NeuroGym Environments

This guide shows how to train MPN-DQN agents on NeuroGym cognitive neuroscience tasks using **trial-based replay**.

## Quick Start

```bash
# Train on ContextDecisionMaking (default)
python main.py train-neurogym

# Train on specific environment
python main.py train-neurogym --env-name DelayMatchSample-v0

# Custom experiment with more episodes
python main.py train-neurogym \
    --experiment-name my-neurogym-run \
    --num-episodes 2000 \
    --hidden-dim 256 \
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

### Trial-Based Replay
- Stores **complete trial sequences** (not individual transitions)
- Replays trials from scratch (regenerates fresh MPN states)
- Solves the "stale state" problem for recurrent networks

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
- `--device` - Device: auto/cuda/cpu (default: auto)

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
# Larger network, stronger memory
python3 main.py train-neurogym \
    --hidden-dim 256 \
    --eta 0.15 \
    --lambda-decay 0.98 \
    --trial-batch-size 8 \
    --buffer-size 1000
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

## Understanding Trial-Based Training

### What's Different?
1. **Collection**: Collects complete trials (100-800 timesteps)
2. **Storage**: Buffer stores trials (not individual transitions)
3. **Sampling**: Samples complete trials for training
4. **Replay**: Regenerates MPN states from scratch during replay
5. **State Reset**: MPN state reset at each trial boundary

### Why Trial-Based?
Standard DQN replay breaks for recurrent networks because:
- ❌ Random sampling uses "stale" MPN states (computed by old network)
- ❌ Temporal coherence broken
- ❌ Difficult with sparse rewards

Trial-based replay solves this:
- ✅ Fresh states regenerated with current network
- ✅ Temporal coherence preserved
- ✅ Natural for NeuroGym's trial structure
- ✅ Handles sparse rewards naturally

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
