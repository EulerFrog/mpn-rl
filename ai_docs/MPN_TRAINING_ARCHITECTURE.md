# MPN Training Architecture for NeuroGym

**Date**: 2025-10-20
**Status**: Implemented and tested

---

## Overview

This document describes the training architecture for Multi-Plasticity Networks (MPNs) on NeuroGym cognitive tasks. MPNs are **not standard RNNs** - they use a unique combination of backpropagation (for weight matrix W) and Hebbian plasticity (for modulation matrix M).

## Key Architectural Differences: MPN vs RNN

### Standard RNN
```
h_t = activation(W_h * h_{t-1} + W_x * x_t + b)
All parameters learned via backpropagation
```

### MPN Architecture
```
h_t = activation(W * (M_t + 1) * x_t + b)
M_t = λ * M_{t-1} + η * h_t ⊗ x_t  (Hebbian update)

W, b: Learned via backpropagation
M:    Updated via Hebbian rule (during forward pass)
```

**Key insight**: The M matrix provides fast, within-episode adaptation via Hebbian plasticity, while W provides slow, across-episode learning via backpropagation.

---

## Implementation Details

### State Management

**During Episode Collection:**
- MPN state (M matrix) **persists across trials** within an episode
- State is only reset at episode boundaries (`done=True`)
- This allows the MPN to maintain memory across multiple trials

```python
# Episode start
episode_state = mpn.init_state(batch_size=1, device=device)

while not done:
    # Step environment
    q_values, next_state = mpn(obs, episode_state)
    obs, reward, done, truncated, info = env.step(action)

    # Update state (persists across trials!)
    episode_state = next_state

    # Trial boundary - state DOES NOT reset
    if info['new_trial']:
        # Store trial, but keep state
        pass

    # Episode boundary - state RESETS
    if done:
        episode_state = mpn.init_state(batch_size=1, device=device)
```

**During Replay/Training:**
- Each trial is replayed from **fresh zero state**
- This is simpler than storing initial states (à la SB3 RecurrentPPO)
- The MPN learns to quickly rebuild appropriate M states
- W parameters adapt to this replay strategy

### Truncated BPTT (Backpropagation Through Time)

For long sequences (100-800 timesteps), we use Truncated BPTT to manage memory and gradients:

```python
def compute_td_loss_trial(dqn, target_dqn, trial_batch,
                          gamma=0.99, device='cpu',
                          bptt_chunk_size=None):
    """
    Args:
        bptt_chunk_size: If None, full BPTT through trial.
                        If set (e.g., 20-50), breaks into chunks.
    """
    # For each trial:
    state = dqn.init_state(batch_size=1, device=device)

    # Break trial into chunks
    for chunk_start, chunk_end in chunks:
        # Forward pass through chunk
        for t in range(chunk_start, chunk_end):
            q_values, state = dqn(obs[t], state)
            # ... compute loss ...

        # Truncate gradients between chunks
        state = state.detach()  # State continues, gradients don't!
```

**Key points:**
- State (M matrix) continues across chunks
- Gradients are detached between chunks
- This limits backprop to `bptt_chunk_size` timesteps
- Recommended chunk size: 20-50 for memory efficiency

### Training Configuration

```bash
# Basic training (full BPTT)
python main.py train-neurogym --env-name ContextDecisionMaking-v0

# With Truncated BPTT (recommended for long trials)
python main.py train-neurogym \
    --env-name ContextDecisionMaking-v0 \
    --bptt-chunk-size 30 \
    --num-episodes 1000
```

**Hyperparameters:**
- `--bptt-chunk-size`: Chunk size for Truncated BPTT (default: None = full BPTT)
- `--hidden-dim`: MPN hidden dimension (default: 128)
- `--eta`: Hebbian learning rate (default: 0.1)
- `--lambda-decay`: M matrix decay factor (default: 0.95)
- `--trial-batch-size`: Number of trials per training batch (default: 4)

---

## Extending to Parallel Trial Processing

### Current Implementation: Sequential Processing

Currently, trials are processed one at a time:

```python
for trial in trial_batch:
    state = dqn.init_state(batch_size=1, device=device)

    for t in range(trial_length):
        q_values, state = dqn(obs[t], state)
        # ... compute loss ...
```

**Pros:** Simple, handles variable-length trials easily
**Cons:** Slower than parallel processing

### Future Extension: Parallel Processing

To process multiple trials in parallel, we need:

#### 1. **Padding/Masking for Variable Lengths**

```python
def pad_trials(trial_batch):
    """Pad trials to same length."""
    max_length = max(trial.obs_list.shape[0] for trial in trial_batch)
    batch_size = len(trial_batch)

    # Create padded tensors
    obs_padded = torch.zeros(batch_size, max_length, obs_dim)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)

    for i, trial in enumerate(trial_batch):
        length = trial.obs_list.shape[0]
        obs_padded[i, :length] = trial.obs_list
        mask[i, :length] = True

    return obs_padded, mask
```

#### 2. **Batched Forward Pass**

```python
def compute_td_loss_trial_parallel(dqn, target_dqn, trial_batch,
                                   gamma=0.99, device='cpu',
                                   bptt_chunk_size=None):
    """Process multiple trials in parallel."""
    batch_size = len(trial_batch)

    # Pad trials
    obs_padded, mask = pad_trials(trial_batch)

    # Initialize state for all trials
    state = dqn.init_state(batch_size=batch_size, device=device)

    # Forward pass (all trials in parallel)
    for t in range(max_length):
        obs_t = obs_padded[:, t, :]  # [batch_size, obs_dim]
        q_values, state = dqn(obs_t, state)  # Batched!

        # Apply mask to ignore padded timesteps
        # ... compute masked loss ...
```

#### 3. **Masked Loss Computation**

```python
# Only compute loss for valid timesteps
valid_mask = mask[:, t]
current_q = q_values.gather(1, actions[:, t].unsqueeze(1)).squeeze(1)
current_q = current_q[valid_mask]  # Filter out padding
target_q = target_q[valid_mask]

loss = F.smooth_l1_loss(current_q, target_q)
```

#### 4. **Implementation Steps**

To add parallel processing:

1. **Modify `TrialReplayBuffer.sample()`**:
   - Return trials as a batch-friendly format
   - Consider max_length constraints

2. **Add `compute_td_loss_trial_parallel()`**:
   - Implement padding/masking logic
   - Batched forward passes
   - Masked loss computation

3. **Add command-line flag**:
   ```python
   parser.add_argument('--parallel-trials', action='store_true',
                       help='Process trials in parallel (requires padding)')
   ```

4. **Test thoroughly**:
   - Verify loss matches sequential version
   - Benchmark speedup vs memory usage
   - Test with various trial length distributions

**Expected benefits:**
- 2-4x speedup for batch_size=4
- Better GPU utilization
- More efficient training

**Trade-offs:**
- More complex implementation
- Higher memory usage (padding)
- Requires masking logic

---

## Architecture Comparison

### Collection Phase (Same for both)

```
NeuroGym Env
    ↓
NeuroGymWrapper (handles trial boundaries)
    ↓
MPN-DQN (state persists across trials)
    ↓
TrialReplayBuffer
```

### Training Phase

**Sequential (Current):**
```
Sample trial_batch → For each trial:
    Initialize state → Forward through trial → Compute loss
    → Truncate gradients at chunks
```

**Parallel (Future):**
```
Sample trial_batch → Pad trials → Batch forward pass
    → Masked loss computation
    → Truncate gradients at chunks
```

---

## Debugging and Monitoring

### Key Metrics

1. **State persistence**: Check that M matrix evolves across trials
   ```python
   print(f"M matrix norm: {episode_state.norm().item():.4f}")
   ```

2. **Gradient flow**: Verify gradients are truncated at chunks
   ```python
   # After chunk boundary
   assert not state.requires_grad  # Should be detached
   ```

3. **Trial lengths**: Monitor trial length distribution
   ```python
   print(f"Trial lengths: mean={np.mean(lengths):.1f}, "
         f"max={np.max(lengths)}")
   ```

### Common Issues

1. **OOM (Out of Memory)**:
   - Reduce `bptt_chunk_size`
   - Reduce `trial_batch_size`
   - Use smaller `hidden_dim`

2. **Slow training**:
   - Enable Truncated BPTT (`--bptt-chunk-size 30`)
   - Reduce `trial_batch_size` if too slow
   - Consider parallel processing (future)

3. **Poor learning**:
   - Check that state persists across trials (not reset)
   - Verify Hebbian parameters (eta, lambda_decay)
   - Ensure sufficient exploration (epsilon_start)

---

## References

### Related Documentation
- `ai_docs/2025-10-13/neurogym_integration_progress.md` - Original design decisions
- `ai_docs/replay_buffer_insight.md` - Why standard replay breaks for MPNs
- `NEUROGYM_README.md` - User guide for training

### Papers
- MPN Paper: eLife 2023, Multi-Plasticity Networks
- Truncated BPTT: Williams & Peng, 1990
- Recurrent Experience Replay: Kapturowski et al., 2018

### Similar Implementations
- Stable-Baselines3 RecurrentPPO: Alternative approach storing hidden states
- R2D2 (Deepmind): Stores sequences with overlapping windows

---

## Summary

**Current Implementation (2025-10-20):**
- ✅ MPN state persists across trials (only resets at episode boundaries)
- ✅ Truncated BPTT with configurable chunk size
- ✅ Sequential trial processing (simple, robust)
- ✅ Replays from zero state (simpler than storing states)

**Future Extensions:**
- ⏳ Parallel trial processing (2-4x speedup)
- ⏳ Store initial states for replay (à la SB3)
- ⏳ Curriculum learning on trial lengths
- ⏳ Meta-learning across episodes

**Key Takeaway**: This architecture respects MPN's unique combination of Hebbian plasticity (M) and backpropagation (W), enabling investigation of how internal states differ from standard RNNs.
