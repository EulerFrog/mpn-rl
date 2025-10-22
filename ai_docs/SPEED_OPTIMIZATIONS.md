# Training Speed Optimizations

**Date**: 2025-10-20

## Summary

Implemented major speed optimizations for sequence-based replay training, primarily by parallelizing batch processing on GPU.

---

## Optimization #1: Parallelized Sequence Processing ⭐ **BIGGEST WIN**

### Before (Slow)
```python
for sequence in sequences:  # Process one sequence at a time
    state = dqn.init_state(batch_size=1, device=device)  # batch_size=1!
    for transition in sequence:
        q_values, state = dqn(obs.unsqueeze(0), state)
```

**Problem**: Processed 32 sequences sequentially with batch_size=1
- 32 separate forward passes through the network
- Poor GPU utilization (GPUs are designed for parallel operations)

### After (Fast)
```python
# Stack all sequences into batch
obs_batch = torch.stack(...)  # [batch_size, seq_len, obs_dim]
state = dqn.init_state(batch_size=32, device=device)  # batch_size=32!

for t in range(seq_len):
    q_values, state = dqn(obs_batch[:, t, :], state)  # Process all 32 in parallel
```

**Improvement**: Process all 32 sequences in parallel with batch_size=32
- Single forward pass processes entire batch
- Full GPU utilization
- **Expected speedup: 10-30x for loss computation**

---

## Optimization #2: Increased Default Batch Size

### Changed
- Sequence batch size: **32 → 64**
- Larger batches = better GPU utilization
- Can increase further if GPU memory allows (try 128, 256)

### Trade-off
- More memory usage
- Better parallelization
- Slightly more compute per step, but much faster overall

---

## Optimization #3: Configurable Training Frequency

### New Parameter: `--train-freq N`

```python
# Train every N environment steps (default: 1)
if total_env_steps % train_freq == 0:
    # Do training update
```

**Benefits**:
- `--train-freq 1`: Train every step (default, most sample efficient)
- `--train-freq 4`: Train every 4 steps (4x faster, slightly less stable)
- `--train-freq 8`: Train every 8 steps (8x faster, may impact learning)

**Recommendation**: Start with 4-8 for faster initial training, reduce to 1 later

---

## Speed Comparison Estimates

| Configuration | Relative Speed | Notes |
|--------------|----------------|-------|
| Old (sequential) | 1x (baseline) | batch_size=1 per sequence |
| New (default) | **15-25x** | Parallel batching + larger batch |
| New + train_freq=4 | **60-100x** | Parallel + skip training steps |
| New + train_freq=8 | **120-200x** | Maximum speed, may hurt learning |

---

## Recommended Fast Training Command

```bash
# Fast training (good balance of speed vs learning quality)
python3 main.py train-neurogym \
    --use-sequence-replay \
    --sequence-length 10 \
    --sequence-batch-size 64 \
    --train-freq 4 \
    --num-episodes 100 \
    --max-episode-steps 200 \
    --print-freq 5 \
    --checkpoint-freq 25 \
    --experiment-name fast-sequence-test \
    --device cuda
```

```bash
# Maximum speed (for quick experiments)
python3 main.py train-neurogym \
    --use-sequence-replay \
    --sequence-length 10 \
    --sequence-batch-size 128 \
    --train-freq 8 \
    --num-episodes 100 \
    --max-episode-steps 200 \
    --print-freq 5 \
    --checkpoint-freq 25 \
    --experiment-name ultra-fast-test \
    --device cuda
```

```bash
# High quality (slower but better learning)
python3 main.py train-neurogym \
    --use-sequence-replay \
    --sequence-length 10 \
    --sequence-batch-size 32 \
    --train-freq 1 \
    --num-episodes 200 \
    --max-episode-steps 200 \
    --print-freq 10 \
    --checkpoint-freq 50 \
    --experiment-name quality-sequence-test \
    --device cuda
```

---

## Additional Optimizations (Future)

### Not Yet Implemented

1. **Mixed Precision Training (FP16)**
   - Use `torch.cuda.amp` for automatic mixed precision
   - Expected: 2-3x additional speedup
   - Requires code changes in training loop

2. **Gradient Accumulation**
   - Accumulate gradients over multiple batches
   - Allows larger effective batch sizes
   - Better for stability

3. **Multi-GPU Training**
   - Use `torch.nn.DataParallel`
   - Near-linear speedup with GPU count

4. **Compiled Forward Pass**
   - Use `torch.compile()` (PyTorch 2.0+)
   - Expected: 1.5-2x speedup
   - May not work with dynamic MPN architecture

---

## Memory Usage

With default settings (batch_size=64, seq_len=10):
- Estimated GPU memory: ~1-2 GB
- Can increase batch_size if you have more VRAM:
  - RTX 3090 (24GB): Try `--sequence-batch-size 256`
  - RTX 4090 (24GB): Try `--sequence-batch-size 512`
  - A100 (40/80GB): Try `--sequence-batch-size 1024`

If you get OOM errors, reduce batch size:
```bash
--sequence-batch-size 32  # or even 16
```

---

## Implementation Details

### Files Changed
1. **model_utils.py** (lines 279-392)
   - Rewrote `compute_td_loss_sequences()` for parallel processing
   - Batched tensor operations instead of loops

2. **main.py** (lines 1172-1175, 345-346, 406, 448)
   - Added `--train-freq` parameter
   - Added `total_env_steps` tracking
   - Conditional training based on frequency

### Key Code Change
```python
# Before: Sequential processing
for sequence in sequences:
    state = dqn.init_state(batch_size=1, device=device)
    for transition in sequence:
        q_values, state = dqn(obs.unsqueeze(0), state)
        # ... compute loss ...

# After: Parallel processing
batch_size = len(sequences)
state = dqn.init_state(batch_size=batch_size, device=device)
for t in range(seq_len):
    q_values, state = dqn(obs_batch[:, t, :], state)  # All sequences at once!
    # ... compute loss in parallel ...
```

---

## Verification

To verify the speedup, compare wall-clock time for same number of episodes:

```bash
# Time the old way (trial-based)
time python3 main.py train-neurogym \
    --num-episodes 10 \
    --experiment-name timing-old

# Time the new way (sequence replay)
time python3 main.py train-neurogym \
    --use-sequence-replay \
    --num-episodes 10 \
    --experiment-name timing-new
```

Should see significant reduction in training time per episode.
