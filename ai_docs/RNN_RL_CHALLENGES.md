# Challenges of Training Recurrent Networks for RL

**Date**: 2025-10-20
**Context**: Investigating why MPN-DQN shows poor/no learning on NeuroGym tasks

---

## Our Observations

### Training Results Summary

| Experiment | Episode Steps | BPTT Chunk | Reward Improvement |
|------------|--------------|------------|-------------------|
| optimized-training | 100 | 30 | +0.88 (1.65→2.53) |
| learning-test | 200 | 30 | +2.18 (2.81→4.99) ✅ |
| long-episodes | 500 | 100 | **-0.29** (11.26→10.97) ❌ |

**Key Finding**: Longer episodes with larger BPTT windows actually **hurt** performance!

---

## Fundamental Challenges of Recurrent RL

### 1. **Vanishing/Exploding Gradients Through Time**

**Problem**: With BPTT, gradients must flow backward through many timesteps
- At 500 steps, gradients pass through 500 network applications
- Even with gradient clipping, information degrades
- LSTM/GRU were designed to help, but still struggle at 100+ steps

**Our Case**:
- MPN uses Hebbian plasticity (M matrix) which compounds the gradient issue
- M_t = λ * M_{t-1} + η * h_t ⊗ x_t
- Gradients through M involve products across time

**Evidence**: long-episodes (500 steps, BPTT=100) shows NO learning

---

### 2. **Non-Stationarity from Replay**

**Problem**: Q-learning with experience replay creates non-stationary targets

Standard DQN:
```python
# Sample random transitions
batch = replay_buffer.sample()
# Target: r + γ * max Q_target(s', a')
```

Recurrent DQN:
```python
# Replay trial from scratch with CURRENT network
# But the trial was collected with an OLD policy
# States evolve differently during replay vs collection!
```

**Our Case**:
- We replay trials from zero state
- During collection, M matrix builds up across trial
- During replay, M matrix rebuilds differently (different network params)
- This creates mismatch between collection and training

---

### 3. **Sparse Rewards + Long Sequences**

**Problem**: Credit assignment becomes nearly impossible

NeuroGym tasks:
```
Timestep 0-20:   reward = 0 (fixation)
Timestep 21-40:  reward = 0 (stimulus)
Timestep 41-60:  reward = 0 (delay)
Timestep 61:     reward = +1 or 0 (response)
```

With 500-step episodes containing ~25-30 trials:
- Reward only at end of each trial
- Gradients must propagate backward through intervening trials
- Signal dilutes exponentially

**Evidence**: Longer episodes show worse performance

---

### 4. **State Aliasing Without Memory Reset**

**Problem**: When state persists across trials, old information can interfere

Consider two trials:
- Trial 1: Stimulus A → Correct response = LEFT
- Trial 2: Stimulus B → Correct response = RIGHT

If M matrix retains information from Trial 1, it might bias Trial 2

**Our Current Approach**:
- State persists across trials (no reset)
- Could accumulate misleading information

---

## How Successful RNN-based RL Algorithms Handle This

### R2D2 (Recurrent Replay Distributed DQN)

**Paper**: Kapturowski et al., 2018

**Key Innovations**:

1. **Burn-in Period**
   ```python
   # Don't compute loss on first K timesteps
   # Let RNN state "warm up"
   burn_in = 40  # steps
   loss = compute_loss(sequence[burn_in:])
   ```

2. **Stored State + Short Sequences**
   - Store RNN state with each sequence
   - Sample short sequences (40-80 steps)
   - Use stored initial state during replay

3. **Prioritized Sequence Replay**
   - Prioritize sequences with high TD error
   - Helps with sparse rewards

### RecurrentPPO (Stable-Baselines3)

**Approach**: On-policy, so no replay issues!

**Key Points**:
1. Collect rollouts with current policy
2. Train on sequences immediately (no stale states)
3. Use episode_starts flags to reset LSTM
4. GAE (Generalized Advantage Estimation) helps credit assignment

---

## What We Should Try

### Option 1: Add Burn-In Period ⭐

**Implementation**:
```python
def compute_td_loss_trial(..., burn_in_steps=20):
    for trial in trial_batch:
        # Warm-up phase
        state = dqn.init_state(batch_size=1, device=device)
        for t in range(min(burn_in_steps, trial_length)):
            _, state = dqn(obs[t], state)
            # Don't compute loss here!

        # Training phase
        for t in range(burn_in_steps, trial_length):
            q_values, state = dqn(obs[t], state)
            # Compute loss only here
```

**Why**: Gives MPN time to build appropriate M state before computing loss

### Option 2: Shorter Sequences + Stored States

**Current**:
- Sample full trials (10-30 steps)
- Replay from zero state

**Better**:
- Sample fixed-length sequences (20 steps)
- Store initial MPN state with each sequence
- Use stored state during replay (like R2D2)

### Option 3: Reset State Between Trials ⭐

**Current**: State persists across trials
**Alternative**: Reset M matrix at each trial boundary

**Rationale**:
- Each trial is independent task
- Persisting state might cause interference
- Matches how neuroscience experiments work

### Option 4: Switch to On-Policy (PPO)

**Radical change**: Implement RecurrentPPO instead of DQN

**Advantages**:
- No replay = no stale state problem
- Better for continuous learning
- Proven to work with recurrent networks

**Disadvantages**:
- Need to implement PPO from scratch
- Can't compare directly with original MPN paper (used DQN)

---

## Specific Issues in Our Implementation

### 1. Trial Replay from Zero State

```python
# In compute_td_loss_trial:
state = dqn.init_state(batch_size=1, device=device)  # Always zero!
```

**Problem**: During collection, state built up from previous trial.
During replay, we start fresh. Mismatch!

### 2. No Burn-In

We immediately start computing loss from timestep 0. RNN needs warm-up!

### 3. Large BPTT Windows

100-step BPTT through MPN is very deep:
- 100 applications of M_t = λ * M_{t-1} + η * h_t ⊗ x_t
- Gradients degrade significantly

---

## Recommended Next Steps

### Immediate Actions

1. **Add burn-in period** (easiest to implement)
   - Start with 10-20 steps
   - Should help significantly

2. **Reset state between trials** (test hypothesis)
   - Compare performance with/without reset
   - See if interference is the issue

3. **Reduce episode length back to 100-200**
   - Sweet spot seems to be 200 steps (learning-test)
   - Shorter sequences = better credit assignment

### Medium-Term

4. **Store initial states** (like R2D2)
   - More complex but theoretically sound
   - Eliminates state mismatch

5. **Prioritized replay**
   - Sample trials with high TD error
   - Helps with sparse rewards

### Long-Term

6. **Compare with standard LSTM-DQN**
   - Implement same architecture with LSTM instead of MPN
   - See if issue is MPN-specific or general RNN problem

7. **Try on-policy learning**
   - Implement A2C or PPO variant
   - Benchmark against DQN approach

---

## Key Questions

1. **Is the problem MPN-specific or general RNN issue?**
   - Test: Implement LSTM-DQN with same setup

2. **Does state persistence help or hurt?**
   - Test: Compare reset vs. persist at trial boundaries

3. **What's the optimal sequence length?**
   - Test: Sweep 50, 100, 200, 500 steps

4. **Is our BPTT implementation correct?**
   - Verify: Gradients are flowing properly
   - Check: No detached tensors where shouldn't be

---

## References

### Papers
- **R2D2**: "Recurrent Experience Replay in Distributed Reinforcement Learning" (Kapturowski et al., 2018)
- **RecurrentPPO**: Stable-Baselines3 implementation
- **DRQN**: "Deep Recurrent Q-Learning for Partially Observable MDPs" (Hausknecht & Stone, 2015)

### Code References
- R2D2 (DeepMind): Not open-sourced
- RecurrentPPO (SB3): https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/
- DRQN implementations: Various on GitHub

---

## Summary

Our poor training results likely stem from:
1. **No burn-in period** - RNN needs warm-up
2. **Zero-state replay** - Mismatch with collection
3. **Too-long sequences** - Gradient degradation
4. **State persistence** - Possible interference between trials

**Most promising fix**: Add burn-in period + reduce episode length to 200 steps.
