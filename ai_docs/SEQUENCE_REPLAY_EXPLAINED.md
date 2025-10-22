# Sequence Replay Buffer & Backpropagation Explained

**Date**: 2025-10-21
**Context**: How our DRQN-inspired sequence replay system works

---

## Overview

The sequence replay buffer stores individual transitions and samples fixed-length sequences for training, with automatic zero-padding at episode boundaries to prevent invalid cross-episode sequences.

---

## 1. Storing Transitions (Collection Phase)

### During Episode Execution:

```python
for episode in range(num_episodes):
    episode_state = init_state()  # MPN state persists across trials

    for step in episode:
        # Take action
        obs, action, reward, next_obs, done = env.step(...)

        # Store individual transition with episode ID
        replay_buffer.push(
            obs=obs,           # Current observation [obs_dim]
            action=action,     # Action taken (int)
            reward=reward,     # Reward received (float)
            next_obs=next_obs, # Next observation [obs_dim]
            done=done,         # Episode termination flag
            episode_id=episode # Track which episode this came from
        )

        # MPN state persists across trial boundaries!
        episode_state = next_state
```

### Buffer Storage:

```
Buffer (deque with maxlen=50000):
[
    {obs: tensor, action: 0, reward: 0.0, next_obs: tensor, done: False, episode_id: 0},  # Episode 0, step 0
    {obs: tensor, action: 1, reward: 0.0, next_obs: tensor, done: False, episode_id: 0},  # Episode 0, step 1
    ...
    {obs: tensor, action: 2, reward: 1.0, next_obs: tensor, done: True,  episode_id: 0},  # Episode 0, step 199 (END)
    {obs: tensor, action: 0, reward: 0.0, next_obs: tensor, done: False, episode_id: 1},  # Episode 1, step 0 (START)
    {obs: tensor, action: 1, reward: 0.0, next_obs: tensor, done: False, episode_id: 1},  # Episode 1, step 1
    ...
]
```

Key: Each transition knows which **episode** it came from via `episode_id`.

---

## 2. Sampling Sequences (Training Phase)

### Sampling Process:

```python
def sample(batch_size=32, sequence_length=10):
    sequences = []

    for _ in range(batch_size):  # Sample 32 sequences
        # Pick random end position in buffer
        end_idx = random.randint(sequence_length - 1, len(buffer) - 1)

        # Get episode ID at end position
        current_episode = buffer[end_idx]['episode_id']

        # Build sequence by looking back L=10 steps
        sequence = []
        for i in range(sequence_length):
            idx = end_idx - (sequence_length - 1) + i
            transition = buffer[idx]

            # CRITICAL: Zero-pad if from different episode
            if transition['episode_id'] != current_episode:
                sequence.append(ZERO_TRANSITION)  # Padded!
            else:
                sequence.append(transition)  # Valid

        sequences.append(sequence)

    return sequences
```

### Example Sampled Sequence:

Suppose we sample ending at Episode 1, step 7 with L=10:

```
Buffer indices:     [..., 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, ...]
Episode IDs:        [...,  0,   0,   0,   1,   1,   1,   1,   1,   1,   1,   1, ...]
                                      ↑ Ep 0 ends        ↑ We sample here (end_idx=207)

Sampled sequence (looking back 10 steps from 207):
[
    idx=198: episode_id=0 → ZERO (different episode!)
    idx=199: episode_id=0 → ZERO (different episode!)
    idx=200: episode_id=1 → VALID ✓
    idx=201: episode_id=1 → VALID ✓
    idx=202: episode_id=1 → VALID ✓
    idx=203: episode_id=1 → VALID ✓
    idx=204: episode_id=1 → VALID ✓
    idx=205: episode_id=1 → VALID ✓
    idx=206: episode_id=1 → VALID ✓
    idx=207: episode_id=1 → VALID ✓ (end)
]
```

Result: **2 zero-padded** + **8 valid** transitions.

---

## 3. Computing Loss (Forward & Backward Pass)

### Loss Computation Function:

```python
def compute_td_loss_sequences(dqn, target_dqn, sequences, gamma=0.99, device='cuda'):
    total_loss = 0.0
    total_timesteps = 0

    for sequence in sequences:  # Process each sequence independently
        # Step 1: Initialize MPN state to ZERO
        state = dqn.init_state(batch_size=1, device=device)
        # M matrix = zeros, hidden = zeros

        # Step 2: Forward pass through ENTIRE sequence (full BPTT)
        q_values_list = []
        for t, transition in enumerate(sequence):  # t=0 to t=9
            obs = transition['obs'].unsqueeze(0)  # [1, obs_dim]
            q_values, state = dqn(obs, state)
            # MPN updates: h_t, M_t = f(obs, M_{t-1})
            q_values_list.append(q_values)

        # Step 3: Compute TD targets with target network
        with torch.no_grad():
            target_state = target_dqn.init_state(batch_size=1, device=device)
            target_q_list = []
            for transition in sequence:
                next_obs = transition['next_obs'].unsqueeze(0)
                target_q, target_state = target_dqn(next_obs, target_state)
                target_q_list.append(target_q)

        # Step 4: Compute loss for each timestep (SKIP ZERO-PADDED!)
        for t, transition in enumerate(sequence):
            if transition['episode_id'] == -1:  # Zero-padded
                continue  # Skip! Don't compute loss on invalid data

            # Current Q-value for action taken
            current_q = q_values_list[t][0, transition['action']]

            # TD target: r + γ * max Q_target(s', a')
            if not transition['done']:
                next_action = q_values_list[t+1].argmax()
                target_q = transition['reward'] + gamma * target_q_list[t+1][0, next_action]
            else:
                target_q = transition['reward']

            # Accumulate loss
            loss = smooth_l1_loss(current_q, target_q)
            total_loss += loss
            total_timesteps += 1

    # Average loss
    return total_loss / total_timesteps
```

---

## 4. Backpropagation Through Time (BPTT)

### Gradient Flow:

```
Sequence (L=10):
[ZERO, ZERO, s0, s1, s2, s3, s4, s5, s6, s7]
  ↓     ↓    ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
  (skip)(skip) Forward pass through MPN →

MPN State Evolution:
t=0 (ZERO): M_0 = 0,    h_0 = 0           → q_0 (no loss computed)
t=1 (ZERO): M_1 = 0,    h_1 = 0           → q_1 (no loss computed)
t=2 (s0):   M_2 = λM_1 + η(h_2⊗s0)        → q_2 → loss_2 ✓
t=3 (s1):   M_3 = λM_2 + η(h_3⊗s1)        → q_3 → loss_3 ✓
t=4 (s2):   M_4 = λM_3 + η(h_4⊗s2)        → q_4 → loss_4 ✓
...
t=9 (s7):   M_9 = λM_8 + η(h_9⊗s7)        → q_9 → loss_9 ✓

Total Loss = loss_2 + loss_3 + ... + loss_9  (8 terms, skip first 2)
```

### Backward Pass:

```python
total_loss.backward()  # Triggers BPTT

# Gradients flow backward through:
# 1. Loss at t=9 → ∂L/∂W, ∂L/∂M_9
# 2. M_9 = λM_8 + ... → ∂L/∂M_8
# 3. M_8 = λM_7 + ... → ∂L/∂M_7
# ...
# 9. M_2 = λM_1 + ... → ∂L/∂M_1
# 10. M_1 = 0 (STOP! Zero-padded states block gradients)

# Gradients accumulate through:
# - MPN weights W (learned via backprop)
# - Chain through M matrices (Hebbian updates)
# - NO gradients flow into zero-padded timesteps
```

### Key Points:

1. **Full BPTT** through valid sequence (t=2 to t=9 = 8 steps)
2. **Gradients stop** at zero-padded transitions (t=0, t=1)
3. **MPN state chains** through M_t = λM_{t-1} + η(h_t⊗x_t)
4. **Short sequences (L=10)** prevent vanishing gradients

---

## 5. Why This Works Better Than Trial-Based

### Old Trial-Based Approach:

```
Sample: Complete trial (variable length 14-31 steps)
Problem:
- Inconsistent sequence lengths
- No episode boundary handling
- Could sample trials from different episodes without zero-padding
- Longer sequences → vanishing gradients
```

### New Sequence Approach:

```
Sample: Fixed L=10 sequences with zero-padding
Benefits:
✓ Consistent BPTT window (always 10 steps)
✓ Episode boundaries handled via zero-padding
✓ Shorter sequences → less vanishing gradient
✓ Can sample ANY 10-step window (more data diversity)
✓ Matches DRQN paper approach
```

---

## 6. Example Training Step

### Full Training Step:

```python
# Environment interaction (collect data)
obs, reward, done, info = env.step(action)
replay_buffer.push(obs, action, reward, next_obs, done, episode_id=current_episode)

# Training (every step or every N steps)
if len(replay_buffer) >= 32 and step % train_freq == 0:
    # Sample 32 sequences of length 10
    sequences = replay_buffer.sample(batch_size=32)
    # Each sequence: list of 10 transitions (some may be zero-padded)

    # Compute loss with full BPTT through each sequence
    loss = compute_td_loss_sequences(dqn, target_dqn, sequences)
    # Forward: 32 sequences × 10 timesteps = 320 MPN forward passes
    # Backward: BPTT through valid timesteps only

    # Update weights
    optimizer.zero_grad()
    loss.backward()  # BPTT happens here!
    torch.nn.utils.clip_grad_norm_(dqn.parameters(), 10.0)  # Prevent explosion
    optimizer.step()  # Update W (Hebbian M is updated during forward pass)

    total_training_steps += 1

    # Update target network every 10,000 training steps
    if total_training_steps % 10000 == 0:
        target_dqn.load_state_dict(dqn.state_dict())
```

---

## 7. Memory & Computational Cost

### Memory:

- **Buffer capacity**: 50,000 transitions
- **Each transition**: ~obs_dim×4 + metadata ≈ 200 bytes (for obs_dim=20)
- **Total buffer**: ~10 MB

### Computation per training step:

- **Batch size**: 32 sequences
- **Sequence length**: 10 timesteps
- **Forward passes**: 32 × 10 = 320 MPN evaluations
- **Backward passes**: BPTT through ~8-10 valid timesteps per sequence
- **Total**: ~10-15ms per training step on GPU

---

## 8. Key Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| **sequence_length** | 10 | Short enough to avoid vanishing gradients, long enough for MPN context |
| **batch_size** | 32 | Matches DRQN paper, balances compute/memory |
| **buffer_size** | 50,000 transitions | ~250 episodes worth of data |
| **learning_rate** | 0.00025 | 4x lower than typical (from DRQN paper) |
| **target_update_freq** | 10,000 steps | Update based on training steps, not episodes |

---

## Summary

The sequence replay buffer:
1. ✅ Stores individual transitions with episode tracking
2. ✅ Samples fixed-length sequences (L=10)
3. ✅ Zero-pads at episode boundaries automatically
4. ✅ Enables full BPTT through valid timesteps only
5. ✅ Prevents training on invalid cross-episode sequences
6. ✅ Matches DRQN paper approach for stable recurrent Q-learning

This solves the critical issues identified in the DRQN paper and should dramatically improve training stability!
