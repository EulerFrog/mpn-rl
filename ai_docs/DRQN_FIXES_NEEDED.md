# Critical Fixes Based on DRQN Paper Analysis

**Date**: 2025-10-20
**Source**: "Deep Q-Learning with Recurrent Neural Networks" (Chen, Ying, Laird)

---

## Summary of Issues

After reading the DRQN paper, we identified **5 critical implementation errors** that explain our poor learning performance.

---

## Fix #1: Handle Episode Boundaries with Zero-Padding ⭐ HIGHEST PRIORITY

### Current Problem
When sampling from replay buffer, we sample a random transition and take the previous L states. If those states cross an episode boundary, we're training on invalid sequences!

### DRQN Paper Solution (Page 3)
```
"We propose a simple solution where we sample et ∼ U(D), take the previous L states,
{st−(L+1), . . . , st}, and then zero out states from previous games. For example,
if st−i was the end of the previous game, then we would have states
{0, . . . , 0, st−(i+1), . . . , st}"
```

### Implementation

**Current (broken):**
```python
def sample(self, batch_size):
    # Samples random complete trials
    return random.sample(self.buffer, batch_size)
```

**Fixed version:**
```python
class SequenceReplayBuffer:
    """Stores transitions with episode tracking, samples fixed-length sequences"""

    def __init__(self, capacity, sequence_length=10):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def push(self, obs, action, reward, next_obs, done, episode_id):
        """Store transition with episode ID"""
        self.buffer.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done,
            'episode_id': episode_id
        })

    def sample(self, batch_size):
        """Sample fixed-length sequences with zero-padding at episode boundaries"""
        sequences = []

        for _ in range(batch_size):
            # Sample random end position
            end_idx = random.randint(self.sequence_length, len(self.buffer))

            # Get sequence
            sequence = []
            current_episode = self.buffer[end_idx - 1]['episode_id']

            for i in range(self.sequence_length):
                idx = end_idx - self.sequence_length + i
                transition = self.buffer[idx]

                # Zero out if different episode
                if transition['episode_id'] != current_episode:
                    sequence.append(self._zero_transition())
                else:
                    sequence.append(transition)

            sequences.append(sequence)

        return sequences

    def _zero_transition(self):
        """Return zero-padded transition"""
        return {
            'obs': torch.zeros_like(self.buffer[0]['obs']),
            'action': 0,
            'reward': 0.0,
            'next_obs': torch.zeros_like(self.buffer[0]['obs']),
            'done': False,
            'episode_id': -1
        }
```

---

## Fix #2: Use Fixed Sequence Length ⭐ HIGH PRIORITY

### Current Problem
We replay entire trials (variable 10-30 steps). DRQN uses fixed L=16.

### DRQN Paper Reasoning (Page 4)
```
"We wanted to let the model look over enough previous states to make an informed decision,
but not so many that RNN suffers from vanishing gradients and training time issues which
is why we chose L = 16."
```

### Recommendation
- **For NeuroGym**: Use L = 10 or L = 20
- **Rationale**:
  - Trials are 10-30 steps
  - L=10 captures single trial
  - L=20 captures across trial boundaries
  - Avoids vanishing gradients from L=100+

### Implementation
```python
# In train_neurogym argument parser
train_ng_parser.add_argument('--sequence-length', type=int, default=10,
                            help='Fixed sequence length for BPTT (default: 10)')
```

---

## Fix #3: Reduce Learning Rate ⭐ MEDIUM PRIORITY

### Current Problem
- Our LR: 0.001
- DRQN LR: 0.00025
- **We're 4x too high!**

### DRQN Hyperparameters (Appendix A)
- learning rate: 0.00025
- optimizer: RMSProp with decay=0.99, β=0.01

### Implementation
```python
# Change default
train_ng_parser.add_argument('--learning-rate', type=float, default=0.00025,
                            help='Learning rate (DRQN used 0.00025)')

# Use RMSProp instead of Adam (optional)
optimizer = torch.optim.RMSprop(
    online_dqn.parameters(),
    lr=args.learning_rate,
    alpha=0.99,  # decay
    eps=0.01     # β
)
```

---

## Fix #4: Update Target Network by Steps, Not Episodes ⭐ MEDIUM PRIORITY

### Current Problem
- We update every 10 **episodes**
- DRQN updates every 10,000 **parameter updates** (SGD steps)

### DRQN Paper (Appendix A)
```
target network update frequency: 10000
number of parameter updates after which the target network updates
```

### Calculation
- 100 steps/episode × 10 episodes = 1,000 steps (WE DO THIS)
- vs 10,000 steps (DRQN)
- **We're updating target 10x too frequently!**

### Implementation
```python
# Track total training steps
total_steps = 0

for episode in range(args.num_episodes):
    # ... episode loop ...

    # Train
    if len(replay_buffer) >= batch_size:
        batch = replay_buffer.sample(batch_size)
        loss = compute_loss(...)
        optimizer.step()

        total_steps += 1

        # Update target network every 10,000 steps
        if total_steps % args.target_update_freq == 0:
            target_dqn.load_state_dict(online_dqn.state_dict())
```

---

## Fix #5: Consistent Minibatch Size ⭐ LOW PRIORITY

### Current Problem
We sample N trials (variable total transitions). DRQN samples fixed 32 sequences.

### DRQN Paper (Appendix A)
```
minibatch size: 32
number of experiences for SGD update
```

Each "experience" = one L-length sequence

### Implementation
```python
# Sample fixed number of sequences
BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

sequences = replay_buffer.sample(BATCH_SIZE)
# Each sequence has SEQUENCE_LENGTH timesteps
# Total: 32 sequences × 10 steps = 320 transitions
```

---

## Fix #6: Proper BPTT Implementation

### Current Problem
Our chunked BPTT doesn't match DRQN's approach.

### DRQN Approach
- Process entire sequence of L=16
- Full BPTT through all L steps
- Only detach gradients between different sampled sequences

### Our Approach
- Process variable-length trials
- Chunk with truncated BPTT
- More complex

### Recommendation
With fixed L=10-20, we can use **full BPTT** (no chunking needed):

```python
def compute_td_loss_sequences(dqn, target_dqn, sequences, gamma, device):
    """
    Compute loss over fixed-length sequences.

    Args:
        sequences: List of length BATCH_SIZE, each containing SEQUENCE_LENGTH transitions
    """
    total_loss = 0.0

    for sequence in sequences:
        # Initialize state (zero at start of each sequence)
        state = dqn.init_state(batch_size=1, device=device)

        # Forward pass through entire sequence (full BPTT)
        q_values_list = []
        for transition in sequence:
            obs = transition['obs'].unsqueeze(0).to(device)
            q_values, state = dqn(obs, state)
            q_values_list.append(q_values)

        # Compute TD targets
        with torch.no_grad():
            target_state = target_dqn.init_state(batch_size=1, device=device)
            target_q_list = []
            for transition in sequence:
                next_obs = transition['next_obs'].unsqueeze(0).to(device)
                target_q, target_state = target_dqn(next_obs, target_state)
                target_q_list.append(target_q)

        # Compute loss for each timestep
        for t, transition in enumerate(sequence):
            if transition['episode_id'] == -1:  # Zero-padded
                continue

            current_q = q_values_list[t][0, transition['action']]

            if t < len(sequence) - 1 and not transition['done']:
                target_q = transition['reward'] + gamma * target_q_list[t+1].max()
            else:
                target_q = transition['reward']

            total_loss += F.smooth_l1_loss(current_q, target_q)

    return total_loss / len(sequences)
```

---

## Priority Order

1. **Fix #1**: Episode boundary zero-padding (CRITICAL)
2. **Fix #2**: Fixed sequence length L=10-20 (HIGH)
3. **Fix #3**: Learning rate 0.001 → 0.00025 (MEDIUM)
4. **Fix #4**: Target update 10,000 steps (MEDIUM)
5. **Fix #6**: Simplified BPTT with fixed sequences (MEDIUM)
6. **Fix #5**: Consistent batch size (LOW)

---

## Expected Impact

Based on DRQN paper results:
- **Still expect modest improvements** (DRQN: 850 vs DQN: 700 = 21% improvement)
- **Training will be noisy** (their graphs show high variance)
- **May take long time** (they trained 6 days for convergence)

DRQN is not a magic solution - recurrent Q-learning is inherently difficult!

---

## Alternative: Consider On-Policy Learning

Given the challenges with recurrent Q-learning, consider:
- **RecurrentPPO** (on-policy, no replay issues)
- **A2C with LSTM** (simpler, on-policy)

On-policy methods avoid the replay mismatch problem entirely.

---

## Next Steps

Recommend implementing fixes in order:
1. Start with Fix #1 + #2 (episode boundaries + fixed length)
2. Test if learning improves
3. Add Fix #3 + #4 (hyperparameters)
4. Compare results

If still poor results, consider switching to on-policy approach.
