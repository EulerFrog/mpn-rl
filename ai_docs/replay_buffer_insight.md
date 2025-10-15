# Replay Buffer Insight: Why Standard DQN Replay Breaks for Recurrent Networks

**Date**: 2025-10-13

---

## The Core Problem

Standard experience replay, as used in DQN, **breaks temporal coherence** when training recurrent networks like MPNs.

### Standard DQN Replay Buffer

```python
class ReplayBuffer:
    def push(self, obs, action, reward, next_obs, done):
        # Store individual transitions
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        # Sample random transitions
        return random.sample(self.buffer, batch_size)
```

**What happens**: Transitions from different timesteps and different episodes are randomly mixed together.

---

## Why This Breaks for MPNs

### The MPN Has Recurrent State

The MPN maintains internal state across time:
- **h_t**: Hidden activations
- **M_t**: Hebbian plasticity matrix

These evolve as: `h_t, M_t = MPN(obs_t, h_{t-1}, M_{t-1})`

### The Problem: Stale States

When we store transitions in the replay buffer:

```python
# During episode collection (e.g., timestep 100 in a trial)
obs_100, action_100, reward_100, next_obs_100, done = env.step(...)
state_100 = current_mpn_state  # (h_100, M_100)

# Store in buffer
buffer.push(obs_100, action_100, reward_100, next_obs_100, done, state_100)
```

Later during training:

```python
# Sample random batch - might include transition from timestep 100
batch = buffer.sample(64)

# Problem: state_100 stored in buffer was computed by the OLD network!
# The network has been updated thousands of times since then.
# The stored state no longer matches the current network's dynamics.

# This breaks temporal coherence:
q_values, new_state = mpn(obs_100, state_100)  # state_100 is STALE!
```

---

## Why This Doesn't Affect Standard DQN

Standard DQN has **no recurrent state**:

```python
q_values = dqn(obs)  # No dependence on previous timesteps
```

Each observation is processed independently, so random sampling works fine.

---

## How Stable-Baselines3 RecurrentPPO Solves This

### Key Insight from SB3

RecurrentPPO addresses this by storing LSTM states and episode boundaries in a specialized buffer.

**Source Code References:**

1. **RecurrentRolloutBuffer stores LSTM states**
   - File: `sb3_contrib/common/recurrent/buffers.py`
   - Link: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py#L78-L90

   ```python
   # Hidden and cell states for policy network
   self.hidden_states_pi = np.zeros((self.buffer_size, self.n_envs, *hidden_state_shape), dtype=np.float32)
   self.cell_states_pi = np.zeros((self.buffer_size, self.n_envs, *hidden_state_shape), dtype=np.float32)

   # Episode start flags to know when to reset LSTM
   self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
   ```

2. **Storing LSTM states during rollout collection**
   - File: `sb3_contrib/common/recurrent/buffers.py`
   - Link: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py#L148-L164

   ```python
   def add(
       self,
       # ... standard obs, actions, rewards ...
       lstm_states: RNNStates,  # <-- LSTM states stored here
       episode_starts: np.ndarray,  # <-- Episode boundaries
   ) -> None:
       # Store LSTM states alongside transitions
       self.hidden_states_pi[self.pos] = np.array(lstm_states.pi[0])
       self.cell_states_pi[self.pos] = np.array(lstm_states.pi[1])
       self.episode_starts[self.pos] = episode_start
   ```

3. **Sampling sequences (not random transitions)**
   - File: `sb3_contrib/common/recurrent/buffers.py`
   - Link: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py#L222-L246

   ```python
   def get(self, batch_size: Optional[int] = None) -> Generator[RecurrentRolloutBufferSamples, None, None]:
       # Creates sequences respecting episode boundaries
       # Uses episode_starts to identify sequence breaks
       seq_start, seq_end = create_sequencers(
           self.episode_starts,  # <-- Respects episode boundaries
           self.env_change,
           self.device,
       )
   ```

4. **Maintaining LSTM states between rollouts**
   - File: `sb3_contrib/ppo_recurrent/ppo_recurrent.py`
   - Link: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L165-L172

   ```python
   # Save last LSTM states for continuity across rollouts
   self._last_lstm_states = lstm_states

   # Reset LSTM states at episode boundaries
   lstm_states.pi = (
       lstm_states.pi[0] * (1 - episode_starts),  # Reset if episode_start=1
       lstm_states.pi[1] * (1 - episode_starts),
   )
   ```

5. **Using stored states during training**
   - File: `sb3_contrib/ppo_recurrent/ppo_recurrent.py`
   - Link: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py#L256-L268

   ```python
   # Retrieve stored LSTM states from buffer
   values, log_prob, entropy = self.policy.evaluate_actions(
       rollout_data.observations,
       rollout_data.actions,
       rollout_data.lstm_states,  # <-- Use stored initial states
       rollout_data.episode_starts,  # <-- Reset at boundaries
   )
   ```

---

## Implications for MPN-DQN

We cannot use standard transition-based replay. Instead, we need:

### Option 1: Store MPN States with Transitions (SB3-style)
- Store (h_t, M_t) alongside each transition
- Sample sequences respecting episode boundaries
- Use stored initial states when replaying
- **Similar to RecurrentRolloutBuffer approach**

### Option 2: Trial-Based Replay ⭐
- Store complete trials as units
- Replay trials from scratch (regenerate MPN states)
- No stale state problem
- **Simpler than Option 1**

---

## Why Neurogym Makes This Worse

Neurogym tasks have **sparse rewards** (often only at trial end):

```
Timestep 0-100:  reward = 0 (fixation, stimulus, delay)
Timestep 101:    reward = +1 or 0 (response)
```

With random transition sampling:
- Most sampled transitions have reward = 0
- Temporal context needed to understand trial structure is lost
- Credit assignment through 100+ timesteps is difficult

**Trial-based replay** naturally preserves this temporal structure.

---

## Summary

**The fundamental issue**: Recurrent networks have state that depends on sequence history. Random sampling of individual transitions breaks this dependency and leads to training on "stale" states that no longer match the current network dynamics.

**The solution**: Sample and train on sequences (either with stored initial states or by replaying from scratch) rather than random individual transitions.

---

## References

### Stable-Baselines3 RecurrentPPO Implementation
- Main algorithm: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py
- Recurrent buffer: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py
- Recurrent policies: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/policies.py

### Papers
- DQN (Mnih et al., 2015) - uses feedforward networks: https://arxiv.org/abs/1312.5602
- Recurrent Experience Replay (Kapturowski et al., 2018): https://arxiv.org/abs/1805.09692
