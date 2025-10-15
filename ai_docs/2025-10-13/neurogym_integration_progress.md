# Progress Log: NeuroGym Integration for MPN-DQN

**Date**: 2025-10-13
**Session**: Architecture & Design Phase

---

## Session Goals

Integrate neurogym environments into mpn-rl library using RL versions of the environments.

---

## Key Questions Addressed

### 1. Environment Selection
**Decision**: Focus on memory-based tasks similar to MPN paper
- ContextDecisionMaking-v0 (context-dependent integration)
- DelayMatchSample-v0 (working memory)
- PerceptualDecisionMaking-v0 (evidence integration)

### 2. Trial vs Episode Semantics
**Discussion**: How to map neurogym trials to RL episodes
- Neurogym has internal "trials" (end with `info['new_trial']=True`)
- Gymnasium has "episodes" (termination/truncation)
- **Current approach**: 1 neurogym trial = 1 RL episode (reset MPN state between trials)

### 3. Action Spaces
**Clarified**: Neurogym uses Discrete action spaces (not continuous)
- User confirmed discrete actions are correct

### 4. Observation Processing
**Decision**: Flatten observations automatically for now
- Neurogym has structured observations (e.g., `{'fixation': 0, 'stimulus': [1,2]}`)
- Start with flattening, add structured support later

---

## Major Insight: Replay Buffer Problem

### The Discovery
Examined how Stable-Baselines3's RecurrentPPO handles LSTM states to understand proper recurrent RL training.

### Key Finding
**Standard DQN replay buffer breaks for recurrent networks due to "stale states"**:
- MPN has internal state M_t that evolves over time
- Standard replay samples random transitions from different timesteps/trials
- Stored MPN states were computed by old network parameters
- Using stale states breaks temporal coherence

### How SB3 RecurrentPPO Solves It
1. Stores LSTM states alongside transitions in `RecurrentRolloutBuffer`
2. Tracks episode boundaries with `episode_starts` flags
3. Samples sequences (not random transitions)
4. Uses stored initial states when replaying sequences
5. Resets LSTM at episode boundaries

**Source Code Reference**: `sb3_contrib/common/recurrent/buffers.py`

---

## Proposed Solutions

### Option A: Step-by-Step Training (Current DQN Approach)
- Keep step-by-step updates
- Use experience replay
- Each step is independent sample
- **Problem**: MPN state resets between episodes, loses long-term memory

### Option B: Sequence-Based Training (MPN Paper Style)
- Collect entire trial sequences
- Train on full sequences at once
- Can mask loss to focus on important timesteps
- **Problem**: Can't use standard experience replay easily

### Our Approach: Trial-Based Replay (Hybrid)
**Decision**: Store complete trials, replay from scratch

```python
class TrialReplayBuffer:
    def push_trial(self, obs_list, action_list, reward_list, done_list):
        # Store complete trial as a unit

    def sample(self, batch_size):
        # Sample random complete trials

def compute_td_loss_trial(mpn, target_mpn, trial_batch, gamma):
    # Replay each trial from scratch
    # Regenerate MPN states during replay
    # No stale state problem!
```

**Rationale**:
- ✅ Simpler than storing MPN states
- ✅ Always fresh states (regenerated each replay)
- ✅ Natural for neurogym (trials are natural units)
- ✅ Handles sparse rewards (full trial context available)
- ✅ Matches neurogym structure

---

## Documentation Created

1. **`ai_docs/replay_buffer_insight.md`**
   - Documents why standard replay breaks for recurrent networks
   - Includes specific GitHub links to SB3 RecurrentPPO source code
   - Explains stale state problem
   - Shows how SB3 solves it
   - Proposes our trial-based solution

---

## Next Steps

1. **Create Mermaid architecture diagrams**
   - Current architecture (standard DQN with replay buffer)
   - Proposed architecture (trial-based replay with neurogym)
   - Training flow comparison

2. **Design neurogym wrapper**
   - Handle observation flattening
   - Track trial boundaries
   - Extract `info['new_trial']` flag

3. **Implement TrialReplayBuffer**
   - Store complete trials
   - Sample random trials
   - Handle variable-length sequences

4. **Modify training loop**
   - Collect complete trials
   - Push to trial buffer
   - Train on trial batches

5. **Test on first environment**
   - Start with ContextDecisionMaking-v0
   - Verify trial collection works
   - Validate training with trial replay

---

## Open Questions

1. **Training mode**: Should we support both step-by-step and sequence-based?
   - Start with trial-based (Option 2)
   - Add step-by-step with stored states later if needed

2. **MPN state persistence**: Reset at each trial or persist across trials?
   - Start with reset at each trial (simpler)
   - Make configurable later for meta-learning

3. **Memory concerns**: Trial-based replay uses more memory
   - Accept higher memory for now
   - Add compression/subsampling if needed

4. **Sequence length**: Neurogym trials are 100-800 timesteps
   - Train on full trials initially
   - Add truncation/gradient checkpointing if memory becomes issue

---

## Code Structure Decisions

### Implementation Phases
**Phase 1**: Core integration (single environment)
- TrialReplayBuffer class
- compute_td_loss_trial() function
- Neurogym wrapper
- Modified training loop
- Test on ContextDecisionMaking-v0

**Phase 2**: Multi-environment support
- Test on 3 initial environments
- General neurogym wrapper
- Environment-specific preprocessing

**Phase 3**: Advanced features
- State storage option (SB3-style)
- Multi-trial episodes
- Sequence subsampling
- Task-specific analysis

---

## References

### Source Code
- Stable-Baselines3 RecurrentPPO: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py
- RecurrentRolloutBuffer: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py
- NeuroGym RL Example: `/neurogym/docs/examples/reinforcement_learning.ipynb`

### Papers
- MPN Paper: `mpn/elife-83035-v4 (1).pdf`
- Recurrent Experience Replay: https://arxiv.org/abs/1805.09692

---

## Session Summary

**Status**: Design phase complete, ready to begin implementation

**Key Achievement**: Identified and solved the fundamental replay buffer problem for recurrent RL

**Next Session**: Create architecture diagrams and begin implementation
