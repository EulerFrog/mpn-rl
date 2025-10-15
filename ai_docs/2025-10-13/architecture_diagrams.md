# Architecture Diagrams: Current vs Proposed (NeuroGym Integration)

**Date**: 2025-10-13

---

## Overview

This document presents the architectural design for integrating NeuroGym environments into the MPN-RL library. The key innovation is **trial-based replay** that solves the "stale state" problem inherent in using standard DQN replay buffers with recurrent networks.

---

## 1. Architecture Comparison

Shows the current DQN implementation with transition-based replay (problematic for recurrent networks) vs the proposed trial-based replay system for NeuroGym.

![Architecture Comparison](diagrams/1_architecture_comparison.png)

**Key Differences**:
- **Current**: Stores individual transitions, random sampling breaks temporal coherence
- **Proposed**: Stores complete trials, samples full sequences, regenerates fresh MPN states

**Color Coding**:
- 🔴 Red (light): Problematic components in current approach
- 🟢 Green (light): New trial-based components (solution)
- 🔵 Blue (light): New NeuroGym wrapper layer

---

## 2. Training Flow Comparison

Contrasts the step-by-step training flow with the trial-based training flow, highlighting where the stale state problem occurs and how it's resolved.

![Training Flow Comparison](diagrams/2_training_flow_comparison.png)

**Current Flow Problems**:
- Random sampling of transitions from different episodes
- Stored MPN states were computed by old network parameters
- Temporal coherence broken

**Proposed Flow Benefits**:
- Complete trials collected as sequences
- States regenerated fresh during replay
- Temporal coherence preserved
- Natural handling of sparse rewards

---

## 3. Problem & Solution

Visualizes the core issue with random replay for recurrent networks and how trial-based replay from scratch solves it.

![Problem and Solution](diagrams/3_problem_and_solution.png)

**The Problem**: Stale States
- MPN state evolves over time: `M_t = λM_{t-1} + ηh_t x_t^T`
- Transitions stored with states from old network
- Random sampling mixes timesteps from different episodes
- Using stale states breaks recurrent dynamics

**The Solution**: Trial-Based Replay
- Store complete trial sequences (100-800 timesteps)
- Replay each trial from scratch with `state = init_state()`
- Regenerate MPN states during replay using current network
- Preserves temporal coherence and trial structure

---

## 4. Implementation Plan

Class diagram showing the key components to be implemented: `NeuroGymWrapper`, `TrialReplayBuffer`, and `compute_td_loss_trial()`.

![Implementation Plan](diagrams/4_implementation_plan.png)

**New Components**:

1. **NeuroGymWrapper**
   - Wraps NeuroGym environments
   - Flattens structured observations
   - Tracks trial boundaries via `info['new_trial']`

2. **TrialReplayBuffer**
   - Stores complete trial sequences
   - Samples random trials (not random transitions)
   - Handles variable-length sequences

3. **compute_td_loss_trial()**
   - Replays each trial from scratch
   - Regenerates MPN states with current network
   - Accumulates TD loss over full sequence

**Unchanged Components**:
- `MPNDQN`: Core Q-network with MPN
- `MPNModule`: Hebbian plasticity implementation
- `ExperimentManager`: Tracking and logging
- `TrainingVisualizer`: Result visualization

---

## Key Design Decisions

### Why Trial-Based Replay?

Compared to Stable-Baselines3's RecurrentPPO approach (which stores LSTM states alongside transitions and samples sequences), trial-based replay is:

✅ **Simpler**: No need to store MPN states (M matrices are large)
✅ **More robust**: Always uses current network to generate states
✅ **Natural fit**: NeuroGym tasks have inherent trial structure
✅ **Handles sparse rewards**: Full trial context available during training

### Tradeoffs

| Aspect | Transition-Based | Trial-Based |
|--------|-----------------|-------------|
| Memory Usage | Low | Higher (stores sequences) |
| Sample Efficiency | Higher (reuse transitions) | Lower (recompute states) |
| Temporal Coherence | ❌ Broken | ✅ Preserved |
| Implementation | Complex state storage | Simpler replay logic |
| Compatibility | Recurrent networks need special handling | Natural for sequential tasks |

---

## Target Environments

Starting with 3 memory-based NeuroGym tasks:

1. **ContextDecisionMaking-v0**: Context-dependent integration (similar to MPN paper)
2. **DelayMatchSample-v0**: Working memory across delay periods
3. **PerceptualDecisionMaking-v0**: Evidence integration over time

All have:
- Discrete action spaces
- Trial structure with fixation, stimulus, delay, response periods
- Sparse rewards (typically only at trial end)
- 100-800 timestep trials

---

## Implementation Status

**Phase**: ✅ Implementation Complete
**Completed Steps**:
1. ✅ Fixed diagram paths (images/ → diagrams/)
2. ✅ Implemented `TrialReplayBuffer` class in `model_utils.py`
3. ✅ Created `NeuroGymWrapper` in new file `neurogym_wrapper.py`
4. ✅ Implemented `compute_td_loss_trial()` function
5. ✅ Added `train_neurogym()` function to `main.py`
6. ✅ Added `train-neurogym` CLI command
7. ✅ Tested wrapper on ContextDecisionMaking-v0

**Ready for Training**: Use `python main.py train-neurogym --help` to see options

See `implementation_summary.md` for detailed documentation of all changes.

---

## References

### Design Documentation
- **Replay Buffer Insight**: `ai_docs/replay_buffer_insight.md` - Why standard replay breaks for recurrent networks
- **Progress Log**: `ai_docs/2025-10-13/neurogym_integration_progress.md` - Session notes

### Source Code References
- **Stable-Baselines3 RecurrentPPO**: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/ppo_recurrent/ppo_recurrent.py
- **RecurrentRolloutBuffer**: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/master/sb3_contrib/common/recurrent/buffers.py
- **NeuroGym Examples**: `/home/eulerfrog/KAM/neurogym/docs/examples/reinforcement_learning.ipynb`

### Papers
- **MPN Paper**: `/home/eulerfrog/KAM/mpn/elife-83035-v4 (1).pdf`
- **Recurrent Experience Replay**: https://arxiv.org/abs/1805.09692

---

## Notes

All diagrams use light background colors with thick colored borders to ensure text visibility:
- 🔴 Light red fill with red border: Problems/current issues
- 🟢 Light green fill with green border: Solutions/new components
- 🔵 Light blue fill with blue border: Wrapper/interface layers
