# Code Changes Log - October 6, 2025

## New Files Created

### 1. `pca_analysis.py`
**Location**: `/home/eulerfrog/KAM/mpn_rl/pca_analysis.py`
**Lines**: 383
**Status**: New file

**Contents**:
- `participation_ratio(variances)` - Function to compute PR metric
- `HiddenStateCollector` class - Data collection during episodes
  - `reset()` - Clear all data
  - `start_episode()` - Begin new episode
  - `record_step(hidden, M_state, obs, action, reward)` - Record timestep
  - `end_episode()` - Finalize episode
  - `get_data()` - Return episode-separated data
  - `get_pooled_data()` - Return concatenated data
- `MPNPCAAnalyzer` class - PCA analysis
  - `fit_hidden_pca(hidden_states, n_components)` - Fit PCA on hidden states
  - `fit_M_pca(M_flat, n_components)` - Fit PCA on M matrices
  - `transform_hidden(hidden_states)` - Transform to PC space
  - `transform_M(M_flat)` - Transform M to PC space
  - `plot_variance_explained(save_path, max_components)` - Plot variance bars
- `collect_episodes(dqn, env, n_episodes, device, epsilon, max_steps)` - Run and collect
- `plot_trajectories_2d(trajectories_pcs, colors, pc_pairs, ...)` - Create visualizations
- `get_cartpole_colors(observations, feature_idx)` - Extract CartPole features
- Test suite at bottom (runs when `__main__`)

**Dependencies**:
```python
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional, List, Dict, Tuple
import gymnasium as gym
from pathlib import Path
```

## Modified Files

### 1. `main.py`
**Location**: `/home/eulerfrog/KAM/mpn_rl/main.py`
**Status**: Modified

**Changes**:

#### A. Updated Module Docstring (Lines 1-29)
Added `analyze` command to documentation:
```python
"""
Main CLI for MPN-DQN training, evaluation, and rendering

Commands:
    train   - Train a new MPN-DQN agent
    resume  - Resume training from checkpoint
    eval    - Evaluate a trained agent
    render  - Render episode(s) to GIF
    analyze - Analyze agent with PCA on hidden states and M matrices  # NEW

Examples:
    ...
    # Analyze with PCA  # NEW
    python main.py analyze --experiment-name my-agent --num-episodes 100 --color-feature 2  # NEW
"""
```

#### B. New Function: `analyze_agent(args)` (Lines 463-602)
**Lines**: ~140 lines
**Purpose**: Main analysis pipeline

**Code Structure**:
```python
def analyze_agent(args):
    """Analyze trained agent with PCA on hidden states and M matrices."""
    # 1. Setup
    exp_manager = ExperimentManager(args.experiment_name)
    config = exp_manager.load_config()
    analysis_dir = exp_manager.exp_dir / "analysis"
    analysis_dir.mkdir(exist_ok=True)

    # 2. Load model
    env = gym.make(config['env_name'], render_mode='rgb_array')
    device = get_device(args.device)
    dqn = MPNDQN(...)
    load_checkpoint_for_eval(exp_manager, dqn, checkpoint_name, device)

    # 3. Collect episodes
    from pca_analysis import (collect_episodes, MPNPCAAnalyzer,
                              plot_trajectories_2d, get_cartpole_colors)
    collector = collect_episodes(dqn, env, args.num_episodes, device, ...)
    data = collector.get_data()
    pooled = collector.get_pooled_data()

    # 4. PCA analysis
    analyzer = MPNPCAAnalyzer()
    if args.analyze_hidden:
        analyzer.fit_hidden_pca(pooled['hidden'], n_components=...)
    if args.analyze_M:
        analyzer.fit_M_pca(pooled['M_flat'], n_components=...)

    # 5. Plot variance
    if args.plot_variance:
        analyzer.plot_variance_explained(save_path=...)

    # 6. Plot trajectories
    if args.plot_trajectories:
        # Get colors (CartPole-specific or default)
        if config['env_name'] == 'CartPole-v1':
            colors = get_cartpole_colors(data['obs'], args.color_feature)
        else:
            colors = [np.full(len(obs), i) for i, obs in enumerate(data['obs'])]

        # Hidden state trajectories
        if args.analyze_hidden:
            hidden_pcs_list = [analyzer.transform_hidden(h) for h in data['hidden']]
            readout_weights = dqn.q_head.weight.detach().cpu().numpy()
            readout_pcs = analyzer.transform_hidden(readout_weights)
            plot_trajectories_2d(hidden_pcs_list, colors, ..., readout_pcs=readout_pcs)

        # M matrix trajectories
        if args.analyze_M:
            M_pcs_list = [analyzer.transform_M(M.reshape(...)) for M in data['M']]
            plot_trajectories_2d(M_pcs_list, colors, ...)

    env.close()
```

#### C. Updated `main()` Function (Lines 604-682)
Added analyze subparser:

**Before Line 653** (after render_parser):
```python
# Analyze command
analyze_parser = subparsers.add_parser('analyze', help='Analyze agent with PCA')
analyze_parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name')
analyze_parser.add_argument('--num-episodes', type=int, default=100, help='Number of episodes to analyze')
analyze_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load (default: best_model.pt)')
analyze_parser.add_argument('--max-steps', type=int, default=500, help='Max steps per episode')
analyze_parser.add_argument('--n-components', type=int, default=100, help='Number of PCA components')
analyze_parser.add_argument('--analyze-hidden', action='store_true', default=True, help='Analyze hidden states')
analyze_parser.add_argument('--analyze-M', action='store_true', default=True, help='Analyze M matrices')
analyze_parser.add_argument('--plot-variance', action='store_true', default=True, help='Plot explained variance')
analyze_parser.add_argument('--plot-trajectories', action='store_true', default=True, help='Plot PC trajectories')
analyze_parser.add_argument('--color-feature', type=int, default=0,
                           help='CartPole feature for coloring (0=pos, 1=vel, 2=angle, 3=ang_vel)')
analyze_parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu (default: auto)')
```

**Updated command dispatcher** (Lines 671-682):
```python
if args.command == 'train':
    train(args)
elif args.command == 'resume':
    resume_training(args)
elif args.command == 'eval':
    evaluate(args)
elif args.command == 'render':
    render_to_gif(args)
elif args.command == 'analyze':  # NEW
    analyze_agent(args)          # NEW
else:
    parser.print_help()
```

## Documentation Files Created

### In `ai_docs/`
1. `IMPLEMENTATION_REFERENCE.md` - Technical details and API reference
2. `USAGE_EXAMPLES.md` - Command-line examples and batch scripts
3. `RESEARCH_QUESTIONS.md` - Potential research questions (hypotheses only)

### In `ai_docs/2025-10-06/`
1. `SESSION_SUMMARY.md` - This session's summary
2. `CODE_CHANGES.md` - This file (detailed code changes)

## Files NOT Modified

The following files were referenced but not modified:
- `mpn_dqn.py` - Used by analyze command (unchanged)
- `mpn_module.py` - Used by MPN-DQN (unchanged)
- `model_utils.py` - ExperimentManager used (unchanged)
- `visualize.py` - Not used by PCA analysis (unchanged)
- `render_utils.py` - Not used by PCA analysis (unchanged)
- `example_usage.py` - Not used by PCA analysis (unchanged)

## Import Additions

### In `main.py`
No new top-level imports needed. Imports added within `analyze_agent()` function:
```python
from pca_analysis import (
    collect_episodes,
    MPNPCAAnalyzer,
    plot_trajectories_2d,
    get_cartpole_colors
)
```

## Directory Structure Changes

### Before
```
mpn_rl/
├── main.py
├── mpn_dqn.py
├── mpn_module.py
├── visualize.py
├── model_utils.py
├── render_utils.py
├── example_usage.py
└── experiments/
```

### After
```
mpn_rl/
├── main.py                    # MODIFIED
├── pca_analysis.py            # NEW
├── mpn_dqn.py
├── mpn_module.py
├── visualize.py
├── model_utils.py
├── render_utils.py
├── example_usage.py
├── experiments/               # Unchanged, but analysis/ subdirs will be created
└── ai_docs/                   # NEW
    ├── 2025-10-06/           # NEW
    │   ├── SESSION_SUMMARY.md
    │   └── CODE_CHANGES.md
    ├── IMPLEMENTATION_REFERENCE.md
    ├── USAGE_EXAMPLES.md
    └── RESEARCH_QUESTIONS.md
```

## Testing Verification

### Module Test
```bash
$ cd /home/eulerfrog/KAM/mpn_rl
$ python3 pca_analysis.py
```
**Result**: ✓ All tests passed

### CLI Test
```bash
$ python3 main.py analyze --help
```
**Result**: ✓ Help text displayed correctly

## Line Count Summary

**New code**:
- `pca_analysis.py`: 383 lines
- `main.py` additions: ~140 lines
- **Total new code**: ~523 lines

**Documentation**:
- `ai_docs/*.md`: ~4 files, substantial content
- **Purpose**: Reference only, no assumptions about results

## Git Status (for reference)
Not committed yet. Changes are:
- New file: `pca_analysis.py`
- Modified: `main.py`
- New directory: `ai_docs/`
- Untracked: All documentation files

## Integration Points

### With Existing Code
1. Uses `ExperimentManager` from `model_utils.py`
2. Uses `MPNDQN` from `mpn_dqn.py`
3. Uses `MPN` from `mpn_module.py` (via MPNDQN)
4. Uses `load_checkpoint_for_eval` from `model_utils.py`
5. Uses `get_device` from `main.py`

### New Output Locations
Analysis outputs saved to: `experiments/<experiment-name>/analysis/`
- Creates directory if doesn't exist
- Saves PNG files with fixed names
- Overwrites previous analysis if re-run

## Backward Compatibility
- No breaking changes to existing commands
- All existing functionality preserved
- New `analyze` command is additive only
- Experiments directory structure unchanged

## Future Extensibility

### Easy Extensions
1. Add new environment support: Implement `get_<env>_colors()` function
2. Add new plot types: Add methods to `MPNPCAAnalyzer` or new functions
3. Add new metrics: Extend `MPNPCAAnalyzer` with new analysis methods

### Harder Extensions
1. Temporal evolution analysis (PCA structure over training)
2. Multi-agent comparison plots
3. Interactive visualizations
4. 3D trajectory plots
5. Fixed point analysis

## Code Quality Notes
- Type hints used throughout
- Comprehensive docstrings
- Modular design
- Follows existing code style
- Error handling included
- Built-in testing
