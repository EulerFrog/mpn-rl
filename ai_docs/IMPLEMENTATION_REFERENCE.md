# PCA Analysis Implementation Reference

## Purpose

This module implements PCA analysis tools for MPN-DQN agents, inspired by the methodology in the MPN paper (eLife-83035). It enables analysis of hidden state dynamics and synaptic modulation matrices during RL task execution.

## Implementation Overview

### Module: `pca_analysis.py`

**Core Classes:**

1. **`HiddenStateCollector`**
   - Collects data during episode rollouts
   - Stores: hidden states, M matrices, observations, actions, rewards
   - Methods:
     - `start_episode()` - Begin new episode collection
     - `record_step(hidden, M_state, obs, action, reward)` - Record single timestep
     - `end_episode()` - Finalize current episode
     - `get_data()` - Returns episode-separated data
     - `get_pooled_data()` - Returns concatenated data across episodes

2. **`MPNPCAAnalyzer`**
   - Performs PCA on collected data
   - Methods:
     - `fit_hidden_pca(hidden_states, n_components)` - Fit PCA on hidden states
     - `fit_M_pca(M_flat, n_components)` - Fit PCA on flattened M matrices
     - `transform_hidden(hidden_states)` - Transform to PC space
     - `transform_M(M_flat)` - Transform M matrices to PC space
     - `plot_variance_explained(save_path)` - Plot variance bar charts

**Core Functions:**

- `participation_ratio(variances)` - Computes PR = (Σλᵢ)² / Σλᵢ²
- `collect_episodes(dqn, env, n_episodes, device, epsilon, max_steps)` - Run episodes and collect data
- `plot_trajectories_2d(trajectories_pcs, colors, pc_pairs, ...)` - Create 2D trajectory visualizations
- `get_cartpole_colors(observations, feature_idx)` - Extract color values from CartPole observations

### Integration: `main.py`

**New Command:**
```bash
python3 main.py analyze [options]
```

**Function: `analyze_agent(args)`**
- Loads trained checkpoint
- Collects episodes using `collect_episodes()`
- Fits PCA on pooled data
- Generates visualizations
- Saves outputs to `experiments/<name>/analysis/`

## Data Flow

1. **Collection Phase:**
   ```
   Load checkpoint → Run episodes → Record (hidden, M, obs, actions, rewards)
   ```

2. **Pooling Phase:**
   ```
   Concatenate all timesteps → Flatten M matrices → Pool across episodes
   ```

3. **PCA Phase:**
   ```
   Fit PCA on pooled data → Compute explained variance → Calculate participation ratio
   ```

4. **Visualization Phase:**
   ```
   Transform episodes to PC space → Color by state feature → Generate plots
   ```

## Command-Line Interface

### Basic Usage
```bash
python3 main.py analyze --experiment-name <name>
```

### All Options
| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-name` | Required | Experiment to analyze |
| `--num-episodes` | 100 | Episodes to collect |
| `--checkpoint` | best_model.pt | Checkpoint file to load |
| `--max-steps` | 500 | Max steps per episode |
| `--n-components` | 100 | Number of PCA components |
| `--analyze-hidden` | True | Analyze hidden states |
| `--analyze-M` | True | Analyze M matrices |
| `--plot-variance` | True | Plot explained variance |
| `--plot-trajectories` | True | Plot PC trajectories |
| `--color-feature` | 0 | CartPole feature index (0-3) |
| `--device` | auto | Compute device |

### CartPole Feature Indices
- 0: Cart Position
- 1: Cart Velocity
- 2: Pole Angle
- 3: Pole Angular Velocity

## Output Files

All outputs saved to: `experiments/<experiment-name>/analysis/`

### 1. `pca_variance.png`
- Two-panel bar chart
- Left: Hidden state explained variance per component
- Right: M matrix explained variance per component
- Title includes participation ratio value

### 2. `trajectories_hidden.png`
- Three panels: (PC0, PC1), (PC0, PC2), (PC1, PC2)
- Gray lines: Temporal trajectories
- Colored points: Colored by selected state feature
- Red dots: Final states
- Arrows: Q-value readout weight vectors in PC space

### 3. `trajectories_M.png`
- Same layout as hidden trajectories
- Shows M matrix dynamics instead of hidden states
- No readout vectors (only applicable to hidden states)

## Technical Details

### Data Dimensions

**Hidden States:**
- Raw: `[n_episodes, varying_T, hidden_dim]`
- Pooled: `[total_timesteps, hidden_dim]`
- Post-PCA: `[total_timesteps, n_components]`

**M Matrices:**
- Raw: `[n_episodes, varying_T, hidden_dim, obs_dim]`
- Flattened & Pooled: `[total_timesteps, hidden_dim * obs_dim]`
- Post-PCA: `[total_timesteps, n_components]`

### PCA Methodology

Following the MPN paper approach:
1. Pool all timesteps from all episodes
2. Fit single PCA model on pooled data
3. Transform individual episodes using fitted PCA
4. Visualize trajectories in low-dimensional PC space

### Participation Ratio

Formula: `PR = (Σλᵢ)² / Σλᵢ²`

where λᵢ are the PCA eigenvalues (explained variances).

Properties:
- Minimum: 1 (all variance in 1 dimension)
- Maximum: N (variance spread uniformly across N dimensions)
- Scale-dependent: depends on number of components computed

## Extending to Other Environments

### Current Implementation
- CartPole-v1: Uses `get_cartpole_colors()` with 4 feature indices
- Other environments: Falls back to episode index coloring

### Adding New Environment Support

1. **Create color extraction function:**
```python
def get_myenv_colors(observations: List[np.ndarray],
                     feature_idx: int = 0) -> List[np.ndarray]:
    """Extract color values from MyEnv observations."""
    return [obs[:, feature_idx] for obs in observations]
```

2. **Update `analyze_agent()` in main.py:**
```python
if config['env_name'] == 'MyEnv-v0':
    feature_names = ['Feature1', 'Feature2', ...]
    colors = get_myenv_colors(data['obs'], feature_idx=args.color_feature)
    color_label = feature_names[args.color_feature]
```

## Testing

Run module tests:
```bash
python3 pca_analysis.py
```

Expected output:
- Participation ratio calculation
- Data collection simulation
- PCA fitting and transformation
- Success messages for all tests

## Dependencies

- `numpy` - Array operations
- `torch` - Neural network operations
- `matplotlib` - Plotting
- `sklearn.decomposition.PCA` - PCA computation
- `gymnasium` - Environment simulation

## Files Created

### Python Modules
- `pca_analysis.py` - Core analysis implementation (383 lines)

### Documentation (ai_docs/)
- `IMPLEMENTATION_REFERENCE.md` - This file
- `USAGE_EXAMPLES.md` - Command examples
- `RESEARCH_QUESTIONS.md` - Potential research directions

### Modified Files
- `main.py` - Added `analyze` command and `analyze_agent()` function

## Known Limitations

1. **Memory**: Large numbers of long episodes may cause memory issues
2. **CartPole-specific**: Color extraction currently hardcoded for CartPole
3. **2D Visualization Only**: Only 2D PC projections implemented
4. **No Temporal Analysis**: Does not analyze how PCA structure changes over training
5. **No Comparison Tools**: Cannot directly compare multiple agents/checkpoints

## Future Extensions (Not Yet Implemented)

Potential additions:
- Temporal evolution of participation ratio during training
- Direct comparison plots across multiple checkpoints
- 3D trajectory visualizations
- Fixed point analysis
- State-space partitioning analysis
- Clustering analysis of final states
- Correlation analysis between PC loadings and task variables
