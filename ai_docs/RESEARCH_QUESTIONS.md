# Potential Research Questions

This document outlines potential research questions that can be investigated using the PCA analysis tools. These are hypotheses and open questions - actual results should be documented separately after data collection.

## Dimensionality Questions

### Hidden State Dimensionality
- What is the effective dimensionality (participation ratio) of MPN-DQN hidden states during CartPole?
- How does hidden state dimensionality compare to the network's nominal hidden dimension?
- Does dimensionality change as training progresses?
- How does dimensionality compare between successful and failed episodes?

### M Matrix Dimensionality
- What is the effective dimensionality of the M matrix space?
- Is M matrix dimensionality higher or lower than hidden state dimensionality?
- How does the high-dimensional M matrix space (hidden_dim × obs_dim) collapse to a lower-dimensional manifold?

### Comparative Dimensionality
- How does MPN-DQN dimensionality compare to standard RNN-based DQN?
- Does the Hebbian plasticity mechanism (M matrix) enable lower-dimensional representations than standard recurrence?

## Trajectory Structure Questions

### Temporal Dynamics
- Do hidden state trajectories converge to fixed points or attractors?
- Are there distinct trajectory patterns for different task phases (e.g., balancing vs. recovering)?
- How do trajectories evolve from episode start to episode end?

### State Space Organization
- Do trajectories cluster by task outcome (success/failure, reward level)?
- Are there separable regions in PC space corresponding to different behavioral regimes?
- Is there a clear boundary between "balanced" and "falling" states in PC space?

### Readout Alignment
- How are Q-value readout vectors oriented relative to trajectory structure?
- Do action-selection decision boundaries align with trajectory clustering?
- Can we predict action selection from position in PC space?

## State Feature Correlations

### CartPole Features
Questions for each feature (position, velocity, angle, angular velocity):
- How does this state feature correlate with position in PC space?
- Which principal components encode which state features?
- Are features encoded separately or jointly in the hidden state?

### Feature Encoding Hypotheses
- Is pole angle the primary feature encoded (as it's most critical for balancing)?
- Are position and velocity encoded as secondary features?
- Do angular features (angle, angular velocity) cluster together in PC space?

## M Matrix vs Hidden State Dynamics

### Functional Differences
- How do M matrix trajectories differ from hidden state trajectories?
- Does the M matrix capture "context" or "memory" distinct from instantaneous state?
- Do M matrix principal components capture different aspects of the task than hidden state PCs?

### Temporal Evolution
- Does the M matrix evolve more slowly than hidden states (due to λ decay)?
- Can we observe "memory effects" where M matrix state predicts future hidden states?
- How quickly does the M matrix adapt to changing observations?

## Training Progression Questions

### Dimensionality Evolution
- Does effective dimensionality decrease as training progresses (suggesting compression)?
- Does the participation ratio stabilize at later training stages?
- Is there a relationship between dimensionality and task performance?

### Trajectory Evolution
- Do trajectories become more stereotyped (less variable) with training?
- Do final states cluster more tightly as the policy improves?
- How does trajectory structure change from early to late training?

### Representational Changes
- Which principal components emerge first during training?
- Do early PCs capture coarse features and later PCs capture fine details?
- Is there evidence of representational reorganization during training?

## Comparison with MPN Paper Results

### Methodological Parallels
- Can we replicate the type of structure observed in the MPN paper (eLife-83035) for supervised tasks?
- How does RL task structure differ from supervised classification task structure?
- Are participation ratios comparable between RL and supervised settings?

### Context-Dependence
The MPN paper analyzes context-dependent tasks. For RL:
- Does the M matrix capture "context" analogous to the paper's task structure?
- Are there distinct M matrix states for different phases of the CartPole episode?
- Can we identify "task modes" in the M matrix space?

## Hyperparameter Questions

### Eta (Hebbian Learning Rate)
- How does eta affect the dimensionality of M matrix representations?
- Does higher eta lead to more rapid M matrix changes and different trajectory structure?
- Is there an optimal eta for compact representations?

### Lambda (Decay Factor)
- How does lambda affect the temporal structure of M matrix trajectories?
- Does higher lambda (slower decay) lead to longer "memory" visible in trajectory structure?
- How does lambda interact with episode length and task dynamics?

### Hidden Dimension
- Does increasing hidden_dim increase effective dimensionality proportionally?
- Is there a plateau where additional hidden units don't increase participation ratio?
- How does hidden dimension affect trajectory clustering?

## Hypothesis Testing Framework

For any investigation, consider:

1. **Null Hypothesis**: What would we expect from a random/unstructured network?
2. **Alternative Hypothesis**: What specific structure do we expect to observe?
3. **Metrics**: How will we quantify the observation (PR, cluster separation, correlation, etc.)?
4. **Controls**: What comparisons are needed (random network, standard RNN, different hyperparameters)?
5. **Statistical Tests**: How will we assess significance (if applicable)?

## Open Questions for Future Investigation

### Mechanistic Understanding
- What computations are performed in the principal component space?
- Can we decode task-relevant variables directly from PC coordinates?
- How does the learned PC structure relate to the task's optimal control structure?

### Generalization
- Do the PC structures generalize across different CartPole variants?
- If we train on short episodes, do the PCs still capture long episode dynamics?
- How do the representations differ for other continuous control tasks?

### Comparison with Biological Systems
- Does the MPN-DQN dimensionality match typical neural dimensionality in biological systems?
- Are the trajectories qualitatively similar to neural population trajectories during motor control?
- Does the separation of W (slow) and M (fast) weights mirror biological learning timescales?

## Experimental Design Considerations

When investigating these questions:

1. **Episode Count**: More episodes needed for stable PCA estimates (suggest 100+ for initial analyses)
2. **Checkpoint Selection**: Compare early, mid, late, and final training checkpoints
3. **Feature Selection**: Analyze all 4 CartPole features to get complete picture
4. **Replication**: Multiple training runs needed to assess variability
5. **Visualization**: Use multiple PC pairs beyond (PC0, PC1) to avoid missing structure

## Notes on Interpretation

Remember:
- PCA finds linear structure; nonlinear structure may not be captured
- Participation ratio is scale-dependent and sensitive to noise
- Trajectory clustering depends on episode diversity during evaluation
- Visual patterns should be quantified with metrics for rigorous conclusions
- Correlation ≠ causation; structure in PC space doesn't imply computational necessity

## Current Status

**Implementation**: Complete ✓
**Data Collection**: Not yet performed
**Analysis**: Not yet performed
**Conclusions**: None yet - awaiting data

All questions above are hypotheses to investigate, not established findings.
