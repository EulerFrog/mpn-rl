"""
Visualize the GoNogo task structure using actual experimental data.

Runs actual trials and shows:
1. Real observation patterns for Go and NoGo trials
2. Actual reward timings
3. Trial structure with real data
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import neurogym as ngym
import numpy as np
import torch
from torchrl.envs import Compose, StepCounter, InitTracker, TransformedEnv
from torchrl.envs.libs.gym import GymEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create environment
gymenv = GymEnv("GoNogo-v0", device=device)
gymenv._env = NeuroGymInfoWrapper(gymenv._env)

env = TransformedEnv(
    gymenv,
    Compose(
        NeuroGymInfoTransform(),
        StepCounter(),
        InitTracker(),
    )
)

print("Collecting experimental data from GoNogo task...")


def collect_trial_data(env, action_policy='optimal', max_steps=100, seed=None):
    """
    Collect data from a single trial.

    action_policy: 'optimal', 'always_go', 'always_fixate', 'random'
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    td = env.reset()

    observations = []
    rewards = []
    actions = []
    ground_truths = []
    new_trials = []

    # Store the initial ground truth if available
    if "gt" in td.keys():
        initial_gt = td["gt"].cpu().numpy()
    elif "ground_truth" in td.keys():
        initial_gt = td["ground_truth"].cpu().numpy()
    else:
        initial_gt = None

    for step in range(max_steps):
        # Store data
        obs = td["observation"].cpu().numpy()
        observations.append(obs)

        # Check various keys for ground truth
        if "gt" in td.keys():
            ground_truths.append(td["gt"].cpu().numpy())
        elif "ground_truth" in td.keys():
            ground_truths.append(td["ground_truth"].cpu().numpy())
        elif initial_gt is not None:
            ground_truths.append(initial_gt)

        if "new_trial" in td.keys():
            new_trials.append(td["new_trial"].cpu().numpy())

        # Select action based on policy
        if action_policy == 'optimal':
            # Optimal policy: fixate during fixation/stimulus/delay, respond based on stimulus during decision
            # Check if we're in decision period (when fixation signal is 0)
            if obs[0] == 0:  # Decision period
                # Get ground truth from previous observation
                if len(ground_truths) > 0:
                    gt = ground_truths[-1]
                    action_choice = int(gt)  # Go if gt=1, Fixate if gt=0
                else:
                    action_choice = 0
            else:
                action_choice = 0  # Fixate during other periods
        elif action_policy == 'always_go':
            action_choice = 1
        elif action_policy == 'always_fixate':
            action_choice = 0
        elif action_policy == 'random':
            action_choice = np.random.randint(0, 2)

        actions.append(action_choice)

        # Create action tensor
        action = torch.zeros(env.action_spec.shape, device=device)
        action[action_choice] = 1.0

        # Take step
        td["action"] = action
        td = env.step(td)

        # Get reward
        reward = td["next", "reward"].item()
        rewards.append(reward)

        # Check if new trial started
        if "new_trial" in td["next"].keys() and td["next", "new_trial"].item() > 0.5:
            break

        # Check if done
        done = td["next", "done"].item() if "done" in td["next"].keys() else False
        terminated = td["next", "terminated"].item() if "terminated" in td["next"].keys() else False

        if done or terminated:
            break

        td = env.step_mdp(td)

    return {
        'observations': np.array(observations),
        'rewards': np.array(rewards),
        'actions': np.array(actions),
        'ground_truths': np.array(ground_truths) if ground_truths else None,
    }


# Collect multiple trials with optimal policy
print("\nCollecting trial examples...")
trials = []

# Try to collect at least one Go and one NoGo trial
for seed in range(20):
    trial_data = collect_trial_data(env, action_policy='optimal', max_steps=100, seed=seed)

    # Determine trial type from ground truth
    if trial_data['ground_truths'] is not None and len(trial_data['ground_truths']) > 0:
        trial_type = trial_data['ground_truths'][0]
        trial_data['trial_type'] = 'GO' if trial_type == 1 else 'NOGO'
        trials.append(trial_data)

        # Print trial info
        total_reward = trial_data['rewards'].sum()
        print(f"  Trial {len(trials)}: {trial_data['trial_type']} - Reward: {total_reward:.2f}, Steps: {len(trial_data['observations'])}")

        # Stop if we have at least one of each type
        go_trials = [t for t in trials if t['trial_type'] == 'GO']
        nogo_trials = [t for t in trials if t['trial_type'] == 'NOGO']

        if len(go_trials) >= 3 and len(nogo_trials) >= 3:
            break

# Separate trials by type
go_trials = [t for t in trials if t['trial_type'] == 'GO']
nogo_trials = [t for t in trials if t['trial_type'] == 'NOGO']

print(f"\nCollected {len(go_trials)} Go trials and {len(nogo_trials)} NoGo trials")

# ============================================================================
# Create Visualization
# ============================================================================
fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(5, 2, hspace=0.35, wspace=0.3, height_ratios=[0.8, 1.2, 1.2, 1, 0.8])

# ============================================================================
# Title and description
# ============================================================================
fig.suptitle('Go/No-Go Task: Experimental Data Analysis',
             fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# Row 1: Task Description (spans both columns)
# ============================================================================
ax_desc = fig.add_subplot(gs[0, :])
ax_desc.axis('off')

desc_text = """
TASK STRUCTURE: Each trial has 4 periods: Fixation (variable) → Stimulus (500ms) → Delay (500ms) → Decision (500ms)
OBSERVATIONS: [fixation_signal, nogo_stimulus, go_stimulus] - values are 0 or 1
ACTIONS: 0 = Fixate, 1 = Go
GOAL: During decision period, respond (Go=1) only if Go stimulus was shown; otherwise maintain fixation (Fixate=0)
"""

ax_desc.text(0.5, 0.5, desc_text, ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', edgecolor='black', linewidth=2),
            family='monospace')

# ============================================================================
# Row 2: Example Go Trial
# ============================================================================
if len(go_trials) > 0:
    trial = go_trials[0]

    # Left: Observations
    ax_go_obs = fig.add_subplot(gs[1, 0])
    ax_go_obs.set_title(f'GO TRIAL - Observations (Reward: {trial["rewards"].sum():.2f})',
                       fontsize=12, fontweight='bold', color='darkgreen')
    ax_go_obs.set_xlabel('Timestep', fontsize=10)
    ax_go_obs.set_ylabel('Value (offset)', fontsize=10)

    timesteps = np.arange(len(trial['observations']))
    obs = trial['observations']

    # Plot observations with offsets
    ax_go_obs.plot(timesteps, obs[:, 0] + 0, label='Fixation Signal', linewidth=2, color='blue')
    ax_go_obs.plot(timesteps, obs[:, 1] + 2, label='NoGo Stimulus', linewidth=2, color='red')
    ax_go_obs.plot(timesteps, obs[:, 2] + 4, label='Go Stimulus', linewidth=2, color='green')

    ax_go_obs.legend(loc='upper right', fontsize=9)
    ax_go_obs.grid(True, alpha=0.3)
    ax_go_obs.set_ylim(-0.5, 6)

    # Right: Rewards and Actions
    ax_go_rew = fig.add_subplot(gs[1, 1])
    ax_go_rew.set_title('GO TRIAL - Actions and Rewards', fontsize=12, fontweight='bold', color='darkgreen')
    ax_go_rew.set_xlabel('Timestep', fontsize=10)

    # Plot rewards as bars
    reward_colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in trial['rewards']]
    ax_go_rew.bar(timesteps, trial['rewards'], color=reward_colors, alpha=0.6, label='Reward', width=1.0)
    ax_go_rew.set_ylabel('Reward', fontsize=10, color='black')
    ax_go_rew.tick_params(axis='y', labelcolor='black')

    # Plot actions on secondary axis
    ax_go_act = ax_go_rew.twinx()
    ax_go_act.plot(timesteps, trial['actions'], 'o-', color='purple', linewidth=2,
                  markersize=5, label='Action', alpha=0.7)
    ax_go_act.set_ylabel('Action (0=Fixate, 1=Go)', fontsize=10, color='purple')
    ax_go_act.set_ylim(-0.2, 1.4)
    ax_go_act.tick_params(axis='y', labelcolor='purple')
    ax_go_act.set_yticks([0, 1])
    ax_go_act.set_yticklabels(['Fixate', 'Go'])

    ax_go_rew.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_go_rew.grid(True, alpha=0.3, axis='y')
    ax_go_rew.legend(loc='upper left', fontsize=9)
    ax_go_act.legend(loc='upper right', fontsize=9)

# ============================================================================
# Row 3: Example NoGo Trial
# ============================================================================
if len(nogo_trials) > 0:
    trial = nogo_trials[0]

    # Left: Observations
    ax_nogo_obs = fig.add_subplot(gs[2, 0])
    ax_nogo_obs.set_title(f'NO-GO TRIAL - Observations (Reward: {trial["rewards"].sum():.2f})',
                         fontsize=12, fontweight='bold', color='darkred')
    ax_nogo_obs.set_xlabel('Timestep', fontsize=10)
    ax_nogo_obs.set_ylabel('Value (offset)', fontsize=10)

    timesteps = np.arange(len(trial['observations']))
    obs = trial['observations']

    # Plot observations with offsets
    ax_nogo_obs.plot(timesteps, obs[:, 0] + 0, label='Fixation Signal', linewidth=2, color='blue')
    ax_nogo_obs.plot(timesteps, obs[:, 1] + 2, label='NoGo Stimulus', linewidth=2, color='red')
    ax_nogo_obs.plot(timesteps, obs[:, 2] + 4, label='Go Stimulus', linewidth=2, color='green')

    ax_nogo_obs.legend(loc='upper right', fontsize=9)
    ax_nogo_obs.grid(True, alpha=0.3)
    ax_nogo_obs.set_ylim(-0.5, 6)

    # Right: Rewards and Actions
    ax_nogo_rew = fig.add_subplot(gs[2, 1])
    ax_nogo_rew.set_title('NO-GO TRIAL - Actions and Rewards', fontsize=12, fontweight='bold', color='darkred')
    ax_nogo_rew.set_xlabel('Timestep', fontsize=10)

    # Plot rewards as bars
    reward_colors = ['green' if r > 0 else 'red' if r < 0 else 'gray' for r in trial['rewards']]
    ax_nogo_rew.bar(timesteps, trial['rewards'], color=reward_colors, alpha=0.6, label='Reward', width=1.0)
    ax_nogo_rew.set_ylabel('Reward', fontsize=10, color='black')
    ax_nogo_rew.tick_params(axis='y', labelcolor='black')

    # Plot actions on secondary axis
    ax_nogo_act = ax_nogo_rew.twinx()
    ax_nogo_act.plot(timesteps, trial['actions'], 'o-', color='purple', linewidth=2,
                    markersize=5, label='Action', alpha=0.7)
    ax_nogo_act.set_ylabel('Action (0=Fixate, 1=Go)', fontsize=10, color='purple')
    ax_nogo_act.set_ylim(-0.2, 1.4)
    ax_nogo_act.tick_params(axis='y', labelcolor='purple')
    ax_nogo_act.set_yticks([0, 1])
    ax_nogo_act.set_yticklabels(['Fixate', 'Go'])

    ax_nogo_rew.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_nogo_rew.grid(True, alpha=0.3, axis='y')
    ax_nogo_rew.legend(loc='upper left', fontsize=9)
    ax_nogo_act.legend(loc='upper right', fontsize=9)

# ============================================================================
# Row 4: Reward Statistics from Collected Trials
# ============================================================================
ax_stats = fig.add_subplot(gs[3, :])
ax_stats.axis('off')

# Analyze rewards across all trials
go_rewards = [t['rewards'].sum() for t in go_trials]
nogo_rewards = [t['rewards'].sum() for t in nogo_trials]

stats_text = f"""
EXPERIMENTAL RESULTS (Optimal Policy):

GO TRIALS ({len(go_trials)} trials):
  • Mean reward per trial: {np.mean(go_rewards):.2f} ± {np.std(go_rewards):.2f}
  • Range: [{np.min(go_rewards):.2f}, {np.max(go_rewards):.2f}]
  • Mean trial length: {np.mean([len(t['observations']) for t in go_trials]):.1f} steps

NO-GO TRIALS ({len(nogo_trials)} trials):
  • Mean reward per trial: {np.mean(nogo_rewards):.2f} ± {np.std(nogo_rewards):.2f}
  • Range: [{np.min(nogo_rewards):.2f}, {np.max(nogo_rewards):.2f}]
  • Mean trial length: {np.mean([len(t['observations']) for t in nogo_trials]):.1f} steps

OPTIMAL STRATEGY: Fixate during fixation/stimulus/delay periods, then respond based on stimulus type during decision period.
"""

ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', edgecolor='black', linewidth=2),
             family='monospace')

# ============================================================================
# Row 5: Reward Contingency Summary (from code analysis)
# ============================================================================
ax_reward_table = fig.add_subplot(gs[4, :])
ax_reward_table.axis('off')
ax_reward_table.set_title('Reward Contingencies', fontsize=12, fontweight='bold')

table_data = [
    ['Period', 'Trial Type', 'Action', 'Outcome', 'Reward'],
    ['Fixation', 'Any', 'Go (1)', 'ABORT', '-0.1'],
    ['Fixation', 'Any', 'Fixate (0)', 'Continue', '0.0'],
    ['Decision', 'GO', 'Go (1)', 'CORRECT', '+1.0'],
    ['Decision', 'GO', 'Fixate (0)', 'MISS', '-0.5'],
    ['Decision', 'NO-GO', 'Go (1)', 'FAIL', '-0.5'],
    ['Decision', 'NO-GO', 'Fixate (0)', 'CORRECT', '0.0'],
]

row_colors = [
    '#D3D3D3',  # Header
    '#FFB6C1',  # Abort
    '#E8E8E8',  # Continue
    '#90EE90',  # Correct
    '#FFB6C1',  # Miss
    '#FFB6C1',  # Fail
    '#E8F8E8',  # Correct rejection
]

table = ax_reward_table.table(
    cellText=table_data,
    cellLoc='center',
    loc='center',
    cellColours=[[row_colors[i]] * 5 for i in range(len(table_data))],
    colWidths=[0.15, 0.15, 0.15, 0.15, 0.15]
)

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

for i in range(5):
    cell = table[(0, i)]
    cell.set_text_props(weight='bold')
    cell.set_facecolor('#A9A9A9')
    cell.set_edgecolor('black')
    cell.set_linewidth(2)

for i in range(len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        cell.set_edgecolor('black')
        cell.set_linewidth(1)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

# Save
output_path = '/home/ewertj2/KAM/mpn-rl/gonogo_task_from_experiments.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved experimental visualization to: {output_path}")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print(f"The plot shows actual experimental data from the GoNogo task.")
print(f"You can see exactly when stimuli appear, when decisions are made,")
print(f"and when rewards are given based on real trial executions.")
print("="*80)

plt.show()
