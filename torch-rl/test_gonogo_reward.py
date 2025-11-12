import os
import sys

import matplotlib.pyplot as plt
import neurogym as ngym  # Import neurogym to register environments
import numpy as np
import torch
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.envs import (Compose, ExplorationType, InitTracker, StepCounter,
                          TransformedEnv, set_exploration_type)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, LSTMModule, QValueModule

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mpn_torchrl_module import MPNModule
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create environment
env = TransformedEnv(
    GymEnv("GoNogo-v0", device=device),
    Compose(
        StepCounter(),
        InitTracker(),
    )
)

print(f"Action space: {env.action_spec}")
print(f"Observation space: {env.observation_spec}")

# Test parameters
num_rollouts = 50
rollout_length = 500

def run_policy(env, action_choice, num_rollouts, rollout_length):
    """
    Run a constant action policy for multiple rollouts.

    Args:
        env: The environment
        action_choice: The action to always take (0 or 1)
        num_rollouts: Number of rollouts to run
        rollout_length: Number of steps per rollout

    Returns:
        rollout_rewards: List of total rewards per rollout
        rollout_reward_sequences: List of reward sequences for each rollout
    """
    rollout_rewards = []
    rollout_reward_sequences = []

    for rollout_idx in range(num_rollouts):
        td = env.reset()
        rollout_reward = 0
        reward_sequence = []

        for step in range(rollout_length):
            # Create action tensor
            action = torch.zeros(env.action_spec.shape, device=device)
            action[action_choice] = 1.0

            # Add action to tensordict
            td["action"] = action

            # Step environment
            td = env.step(td)

            # Get reward
            reward = td["next", "reward"].item()
            rollout_reward += reward
            reward_sequence.append(reward)

            # Check if episode boundary is reached
            done = td["next", "done"].item() if "done" in td["next"].keys() else False
            terminated = td["next", "terminated"].item() if "terminated" in td["next"].keys() else False

            if done or terminated:
                # Reset environment but continue the rollout
                td = env.reset()
            else:
                # Prepare for next step - use step_mdp to move "next" to current
                td = env.step_mdp(td)

        rollout_rewards.append(rollout_reward)
        rollout_reward_sequences.append(reward_sequence)

        print(f"  Rollout {rollout_idx + 1}/{num_rollouts} complete, total reward: {rollout_reward:.3f}")

    return rollout_rewards, rollout_reward_sequences


def run_random_policy(env, num_rollouts, rollout_length):
    """Run a random action policy."""
    rollout_rewards = []
    rollout_reward_sequences = []

    for rollout_idx in range(num_rollouts):
        td = env.reset()
        rollout_reward = 0
        reward_sequence = []

        for step in range(rollout_length):
            # Random action
            action = torch.zeros(env.action_spec.shape, device=device)
            action_choice = np.random.randint(0, env.action_spec.shape[0])
            action[action_choice] = 1.0

            td["action"] = action
            td = env.step(td)

            reward = td["next", "reward"].item()
            rollout_reward += reward
            reward_sequence.append(reward)

            done = td["next", "done"].item() if "done" in td["next"].keys() else False
            terminated = td["next", "terminated"].item() if "terminated" in td["next"].keys() else False

            if done or terminated:
                td = env.reset()
            else:
                td = env.step_mdp(td)

        rollout_rewards.append(rollout_reward)
        rollout_reward_sequences.append(reward_sequence)

        if (rollout_idx + 1) % 10 == 0:
            print(f"  Rollout {rollout_idx + 1}/{num_rollouts} complete, total reward: {rollout_reward:.3f}")

    return rollout_rewards, rollout_reward_sequences


def run_trained_policy(env, policy, num_rollouts, rollout_length):
    """Run a trained model policy."""
    rollout_rewards = []
    rollout_reward_sequences = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for rollout_idx in range(num_rollouts):
            td = env.reset()
            rollout_reward = 0
            reward_sequence = []

            for step in range(rollout_length):
                # Use trained policy to select action
                td = policy(td)
                td = env.step(td)

                reward = td["next", "reward"].item()
                rollout_reward += reward
                reward_sequence.append(reward)

                done = td["next", "done"].item() if "done" in td["next"].keys() else False
                terminated = td["next", "terminated"].item() if "terminated" in td["next"].keys() else False

                if done or terminated:
                    td = env.reset()
                else:
                    td = env.step_mdp(td)

            rollout_rewards.append(rollout_reward)
            rollout_reward_sequences.append(reward_sequence)

            if (rollout_idx + 1) % 10 == 0:
                print(f"  Rollout {rollout_idx + 1}/{num_rollouts} complete, total reward: {rollout_reward:.3f}")

    return rollout_rewards, rollout_reward_sequences


def load_trained_mpn_model(checkpoint_path, env, device):
    """Load trained MPN model from checkpoint."""
    print(f"\nLoading MPN model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters from checkpoint
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 1)
    eta = config.get('eta', 0.1)
    lambda_decay = config.get('lambda_decay', 0.9)
    activation = config.get('activation', 'tanh')

    print(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}, eta={eta}, lambda_decay={lambda_decay}")

    # Get dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Create stacked MPN layers
    layers = []
    for layer_idx in range(num_layers):
        in_key = "observation" if layer_idx == 0 else f"embed_{layer_idx-1}"
        out_key = f"embed_{layer_idx}"

        in_keys = [in_key, f"recurrent_state_{layer_idx}"]
        out_keys = [out_key, ("next", f"recurrent_state_{layer_idx}")]

        mpn_layer = MPNModule(
            input_size=obs_dim if layer_idx == 0 else hidden_dim,
            hidden_size=hidden_dim,
            eta=eta,
            lambda_decay=lambda_decay,
            activation=activation,
            freeze_plasticity=False,
            device=device,
            in_keys=in_keys,
            out_keys=out_keys,
        )
        layers.append(mpn_layer)
        env.append_transform(mpn_layer.make_tensordict_primer())

    recurrent_module = Seq(*layers)

    # Create MLP head
    mlp = MLP(
        out_features=action_dim,
        num_cells=[hidden_dim],
        device=device,
    )
    mlp[-1].bias.data.fill_(0.0)
    mlp_module = Mod(mlp, in_keys=[f"embed_{num_layers-1}"], out_keys=["action_value"])

    # Q-value module
    qval = QValueModule(spec=env.action_spec)

    # Build policy
    policy = Seq(recurrent_module, mlp_module, qval)

    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    print(f"MPN model loaded successfully!")
    return policy


def load_trained_lstm_model(checkpoint_path, env, device):
    """Load trained LSTM model from checkpoint."""
    print(f"\nLoading LSTM model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters from checkpoint
    config = checkpoint.get('config', {})
    hidden_dim = config.get('hidden_dim', 64)
    num_layers = config.get('num_layers', 1)

    print(f"Model config: hidden_dim={hidden_dim}, num_layers={num_layers}")

    # Get dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Create LSTM module
    lstm_module = LSTMModule(
        input_size=obs_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        device=device,
        in_key="observation",
        out_key=f"embed_{num_layers-1}",
    )
    env.append_transform(lstm_module.make_tensordict_primer())

    # Create MLP head
    mlp = MLP(
        out_features=action_dim,
        num_cells=[hidden_dim],
        device=device,
    )
    mlp[-1].bias.data.fill_(0.0)
    mlp_module = Mod(mlp, in_keys=[f"embed_{num_layers-1}"], out_keys=["action_value"])

    # Q-value module
    qval = QValueModule(spec=env.action_spec)

    # Build policy
    policy = Seq(lstm_module, mlp_module, qval)

    # Load weights
    policy.load_state_dict(checkpoint['model_state_dict'])
    policy.eval()

    print(f"LSTM model loaded successfully!")
    return policy


# Run always-go policy
print("\nRunning always-GO policy...")
go_rollout_rewards, go_reward_sequences = run_policy(env, action_choice=1,
                                                       num_rollouts=num_rollouts,
                                                       rollout_length=rollout_length)

# Reset environment
env.reset()

# Run always-nogo policy
print("\nRunning always-NOGO policy...")
nogo_rollout_rewards, nogo_reward_sequences = run_policy(env, action_choice=0,
                                                           num_rollouts=num_rollouts,
                                                           rollout_length=rollout_length)

# Run random policy
print("\nRunning random policy...")
random_rollout_rewards, random_reward_sequences = run_random_policy(env, num_rollouts=num_rollouts,
                                                                     rollout_length=rollout_length)

# Load and run trained MPN model
mpn_checkpoint_path = "/home/ewertj2/KAM/mpn-rl/experiments/gonogo-mpn-spicy_layer2/checkpoints/best_model.pt"
mpn_env = TransformedEnv(
    GymEnv("GoNogo-v0", device=device),
    Compose(
        StepCounter(),
        InitTracker(),
    )
)
mpn_policy = load_trained_mpn_model(mpn_checkpoint_path, mpn_env, device)

print("\nRunning trained MPN policy...")
mpn_rollout_rewards, mpn_reward_sequences = run_trained_policy(mpn_env, mpn_policy,
                                                                 num_rollouts=num_rollouts,
                                                                 rollout_length=rollout_length)

# Load and run trained LSTM model
# lstm_checkpoint_path = "/home/ewertj2/KAM/mpn-rl/experiments/gonogo-lstm-spicy_layer1/checkpoints/best_model.pt"
# lstm_env = TransformedEnv(
#     GymEnv("GoNogo-v0", device=device),
#     Compose(
#         StepCounter(),
#         InitTracker(),
#     )
# )
# lstm_policy = load_trained_lstm_model(lstm_checkpoint_path, lstm_env, device)

# print("\nRunning trained LSTM policy...")
# lstm_rollout_rewards, lstm_reward_sequences = run_trained_policy(lstm_env, lstm_policy,
#                                                                    num_rollouts=num_rollouts,
#                                                                    rollout_length=rollout_length)

# Calculate statistics
print("\n" + "="*80)
print("RESULTS:")
print("="*80)

policies = {
    'Always-GO': go_rollout_rewards,
    'Always-NOGO': nogo_rollout_rewards,
    'Random': random_rollout_rewards,
    'Trained MPN': mpn_rollout_rewards,
    # 'Trained LSTM': lstm_rollout_rewards,
}

for policy_name, rewards in policies.items():
    print(f"\n{policy_name} Policy:")
    print(f"  Total rollouts: {len(rewards)}")
    print(f"  Rollout length: {rollout_length} steps")
    print(f"  Mean reward per rollout: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"  Mean reward per step: {np.mean(rewards) / rollout_length:.4f}")
    print(f"  Min rollout reward: {np.min(rewards):.3f}")
    print(f"  Max rollout reward: {np.max(rewards):.3f}")

# Create visualizations
plt.rcParams['font.family'] = 'serif'
fig, axs = plt.subplots(2, 1, figsize=(14, 10))

# Define colors for each policy
colors = {
    'go': 'blue',
    'nogo': 'red',
    'random': 'green',
    'mpn': 'purple',
    'lstm': 'orange'
}

# Plot 1: Cumulative reward over steps for each rollout
reward_data = [
    (go_reward_sequences, 'Always GO', colors['go']),
    (nogo_reward_sequences, 'Always NOGO', colors['nogo']),
    (random_reward_sequences, 'Random', colors['random']),
    (mpn_reward_sequences, 'Trained MPN', colors['mpn']),
    # (lstm_reward_sequences, 'Trained LSTM', colors['lstm']),
]

for sequences, label, color in reward_data:
    # Plot individual rollout trajectories with low alpha
    for seq in sequences:
        cumsum = np.cumsum(seq)
        axs[0].plot(cumsum, color=color, alpha=0.15, linewidth=0.5)

    # Plot mean trajectory with high alpha
    mean_cumsum = np.cumsum(np.mean(sequences, axis=0))
    axs[0].plot(mean_cumsum, color=color, linewidth=3, label=f'{label} (μ={np.mean([sum(s) for s in sequences]):.1f})')

axs[0].set_xlabel('Step within Rollout', fontsize=12)
axs[0].set_ylabel('Cumulative Reward', fontsize=12)
axs[0].set_title(f'Cumulative Reward Trajectories ({num_rollouts} rollouts × {rollout_length} steps)', fontsize=14, fontweight='bold')
axs[0].legend(fontsize=10, loc='best')
axs[0].grid(True, alpha=0.3)
axs[0].axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

# Plot 2: Reward distribution comparison
all_rewards = [go_rollout_rewards, nogo_rollout_rewards, random_rollout_rewards, mpn_rollout_rewards, lstm_rollout_rewards]
bins = np.linspace(
    min([min(r) for r in all_rewards]) - 5,
    max([max(r) for r in all_rewards]) + 5,
    25
)

axs[1].hist(go_rollout_rewards, bins=bins, alpha=0.5, color=colors['go'], edgecolor='black',
            label=f'Always GO (μ={np.mean(go_rollout_rewards):.1f}±{np.std(go_rollout_rewards):.1f})')
axs[1].hist(nogo_rollout_rewards, bins=bins, alpha=0.5, color=colors['nogo'], edgecolor='black',
            label=f'Always NOGO (μ={np.mean(nogo_rollout_rewards):.1f}±{np.std(nogo_rollout_rewards):.1f})')
axs[1].hist(random_rollout_rewards, bins=bins, alpha=0.5, color=colors['random'], edgecolor='black',
            label=f'Random (μ={np.mean(random_rollout_rewards):.1f}±{np.std(random_rollout_rewards):.1f})')
axs[1].hist(mpn_rollout_rewards, bins=bins, alpha=0.5, color=colors['mpn'], edgecolor='black',
            label=f'Trained MPN (μ={np.mean(mpn_rollout_rewards):.1f}±{np.std(mpn_rollout_rewards):.1f})')
# axs[1].hist(lstm_rollout_rewards, bins=bins, alpha=0.5, color=colors['lstm'], edgecolor='black',
#             label=f'Trained LSTM (μ={np.mean(lstm_rollout_rewards):.1f}±{np.std(lstm_rollout_rewards):.1f})')

# Add vertical lines for means
axs[1].axvline(x=np.mean(go_rollout_rewards), color=colors['go'], linestyle='--', linewidth=2, alpha=0.8)
axs[1].axvline(x=np.mean(nogo_rollout_rewards), color=colors['nogo'], linestyle='--', linewidth=2, alpha=0.8)
axs[1].axvline(x=np.mean(random_rollout_rewards), color=colors['random'], linestyle='--', linewidth=2, alpha=0.8)
axs[1].axvline(x=np.mean(mpn_rollout_rewards), color=colors['mpn'], linestyle='--', linewidth=2, alpha=0.8)
# axs[1].axvline(x=np.mean(lstm_rollout_rewards), color=colors['lstm'], linestyle='--', linewidth=2, alpha=0.8)

axs[1].set_xlabel('Total Reward per Rollout', fontsize=12)
axs[1].set_ylabel('Frequency', fontsize=12)
axs[1].set_title('Reward Distribution Comparison', fontsize=14, fontweight='bold')
axs[1].legend(fontsize=10, loc='best')
axs[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save the figure
output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(output_dir, 'gonogo_reward_analysis.png')
plt.savefig(output_path, dpi=150)
plt.close()

print(f"\nSaved reward analysis plot to {output_path}")
print("\nAnalysis complete!")
