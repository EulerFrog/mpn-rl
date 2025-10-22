"""
Minimal training test to isolate the hanging issue.
"""

import torch
import numpy as np
from mpn_dqn import MPNDQN
from neurogym_wrapper import make_neurogym_env
from model_utils import TrialReplayBuffer, compute_td_loss_trial

print("="*60)
print("Minimal Training Test")
print("="*60)

# Create environment
print("\n1. Creating environment...")
env = make_neurogym_env('ContextDecisionMaking-v0')
obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print(f"   Obs dim: {obs_dim}, Action dim: {action_dim}")

# Create network
print("\n2. Creating MPN-DQN...")
device = torch.device('cpu')
mpn = MPNDQN(obs_dim=obs_dim, hidden_dim=32, action_dim=action_dim,
             eta=0.1, lambda_decay=0.95).to(device)
target_mpn = MPNDQN(obs_dim=obs_dim, hidden_dim=32, action_dim=action_dim,
                    eta=0.1, lambda_decay=0.95).to(device)
target_mpn.load_state_dict(mpn.state_dict())
print("   Created")

# Create buffer
print("\n3. Creating trial buffer...")
buffer = TrialReplayBuffer(capacity=10)
print("   Created")

# Collect one trial
print("\n4. Collecting one trial...")
obs, info = env.reset()
obs = torch.FloatTensor(obs).to(device)
state = mpn.init_state(batch_size=1, device=device)

trial_obs = []
trial_actions = []
trial_rewards = []
trial_dones = []

for step in range(100):  # Max 100 steps
    with torch.no_grad():
        q_values, next_state = mpn(obs.unsqueeze(0), state)

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    trial_obs.append(obs.cpu())
    trial_actions.append(action)
    trial_rewards.append(reward)
    trial_dones.append(done)

    obs = torch.FloatTensor(next_obs).to(device)
    state = next_state

    if info.get('new_trial', False) or done:
        print(f"   Trial ended at step {step}")
        break

# Push to buffer
print("\n5. Pushing trial to buffer...")
buffer.push_trial(trial_obs, trial_actions, trial_rewards, trial_dones)
print(f"   Buffer size: {len(buffer)}")

# Collect a few more trials
print("\n6. Collecting 3 more trials...")
for trial_num in range(3):
    obs, info = env.reset()
    obs = torch.FloatTensor(obs).to(device)
    state = mpn.init_state(batch_size=1, device=device)

    trial_obs = []
    trial_actions = []
    trial_rewards = []
    trial_dones = []

    for step in range(100):
        with torch.no_grad():
            q_values, next_state = mpn(obs.unsqueeze(0), state)

        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        trial_obs.append(obs.cpu())
        trial_actions.append(action)
        trial_rewards.append(reward)
        trial_dones.append(done)

        obs = torch.FloatTensor(next_obs).to(device)
        state = next_state

        if info.get('new_trial', False) or done:
            print(f"   Trial {trial_num+1} ended at step {step}")
            break

    buffer.push_trial(trial_obs, trial_actions, trial_rewards, trial_dones)

print(f"   Buffer size: {len(buffer)}")

# Try computing loss
print("\n7. Computing TD loss on trial batch...")
print("   Sampling batch...")
trial_batch = buffer.sample(2)
print(f"   Batch size: {len(trial_batch)}")

print("   Computing loss (THIS IS WHERE IT MIGHT HANG)...")
loss = compute_td_loss_trial(mpn, target_mpn, trial_batch, gamma=0.99, device=device)
print(f"   Loss: {loss.item():.4f}")

print("\n" + "="*60)
print("✓ Test completed successfully!")
print("="*60)

env.close()
