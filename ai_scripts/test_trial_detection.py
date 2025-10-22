#!/usr/bin/env python3
"""
Quick test to verify that info.get('new_trial') is being triggered by NeuroGym environments.
"""

from neurogym_wrapper import make_neurogym_env

print("Testing trial boundary detection in NeuroGym...")
print("=" * 60)

# Create environment
env = make_neurogym_env('ContextDecisionMaking-v0', max_episode_steps=200)

# Run one episode and count trials
obs, info = env.reset()
print(f"Episode started. Initial info: {info}")

num_trials = 0
step_count = 0
max_steps = 200

print("\nRunning episode and watching for 'new_trial' events...")
print("-" * 60)

for step in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    # Check for trial boundary
    if info.get('new_trial', False):
        num_trials += 1
        print(f"Step {step:3d}: ✓ NEW TRIAL DETECTED (trial #{num_trials})")

    if terminated or truncated:
        print(f"Step {step:3d}: Episode ended (terminated={terminated}, truncated={truncated})")
        break

print("-" * 60)
print(f"\nResults:")
print(f"  Total steps: {step_count}")
print(f"  Trials detected: {num_trials}")
print(f"  Average steps per trial: {step_count / num_trials if num_trials > 0 else 'N/A'}")

if num_trials == 0:
    print("\n⚠️  WARNING: No trials detected! The 'new_trial' flag may not be working.")
else:
    print(f"\n✓ Trial detection working! Found {num_trials} trials in {step_count} steps.")

env.close()
print("\nTest complete!")
print("=" * 60)
