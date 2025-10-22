"""
Test script to understand NeuroGym trial behavior.

This helps debug the infinite loop issue in train_neurogym by showing
exactly when 'new_trial' is set to True.
"""

import neurogym as ngym
from neurogym_wrapper import make_neurogym_env

print("="*60)
print("Testing NeuroGym Trial Behavior")
print("="*60)

# Create environment
env_name = 'ContextDecisionMaking-v0'
print(f"\nCreating environment: {env_name}")
env = make_neurogym_env(env_name)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
print(f"Actions: 0=fixation, 1-{env.action_space.n-1}=choices\n")

# Reset environment
obs, info = env.reset()
print("Environment reset")
print(f"Initial info: {info}\n")

# Run with random actions
print("Running with random actions...")
print("-"*60)

step = 0
trial_count = 0
max_steps = 200

while step < max_steps:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step {step:3d}: action={action}, reward={reward:5.2f}, "
          f"new_trial={info.get('new_trial', False)}, "
          f"terminated={terminated}, truncated={truncated}")

    if info.get('new_trial', False):
        trial_count += 1
        print(f"  → Trial {trial_count} ended at step {step}!")
        print(f"  → Resetting environment...")
        obs, info = env.reset()
        print(f"  → Reset complete\n")

    if terminated or truncated:
        print(f"  → Episode ended!")
        break

    step += 1

print("\n" + "="*60)
print(f"Test completed after {step} steps")
print(f"Trials completed: {trial_count}")
print("="*60)

# Test 2: Run with mostly fixation (action=0) to see if it hangs
print("\n\n" + "="*60)
print("Test 2: Mostly fixation actions (should show the problem)")
print("="*60)

env = make_neurogym_env(env_name)
obs, info = env.reset()

step = 0
max_steps = 50
print(f"\nTaking only fixation actions (action=0) for {max_steps} steps...")
print("-"*60)

while step < max_steps:
    action = 0  # Always fixation
    obs, reward, terminated, truncated, info = env.step(action)

    if step % 10 == 0:  # Print every 10 steps
        print(f"Step {step:3d}: action={action}, reward={reward:5.2f}, "
              f"new_trial={info.get('new_trial', False)}")

    if info.get('new_trial', False):
        print(f"  → Trial ended at step {step}!")
        break

    step += 1

if step >= max_steps:
    print(f"\n⚠️  WARNING: Trial did NOT end after {max_steps} fixation actions!")
    print("   This confirms the infinite loop problem.")
else:
    print(f"\n✓ Trial ended naturally at step {step}")

print("\n" + "="*60)
print("Testing complete!")
print("="*60)

env.close()
