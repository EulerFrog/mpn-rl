"""
Quick debug script to test NeuroGym environment behavior
"""
import sys
from neurogym_wrapper import make_neurogym_env

print("Creating NeuroGym environment with TimeLimit...", flush=True)
env = make_neurogym_env('ContextDecisionMaking-v0', max_episode_steps=100)
print(f"Environment created successfully!", flush=True)
print(f"Observation space: {env.observation_space}", flush=True)
print(f"Action space: {env.action_space}", flush=True)

print("\nRunning 1 episode to observe trial boundaries and termination...", flush=True)
obs, info = env.reset()
print(f"Reset - obs shape: {obs.shape}, info: {info}", flush=True)

step_count = 0
trial_count = 0
max_steps = 200  # Higher than TimeLimit to test truncation

for step in range(max_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1

    if info.get('new_trial', False):
        trial_count += 1
        print(f"  Step {step}: NEW TRIAL detected! (trial #{trial_count})", flush=True)

    if step % 20 == 0:
        print(f"  Step {step}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}", flush=True)

    if terminated or truncated:
        print(f"  Step {step}: EPISODE ENDED (terminated={terminated}, truncated={truncated})", flush=True)
        break

print(f"\nSummary:", flush=True)
print(f"  Total steps: {step_count}", flush=True)
print(f"  Total trials: {trial_count}", flush=True)
print(f"  Episode ended: {terminated or truncated}", flush=True)

env.close()
print("\nTest complete!", flush=True)
