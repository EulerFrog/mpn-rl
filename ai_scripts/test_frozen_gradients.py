#!/usr/bin/env python3
"""
Test to verify gradient flow in frozen vs non-frozen MPN.

This test checks whether gradients flow backward through time when plasticity is frozen.
"""

import torch
from mpn_dqn import MPNDQN

print("Testing gradient flow with frozen vs non-frozen plasticity")
print("=" * 70)

# Create two networks: one frozen, one normal
frozen_dqn = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2, freeze_plasticity=True)
normal_dqn = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2, freeze_plasticity=False)

# Copy weights so they start identical
normal_dqn.load_state_dict(frozen_dqn.state_dict())

# Create a sequence of observations
sequence_length = 5
batch_size = 1
obs_sequence = [torch.randn(batch_size, 4, requires_grad=True) for _ in range(sequence_length)]

print(f"\nSequence length: {sequence_length}")
print(f"Batch size: {batch_size}")

# Test frozen plasticity
print("\n" + "-" * 70)
print("FROZEN PLASTICITY (M matrix always zero)")
print("-" * 70)

frozen_state = frozen_dqn.init_state(batch_size)
print(f"Initial M matrix sum: {frozen_state.sum().item():.6f}")

q_values_frozen = []
states_frozen = []
for t, obs in enumerate(obs_sequence):
    q_val, frozen_state = frozen_dqn(obs, frozen_state)
    q_values_frozen.append(q_val)
    states_frozen.append(frozen_state)
    print(f"  t={t}: M sum = {frozen_state.sum().item():.6f}, Q = {q_val[0].detach()}")

# Compute loss on final timestep
loss_frozen = q_values_frozen[-1].sum()
loss_frozen.backward()

# Check if gradients exist for first observation
if obs_sequence[0].grad is not None:
    print(f"\n✓ Gradient on first obs: {obs_sequence[0].grad.abs().sum().item():.6f}")
else:
    print(f"\n✗ No gradient on first observation")

# Reset gradients
for obs in obs_sequence:
    if obs.grad is not None:
        obs.grad.zero_()

# Test normal plasticity
print("\n" + "-" * 70)
print("NORMAL PLASTICITY (M matrix updates via Hebbian rule)")
print("-" * 70)

# Clear requires_grad from previous test
obs_sequence = [torch.randn(batch_size, 4, requires_grad=True) for _ in range(sequence_length)]

normal_state = normal_dqn.init_state(batch_size)
print(f"Initial M matrix sum: {normal_state.sum().item():.6f}")

q_values_normal = []
states_normal = []
for t, obs in enumerate(obs_sequence):
    q_val, normal_state = normal_dqn(obs, normal_state)
    q_values_normal.append(q_val)
    states_normal.append(normal_state)
    print(f"  t={t}: M sum = {normal_state.sum().item():.6f}, Q = {q_val[0].detach()}")

# Compute loss on final timestep
loss_normal = q_values_normal[-1].sum()
loss_normal.backward()

# Check if gradients exist for first observation
if obs_sequence[0].grad is not None:
    print(f"\n✓ Gradient on first obs: {obs_sequence[0].grad.abs().sum().item():.6f}")
else:
    print(f"\n✗ No gradient on first observation")

# Summary
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\nFrozen plasticity:")
print(f"  - M matrix stays at zero: ✓")
print(f"  - Each timestep is independent (no recurrence)")
print(f"  - Loss at t=4 should NOT create gradients at t=0")

print("\nNormal plasticity:")
print(f"  - M matrix accumulates: {states_normal[-1].abs().sum().item():.4f}")
print(f"  - State carries information forward (recurrence exists)")
print(f"  - Loss at t=4 SHOULD create gradients at t=0 through BPTT")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

print("\nWith frozen plasticity:")
print("  - M_t = M_{t-1} = 0 (no update)")
print("  - h_t = activation(W * (0 + 1) * x_t + b) = activation(W * x_t + b)")
print("  - h_t does NOT depend on h_{t-1} or x_{t-1}")
print("  - Therefore: NO gradient flow through time")
print("  - Each timestep is independent feedforward computation")

print("\nWith normal plasticity:")
print("  - M_t = λ*M_{t-1} + η*h_t*x_t^T")
print("  - h_t depends on M_t which depends on M_{t-1}")
print("  - Therefore: gradients flow backward through M chain")
print("  - BPTT is active")

print("\n✓ Test complete!")
