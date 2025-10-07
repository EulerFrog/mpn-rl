"""
Quick script to test GPU acceleration

Run with: python test_gpu.py
"""

import torch
from mpn_dqn import MPNDQN

print("="*60)
print("Testing GPU Acceleration for MPN-DQN")
print("="*60)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Check memory
    props = torch.cuda.get_device_properties(0)
    print(f"Total GPU memory: {props.total_memory / 1e9:.2f} GB")
    print(f"GPU compute capability: {props.major}.{props.minor}")
else:
    print("\nNo CUDA device detected. Training will use CPU.")
    print("To enable GPU:")
    print("  1. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  2. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")

# Test model on GPU
print("\n" + "="*60)
print("Testing Model on Device")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Create model
model = MPNDQN(obs_dim=4, hidden_dim=64, action_dim=2).to(device)
print(f"Model created and moved to {device}")

# Test forward pass
obs = torch.randn(1, 4).to(device)
state = model.init_state(batch_size=1, device=device)

q_values, new_state = model(obs, state)

print(f"\nForward pass successful!")
print(f"  Input device: {obs.device}")
print(f"  State device: {state.device}")
print(f"  Q-values device: {q_values.device}")
print(f"  Output device: {new_state.device}")

if torch.cuda.is_available():
    print(f"\n✓ GPU acceleration is working!")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e6:.2f} MB")
    print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e6:.2f} MB")
else:
    print(f"\n✓ CPU mode is working!")

print("\n" + "="*60)
print("GPU Test Completed")
print("="*60)
