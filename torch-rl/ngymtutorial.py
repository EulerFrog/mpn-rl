import multiprocessing
import os
import sys
import time

import imageio
import neurogym as ngym
import numpy as np
import torch
import tqdm
from matplotlib import pyplot as plt
from pynput import keyboard
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (Compose, ExplorationType, GrayScale, InitTracker,
                          ObservationNorm, Resize, RewardScaling, StepCounter,
                          ToTensorImage, TransformedEnv, set_exploration_type)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, ConvNet, EGreedyModule, QValueModule

sys.path.append('/home/eulerfrog/KAM/mpn-rl')
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.modules import LSTMModule

TOTAL_FRAMES = 10_000
global exit
exit = False

def on_press(key):
    global exit
    try:
        if key.char == '+':
            exit = True
    except AttributeError:
        pass
        
listener = keyboard.Listener(on_press=on_press)
listener.start()

is_fork = multiprocessing.get_start_method() == "fork"
device = (
    torch.device(0)
    if torch.cuda.is_available() and not is_fork
    else torch.device("cpu")
)
print(device)



# transforms environment so that only pixels are shown
env = TransformedEnv(
    GymEnv("DelayMatchSample-v0", device=device),
    Compose(
        StepCounter(),
        InitTracker(),
    )
)

td = env.reset()

# first layer is conv layer
n_cells = env.reset()["observation"].shape[0]

print(env.reset()["observation"].shape[0])

# LSTM takes input from observation and outputs to embed layer
# LSTM has two hidden states: hidden state (h) and cell state (c)
lstm = LSTMModule(
    input_size=n_cells,
    hidden_size=64,
    device=device,
    in_key="observation",
    out_key="embed",
)

# print("in_keys", lstm.in_keys)
# print("out_keys", lstm.out_keys)

env.append_transform(lstm.make_tensordict_primer())


mlp = MLP(
    out_features=3,
    num_cells=[
        64,
    ],
    device=device,
)

mlp[-1].bias.data.fill_(0.0)

mlp = Mod(mlp, in_keys=["embed"], out_keys=["action_value"])

qval = QValueModule(spec=env.action_spec)


stoch_policy = Seq(lstm, mlp, qval)

exploration_module = EGreedyModule(
    annealing_num_steps=1_000_000, spec=env.action_spec, eps_init=0.2
)

stoch_policy = Seq(
    stoch_policy,
    exploration_module,
)

policy = Seq(lstm, mlp, qval)

policy(env.reset())

loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

# what is the updater?
updater = SoftUpdate(loss_fn, eps=0.95)

optim = torch.optim.Adam(policy.parameters(), lr=3e-4)

collector = SyncDataCollector(env, stoch_policy, frames_per_batch=50, total_frames=TOTAL_FRAMES, device=device)

rb = TensorDictReplayBuffer(
    storage=LazyMemmapStorage(20_000), batch_size=4, prefetch=10
)

utd = 64

# progress bar to show how much time is left in the training process
pbar = tqdm.tqdm(total=TOTAL_FRAMES)
longest = 0
sample_trajectory = None
random_trajectory = None

max_reward = 0

loss = []
reward = []
misses = []
for i, data in enumerate(collector):
    if i == 0:
        print(
            "Let us print the first batch of data.\nPay attention to the key names "
            "which will reflect what can be found in this data structure, in particular: "
            "the output of the QValueModule (action_values, action and chosen_action_value),"
            "the 'is_init' key that will tell us if a step is initial or not, and the "
            "recurrent_state keys.\n",
            data,
        )
    pbar.update(data.numel())
    # it is important to pass data that is not flattened
    rb.extend(data.unsqueeze(0).to_tensordict().cpu())
    for _ in range(utd):
        s = rb.sample().to(device, non_blocking=True)
        # print(s)
        loss_vals = loss_fn(s)
        loss_vals["loss"].backward()
        optim.step()
        optim.zero_grad()
        
    
    pbar.set_description(
        f"loss_val: {loss_vals['loss'].item(): 4.4f}, action_spread: {data['action'].sum(0)}"
    )
    loss.append(loss_vals['loss'].item())
    exploration_module.step(data.numel())
    updater.step()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # env.rollout() runs the simulation on 1000 frames
        rollout = env.rollout(500, stoch_policy)
        curr_reward = torch.sum(rollout.get(("next", "reward"))).item()
        missed = (rollout.get(("next","reward")) < 0).sum().item()
        reward.append(curr_reward)
        misses.append(missed)
        
        rewards = rollout.get(("next","reward"))
        observations = rollout.get(("next","observation"))
        actions = torch.argmax(rollout.get("action"), dim=1)
        
        if curr_reward > max_reward:
            max_reward = curr_reward
            sample_trajectory = torch.hstack((observations, actions.unsqueeze(1), rewards))
    
    if exit:
        break
       
listener.stop() 


sample_trajectory = sample_trajectory.cpu()

labels = ["fixation","mod1","mod2","action"]
plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', etc.

fig, axs = plt.subplots(2,1)

length = 300

axs[0].set_title("State")
for i in range(4):
    axs[0].plot(sample_trajectory[:length, i] + 2*i, label=labels[i])

axs[1].plot(sample_trajectory[:length, 4])
axs[1].set_title("Reward")
axs[0].legend()

# Save plots to the parent directory (mpn-rl root)
output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
trajectory_path = os.path.join(output_dir, 'lstm_ngym_trajectory.png')
plt.savefig(trajectory_path)
plt.close()

print(f"Saved trajectory plot to {trajectory_path}")
print(f"Misses: {misses}")

fig, axs = plt.subplots(3,1, figsize=(10, 8))

axs[0].plot(loss)
axs[0].set_title("Loss")
axs[1].plot(reward)
axs[1].set_title("Reward")
axs[2].plot(misses)
axs[2].set_title("Misses")
plt.tight_layout()

training_path = os.path.join(output_dir, 'lstm_ngym_training.png')
plt.savefig(training_path)
plt.close()

print(f"Saved training plots to {training_path}")
print("\nLSTM training completed successfully!")