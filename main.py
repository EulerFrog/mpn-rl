"""
Main CLI for MPN-DQN training with TorchRL

Commands:
    train-neurogym - Train MPN/RNN agent on NeuroGym environments using TorchRL
    eval           - Evaluate a trained agent
    render         - Render episode(s) to visualization

Examples:
    # Train on NeuroGym with default settings
    python main.py train-neurogym --env-name GoNogo-v0

    # Train with custom hyperparameters
    python main.py train-neurogym --experiment-name my-agent --num-episodes 1000 --eta 0.1

    # Evaluate
    python main.py eval --experiment-name my-agent --num-eval-episodes 10

    # Render
    python main.py render --experiment-name my-agent --output render.png
"""

import argparse
import math
import sys
from pathlib import Path

# Matplotlib for rendering
import matplotlib.pyplot as plt
import neurogym
import numpy as np
import torch
import tqdm
# TorchRL imports
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (Compose, ExplorationType, InitTracker, ParallelEnv,
                          StepCounter, TransformedEnv, set_exploration_type)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, EGreedyModule, LSTMModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

# Local imports
from model_utils import ExperimentManager
from mpn_torchrl_module import MPNModule
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper
from rnn_module import RNNModule


def get_epsilon_for_annealing(total_frames, epsilon_start, epsilon_end, annealing_frames=None):
    """
    Calculate annealing_num_steps for EGreedyModule.

    Args:
        total_frames: Total number of training frames
        epsilon_start: Starting epsilon
        epsilon_end: Final epsilon
        annealing_frames: Frames over which to anneal (default: 30% of total)

    Returns:
        annealing_num_steps for EGreedyModule
    """
    if annealing_frames is None:
        annealing_frames = int(total_frames * 0.3)
    return annealing_frames


def get_device(device_str='cpu'):
    """
    Get PyTorch device.

    Args:
        device_str: 'gpu' or 'cpu'

    Returns:
        torch.device
    """
    # Map 'gpu' to 'cuda' before creating device
    if device_str == 'gpu':
        device_str = 'cuda'

    device = torch.device(device_str)

    if device.type == 'cuda' and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(device.index or 0).total_memory / 1e9:.2f} GB")
    elif device.type == 'cuda':
        print("Warning: GPU requested but not available, falling back to CPU")
        device = torch.device('cpu')
    elif device.type == 'cpu':
        print("Using CPU")

    return device


def create_model(env, model_type, hidden_dim, num_layers, eta, lambda_decay, activation, device):
    """
    Create policy model with stacked recurrent layers.

    Args:
        env: Environment (for getting observation/action dimensions)
        model_type: 'mpn', 'mpn-frozen', 'rnn', or 'lstm'
        hidden_dim: Hidden layer dimension
        num_layers: Number of recurrent layers to stack
        eta: Hebbian learning rate (MPN only)
        lambda_decay: M matrix decay (MPN only)
        activation: Activation function
        device: Device to create model on

    Returns:
        policy: The policy network (stacked recurrent layers -> mlp -> qval)
    """
    # Get environment dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Determine model configuration
    use_rnn = (model_type == 'rnn')
    use_lstm = (model_type == 'lstm')
    freeze_plasticity = (model_type == 'mpn-frozen')

    # Create stacked recurrent layers
    layers = []

    if use_rnn:
        # RNN supports multiple layers natively
        recurrent_module = RNNModule(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=activation,
            device=device,
            in_key="observation",
            out_key=f"embed_{num_layers-1}",
        )
        layers.append(recurrent_module)
        env.append_transform(recurrent_module.make_tensordict_primer())
    elif use_lstm:
        # LSTM supports multiple layers natively
        recurrent_module = LSTMModule(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            device=device,
            in_key="observation",
            out_key=f"embed_{num_layers-1}",
        )
        layers.append(recurrent_module)
        env.append_transform(recurrent_module.make_tensordict_primer())
    else:
        # Stack multiple MPN layers
        for layer_idx in range(num_layers):
            in_key = "observation" if layer_idx == 0 else f"embed_{layer_idx-1}"
            out_key = f"embed_{layer_idx}"

            # Recurrent state keys for this layer
            in_keys = [in_key, f"recurrent_state_{layer_idx}"]
            out_keys = [out_key, ("next", f"recurrent_state_{layer_idx}")]

            mpn_layer = MPNModule(
                input_size=obs_dim if layer_idx == 0 else hidden_dim,
                hidden_size=hidden_dim,
                eta=eta,
                lambda_decay=lambda_decay,
                activation=activation,
                freeze_plasticity=freeze_plasticity,
                device=device,
                in_keys=in_keys,
                out_keys=out_keys,
            )
            layers.append(mpn_layer)
            env.append_transform(mpn_layer.make_tensordict_primer())

    # Combine layers into sequence
    recurrent_module = Seq(*layers)

    # Create MLP head for Q-values (reads from final layer output)
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

    return policy


def evaluate_policy(policy, eval_env, num_steps, num_episodes=1):
    """
    Evaluate policy by doing multiple rollouts and averaging rewards.

    Args:
        policy: The policy network to evaluate
        eval_env: The evaluation environment (can be single or parallel)
        num_steps: Number of steps to run per episode
        num_episodes: Number of episodes (only used for reporting, env should match)

    Returns:
        avg_reward: Average total reward across all episodes
        std_reward: Standard deviation of rewards across episodes
    """
    policy.eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # Do rollout(s) on the eval environment
        rollout = eval_env.rollout(num_steps, policy)

        # Extract rewards from rollout
        rewards = rollout.get(("next", "reward"))
        rewards = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.array(rewards)

        if num_episodes == 1:
            # Single episode - sum all rewards
            rewards = rewards.squeeze()
            total_reward = float(rewards.sum())
            policy.train()
            return total_reward, 0.0
        else:
            # Multiple episodes - sum each episode and compute stats
            # rewards shape is [num_episodes, num_steps] after rollout
            episode_rewards = rewards.sum(axis=1)  # Sum over time dimension
            avg_reward = float(np.mean(episode_rewards))
            std_reward = float(np.std(episode_rewards))
            policy.train()
            return avg_reward, std_reward


def train_neurogym(args):
    """Train MPN/RNN-DQN on NeuroGym environment using TorchRL."""
    print("="*60)
    print("Training with TorchRL on NeuroGym")
    print("="*60)

    # Create experiment manager
    exp_manager = ExperimentManager(args.experiment_name)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Directory: {exp_manager.exp_dir}\n")

    # Save configuration
    config = vars(args)
    config['command'] = 'train-neurogym'
    exp_manager.save_config(config)

    # Setup device
    device = get_device(args.device)
    print()

    # Create NeuroGym environment using TorchRL's GymEnv
    print(f"Creating NeuroGym environment: {args.env_name}")

    # Create environment factory function for parallel evaluation
    def make_env():
        gymenv = GymEnv(args.env_name, device=device)
        gymenv._env = NeuroGymInfoWrapper(gymenv._env)
        return TransformedEnv(
            gymenv,
            Compose(
                StepCounter(max_steps=args.max_episode_steps),
                InitTracker(),
                NeuroGymInfoTransform(),
            )
        )

    # Create main training environment (parallel if num_envs > 1)
    if args.num_envs > 1:
        print(f"Creating {args.num_envs} parallel training environments")
        env = ParallelEnv(args.num_envs, make_env, device=device)
    else:
        env = make_env()

    # Get environment dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Print model configuration
    use_rnn = (args.model_type == 'rnn')
    use_lstm = (args.model_type == 'lstm')
    freeze_plasticity = (args.model_type == 'mpn-frozen')

    print(f"Algorithm: DQN with TorchRL")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Environment: {args.env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Max episode steps: {args.max_episode_steps}")

    if use_rnn:
        print(f"RNN: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, activation={args.activation}\n")
    elif use_lstm:
        print(f"LSTM: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}\n")
    else:
        print(f"MPN: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, eta={args.eta}, lambda={args.lambda_decay}")
        print(f"Activation: {args.activation}, Plasticity frozen: {freeze_plasticity}\n")

    # Create policy model
    policy = create_model(
        env=env,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        eta=args.eta,
        lambda_decay=args.lambda_decay,
        activation=args.activation,
        device=device
    )

    # Exploration module
    total_frames = args.total_frames
    annealing_frames = get_epsilon_for_annealing(
        total_frames, args.epsilon_start, args.epsilon_end
    )

    exploration_module = EGreedyModule(
        annealing_num_steps=annealing_frames,
        spec=env.action_spec,
        eps_init=args.epsilon_start,
        eps_end=args.epsilon_end,
        device=device
    )

    stoch_policy = Seq(policy, exploration_module)

    # Test policy
    print("Testing policy forward pass...")
    test_td = env.reset()
    test_td = stoch_policy(test_td)
    print(f"Policy output keys: {list(test_td.keys())}\n")

    # Create DQN loss
    loss_fn = DQNLoss(policy, action_space=env.action_spec, delay_value=True)

    # Target network updater
    updater = SoftUpdate(loss_fn, eps=args.target_update_tau)

    # Optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    # Data collector
    collector = SyncDataCollector(
        env,
        stoch_policy,
        frames_per_batch=args.frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # Replay buffer
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(args.buffer_size),
        batch_size=args.batch_size,
        prefetch=10,
    )

    print(f"Replay buffer: capacity={args.buffer_size}, batch_size={args.batch_size}")
    print(f"UTD (updates-to-data): {args.utd}")
    print(f"Epsilon: {args.epsilon_start} → {args.epsilon_end} over {annealing_frames} frames")
    print(f"Target update tau: {args.target_update_tau}")
    print(f"Total frames: {total_frames}")
    print("Starting training...")
    print("-"*60)

    # Create evaluation environment (parallel if num_eval_episodes > 1)
    if args.num_eval_episodes > 1:
        print(f"Creating {args.num_eval_episodes} parallel evaluation environments")
        eval_env = ParallelEnv(args.num_eval_episodes, make_env, device=device)
    else:
        eval_env = make_env()

    # Training metrics
    eval_rewards = []
    eval_reward_stds = []
    eval_losses = []
    recent_losses = []  # Track losses between evaluations
    best_reward = -float('inf')
    frames_collected = 0
    last_eval_frame = 0

    # Progress bar
    pbar = tqdm.tqdm(total=total_frames, desc="Training", unit="frames")

    # Training loop
    for i, data in enumerate(collector):
        # Add data to replay buffer
        replay_buffer.extend(data.unsqueeze(0).to_tensordict().cpu())
        

        frames_collected += data.numel()

        # Perform multiple gradient updates (UTD)
        batch_loss_vals = []
        for _ in range(args.utd):
            if len(replay_buffer) >= args.batch_size:
                sample = replay_buffer.sample().to(device, non_blocking=True)
                loss_vals = loss_fn(sample)

                optimizer.zero_grad()
                loss_vals["loss"].backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()

                batch_loss_vals.append(loss_vals["loss"].item())

                # Update target network
                updater.step()

        # Update exploration
        exploration_module.step(data.numel())

        # Track metrics per batch
        batch_loss = np.mean(batch_loss_vals) if batch_loss_vals else 0.0
        # Convert epsilon to Python float (may be tensor)
        current_epsilon = float(exploration_module.eps.item()) if torch.is_tensor(exploration_module.eps) else float(exploration_module.eps)

        # Store recent losses for evaluation logging
        if batch_loss_vals:
            recent_losses.append(batch_loss)

        # Update progress bar
        pbar.update(data.numel())
        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'eps': f'{current_epsilon:.3f}',
            'buffer': len(replay_buffer)
        })

        # Print progress and evaluate every N frames
        if frames_collected - last_eval_frame >= args.print_freq:
            # Run evaluation rollout (using max_episode_steps)
            eval_reward, eval_reward_std = evaluate_policy(policy, eval_env, args.max_episode_steps, args.num_eval_episodes)

            # Calculate average loss since last evaluation
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0

            # Store evaluation metrics (aligned arrays)
            eval_rewards.append(eval_reward)
            eval_reward_stds.append(eval_reward_std)
            eval_losses.append(avg_loss)

            # Save to history
            exp_manager.append_training_history(
                int(frames_collected),
                float(eval_reward),
                int(data.numel()),
                float(avg_loss),
                float(current_epsilon)
            )

            # Print combined progress and evaluation
            tqdm.tqdm.write(f"Frames {frames_collected:7d}/{total_frames} | "
                            f"Eval Reward: {eval_reward:7.2f} ± {eval_reward_std:5.2f} | "
                            f"Loss: {avg_loss:6.4f} | "
                            f"ε: {current_epsilon:.3f}")

            # Reset recent losses and update last eval frame
            recent_losses = []
            last_eval_frame = frames_collected

            # Check if this is the best eval performance
            if eval_reward > best_reward:
                best_reward = eval_reward

                # Save as best model
                exp_manager.save_model(
                    policy,
                    optimizer=optimizer,
                    checkpoint_name="best_model.pt",
                    metadata={
                        'frames': int(frames_collected),
                        'eval_reward': float(eval_reward),
                        'epsilon': float(current_epsilon),
                    }
                )
                tqdm.tqdm.write(f"  → New best model! Reward: {eval_reward:.2f}")

        # Checkpoint every N frames
        if frames_collected % args.checkpoint_freq == 0 and frames_collected > 0:
            # Save periodic checkpoint
            exp_manager.save_model(
                policy,
                optimizer=optimizer,
                checkpoint_name=f"checkpoint_{frames_collected}.pt",
                metadata={
                    'frames': int(frames_collected),
                    'epsilon': float(current_epsilon),
                }
            )
            tqdm.tqdm.write(f"  → Checkpoint saved at {frames_collected} frames")

    # Close progress bar
    pbar.close()

    # Save final model
    final_eval_reward = eval_rewards[-1] if eval_rewards else 0.0
    exp_manager.save_model(
        policy,
        optimizer=optimizer,
        checkpoint_name="final_model.pt",
        metadata={
            'frames': int(frames_collected),
            'final_eval_reward': float(final_eval_reward),
        }
    )

    # Save training metrics for plotting
    import json
    metrics_path = exp_manager.exp_dir / "training_metrics.json"
    metrics = {
        'eval_rewards': eval_rewards,
        'eval_reward_stds': eval_reward_stds,
        'eval_losses': eval_losses,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    # Close environments
    env.close()
    eval_env.close()
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total frames: {frames_collected}")
    print(f"Best eval reward: {best_reward:.2f}")
    print(f"Final eval reward: {final_eval_reward:.2f}")
    print(f"Results saved to: {exp_manager.exp_dir}")
    print("="*60)


def evaluate(args):
    """Evaluate trained agent."""
    print("="*60)
    print("Evaluating Agent")
    print("="*60)

    # Load experiment
    exp_manager = ExperimentManager(args.experiment_name)
    config = exp_manager.load_config()

    # Detect model type
    model_type = config.get('model_type', 'mpn')
    use_rnn = (model_type == 'rnn')
    freeze_plasticity = (model_type == 'mpn-frozen')

    print(f"Model type: {model_type.upper()}")

    # Setup device
    device = get_device(args.device)
    print()

    # Create environment
    env = TransformedEnv(
        GymEnv(config['env_name'], device=device),
        Compose(
            StepCounter(max_steps=args.max_episode_steps),
            InitTracker(),
        )
    )

    # Create policy model
    policy = create_model(
        env=env,
        model_type=config.get('model_type', 'mpn'),
        hidden_dim=config['hidden_dim'],
        num_layers=config.get('num_layers', 1),
        eta=config.get('eta', 0.1),
        lambda_decay=config.get('lambda_decay', 0.95),
        activation=config.get('activation', 'tanh'),
        device=device
    )

    # Load checkpoint
    checkpoint_name = args.checkpoint if args.checkpoint else "best_model.pt"
    exp_manager.load_model(policy, checkpoint_name=checkpoint_name, device=str(device))

    policy.eval()

    print(f"Loaded checkpoint: {checkpoint_name}")
    print(f"Max episode steps: {args.max_episode_steps}")
    print(f"Evaluating for {args.num_eval_episodes} episodes\n")

    # Evaluate
    rewards = []
    lengths = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for ep in tqdm.tqdm(range(args.num_eval_episodes), desc="Evaluating", unit="episode"):
            td = env.reset()
            episode_reward = 0
            episode_length = 0

            while episode_length < args.max_episode_steps:
                td = policy(td)

                # Check for termination
                if td.get("done", torch.tensor(False)).item():
                    break

                # Get reward
                episode_reward += td.get(("next", "reward"), torch.tensor(0.0)).item()
                episode_length += 1

                # Move to next step
                td = env.step(td)

            rewards.append(episode_reward)
            lengths.append(episode_length)
            tqdm.tqdm.write(f"Episode {ep+1}/{args.num_eval_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    env.close()

    print("\n" + "="*60)
    print("Evaluation Results:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"  Min reward:  {np.min(rewards):.2f}")
    print(f"  Max reward:  {np.max(rewards):.2f}")
    print("="*60)


def render_to_plot(args):
    """Render episode to static plot (for NeuroGym environments)."""
    print("="*60)
    print("Rendering Agent Episode")
    print("="*60)

    # Load experiment
    exp_manager = ExperimentManager(args.experiment_name)
    config = exp_manager.load_config()

    # Detect model type
    model_type = config.get('model_type', 'mpn')
    use_rnn = (model_type == 'rnn')
    freeze_plasticity = (model_type == 'mpn-frozen')

    print(f"Model type: {model_type.upper()}")

    # Setup device (CPU for rendering)
    device = torch.device('cpu')
    print("Using CPU for rendering\n")

    # Create environment
    env = TransformedEnv(
        GymEnv(config['env_name'], device=device),
        Compose(
            StepCounter(max_steps=args.max_episode_steps),
            InitTracker(),
        )
    )

    # Create policy model
    policy = create_model(
        env=env,
        model_type=config.get('model_type', 'mpn'),
        hidden_dim=config['hidden_dim'],
        num_layers=config.get('num_layers', 1),
        eta=config.get('eta', 0.1),
        lambda_decay=config.get('lambda_decay', 0.95),
        activation=config.get('activation', 'tanh'),
        device=device
    )

    # Load checkpoint
    checkpoint_name = args.checkpoint if args.checkpoint else "best_model.pt"
    exp_manager.load_model(policy, checkpoint_name=checkpoint_name, device='cpu')

    policy.eval()

    print(f"Loaded checkpoint: {checkpoint_name}")
    print(f"Max episode steps: {args.max_episode_steps}\n")

    # Determine output path
    if args.output is None:
        output_path = exp_manager.plot_dir / "episode_render.png"
    else:
        output_path = Path(args.output)

    # Rollout episode
    print("Running episode...")
    observations = []
    actions = []
    rewards = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        td = env.reset()

        for step in tqdm.tqdm(range(args.max_episode_steps), desc="Recording episode", unit="step"):
            # Store observation
            observations.append(td["observation"].cpu().numpy())

            # Get action
            td = policy(td)
            action = torch.argmax(td["action"], dim=-1).item()
            actions.append(action)

            # Step environment
            td = env.step(td)

            # Store reward
            reward = td.get(("next", "reward"), torch.tensor(0.0)).item()
            rewards.append(reward)

            # Check for done
            if td.get("done", torch.tensor(False)).item():
                break

    env.close()

    # Convert to arrays
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)

    # Create plot
    print(f"Creating plot with {len(observations)} timesteps...")

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))

    # Plot observations
    axs[0].plot(observations)
    axs[0].set_title("Observations")
    axs[0].set_ylabel("Value")
    axs[0].legend([f"Obs {i}" for i in range(observations.shape[1])], loc='upper right', ncol=observations.shape[1])

    # Plot actions
    axs[1].step(range(len(actions)), actions, where='post')
    axs[1].set_title("Actions")
    axs[1].set_ylabel("Action")
    axs[1].set_ylim(-0.5, action_dim - 0.5)

    # Plot rewards
    axs[2].plot(rewards)
    axs[2].set_title(f"Rewards (Total: {np.sum(rewards):.2f})")
    axs[2].set_ylabel("Reward")
    axs[2].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    print(f"Episode reward: {np.sum(rewards):.2f}")
    print(f"Episode length: {len(rewards)}")


def main():
    parser = argparse.ArgumentParser(description="MPN-DQN with TorchRL")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train-neurogym', help='Train on NeuroGym environment')
    train_parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name (random if not provided)')
    train_parser.add_argument('--env-name', type=str, default='GoNogo-v0', help='NeuroGym environment name')
    train_parser.add_argument('--max-episode-steps', type=int, default=500, help='Maximum steps per episode')
    train_parser.add_argument('--total-frames', type=int, default=50000, help='Total number of training frames')
    train_parser.add_argument('--model-type', type=str, default='mpn',
                             choices=['mpn', 'mpn-frozen', 'rnn', 'lstm'],
                             help='Model type: mpn, mpn-frozen (no plasticity), rnn, lstm')
    train_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    train_parser.add_argument('--num-layers', type=int, default=1, help='Number of recurrent layers')
    train_parser.add_argument('--eta', type=float, default=0.1, help='Hebbian learning rate (MPN only)')
    train_parser.add_argument('--lambda-decay', type=float, default=0.95, help='M matrix decay (MPN only)')
    train_parser.add_argument('--activation', type=str, default='tanh',
                             choices=['relu', 'tanh', 'sigmoid'], help='Activation function')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--epsilon-start', type=float, default=0.2, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final exploration rate')
    train_parser.add_argument('--frames-per-batch', type=int, default=50, help='Frames per collection batch')
    train_parser.add_argument('--buffer-size', type=int, default=20000, help='Replay buffer capacity')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    train_parser.add_argument('--utd', type=int, default=64, help='Updates-to-data ratio')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--grad-clip', type=float, default=10.0, help='Gradient clipping')
    train_parser.add_argument('--target-update-tau', type=float, default=0.95, help='Target network soft update tau')
    train_parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint frequency (frames)')
    train_parser.add_argument('--print-freq', type=int, default=500, help='Print and evaluation frequency (frames)')
    train_parser.add_argument('--num-eval-episodes', type=int, default=3, help='Number of evaluation episodes to average')
    train_parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel training environments')
    train_parser.add_argument('--device', type=str, default='cpu', help='Device: gpu or cpu')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name')
    eval_parser.add_argument('--num-eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    eval_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load (default: best_model.pt)')
    eval_parser.add_argument('--max-episode-steps', type=int, default=500, help='Maximum steps per episode')
    eval_parser.add_argument('--device', type=str, default='cpu', help='Device: gpu or cpu')

    # Render command
    render_parser = subparsers.add_parser('render', help='Render episode to plot')
    render_parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name')
    render_parser.add_argument('--output', type=str, default=None, help='Output path (default: experiments/{name}/plots/episode_render.png)')
    render_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load (default: best_model.pt)')
    render_parser.add_argument('--max-episode-steps', type=int, default=500, help='Maximum steps per episode')

    args = parser.parse_args()

    if args.command == 'train-neurogym':
        train_neurogym(args)
    elif args.command == 'eval':
        evaluate(args)
    elif args.command == 'render':
        render_to_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
