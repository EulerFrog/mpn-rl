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
import sys
import math
import torch
import numpy as np
from pathlib import Path
import neurogym
import tqdm

# TorchRL imports
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (Compose, ExplorationType, InitTracker,
                          StepCounter, TransformedEnv, set_exploration_type)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import MLP, EGreedyModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

# Local imports
from model_utils import ExperimentManager
from mpn_torchrl_module import MPNModule
from rnn_module import RNNModule
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper

# Matplotlib for rendering
import matplotlib.pyplot as plt


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


def get_device(device_str='auto'):
    """
    Get PyTorch device with automatic GPU detection.

    Args:
        device_str: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'

    Returns:
        torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            print("GPU not available, using CPU")
    else:
        device = torch.device(device_str)
        if device.type == 'cuda' and torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
        elif device.type == 'cuda':
            print("Warning: CUDA device requested but not available, falling back to CPU")
            device = torch.device('cpu')

    return device


def create_model(env, model_type, hidden_dim, num_layers, eta, lambda_decay, activation, device):
    """
    Create policy model with stacked recurrent layers.

    Args:
        env: Environment (for getting observation/action dimensions)
        model_type: 'mpn', 'mpn-frozen', or 'rnn'
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


def evaluate_policy(policy, env, num_steps, device):
    """
    Evaluate policy on GoNogo task by doing a rollout for a fixed number of steps.

    Uses ground truth ('gt') and trial completion ('new_trial') to properly count:
    - True Positives (TP): Correct Go responses (reward = +1.0, gt = 1)
    - False Negatives (FN): Missed Go trials (no response when gt = 1)
    - True Negatives (TN): Correct No-go (no response when gt = 0)
    - False Positives (FP): Incorrect No-go responses (reward = -0.5, gt = 0)

    Returns:
        accuracy: (TP + TN) / total_trials
        true_positives: Number of correct Go responses (TP)
        true_negatives: Number of correct No-go rejections (TN)
        false_negatives: Number of missed Go trials (FN)
        false_positives: Number of incorrect No-go responses (FP)
        total_reward: Total cumulative reward
        total_trials: Total number of completed trials
    """
    policy.eval()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # Do a full rollout
        rollout = env.rollout(num_steps, policy)

        # Extract data from rollout as numpy arrays
        rewards = rollout.get(("next", "reward"))
        rewards = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.array(rewards)
        rewards = rewards.squeeze()  # Remove extra dimensions

        # Get ground truth and new_trial flags (both in "next" dict)
        gt = rollout.get(("next", "gt"))
        gt = gt.cpu().numpy() if torch.is_tensor(gt) else np.array(gt)

        new_trial = rollout.get(("next", "new_trial"))
        new_trial = new_trial.cpu().numpy() if torch.is_tensor(new_trial) else np.array(new_trial)

        # Count trial outcomes at trial completion points
        # Exclude abort trials (reward = -0.1) from analysis
        trial_complete_mask = new_trial.astype(bool)
        abort_mask = (rewards > -0.2) & (rewards < -0.05)  # reward ~ -0.1
        valid_trial_mask = trial_complete_mask & ~abort_mask

        # Go trials (gt == 1) and No-go trials (gt == 0)
        go_trial_mask = (gt == 1) & valid_trial_mask
        nogo_trial_mask = (gt == 0) & valid_trial_mask

        # True Positives: correct Go responses (reward > 0 at Go trial completion)
        true_positives = ((rewards > 0.5) & go_trial_mask).sum()

        # False Negatives: failed to respond on Go trials (reward == 0 at Go trial completion)
        false_negatives = ((rewards == 0) & go_trial_mask).sum()

        # True Negatives: correctly withheld on No-go trials (reward == 0 at No-go completion)
        true_negatives = ((rewards == 0) & nogo_trial_mask).sum()

        # False Positives: incorrectly responded on No-go trials (reward < 0, excluding aborts)
        false_positives = ((rewards < -0.4) & nogo_trial_mask).sum()

        # Total trials (excluding aborts)
        total_trials = valid_trial_mask.sum()

        # Calculate accuracy
        accuracy = (true_positives + true_negatives) / total_trials if total_trials > 0 else 0.0

        # Total reward
        total_reward = rewards.sum()

    policy.train()
    return accuracy, int(true_positives), int(true_negatives), int(false_negatives), int(false_positives), float(total_reward), int(total_trials)


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

    # Create GymEnv first with the env name
    gymenv = GymEnv(args.env_name, device=device)

    # Then wrap the internal gym environment to capture info dict
    gymenv._env = NeuroGymInfoWrapper(gymenv._env)

    env = TransformedEnv(
        gymenv,
        Compose(
            StepCounter(max_steps=args.max_episode_steps),
            InitTracker(),
            NeuroGymInfoTransform(),  # Extract gt and new_trial from info
        )
    )

    # Get environment dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Print model configuration
    use_rnn = (args.model_type == 'rnn')
    freeze_plasticity = (args.model_type == 'mpn-frozen')

    print(f"Algorithm: DQN with TorchRL")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Environment: {args.env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Max episode steps: {args.max_episode_steps}")

    if use_rnn:
        print(f"RNN: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, activation={args.activation}\n")
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

    # Training metrics
    batch_rewards = []
    batch_losses = []
    best_accuracy = -float('inf')
    frames_collected = 0

    # Evaluation metrics tracking
    eval_accuracies = []
    eval_true_positives = []
    eval_true_negatives = []
    eval_false_negatives = []
    eval_false_positives = []
    eval_total_trials = []

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
        batch_reward = data["next", "reward"].sum().item()
        batch_loss = np.mean(batch_loss_vals) if batch_loss_vals else 0.0
        # Convert epsilon to Python float (may be tensor)
        current_epsilon = float(exploration_module.eps.item()) if torch.is_tensor(exploration_module.eps) else float(exploration_module.eps)

        # Store metrics
        batch_rewards.append(batch_reward)
        if batch_loss_vals:
            batch_losses.append(batch_loss)

        # Update progress bar
        pbar.update(data.numel())
        pbar.set_postfix({
            'reward': f'{batch_reward:.2f}',
            'loss': f'{batch_loss:.4f}',
            'eps': f'{current_epsilon:.3f}',
            'buffer': len(replay_buffer)
        })

        # Save to history every batch
        exp_manager.append_training_history(
            int(frames_collected),
            float(batch_reward),
            int(data.numel()),
            float(batch_loss),
            float(current_epsilon)
        )

        # Print progress and evaluate every N frames
        if frames_collected % args.print_freq == 0 and frames_collected > 0:
            # Run evaluation rollout (using max_episode_steps)
            eval_accuracy, eval_tp, eval_tn, eval_fn, eval_fp, eval_total_reward, eval_trials = evaluate_policy(
                policy, env, args.max_episode_steps, device
            )

            # Store evaluation metrics
            eval_accuracies.append(eval_accuracy)
            eval_true_positives.append(eval_tp)
            eval_true_negatives.append(eval_tn)
            eval_false_negatives.append(eval_fn)
            eval_false_positives.append(eval_fp)
            eval_total_trials.append(eval_trials)

            # Calculate recent batch averages for comparison
            recent_window = min(100, len(batch_rewards))
            avg_batch_reward = np.mean(batch_rewards[-recent_window:]) if batch_rewards else 0.0
            avg_loss = np.mean(batch_losses[-recent_window:]) if batch_losses else 0.0

            # Print combined progress and evaluation
            tqdm.tqdm.write(f"Frames {frames_collected:7d}/{total_frames} | "
                            f"Accuracy: {eval_accuracy:7.4f} | "
                            f"Total Reward: {eval_total_reward:7.2f} | "
                            f"Trials: {eval_trials:3d} | "
                            f"TP: {eval_tp:3d} | "
                            f"TN: {eval_tn:3d} | "
                            f"FN: {eval_fn:3d} | "
                            f"FP: {eval_fp:3d} | "
                            f"Loss: {avg_loss:6.4f} | "
                            f"ε: {current_epsilon:.3f}")

            # Check if this is the best eval performance
            if eval_accuracy > best_accuracy:
                best_accuracy = eval_accuracy

                # Save as best model
                exp_manager.save_model(
                    policy,
                    optimizer=optimizer,
                    checkpoint_name="best_model.pt",
                    metadata={
                        'frames': int(frames_collected),
                        'eval_accuracy': float(eval_accuracy),
                        'eval_total_reward': float(eval_total_reward),
                        'eval_trials': int(eval_trials),
                        'eval_true_positives': int(eval_tp),
                        'eval_true_negatives': int(eval_tn),
                        'eval_false_negatives': int(eval_fn),
                        'eval_false_positives': int(eval_fp),
                        'epsilon': float(current_epsilon),
                    }
                )
                tqdm.tqdm.write(f"  → New best model! Accuracy: {eval_accuracy:.4f}")

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
    final_avg_reward = float(np.mean(batch_rewards[-100:])) if len(batch_rewards) >= 100 else 0.0
    exp_manager.save_model(
        policy,
        optimizer=optimizer,
        checkpoint_name="final_model.pt",
        metadata={
            'frames': int(frames_collected),
            'avg_reward': float(final_avg_reward),
        }
    )

    # Save training metrics for plotting
    import json
    metrics_path = exp_manager.exp_dir / "training_metrics.json"
    metrics = {
        'eval_accuracies': eval_accuracies,
        'eval_true_positives': eval_true_positives,
        'eval_true_negatives': eval_true_negatives,
        'eval_false_negatives': eval_false_negatives,
        'eval_false_positives': eval_false_positives,
        'eval_total_trials': eval_total_trials,
        'batch_rewards': batch_rewards,
        'batch_losses': batch_losses,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    env.close()
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Total frames: {frames_collected}")
    if eval_accuracies:
        print(f"Best eval accuracy: {best_accuracy:.4f}")
        print(f"Final eval accuracy: {eval_accuracies[-1]:.4f}")
        print(f"Final eval trials: {eval_total_trials[-1]}")
        print(f"Final eval false negatives: {eval_false_negatives[-1]}")
        print(f"Final eval false positives: {eval_false_positives[-1]}")
    else:
        print(f"Final batch avg reward: {final_avg_reward:.2f}")
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
                             choices=['mpn', 'mpn-frozen', 'rnn'],
                             help='Model type: mpn, mpn-frozen (no plasticity), rnn')
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
    train_parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu')

    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained agent')
    eval_parser.add_argument('--experiment-name', type=str, required=True, help='Experiment name')
    eval_parser.add_argument('--num-eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    eval_parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint to load (default: best_model.pt)')
    eval_parser.add_argument('--max-episode-steps', type=int, default=500, help='Maximum steps per episode')
    eval_parser.add_argument('--device', type=str, default='auto', help='Device: auto, cuda, cpu')

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
