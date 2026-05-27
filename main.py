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
    python main.py train-neurogym --experiment-name my-agent --total-frames 50000

    # Evaluate
    python main.py eval --experiment-name my-agent --num-eval-episodes 10

    # Render
    python main.py render --experiment-name my-agent --output render.png
"""

import argparse
import copy
import math
import sys
import uuid
from pathlib import Path

# Matplotlib for rendering
import matplotlib.pyplot as plt
import neurogym
import numpy as np
import torch
import tqdm
import wandb
# TorchRL imports
from tensordict.nn import TensorDictModule as Mod
from tensordict.nn import TensorDictSequential as Seq
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import (Compose, ExplorationType, InitTracker, ParallelEnv,
                          StepCounter, TransformedEnv, set_exploration_type)
from torchrl.envs.libs.gym import GymEnv
from tensordict.nn import TensorDictModuleBase
from torchrl.modules import MLP, EGreedyModule, LSTMModule, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate

# Local imports
from model_utils import ExperimentManager
from mpn_module import RandomInputProjection
from mpn_torchrl_module import MPNModule, MPNPolyModule
import temporal_order_env  # registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0
from neurogym_transform import NeuroGymInfoTransform
from neurogym_wrapper import NeuroGymInfoWrapper
from oracle_agents import get_oracle_reward
from rnn_module import RNNModule


class TrialEGreedyModule(TensorDictModuleBase):
    """
    Epsilon-greedy exploration that commits to one action per trial.

    Unlike EGreedyModule which independently randomizes every step, this module:
      1. At the start of each trial, flips the epsilon coin once.
      2. If random: samples one action and repeats it for every step of the trial.
      3. If greedy: leaves the QValueModule's action unchanged for the whole trial.

    Trial starts are detected by:
      - new_trial == True  (promoted from ("next","new_trial") by step_mdp)
      - is_init == True    (InitTracker marks the first step of a new episode)

    The module exposes an `eps` buffer and a `step()` method so the training
    loop can read and anneal epsilon exactly like EGreedyModule.
    """

    in_keys = ["action", "new_trial", "is_init"]
    out_keys = ["action"]

    def __init__(self, spec, eps_init=1.0, eps_end=0.01, annealing_num_steps=1000, device=None):
        super().__init__()
        self.spec = spec
        self.eps_init = float(eps_init)
        self.eps_end = float(eps_end)
        self.annealing_num_steps = max(1, int(annealing_num_steps))

        self.register_buffer("eps", torch.tensor(eps_init, dtype=torch.float32))

        # Per-trial state — Python scalars, not buffers (ephemeral, not serialised)
        self._trial_started = False   # False until we've seen the first trial start
        self._in_random_trial = False
        self._random_action = None    # action tensor sampled at trial start

        if device is not None:
            self.to(device)

    def forward(self, tensordict):
        new_trial = tensordict.get("new_trial", None)
        is_init   = tensordict.get("is_init",   None)

        # Start of a new trial: either the env signalled trial end on the
        # previous step (new_trial=True after step_mdp) or it's an episode
        # reset (is_init=True), or we haven't started any trial yet.
        is_trial_start = (
            not self._trial_started
            or (new_trial is not None and bool(new_trial.any()))
            or (is_init   is not None and bool(is_init.any()))
        )

        if is_trial_start:
            self._trial_started = True
            if torch.rand(1).item() < self.eps.item():
                self._in_random_trial = True
                self._random_action = self.spec.rand().to(tensordict.device)
            else:
                self._in_random_trial = False
                self._random_action = None

        if self._in_random_trial and self._random_action is not None:
            # Broadcast the stored random action to the tensordict's batch shape
            action = self._random_action
            if tensordict.batch_size:
                action = action.expand(tensordict.batch_size)
            tensordict.set("action", action)
        # else: greedy — leave the QValueModule's action untouched

        return tensordict

    def step(self, frames: int) -> None:
        """Anneal epsilon linearly (same signature as EGreedyModule.step)."""
        progress = min(1.0, frames / self.annealing_num_steps)
        new_eps = self.eps_init + (self.eps_end - self.eps_init) * progress
        self.eps.fill_(float(max(self.eps_end, new_eps)))


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


def create_model(env, model_type, hidden_dim, num_layers, activation, device, lambda_max=0.99, eta_init=0.01, lambda_init=0.99, num_pre_layers=0, num_mlp_layers=1, random_proj_dim=0):
    """
    Create policy model with stacked recurrent layers.

    Args:
        env: Environment (for getting observation/action dimensions)
        model_type: 'mpn', 'mpn-frozen', 'mpn-poly', 'rnn', or 'lstm'
        hidden_dim: Hidden layer dimension
        num_layers: Number of recurrent layers to stack
        activation: Activation function
        device: Device to create model on
        lambda_max: Maximum value for lambda clamping (MPN only, default 0.99)
        num_pre_layers: Number of Linear+tanh layers before MPN (default 0)
        random_proj_dim: If > 0, project observations through a fixed random
            matrix (Xavier init) before any recurrent layer, per eLife-83035
            Methods 5.1. 0 disables the projection. (default: 0)

    Returns:
        policy: The policy network (stacked recurrent layers -> mlp -> qval)
    """
    # Get environment dimensions
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.space.n

    # Determine model configuration
    use_rnn = (model_type == 'rnn')
    use_lstm = (model_type == 'lstm')
    use_mpn_poly = (model_type == 'mpn-poly')

    freeze_plasticity = (model_type == 'mpn-frozen')

    # Optional fixed random input projection (eLife-83035 Methods 5.1).
    # Projects observations to a higher-dimensional space before any recurrent
    # layer, so that near-zero inputs still produce a non-trivial response.
    # W_rand and b_rand are Xavier-initialised and frozen during training.
    layers = []
    if random_proj_dim > 0:
        proj = RandomInputProjection(obs_dim, random_proj_dim).to(device)
        proj_module = Mod(proj, in_keys=["observation"], out_keys=["observation_proj"])
        layers.append(proj_module)
        recurrent_in_key = "observation_proj"
        effective_obs_dim = random_proj_dim
    else:
        recurrent_in_key = "observation"
        effective_obs_dim = obs_dim

    if use_rnn:
        # RNN supports multiple layers natively
        recurrent_module = RNNModule(
            input_size=effective_obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            nonlinearity=activation,
            device=device,
            in_key=recurrent_in_key,
            out_key=f"embed_{num_layers-1}",
        )
        layers.append(recurrent_module)
        env.append_transform(recurrent_module.make_tensordict_primer())
    elif use_lstm:
        # LSTM supports multiple layers natively
        recurrent_module = LSTMModule(
            input_size=effective_obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            device=device,
            in_key=recurrent_in_key,
            out_key=f"embed_{num_layers-1}",
        )
        layers.append(recurrent_module)
        env.append_transform(recurrent_module.make_tensordict_primer())
    else:
        # Stack multiple MPN layers
        for layer_idx in range(num_layers):
            in_key = recurrent_in_key if layer_idx == 0 else f"embed_{layer_idx-1}"
            out_key = f"embed_{layer_idx}"

            # Recurrent state keys for this layer
            in_keys = [in_key, f"recurrent_state_{layer_idx}"]
            out_keys = [out_key, ("next", f"recurrent_state_{layer_idx}")]

            if use_mpn_poly:
                common_kwargs = dict(
                    input_size=effective_obs_dim if layer_idx == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    activation=activation,
                    freeze_plasticity=freeze_plasticity,
                    num_pre_layers=num_pre_layers if layer_idx == 0 else 0,
                    device=device,
                    in_keys=in_keys,
                    out_keys=out_keys,
                )
                mpn_layer = MPNPolyModule(**common_kwargs)
            else:
                common_kwargs = dict(
                    input_size=effective_obs_dim if layer_idx == 0 else hidden_dim,
                    hidden_size=hidden_dim,
                    activation=activation,
                    freeze_plasticity=freeze_plasticity,
                    lambda_max=lambda_max,
                    eta_init=eta_init,
                    lambda_init=lambda_init,
                    num_pre_layers=num_pre_layers if layer_idx == 0 else 0,
                    device=device,
                    in_keys=in_keys,
                    out_keys=out_keys,
                )
                mpn_layer = MPNModule(**common_kwargs)

            layers.append(mpn_layer)
            env.append_transform(mpn_layer.make_tensordict_primer())

    # Combine layers into sequence
    recurrent_module = Seq(*layers)

    # Create MLP head for Q-values (reads from final layer output)
    mlp = MLP(
        out_features=action_dim,
        num_cells=[hidden_dim] * num_mlp_layers,
        device=device,
    )
    mlp[-1].bias.data.fill_(0.0)
    mlp_module = Mod(mlp, in_keys=[f"embed_{num_layers-1}"], out_keys=["action_value"])

    # Q-value module
    qval = QValueModule(spec=env.action_spec)

    # Build policy
    policy = Seq(recurrent_module, mlp_module, qval)

    return policy


def _reseed_torchrl_env(env, seed: int) -> None:
    """Walk the TorchRL wrapper chain and reseed the TrialEnv's rng directly.

    TorchRL's set_seed / gymnasium's reset(seed=...) only update np_random,
    but neurogym's TrialEnv drives all trial randomness through self.rng
    (a np.random.RandomState).  We must patch it directly for reproducibility.

    Strategy: find the gym-compatible env inside TorchRL's wrappers (has
    .unwrapped), then call .unwrapped to skip past all gymnasium wrappers
    (e.g. OrderEnforcing) to reach the TrialEnv directly.
    """
    e = env
    while e is not None:
        inner = getattr(e, '_env', None) or getattr(e, 'env', None)
        if inner is not None and hasattr(inner, 'unwrapped'):
            # Found the gym env layer; .unwrapped reaches the base TrialEnv
            base = inner.unwrapped
            if hasattr(base, 'rng'):
                base.rng = np.random.RandomState(seed)
                return
        # recurse into the next TorchRL wrapper layer
        candidate = getattr(e, '_env', None)
        if candidate is None:
            candidate = getattr(e, 'env', None)
        e = candidate
    # fallback: let TorchRL handle it (no TrialEnv found)
    env.set_seed(seed)


def evaluate_policy(policy, eval_env, num_steps, num_episodes=1, seed=None):
    """
    Evaluate policy by running episodes sequentially on a single environment.

    Uses sequential evaluation (not ParallelEnv) to avoid forking N worker
    processes, which was the primary cause of OOM on 8GB condor nodes.

    Args:
        policy: The policy network to evaluate
        eval_env: A single evaluation environment
        num_steps: Maximum steps per episode
        num_episodes: Number of episodes to average over
        seed: Optional RNG seed to pass to the environment before rolling out.
            Pass the same seed to evaluate_oracle() for a fair comparison.

    Returns:
        avg_reward: Average total reward across all episodes
        std_reward: Standard deviation of rewards across episodes
    """
    policy.eval()
    episode_rewards = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for ep_idx in range(num_episodes):
            if seed is not None:
                _reseed_torchrl_env(eval_env, seed + ep_idx)
            rollout = eval_env.rollout(num_steps, policy)
            rewards = rollout.get(("next", "reward"))
            rewards = rewards.cpu().numpy() if torch.is_tensor(rewards) else np.array(rewards)
            episode_rewards.append(float(rewards.sum()))

    policy.train()
    arr = np.array(episode_rewards)
    return float(arr.mean()), float(arr.std())


def train_neurogym(args):
    """Train MPN/RNN-DQN on NeuroGym environment using TorchRL."""
    print("="*60)
    print("Training with TorchRL on NeuroGym")
    print("="*60)

    # Auto-generate experiment ID and append to experiment name
    if args.experiment_id is None:
        args.experiment_id = str(uuid.uuid4())[:8]
    if args.experiment_name is not None:
        args.experiment_name = f"{args.experiment_name}-{args.experiment_id}"

    # Create experiment manager
    exp_manager = ExperimentManager(args.experiment_name)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Experiment ID: {args.experiment_id}")
    if args.tag:
        print(f"Tag: {args.tag}")
    print(f"Directory: {exp_manager.exp_dir}\n")

    # Save configuration
    config = vars(args)
    config['command'] = 'train-neurogym'
    config['algorithm'] = 'dqn'
    exp_manager.save_config(config)

    # Initialise Weights & Biases
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=exp_manager.experiment_name,
            config=config,
            tags=[args.tag] if args.tag else [],
            dir=str(exp_manager.exp_dir),
        )

    # Setup device
    device = get_device(args.device)
    print()

    # Create NeuroGym environment using TorchRL's GymEnv
    print(f"Creating NeuroGym environment: {args.env_name}")

    def _patch_rewards(gymenv):
        """Set incorrect-decision rewards to -1.0 to match reference reward structure."""
        rewards = gymenv._env.unwrapped.rewards
        for key in ('fail', 'miss'):
            if key in rewards:
                rewards[key] = -1.0

    def make_oracle_env():
        import neurogym as ngym
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = ngym.make(args.env_name)
        for key in ('fail', 'miss'):
            if key in env.unwrapped.rewards:
                env.unwrapped.rewards[key] = -1.0
        return env

    # Create environment factory function for parallel evaluation
    def make_env():
        gymenv = GymEnv(args.env_name, device=device)
        _patch_rewards(gymenv)
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
    use_mpn_poly = (args.model_type == 'mpn-poly')
    freeze_plasticity = (args.model_type == 'mpn-frozen')

    print(f"Algorithm: DQN with TorchRL")
    print(f"Model type: {args.model_type.upper()}")
    print(f"Environment: {args.env_name}")
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    print(f"Max episode steps: {args.max_episode_steps}")

    if args.random_proj_dim > 0:
        print(f"Random input projection: {obs_dim} → {args.random_proj_dim} (Xavier init, fixed)")

    if use_rnn:
        print(f"RNN: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, activation={args.activation}\n")
    elif use_lstm:
        print(f"LSTM: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}\n")
    elif use_mpn_poly:
        print(f"MPN-Poly: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")
        print(f"Activation: {args.activation}, Update: a*(M⊙M) + b*M + h⊗x^T (a,b trainable)")
        if args.num_pre_layers > 0:
            print(f"Pre-MPN layers: {args.num_pre_layers} x Linear+tanh")
        print()
    else:
        print(f"MPN: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, lambda_max={args.lambda_max}")
        print(f"Activation: {args.activation}, Plasticity frozen: {freeze_plasticity}")
        print(f"Eta: trainable scalar (init=0.01), Lambda: trainable scalar (init=0.95)")
        if args.num_pre_layers > 0:
            print(f"Pre-MPN layers: {args.num_pre_layers} x Linear+tanh")
        print()

    # Create policy model
    policy = create_model(
        env=env,
        model_type=args.model_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
        device=device,
        lambda_max=args.lambda_max,
        eta_init=args.eta_init,
        lambda_init=args.lambda_init,
        num_pre_layers=args.num_pre_layers,
        random_proj_dim=args.random_proj_dim,
    )

    # Exploration module
    total_frames = args.total_frames
    annealing_frames = get_epsilon_for_annealing(
        total_frames, args.epsilon_start, args.epsilon_end
    )

    exploration_module = TrialEGreedyModule(
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Data collector
    collector = SyncDataCollector(
        env,
        stoch_policy,
        frames_per_batch=args.frames_per_batch,
        total_frames=total_frames,
        device=device,
    )

    # Sequence length for slice-aligned sampling
    seq_len = args.sequence_len if args.sequence_len is not None else args.frames_per_batch

    # Replay buffer with trial-aligned sequence sampling.
    # end_key=("next", "new_trial") makes SliceSampler treat each trial as its
    # own trajectory: it finds all steps where new_trial=True, then samples
    # sequences of length slice_len ending at those steps. This ensures every
    # training sequence is a complete trial (stimulus → delay → decision) and
    # never crosses trial boundaries. Without this, random windows often span
    # two trials, breaking the stimulus→reward association that recurrent memory
    # is supposed to learn — making all models behave like feedforward networks.
    # truncated_key=('next','truncated') (default) additionally prevents sampling
    # across episode boundaries caused by StepCounter hitting max_episode_steps.
    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.buffer_size),
        sampler=SliceSampler(
            slice_len=seq_len,
            end_key=("next", "new_trial"),
            strict_length=False,
        ),
        batch_size=seq_len,  # one trial per gradient update
        prefetch=10,
    )

    print(f"Replay buffer: capacity={args.buffer_size}, batch_size=1 trial (~{seq_len} steps, trial-aligned, end_key=new_trial)")
    print(f"UTD (updates-to-data): {args.utd}")
    print(f"Epsilon: {args.epsilon_start} → {args.epsilon_end} over {annealing_frames} frames")
    print(f"Target update tau: {args.target_update_tau}")
    print(f"Total frames: {total_frames}")

    print("Starting training...")
    print("-"*60)

    # Create a single evaluation environment (sequential eval avoids spawning
    # N worker processes that each load the full Python/PyTorch stack)
    eval_env = make_env()

    # Training metrics
    eval_rewards = []
    eval_reward_stds = []
    eval_losses = []
    eval_oracle_rewards = []  # oracle reward on the same seed as each eval
    recent_losses = []  # Track losses between evaluations
    best_reward = -float('inf')
    best_model_state = {'state_dict': None, 'optimizer_state': None, 'metadata': {}}
    frames_collected = 0
    last_eval_frame = 0
    _eval_rng = np.random.default_rng(seed=42)  # reproducible eval seeds

    # Progress bar
    pbar = tqdm.tqdm(total=total_frames, desc="Training", unit="frames")

    _printed_batch = False  # flag to print one batch for inspection

    # Training loop
    for i, data in enumerate(collector):
        # Add data to replay buffer — store individual frames (not batched)
        # so SliceSampler can reconstruct episode-aligned sequences
        replay_buffer.extend(data.to_tensordict().cpu())

        frames_collected += data.numel()

        # Perform gradient updates by sampling individual trials from the replay buffer.
        batch_loss_vals = []
        for _ in range(args.utd):
            if len(replay_buffer) >= seq_len:
                sample = replay_buffer.sample().to(device, non_blocking=True)
                if not _printed_batch:
                    _printed_batch = True
                    print("\n===== BATCH SAMPLE INSPECTION =====")
                    print(f"Raw sample shape: {sample.shape}")
                    print(f"  Expected: [T] flat — one complete trial (variable length, ends at new_trial=True)")
                    print(f"Keys: {list(sample.keys())}")
                    print(f"  observation shape : {sample['observation'].shape}")
                    print(f"  action shape      : {sample['action'].shape}")
                    print(f"  reward shape      : {sample[('next','reward')].shape}")
                    print(f"  gt shape          : {sample[('next','gt')].shape}")
                    print(f"  new_trial shape   : {sample[('next','new_trial')].shape}")
                    print(f"  is_init shape     : {sample['is_init'].shape}")
                    obs = sample['observation']
                    act = sample['action']
                    rew = sample[('next', 'reward')]
                    gt = sample[('next', 'gt')]
                    nt = sample[('next', 'new_trial')]
                    ii = sample['is_init']
                    n_steps = obs.shape[0]
                    print(f"\n--- All {n_steps} steps (obs | action | reward | gt | new_trial | is_init) ---")
                    for t in range(n_steps):
                        print(f"  t={t:3d}  obs={obs[t].tolist()}  act={act[t].tolist()}  rew={rew[t].item():.2f}  gt={gt[t].item()}  new_trial={int(nt[t].item())}  is_init={int(ii[t].item())}")
                    print("====================================\n")
                # Zero-start heuristic: force is_init=True at the first step so
                # the recurrent module starts with a zeroed hidden state.
                sample["is_init"][0] = True

                # Optionally zero out fixation-period rewards. obs[:,0] is the
                # fixation channel across all NeuroGym environments (1 = active).
                # The -0.1 fixation penalties are noise during epsilon-greedy
                # exploration and corrupt Q-value estimates for the decision period.
                if args.mask_fixation_reward:
                    fixation_mask = sample["observation"][:, 0] > 0.5
                    sample[("next", "reward")][fixation_mask] = 0.0

                loss_vals = loss_fn(sample)

                optimizer.zero_grad()
                loss_vals["loss"].backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()
                updater.step()

                batch_loss_vals.append(loss_vals["loss"].item())

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
            # Draw a shared seed so the policy and oracle see identical trials
            eval_seed = int(_eval_rng.integers(0, 2**31))

            # Run evaluation rollout (using max_episode_steps)
            eval_reward, eval_reward_std = evaluate_policy(
                policy, eval_env, args.max_episode_steps,
                args.num_eval_episodes, seed=eval_seed,
            )

            # Run oracle with the same seed for a fair comparison
            oracle_reward = get_oracle_reward(
                args.env_name,
                n_episodes=args.num_eval_episodes,
                max_steps=args.max_episode_steps,
                seed=eval_seed,
                env_factory=make_oracle_env,
            )

            # Calculate average loss since last evaluation
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0

            # Compute percentage of oracle reward obtained
            pct_oracle = (eval_reward / oracle_reward * 100.0) if oracle_reward > 0 else float("nan")

            # Store evaluation metrics (aligned arrays)
            eval_rewards.append(eval_reward)
            eval_reward_stds.append(eval_reward_std)
            eval_losses.append(avg_loss)
            eval_oracle_rewards.append(oracle_reward)

            # Save to history
            exp_manager.append_training_history(
                int(frames_collected),
                float(eval_reward),
                int(data.numel()),
                float(avg_loss),
                float(current_epsilon),
                oracle_reward=float(oracle_reward),
                pct_oracle=float(pct_oracle) if not math.isnan(pct_oracle) else None,
            )

            # Print combined progress and evaluation
            tqdm.tqdm.write(f"Frames {frames_collected:7d}/{total_frames} | "
                            f"Eval Reward: {eval_reward:7.2f} ± {eval_reward_std:5.2f} | "
                            f"Oracle: {pct_oracle:5.1f}% ({oracle_reward:.1f}) | "
                            f"Loss: {avg_loss:6.4f} | "
                            f"ε: {current_epsilon:.3f}")

            # Log to wandb
            if args.wandb:
                wandb.log({
                    "eval/reward": eval_reward,
                    "eval/reward_std": eval_reward_std,
                    "eval/oracle_reward": oracle_reward,
                    "eval/pct_oracle": pct_oracle,
                    "train/loss": avg_loss,
                    "train/epsilon": current_epsilon,
                }, step=frames_collected)

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
                if args.wandb:
                    wandb.log({"eval/best_reward": best_reward}, step=frames_collected)

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
            exp_manager.cleanup_checkpoints(args.max_checkpoints)

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
    pct_oracles = [
        (r / o * 100.0) if o > 0 else float("nan")
        for r, o in zip(eval_rewards, eval_oracle_rewards)
    ]
    metrics = {
        'eval_rewards': eval_rewards,
        'eval_reward_stds': eval_reward_stds,
        'eval_losses': eval_losses,
        'eval_oracle_rewards': eval_oracle_rewards,
        'eval_pct_oracle': pct_oracles,
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_path}")

    exp_manager.mark_completed()

    # Finish wandb run
    if args.wandb:
        wandb.finish()

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
    _gymenv = GymEnv(config['env_name'], device=device)
    _gymenv._env.unwrapped.rewards.update({k: -1.0 for k in ('fail', 'miss') if k in _gymenv._env.unwrapped.rewards})
    env = TransformedEnv(
        _gymenv,
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
        activation=config.get('activation', 'tanh'),
        device=device,
        lambda_max=config.get('lambda_max', 0.99),
        num_pre_layers=config.get('num_pre_layers', 0),
        random_proj_dim=config.get('random_proj_dim', 0),
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
    _gymenv = GymEnv(config['env_name'], device=device)
    _gymenv._env.unwrapped.rewards.update({k: -1.0 for k in ('fail', 'miss') if k in _gymenv._env.unwrapped.rewards})
    env = TransformedEnv(
        _gymenv,
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
        activation=config.get('activation', 'tanh'),
        device=device,
        lambda_max=config.get('lambda_max', 0.99),
        num_pre_layers=config.get('num_pre_layers', 0),
        random_proj_dim=config.get('random_proj_dim', 0),
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
                             choices=['mpn', 'mpn-frozen', 'mpn-poly', 'rnn', 'lstm'],
                             help='Model type: mpn, mpn-frozen (no plasticity), mpn-poly (polynomial update), rnn, lstm')
    train_parser.add_argument('--hidden-dim', type=int, default=64, help='Hidden dimension')
    train_parser.add_argument('--num-layers', type=int, default=1, help='Number of recurrent layers')
    train_parser.add_argument('--num-mlp-layers', type=int, default=1, help='Number of hidden layers in MLP head (default: 1)')
    train_parser.add_argument('--num-pre-layers', type=int, default=0, help='Number of Linear+tanh layers before MPN (default: 0)')
    train_parser.add_argument('--random-proj-dim', type=int, default=0, help='Project observations through a fixed random matrix (Xavier init) of this size before any recurrent layer, per eLife-83035 Methods 5.1. Set to 0 to disable. (default: 0)')
    train_parser.add_argument('--lambda-max', type=float, default=0.99, help='Maximum value for lambda clamping (MPN only)')
    train_parser.add_argument('--eta-init', type=float, default=0.01, help='Initial value for eta (Hebbian learning rate, MPN only)')
    train_parser.add_argument('--lambda-init', type=float, default=0.99, help='Initial value for lambda (M decay, MPN only)')
    train_parser.add_argument('--activation', type=str, default='tanh',
                             choices=['relu', 'tanh', 'sigmoid'], help='Activation function')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--epsilon-start', type=float, default=0.2, help='Initial exploration rate')
    train_parser.add_argument('--epsilon-end', type=float, default=0.01, help='Final exploration rate')
    train_parser.add_argument('--frames-per-batch', type=int, default=50, help='Frames per collection batch')
    train_parser.add_argument('--buffer-size', type=int, default=2000, help='Replay buffer capacity in frames')
    train_parser.add_argument('--sequence-len', type=int, default=None, help='Length of sequences sampled from replay buffer. Should match episode/trial length. Defaults to frames_per_batch if not set.')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    train_parser.add_argument('--utd', type=int, default=64, help='Updates-to-data ratio')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--weight-decay', type=float, default=0.0, help='L2 weight decay for Adam optimizer (default: 0.0)')
    train_parser.add_argument('--grad-clip', type=float, default=10.0, help='Gradient clipping')
    train_parser.add_argument('--mask-fixation-reward', action='store_true', default=False,
                              help='Zero out rewards during fixation period (obs[:,0]>0.5) to remove noise from fixation penalties')
    train_parser.add_argument('--target-update-tau', type=float, default=0.95, help='Target network soft update tau')
    train_parser.add_argument('--checkpoint-freq', type=int, default=5000, help='Checkpoint frequency (frames)')
    train_parser.add_argument('--max-checkpoints', type=int, default=1, help='Max periodic checkpoints to keep (oldest deleted)')
    train_parser.add_argument('--print-freq', type=int, default=500, help='Print and evaluation frequency (frames)')
    train_parser.add_argument('--num-eval-episodes', type=int, default=3, help='Number of evaluation episodes to average')
    train_parser.add_argument('--num-envs', type=int, default=1, help='Number of parallel training environments')
    train_parser.add_argument('--device', type=str, default='cpu', help='Device: gpu or cpu')
    train_parser.add_argument('--tag', type=str, default=None, help='Tag to group related experiments (e.g., wm-sweep-v1)')
    train_parser.add_argument('--experiment-id', type=str, default=None, help='Unique experiment ID (auto-generated UUID if not provided)')
    train_parser.add_argument('--wandb', action='store_true', default=False, help='Enable Weights & Biases logging')
    train_parser.add_argument('--wandb-project', type=str, default='mpn-rl', help='W&B project name')
    train_parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (team/username)')

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
