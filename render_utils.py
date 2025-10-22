"""
Rendering utilities for creating GIFs with CartPole + M matrix visualization

Creates side-by-side visualizations:
- Left: CartPole environment rendering
- Right: M matrix heatmap evolution

Requires: imageio, PIL (pillow), matplotlib, numpy
"""

import numpy as np
import torch
from typing import Optional, List, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from PIL import Image
import imageio


def fig_to_array(fig: Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array (RGB)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
    return img_array[:, :, :3]  # Drop alpha channel


def create_m_matrix_frame(M: np.ndarray, figsize=(4, 3), dpi=100, vmin=None, vmax=None) -> np.ndarray:
    """
    Create a heatmap visualization of M matrix as numpy array.

    Args:
        M: M matrix of shape (hidden_dim, obs_dim)
        figsize: Figure size in inches
        dpi: DPI for rendering
        vmin: Minimum value for colormap (if None, computed from M)
        vmax: Maximum value for colormap (if None, computed from M)

    Returns:
        RGB image array
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Plot M matrix with consistent scale and proper axis extent
    if vmin is None or vmax is None:
        abs_max = np.abs(M).max() if np.abs(M).max() > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax

    # Create explicit normalization for consistent coloring across frames
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Set extent so x-axis goes from 0 to n_inputs (e.g., 0 to 4)
    n_hidden, n_inputs = M.shape
    extent = [0, n_inputs, n_hidden, 0]  # [left, right, bottom, top]

    im = ax.imshow(M, cmap='RdBu_r', aspect='auto', norm=norm, extent=extent)

    # Set axis labels and ticks
    ax.set_xlabel('Input Dimension', fontsize=8)
    ax.set_ylabel('Hidden Dimension', fontsize=8)
    ax.set_title('M Matrix (Synaptic Modulation)', fontsize=10)

    # Set x-axis ticks to show integer dimensions
    ax.set_xticks(range(n_inputs + 1))

    ax.tick_params(labelsize=6)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label('Modulation', fontsize=8)

    plt.tight_layout()

    # Convert to array
    img_array = fig_to_array(fig)
    plt.close(fig)

    return img_array


def create_hidden_state_frame(h: np.ndarray, figsize=(4, 3), dpi=100, vmin=None, vmax=None) -> np.ndarray:
    """
    Create a bar plot visualization of RNN hidden state as numpy array.

    Args:
        h: Hidden state vector of shape (hidden_dim,)
        figsize: Figure size in inches
        dpi: DPI for rendering
        vmin: Minimum value for colormap (if None, computed from h)
        vmax: Maximum value for colormap (if None, computed from h)

    Returns:
        RGB image array
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Compute vmin/vmax if not provided
    if vmin is None or vmax is None:
        abs_max = np.abs(h).max() if np.abs(h).max() > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax

    # Create bar colors based on values
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdBu_r
    colors = [cmap(norm(val)) for val in h]

    # Create bar plot
    indices = np.arange(len(h))
    ax.bar(indices, h, color=colors, width=1.0, edgecolor='black', linewidth=0.5)

    # Set axis labels and limits
    ax.set_xlabel('Hidden Unit', fontsize=8)
    ax.set_ylabel('Activation', fontsize=8)
    ax.set_title('RNN Hidden State', fontsize=10)
    ax.set_xlim(-0.5, len(h) - 0.5)
    ax.set_ylim(vmin, vmax)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax.tick_params(labelsize=6)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Convert to array
    img_array = fig_to_array(fig)
    plt.close(fig)

    return img_array


def combine_frames_horizontal(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Combine two frames horizontally with optional resizing to match heights.

    Args:
        frame1: Left image (H1, W1, 3)
        frame2: Right image (H2, W2, 3)

    Returns:
        Combined image (max(H1,H2), W1+W2, 3)
    """
    img1 = Image.fromarray(frame1)
    img2 = Image.fromarray(frame2)

    # Resize to same height
    target_height = max(img1.height, img2.height)

    if img1.height != target_height:
        new_width = int(img1.width * target_height / img1.height)
        img1 = img1.resize((new_width, target_height), Image.Resampling.LANCZOS)

    if img2.height != target_height:
        new_width = int(img2.width * target_height / img2.height)
        img2 = img2.resize((new_width, target_height), Image.Resampling.LANCZOS)

    # Combine horizontally
    total_width = img1.width + img2.width
    combined = Image.new('RGB', (total_width, target_height))
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))

    return np.array(combined)


def render_episode_to_gif(
    dqn,
    env,
    save_path: str,
    max_steps: int = 500,
    fps: int = 30,
    epsilon: float = 0.0,
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 100
) -> float:
    """
    Render an episode to GIF with environment + state visualization.

    Supports both MPNDQN (M matrix heatmap) and RNNDQN (hidden state bar plot).

    Args:
        dqn: Trained model (MPNDQN or RNNDQN)
        env: Gym environment (must support rgb_array rendering)
        save_path: Path to save GIF
        max_steps: Maximum steps per episode
        fps: Frames per second for GIF
        epsilon: Exploration rate (0 = greedy)
        figsize: Size of state visualization plot
        dpi: DPI for state visualization plot

    Returns:
        total_reward: Episode reward
    """
    # Detect model type by checking state shape
    test_state = dqn.init_state(batch_size=1)
    is_mpn = (len(test_state.shape) == 3)  # MPN: (batch, hidden, obs), RNN: (batch, hidden)

    if is_mpn:
        print("First pass: computing M matrix scale...")
    else:
        print("First pass: computing hidden state scale...")

    # Get initial seed to reset to same state later
    reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, info = reset_result
        # Try to get seed from info if available
        seed = info.get('seed', None)
    else:
        obs = reset_result
        seed = None

    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    states = []
    actions_trajectory = []

    for step in range(max_steps):
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax(dim=1).item()

        state_np = new_state.squeeze(0).cpu().numpy()
        states.append(state_np)
        actions_trajectory.append(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = torch.FloatTensor(next_obs)
        state = new_state

        if done:
            break

    # Compute global min/max for consistent scale
    all_state_values = np.concatenate([s.flatten() for s in states])
    abs_max = np.abs(all_state_values).max() if len(all_state_values) > 0 else 1.0
    vmin, vmax = -abs_max, abs_max
    if is_mpn:
        print(f"M matrix scale: [{vmin:.3f}, {vmax:.3f}]")
    else:
        print(f"Hidden state scale: [{vmin:.3f}, {vmax:.3f}]")

    # Second pass: render with consistent scale and frame numbers
    print("Second pass: rendering frames...")
    frames = []
    total_reward = 0

    # Reset to same initial state if possible
    if seed is not None:
        reset_result = env.reset(seed=seed)
    else:
        reset_result = env.reset()

    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    for step in range(max_steps):
        # Render environment
        env_frame = env.render()

        if env_frame is None:
            print("Warning: Environment returned None for render(). Make sure render_mode='rgb_array'")
            break

        # Get state update and compute action (don't use pre-recorded trajectory)
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)

            # Compute action fresh (greedy since epsilon=0.0 by default)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax(dim=1).item()

        # Create state visualization with consistent scale
        state_np = new_state.squeeze(0).cpu().numpy()
        if is_mpn:
            # MPN: visualize M matrix as heatmap
            state_frame = create_m_matrix_frame(state_np, figsize=figsize, dpi=dpi, vmin=vmin, vmax=vmax)
        else:
            # RNN: visualize hidden state as bar plot
            state_frame = create_hidden_state_frame(state_np, figsize=figsize, dpi=dpi, vmin=vmin, vmax=vmax)

        # Combine frames
        combined_frame = combine_frames_horizontal(env_frame, state_frame)

        # Add frame number text
        from PIL import ImageDraw, ImageFont
        img = Image.fromarray(combined_frame)
        draw = ImageDraw.Draw(img)

        # Try to use a nice font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

        # Draw frame number with background for visibility
        text = f"Frame: {step}"
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Position in top-left corner with padding
        x, y = 10, 10
        # Draw semi-transparent background
        draw.rectangle([x-5, y-5, x+text_width+5, y+text_height+5], fill=(0, 0, 0, 180))
        # Draw text
        draw.text((x, y), text, fill=(255, 255, 255), font=font)

        combined_frame = np.array(img)
        frames.append(combined_frame)

        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        obs = torch.FloatTensor(next_obs)
        state = new_state
        total_reward += reward

        if done:
            break

    # Save GIF
    if frames:
        imageio.mimsave(save_path, frames, fps=fps)
        print(f"Saved GIF to {save_path} ({len(frames)} frames, reward={total_reward:.2f})")
    else:
        print("Warning: No frames to save")

    return total_reward


class PeriodicGIFRenderer:
    """
    Renders episodes to GIF periodically based on wall-clock time.

    Example:
        >>> renderer = PeriodicGIFRenderer(dqn, env, save_dir='videos', interval_mins=5)
        >>> for episode in range(1000):
        >>>     # ... training ...
        >>>     renderer.maybe_render(episode)
    """

    def __init__(
        self,
        dqn,
        env,
        save_dir: str,
        interval_mins: float = 5.0,
        max_steps: int = 500,
        fps: int = 30,
        epsilon: float = 0.0
    ):
        """
        Args:
            dqn: MPNDQN model
            env: Gym environment
            save_dir: Directory to save GIFs
            interval_mins: Minutes between GIF renders
            max_steps: Max steps per episode
            fps: FPS for GIF
            epsilon: Exploration rate
        """
        self.dqn = dqn
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.interval_mins = interval_mins
        self.max_steps = max_steps
        self.fps = fps
        self.epsilon = epsilon

        self.last_render_time = None
        self.render_count = 0

    def should_render(self) -> bool:
        """Check if enough time has passed since last render."""
        import time

        if self.last_render_time is None:
            return True

        elapsed_mins = (time.time() - self.last_render_time) / 60.0
        return elapsed_mins >= self.interval_mins

    def maybe_render(self, episode: int) -> Optional[float]:
        """
        Render episode if interval has elapsed.

        Args:
            episode: Current episode number

        Returns:
            Episode reward if rendered, None otherwise
        """
        import time

        if not self.should_render():
            return None

        save_path = self.save_dir / f"episode_{episode:05d}.gif"

        try:
            reward = render_episode_to_gif(
                self.dqn, self.env, str(save_path),
                max_steps=self.max_steps,
                fps=self.fps,
                epsilon=self.epsilon
            )
            self.last_render_time = time.time()
            self.render_count += 1
            return reward
        except Exception as e:
            print(f"Error rendering GIF: {e}")
            return None


def render_neurogym_episode(
    dqn,
    env,
    save_path: str,
    max_steps: int = 500,
    epsilon: float = 0.0,
    seed: Optional[int] = None,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 100
) -> Tuple[float, dict]:
    """
    Render a NeuroGym episode to a static plot with stacked observations and rewards.

    Creates a visualization with:
    - Stacked observations (each channel offset vertically) + rewards

    Args:
        dqn: Trained model (MPNDQN or RNNDQN)
        env: NeuroGym environment
        save_path: Path to save plot (PNG/PDF)
        max_steps: Maximum steps per episode
        epsilon: Exploration rate (0 = greedy)
        seed: Random seed for reproducibility (default: None)
        figsize: Figure size in inches
        dpi: DPI for rendering

    Returns:
        total_reward: Episode reward
        episode_data: Dict with observations, rewards, actions, states
    """
    # Set random seeds for reproducibility
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        # IMPORTANT: NeuroGym requires explicit env.seed() call before reset()
        env.seed(seed)
        print(f"Using seed: {seed}")

    # Detect model type
    test_state = dqn.init_state(batch_size=1)
    is_mpn = (len(test_state.shape) == 3)  # MPN: (batch, hidden, obs), RNN: (batch, hidden)

    print(f"Running episode with {'MPN' if is_mpn else 'RNN'}...")

    # Reset environment
    reset_result = env.reset()

    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result

    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    # Collect episode data
    observations = []
    rewards = []
    actions = []
    states = []
    total_reward = 0

    for step in range(max_steps):
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax(dim=1).item()

        # Store data
        observations.append(obs.cpu().numpy())
        actions.append(action)
        states.append(new_state.squeeze(0).cpu().numpy())

        # Take step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        rewards.append(reward)
        total_reward += reward

        obs = torch.FloatTensor(next_obs)
        state = new_state

        if done:
            break

    # Convert to arrays
    observations = np.array(observations)  # (T, obs_dim)
    rewards = np.array(rewards)  # (T,)
    actions = np.array(actions)  # (T,)
    states = np.array(states)  # (T, hidden_dim) or (T, hidden_dim, obs_dim)

    print(f"Episode finished: {len(observations)} steps, reward={total_reward:.2f}")

    # Create figure with single plot
    fig, ax_obs = plt.subplots(figsize=figsize, dpi=dpi)

    n_obs_channels = observations.shape[1]
    timesteps = np.arange(len(observations))

    # Plot observations with vertical offsets (like in the notebook)
    offset = 2.0
    for i in range(n_obs_channels):
        ax_obs.plot(timesteps, observations[:, i] + i * offset,
                   label=f'Obs {i}', linewidth=1.5, alpha=0.8)

    # Plot rewards on same axis (scaled and offset to be visible)
    reward_offset = n_obs_channels * offset + 2
    ax_obs.plot(timesteps, rewards + reward_offset,
               label='Reward', linewidth=2, color='red', linestyle='--')

    ax_obs.set_xlabel('Timestep', fontsize=12)
    ax_obs.set_ylabel('Stacked Observations + Reward', fontsize=12)
    ax_obs.set_title(f'NeuroGym Episode ({"MPN" if is_mpn else "RNN"}, Total Reward: {total_reward:.2f})',
                    fontsize=14, fontweight='bold')
    ax_obs.legend(loc='upper right', fontsize=9, ncol=2)
    ax_obs.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"Saved NeuroGym episode visualization to {save_path}")
    plt.close(fig)

    # Return data for potential further analysis
    episode_data = {
        'observations': observations,
        'rewards': rewards,
        'actions': actions,
        'states': states,
        'total_reward': total_reward
    }

    return total_reward, episode_data


if __name__ == "__main__":
    print("Testing render utilities...")

    # Test M matrix frame creation
    print("\n1. Testing M matrix frame creation...")
    M = np.random.randn(8, 4) * 0.5
    m_frame = create_m_matrix_frame(M)
    print(f"Created M matrix frame: {m_frame.shape}")

    # Test frame combination
    print("\n2. Testing frame combination...")
    frame1 = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
    frame2 = m_frame[:100, :150]  # Crop to match size
    combined = combine_frames_horizontal(frame1, frame2)
    print(f"Combined frame shape: {combined.shape}")

    # Test with actual CartPole
    import gymnasium as gym
    from mpn_dqn import MPNDQN

    print("\n3. Testing full GIF rendering with CartPole...")
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    dqn = MPNDQN(obs_dim=4, hidden_dim=8, action_dim=2)

    reward = render_episode_to_gif(
        dqn, env, 'test_render.gif',
        max_steps=100, fps=30
    )
    print(f"Rendered episode with reward: {reward:.2f}")
    print("Check 'test_render.gif' to see the result!")

    env.close()

    print("\nRender utilities test completed!")
