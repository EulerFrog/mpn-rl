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

    # Plot M matrix with consistent scale
    if vmin is None or vmax is None:
        abs_max = np.abs(M).max() if np.abs(M).max() > 0 else 1.0
        vmin = -abs_max if vmin is None else vmin
        vmax = abs_max if vmax is None else vmax
    im = ax.imshow(M, cmap='RdBu_r', aspect='auto', vmin=vmin, vmax=vmax)

    ax.set_xlabel('Input Dimension', fontsize=8)
    ax.set_ylabel('Hidden Dimension', fontsize=8)
    ax.set_title('M Matrix (Synaptic Modulation)', fontsize=10)
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
    Render an episode to GIF with CartPole + M matrix visualization.

    Args:
        dqn: Trained MPNDQN model
        env: Gym environment (must support rgb_array rendering)
        save_path: Path to save GIF
        max_steps: Maximum steps per episode
        fps: Frames per second for GIF
        epsilon: Exploration rate (0 = greedy)
        figsize: Size of M matrix plot
        dpi: DPI for M matrix plot

    Returns:
        total_reward: Episode reward
    """
    # First pass: collect all M matrices to compute consistent scale
    print("First pass: computing M matrix scale...")
    obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    m_matrices = []
    actions_trajectory = []

    for step in range(max_steps):
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)

            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = q_values.argmax(dim=1).item()

        M = new_state.squeeze(0).cpu().numpy()
        m_matrices.append(M)
        actions_trajectory.append(action)

        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result

        obs = torch.FloatTensor(next_obs)
        state = new_state

        if done:
            break

    # Compute global min/max for consistent scale
    all_m_values = np.concatenate([m.flatten() for m in m_matrices])
    abs_max = np.abs(all_m_values).max() if len(all_m_values) > 0 else 1.0
    vmin, vmax = -abs_max, abs_max
    print(f"M matrix scale: [{vmin:.3f}, {vmax:.3f}]")

    # Second pass: render with consistent scale and frame numbers
    print("Second pass: rendering frames...")
    frames = []
    total_reward = 0

    obs, _ = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    obs = torch.FloatTensor(obs)
    state = dqn.init_state(batch_size=1)

    for step in range(max_steps):
        # Render environment
        env_frame = env.render()

        if env_frame is None:
            print("Warning: Environment returned None for render(). Make sure render_mode='rgb_array'")
            break

        # Get action from trajectory (must match first pass)
        action = actions_trajectory[step] if step < len(actions_trajectory) else 0

        # Get state update
        with torch.no_grad():
            q_values, new_state = dqn(obs.unsqueeze(0), state)

        # Create M matrix visualization with consistent scale
        M = new_state.squeeze(0).cpu().numpy()
        m_frame = create_m_matrix_frame(M, figsize=figsize, dpi=dpi, vmin=vmin, vmax=vmax)

        # Combine frames
        combined_frame = combine_frames_horizontal(env_frame, m_frame)

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
        step_result = env.step(action)
        if len(step_result) == 5:
            next_obs, reward, terminated, truncated, _ = step_result
            done = terminated or truncated
        else:
            next_obs, reward, done, _ = step_result

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
