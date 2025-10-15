"""
NeuroGym environment wrapper for Gymnasium compatibility.

Provides:
- Flattening of structured observations (dict → vector)
- Trial boundary tracking via info['new_trial']
- Gymnasium-compatible interface for NeuroGym environments
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class NeuroGymWrapper(gym.Wrapper):
    """
    Wrapper for NeuroGym environments to make them compatible with Gymnasium.

    Features:
    - Flattens structured observations into vectors
    - Tracks trial boundaries via info['new_trial']
    - Converts NeuroGym spaces to Gymnasium spaces if needed
    """

    def __init__(self, env, flatten_obs=True):
        """
        Args:
            env: NeuroGym environment
            flatten_obs: Whether to flatten observations (dict → vector)
        """
        super().__init__(env)
        self.flatten_obs = flatten_obs

        # Store original observation space
        self.original_observation_space = env.observation_space

        # If flattening, create flattened observation space
        if self.flatten_obs:
            self.observation_space = self._create_flattened_obs_space()
        else:
            self.observation_space = env.observation_space

        # Action space should already be compatible
        self.action_space = env.action_space

    def _create_flattened_obs_space(self):
        """
        Create flattened observation space from structured space.

        Returns:
            gym.spaces.Box: Flattened observation space
        """
        # NeuroGym uses Box spaces that are already vectors
        if isinstance(self.original_observation_space, gym.spaces.Box):
            # Already a vector space
            return self.original_observation_space
        elif isinstance(self.original_observation_space, gym.spaces.Dict):
            # Flatten dict space
            total_dim = 0
            for key, space in self.original_observation_space.spaces.items():
                if isinstance(space, gym.spaces.Box):
                    total_dim += np.prod(space.shape)
                else:
                    raise ValueError(f"Unsupported space type for key '{key}': {type(space)}")

            return gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(total_dim,),
                dtype=np.float32
            )
        else:
            raise ValueError(f"Unsupported observation space type: {type(self.original_observation_space)}")

    def _flatten(self, obs):
        """
        Flatten observation if needed.

        Args:
            obs: Observation from environment (may be dict or array)

        Returns:
            np.ndarray: Flattened observation vector
        """
        if not self.flatten_obs:
            return obs

        if isinstance(obs, dict):
            # Flatten dict observation
            flattened = []
            for key in sorted(obs.keys()):
                value = obs[key]
                if isinstance(value, np.ndarray):
                    flattened.append(value.flatten())
                else:
                    flattened.append(np.array([value]))
            return np.concatenate(flattened).astype(np.float32)
        elif isinstance(obs, np.ndarray):
            # Already a vector
            return obs.flatten().astype(np.float32)
        else:
            raise ValueError(f"Unsupported observation type: {type(obs)}")

    def _extract_trial_info(self, info):
        """
        Extract trial boundary information from info dict.

        Args:
            info: Info dict from environment

        Returns:
            bool: Whether a new trial is starting
        """
        return info.get('new_trial', False)

    def reset(self, **kwargs):
        """
        Reset environment.

        Returns:
            obs: Flattened observation
            info: Info dict with trial information
        """
        result = self.env.reset(**kwargs)

        # Handle both (obs,) and (obs, info) return formats
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        obs = self._flatten(obs)

        # Add trial info (first timestep is always start of new trial)
        info['new_trial'] = False  # Reset doesn't end a trial, it starts one
        info['trial_start'] = True

        return obs, info

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            obs: Flattened observation
            reward: Reward
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict with trial information
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._flatten(obs)

        # Extract trial boundary info
        new_trial = self._extract_trial_info(info)
        info['new_trial'] = new_trial

        return obs, reward, terminated, truncated, info

    def __repr__(self):
        return f"NeuroGymWrapper({self.env})"


def make_neurogym_env(env_name, **kwargs):
    """
    Convenience function to create a wrapped NeuroGym environment.

    Args:
        env_name: Name of NeuroGym environment (e.g., 'ContextDecisionMaking-v0')
        **kwargs: Additional arguments to pass to neurogym.make()

    Returns:
        NeuroGymWrapper: Wrapped environment

    Examples:
        >>> env = make_neurogym_env('ContextDecisionMaking-v0')
        >>> env = make_neurogym_env('DelayMatchSample-v0', dt=100)
    """
    try:
        import neurogym as ngym
    except ImportError:
        raise ImportError("neurogym is not installed. Install with: pip install neurogym")

    # Create NeuroGym environment
    ngym_env = ngym.make(env_name, **kwargs)

    # Wrap with our wrapper
    wrapped_env = NeuroGymWrapper(ngym_env, flatten_obs=True)

    return wrapped_env


if __name__ == "__main__":
    # Test the wrapper
    print("Testing NeuroGymWrapper...")

    try:
        import neurogym as ngym

        print("\nCreating ContextDecisionMaking-v0...")
        env = make_neurogym_env('ContextDecisionMaking-v0')

        print(f"Environment: {env}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")

        print("\nTesting reset...")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        print(f"Observation: {obs}")
        print(f"Info: {info}")

        print("\nTesting step...")
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {i+1}: action={action}, reward={reward:.2f}, "
                  f"new_trial={info.get('new_trial', False)}, "
                  f"terminated={terminated}, truncated={truncated}")

            if info.get('new_trial', False):
                print("  → Trial boundary detected!")
                obs, info = env.reset()

            if terminated or truncated:
                print("  → Episode ended!")
                break

        env.close()
        print("\n✓ NeuroGymWrapper test completed successfully!")

    except ImportError as e:
        print(f"\n✗ Test failed: {e}")
        print("Install neurogym to run tests: pip install neurogym")
