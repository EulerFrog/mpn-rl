"""
Custom TorchRL transform to extract NeuroGym GoNogo info (gt, new_trial) into tensordict.
"""

import torch
from torchrl.envs.transforms import Transform
from tensordict import TensorDictBase


class NeuroGymInfoTransform(Transform):
    """
    Transform that extracts ground truth and new_trial info from GoNogo environment
    and adds them to the tensordict.
    """

    def __init__(self):
        super().__init__(in_keys=[], out_keys=["gt", "new_trial"])

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Called during forward pass (doesn't modify, just passes through)."""
        return tensordict

    def _step(self, tensordict: TensorDictBase, next_tensordict: TensorDictBase) -> TensorDictBase:
        """
        Called after env.step() to extract info from the underlying GoNogo environment.

        Note: info['gt'] represents the ground truth for the COMPLETED trial when new_trial=True.
        """
        # Access the underlying gym environment
        env = self.parent
        while hasattr(env, 'base_env'):
            env = env.base_env

        # Get the wrapped gym environment (should have NeuroGymInfoWrapper)
        gym_env = env._env

        # Get gt and new_trial from the captured info dict
        # info['gt'] is for the current/completed trial, not the next one
        last_info = gym_env._last_info
        gt = int(last_info.get('gt', 0))
        new_trial = bool(last_info.get('new_trial', False))

        # Add to tensordict
        # Both gt and new_trial come from the same step's info, so both go in next_tensordict
        next_tensordict.set("gt", torch.tensor(gt, dtype=torch.long, device=next_tensordict.device))
        next_tensordict.set("new_trial", torch.tensor(new_trial, dtype=torch.bool, device=next_tensordict.device))

        return next_tensordict

    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        """Called after env.reset()."""
        # Access the underlying gym environment
        env = self.parent
        while hasattr(env, 'base_env'):
            env = env.base_env

        # Get the wrapped gym environment
        gym_env = env._env

        # Get gt from the captured info dict (or from unwrapped.trial as fallback)
        if hasattr(gym_env, '_last_info') and 'gt' in gym_env._last_info:
            gt = int(gym_env._last_info.get('gt', 0))
        else:
            # Fallback to trial dict
            gt = int(gym_env.unwrapped.trial['ground_truth'])

        # Add to tensordict
        tensordict_reset.set("gt", torch.tensor(gt, dtype=torch.long, device=tensordict_reset.device))

        return tensordict_reset
