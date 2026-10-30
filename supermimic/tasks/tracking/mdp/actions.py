"""Joint PD targets, including the original per-episode joint offsets."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class TrackingAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(env.num_envs, 29, device=env.device)
        self._target = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return 29

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._target

    def process_actions(self, actions):
        self._raw.copy_(actions)
        ref = self._env.reference
        self._target = ref.default_q + ref.xi[:, 1:30] + ref.alpha * actions

    def apply_actions(self):
        self._asset.set_joint_position_target_index(target=self._target)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw[env_ids] = 0.0
        self._target[env_ids] = 0.0


@configclass
class TrackingActionCfg(ActionTermCfg):
    class_type: type = TrackingAction
    asset_name: str = "robot"
