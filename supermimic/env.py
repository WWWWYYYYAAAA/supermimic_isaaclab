"""Compatibility adapter from SuperMimic transitions to ManagerBasedRLEnv.

Training keeps its replay-based world-model algorithm. The underlying environment
returns Isaac Lab's standard five-tuple; this adapter exposes its raw transition.
"""

from copy import deepcopy
from .tasks.config import TrackingEnvCfg, TrackingSceneCfg, make_sim
from .tasks.environment import TrackingEnv


__all__ = [
    "TrackingSim",
    "TrackingEnv",
    "TrackingEnvCfg",
    "TrackingSceneCfg",
    "make_sim",
]


class TrackingSim:
    def __init__(
        self,
        sim_cfg,
        num_envs,
        motion,
        mdp,
        gamma=0.99,
        eval_mode=False,
        max_episode_steps=500,
        reset_noise=True,
        goal_frames=10,
        env_cfg=None,
    ):
        cfg = TrackingEnvCfg() if env_cfg is None else deepcopy(env_cfg)
        cfg.prepare()
        cfg.sim = sim_cfg
        cfg.scene.num_envs = num_envs
        cfg.episode_length_s = max_episode_steps * sim_cfg.dt
        cfg.commands.motion.goal_frames = goal_frames
        self.env = TrackingEnv(
            cfg, motion, mdp, gamma, eval_mode, max_episode_steps, reset_noise
        )

    def __getattr__(self, name):
        env = self.__dict__["env"]
        if hasattr(env, name):
            return getattr(env, name)
        return getattr(env.reference, name)

    @property
    def f(self):
        return self.env.reference.f

    @f.setter
    def f(self, frames):
        self.env.reference.f = frames

    def state(self):
        return self.env.reference.state()

    def step(self, action):
        return self.env.step(action)[4]["transition"]

    def reset(self, env_ids, frames=None):
        self.env.reset_reference(env_ids, frames)

    def close(self):
        self.env.close()
