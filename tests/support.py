"""Deterministic CPU physics fixture; Isaac Lab managers themselves are real."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile
import numpy as np
import torch

from supermimic.kinematics import G1Kinematics
from supermimic.motion import MotionLib
from supermimic.mdp import MDP
from supermimic.robot_spec import G1_JOINT_NAMES, default_joint_pos_list


def make_motion():
    kin = G1Kinematics("cpu")
    with tempfile.TemporaryDirectory() as tmp:
        for index, length in enumerate((100, 80)):
            data = np.zeros((length, 36), dtype=np.float32)
            data[:, 0] = np.arange(length) * 0.002
            data[:, 2] = 0.76
            data[:, 6] = 1.0  # CSV quaternion xyzw.
            data[:, 7:] = np.asarray(default_joint_pos_list())
            data[:, 7] += np.sin(np.arange(length) * 0.02) * 0.05
            np.savetxt(Path(tmp) / f"clip{index}.csv", data, delimiter=",")
        motion = MotionLib.from_dir(tmp, kin, "cpu")
    mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
    return kin, motion, mdp


class Robot:
    def __init__(self, n):
        self.joint_names = G1_JOINT_NAMES
        self.data = NS()
        for name, shape in [
            ("root_link_pose_w", (n, 7)),
            ("root_link_vel_w", (n, 6)),
            ("joint_pos", (n, 29)),
            ("joint_vel", (n, 29)),
            ("body_com_pose_b", (n, 1, 7)),
        ]:
            setattr(self.data, name, NS(torch=torch.zeros(shape)))
        self.data.root_link_pose_w.torch[:, 2] = 0.76
        self.data.root_link_pose_w.torch[:, 6] = 1.0
        self.data.default_joint_pos = NS(
            torch=torch.tensor(default_joint_pos_list()).repeat(n, 1)
        )
        self.target = self.data.default_joint_pos.torch.clone()
        self.force_failure = False

    def find_bodies(self, name):
        return [0], [name]

    def set_coms_index(self, coms, body_ids, env_ids=None):
        rows = torch.arange(self.target.shape[0]) if env_ids is None else env_ids
        self.data.body_com_pose_b.torch[
            rows[:, None], torch.tensor(body_ids)[None, :], :3
        ] = coms

    def write_joint_stiffness_to_sim_index(self, stiffness, env_ids=None):
        if not hasattr(self, "stiffness"):
            self.stiffness = torch.zeros_like(self.target)
        self.stiffness[slice(None) if env_ids is None else env_ids] = stiffness

    def write_joint_damping_to_sim_index(self, damping, env_ids=None):
        if not hasattr(self, "damping"):
            self.damping = torch.zeros_like(self.target)
        self.damping[slice(None) if env_ids is None else env_ids] = damping

    def write_root_link_pose_to_sim_index(self, root_pose, env_ids):
        self.data.root_link_pose_w.torch[env_ids] = root_pose

    def write_root_link_velocity_to_sim_index(self, root_velocity, env_ids):
        self.data.root_link_vel_w.torch[env_ids] = root_velocity

    def write_joint_position_to_sim_index(self, position, env_ids):
        self.data.joint_pos.torch[env_ids] = position

    def write_joint_velocity_to_sim_index(self, velocity, env_ids):
        self.data.joint_vel.torch[env_ids] = velocity

    def set_joint_position_target_index(self, target, env_ids=None):
        self.target[slice(None) if env_ids is None else env_ids] = target

    def step(self):
        q = self.data.joint_pos.torch
        self.data.joint_vel.torch[:] = (self.target - q) * 5
        q.add_(self.data.joint_vel.torch * 0.02)
        self.data.root_link_pose_w.torch[:, :3] += (
            self.data.root_link_vel_w.torch[:, :3] * 0.02
        )
        if self.force_failure:
            self.data.root_link_pose_w.torch[0, 2] = 0.0


class Scene(dict):
    def __init__(self, n):
        super().__init__(robot=Robot(n))
        self.num_envs = n
        self.env_origins = torch.zeros(n, 3)
        self.sensors = {}

    def reset(self, env_ids):
        pass

    def write_data_to_sim(self):
        pass

    def update(self, dt):
        pass


def manager_env(
    motion,
    mdp,
    *,
    eval_mode=False,
    max_episode_steps=500,
    reset_noise=True,
    n=2,
    env_cfg=None,
):
    from supermimic.tasks.environment import TrackingEnv
    from supermimic.tasks.config import TrackingEnvCfg, make_sim
    from isaaclab.managers import EventManager

    env = TrackingEnv.__new__(TrackingEnv)
    env._is_closed = True  # Fixture owns no physics resources.
    env.cfg = TrackingEnvCfg() if env_cfg is None else deepcopy(env_cfg).prepare()
    env.cfg.sim = make_sim("cpu", None)
    env.cfg.scene.num_envs = n
    env.cfg.episode_length_s = max_episode_steps * 0.02
    env.scene = Scene(n)
    env.sim = NS(
        device="cpu",
        is_playing=lambda: True,
        is_rendering=False,
        consume_reset_request=lambda: False,
        vis_marker_registry=NS(clear_debug_vis_callback=lambda obj: None),
        step=lambda render=False: env.scene["robot"].step(),
    )
    env.motion, env.mdp, env.gamma = motion, mdp, 0.99
    env.eval_mode, env.episode_steps, env.reset_noise = (
        eval_mode,
        max_episode_steps,
        reset_noise,
    )
    env._reset_frames = env._pending = env._next_state = None
    env._inside_step = False
    env._sim_step_counter = env.common_step_counter = 0
    env._physics_handles_decimation = False
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env.reset_buf = torch.zeros(n, dtype=torch.bool)
    env.render_enabled = False
    env.has_rtx_sensors = False
    env.extras = {}
    env.video_recorders = []
    env.event_manager = EventManager(env.cfg.events, env)

    # Only the backend-specific material write is replaced; startup random draws
    # and every manager's computation/reset run unchanged.
    def setup_friction(command):
        command._mu_view = torch.zeros(n, 1)

    def apply_friction(command):
        command._mu_view[:] = command.xi[:, :1]

    with (
        patch("supermimic.tasks.events._setup_friction", setup_friction),
        patch("supermimic.tasks.events._apply_friction", apply_friction),
    ):
        env.load_managers()
    env.reset_reference(torch.arange(n))
    return env
