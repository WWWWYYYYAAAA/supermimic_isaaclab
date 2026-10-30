"""Reference-motion command, episode sampling and exact reset state distribution."""

from __future__ import annotations
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from supermimic.mdp import MAX_STEP_REWARD
from supermimic.robot_spec import G1_JOINT_NAMES
from ..control_cfg import PDControlCfg, joint_pd_parameters
from ..randomization_cfg import RandomizationCfg
from .events import _sample_vector, reset_context
from supermimic.rotations import mat_to_quat, quat_to_mat, so3_exp


class ReferenceMotionCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.sim, self.scene = env.sim, env.scene
        self.robot = env.scene["robot"]
        assert list(self.robot.joint_names) == G1_JOINT_NAMES, self.robot.joint_names
        self.N = env.num_envs
        self.motion, self.mdp, self.gamma = env.motion, env.mdp, env.gamma
        self.eval_mode = env.eval_mode
        self.max_episode_steps = env.episode_steps if not env.eval_mode else 10**9
        self.control_cfg = getattr(env.cfg, "control", PDControlCfg())
        self.randomization_cfg = getattr(env.cfg, "randomization", RandomizationCfg())
        self.randomization_cfg.validate()
        parameters = joint_pd_parameters(self.control_cfg)
        self.reset_noise = (
            env.reset_noise
            and self.randomization_cfg.reset.enabled
            and not env.eval_mode
        )
        dev = self.device
        self.default_q = self.robot.data.default_joint_pos.torch[0].clone()
        self.alpha = torch.tensor(
            [parameters[n][2] for n in G1_JOINT_NAMES], device=dev
        )
        self.nominal_kp = torch.tensor(
            [parameters[n][0] for n in G1_JOINT_NAMES], device=dev
        )
        self.nominal_kd = torch.tensor(
            [parameters[n][1] for n in G1_JOINT_NAMES], device=dev
        )
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.xi = torch.zeros(self.N, self.randomization_cfg.context_dim, device=dev)
        if self.randomization_cfg.pd_gains.enabled:
            self.xi[:, 33:] = 1.0
        self.f = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_step = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.a_prev = torch.zeros(self.N, 29, device=dev)
        self.start_bin = torch.zeros(self.N, dtype=torch.long, device=dev)
        self.ep_disc_ret = torch.zeros(self.N, device=dev)
        self.ep_ret = torch.zeros(self.N, device=dev)
        self.G_max = getattr(self.mdp, "max_step_reward", MAX_STEP_REWARD) / (
            1 - self.gamma
        )
        self.bin_G = torch.full((self.motion.num_bins,), 0.0, device=dev)
        self.bin_w_avg = torch.full(
            (self.motion.num_bins,), 1.0 / self.motion.num_bins, device=dev
        )
        self.done_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)
        self.failed_eval = torch.zeros(self.N, dtype=torch.bool, device=dev)

    @property
    def command(self):
        return self.motion.goal(self.f, self.cfg.goal_frames)

    def reset(self, env_ids=None):
        # Reset sampling belongs to the reset event, once per episode. Do not
        # call CommandTerm.reset(): its timer draws would change the RNG stream.
        return {}

    def compute(self, dt):
        # Called by ManagerBasedRLEnv after physics/rewards and resets. Finish
        # transitions without resets here; _reset_idx handles terminal ones.
        self._env.finish_transition()

    def _update_metrics(self):
        pass

    def _update_command(self):
        pass

    def _resample_command(self, env_ids):
        self.reset_reference(env_ids)

    def state(self):
        d = self.robot.data
        pose = d.root_link_pose_w.torch.clone()
        vel = d.root_link_vel_w.torch.clone()
        p = pose[:, :3] - self.scene.env_origins
        quat = pose[:, 3:7]
        R = quat_to_mat(torch.cat((quat[:, 3:4], quat[:, :3]), -1))
        return (
            p,
            R,
            vel[:, :3],
            vel[:, 3:6],
            d.joint_pos.torch.clone(),
            d.joint_vel.torch.clone(),
        )

    def sample_frames(self, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.motion
        soft = torch.softmax(-25.0 * self.bin_G / self.G_max, 0)
        self.bin_w_avg.mul_(0.999).add_(0.001 * soft)
        probs = (
            0.5 / m.num_bins + 0.4 * soft + 0.1 * self.bin_w_avg / self.bin_w_avg.sum()
        )
        bins = torch.multinomial(probs, n, replacement=True)
        start = m.bin_start[bins]
        end = torch.minimum(start + 50, m.clip_end[m.bin_clip[bins]])
        f = start + (torch.rand(n, device=self.device) * (end - start).float()).long()
        return (f, bins)

    def reset_reference(
        self, env_ids: torch.Tensor, frames: torch.Tensor | None = None
    ):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        if frames is None:
            (frames, bins) = self.sample_frames(n)
        else:
            bins = self.motion.frame_bin[frames]
        m = self.motion
        (p, R, v, w, q, qd) = (x.clone() for x in m.ref_state(frames))
        if self.reset_noise:
            reset = self.randomization_cfg.reset
            p = p + _sample_vector(n, reset.root_position, dev)
            rv = _sample_vector(n, reset.root_rotation, dev)
            R = so3_exp(rv) @ R
            v = v + _sample_vector(n, reset.root_linear_velocity, dev)
            w = w + _sample_vector(n, reset.root_angular_velocity, dev)
            q = q + torch.empty_like(q).uniform_(*reset.joint_position)
        q = torch.clamp(q, self.mdp.kin.q_lo, self.mdp.kin.q_hi)
        quat = mat_to_quat(R)
        pose = torch.cat(
            (p + self.scene.env_origins[env_ids], quat[:, 1:4], quat[:, 0:1]), -1
        )
        self.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(
            root_velocity=torch.cat((v, w), -1), env_ids=env_ids
        )
        self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=qd, env_ids=env_ids)
        self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)
        self.f[env_ids] = frames
        self.ep_step[env_ids] = 0
        self.a_prev[env_ids] = 0.0
        self.start_bin[env_ids] = bins
        self.ep_disc_ret[env_ids] = 0.0
        self.ep_ret[env_ids] = 0.0
        reset_context(self, env_ids)
        self.scene.reset(env_ids)

    def refresh(self):
        """Propagate written states so that ``state()`` is consistent after a reset."""
        self.sim.forward() if hasattr(self.sim, "forward") else None
        self.scene.update(0.0)


@configclass
class ReferenceMotionCommandCfg(CommandTermCfg):
    class_type: type = ReferenceMotionCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    goal_frames: int = 10
