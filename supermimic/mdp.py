"""Analytic MDP heads shared by the simulator and the world model (paper Appendix A).

State tuple ``S = (p, R, v, w, q, qd)``: pelvis position, rotation matrix, world linear/angular
velocity of the pelvis link frame, joint positions and velocities.
"""

from __future__ import annotations

from .tasks.tracking.reward_cfg import RewardCfg as RewardCfg
from .tasks.tracking.observation_cfg import ObservationCfg

import torch

from .kinematics import G1Kinematics
from .motion import CONTROL_DT, MotionLib
from .rotations import mat_to_6d, quat_to_mat, mat_to_quat, rot_angle_sq, yaw_mat

STATE_DIM = 71
OBS_DIM = 29 + 29 + 3 + 3 + 29
XI_DIM = 33
NUM_J = 29

# Table A.1 sensor noise (uniform half-widths).
OBS_NOISE = torch.cat(
    (
        torch.full((29,), 0.01),
        torch.full((29,), 0.5),
        torch.full((3,), 0.2),
        torch.full((3,), 0.05),
        torch.zeros(29),
    )
)
# normalization of the obs before the policy (qd and angular velocity are large)
OBS_SCALE = torch.cat(
    (
        torch.ones(29),
        torch.full((29,), 0.1),
        torch.full((3,), 0.25),
        torch.ones(3),
        torch.ones(29),
    )
)
MAX_STEP_REWARD = (4 * 1.0 + 2 * 0.5) * CONTROL_DT


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return (
        x[..., 0:3],
        quat_to_mat(x[..., 3:7]),
        x[..., 7:10],
        x[..., 10:13],
        x[..., 13:42],
        x[..., 42:71],
    )


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: height, tilt (6D), heading linear velocity, body angular velocity, joints."""
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (
            p[..., 2:3],
            mat_to_6d(H.transpose(-1, -2) @ R),
            (H.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1),
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            q,
            qd,
        ),
        -1,
    )


INV_DIM = 1 + 6 + 3 + 3 + 29 + 29


class MDP:
    def __init__(
        self,
        kin: G1Kinematics,
        motion: MotionLib,
        default_q: torch.Tensor,
        cfg: RewardCfg | None = None,
        observation_cfg: ObservationCfg | None = None,
        xi_dim: int = XI_DIM,
    ):
        cfg = RewardCfg() if cfg is None else cfg
        cfg.validate()
        self.observation_cfg = (
            ObservationCfg() if observation_cfg is None else observation_cfg
        )
        self.observation_cfg.validate()
        if xi_dim not in (33, 91):
            raise ValueError(
                "xi_dim must be 33, or 91 when joint PD randomization is enabled"
            )
        self.kin, self.motion, self.cfg = kin, motion, cfg
        self.xi_dim = xi_dim
        self.critic_extra = xi_dim + 4
        self.legacy_inputs = False
        self.motion.default_q = default_q
        self.anchor_index = [kin.body_names[i] for i in kin.tracked_idx].index("torso_link")
        self.max_step_reward = cfg.max_reward_rate * CONTROL_DT
        self.default_q = default_q
        mid, half = (kin.q_lo + kin.q_hi) / 2, (kin.q_hi - kin.q_lo) / 2
        self.soft_lo, self.soft_hi = (
            mid - half * cfg.soft_limit_factor,
            mid + half * cfg.soft_limit_factor,
        )
        dev = default_q.device
        selected = [
            (i, width, term)
            for i, (_, width, term) in enumerate(self.observation_cfg.terms())
            if term is not None and term.enabled
        ]
        self._obs_selection = tuple(i for i, _, _ in selected)
        self.obs_dim = self.observation_cfg.dim
        self.obs_widths = tuple(width for _, width, _ in selected)
        self.obs_scale = torch.cat(
            [torch.full((width,), term.scale) for _, width, term in selected]
        ).to(dev)
        self.obs_noise = torch.cat(
            [
                torch.full((width,), term.noise * abs(term.scale))
                for _, width, term in selected
            ]
        ).to(dev)

    # ---------------- observation ----------------
    def obs(self, S, a_prev, xi, f) -> torch.Tensor:
        """Paper proprioception; reference orientation is legacy-checkpoint opt-in."""
        p, R, v, w, q, qd = S
        Rt = R.transpose(-1, -2)
        dq0 = xi[..., 1:30]
        g_b = -R[..., 2, :]  # R^T (0,0,-1)
        values = (
            q - self.default_q + (-dq0 if self.legacy_inputs else dq0),
            qd,
            (Rt @ w.unsqueeze(-1)).squeeze(-1),
            g_b,
            a_prev,
        )
        if self.observation_cfg.reference_orientation is not None and self.observation_cfg.reference_orientation.enabled:
            values += (mat_to_6d(Rt @ self.motion.root_R[f]),)
        return (
            torch.cat(tuple(values[i] for i in self._obs_selection), -1)
            * self.obs_scale
        )

    def add_noise(self, o: torch.Tensor) -> torch.Tensor:
        if not self.observation_cfg.noise_enabled:
            return o
        return o + (2 * torch.rand_like(o) - 1) * self.obs_noise

    # ---------------- reward / termination ----------------
    def reward(self, S, f, a, a_prev, terms: bool = False):
        """Per-step reward, including dt, for differentiable imagined rollouts."""
        result = self.reward_rate(S, f, a, a_prev, terms=terms)
        if terms:
            rate, diagnostics = result
            return rate * CONTROL_DT, diagnostics
        return result * CONTROL_DT

    def reward_rate(self, S, f, a, a_prev, terms: bool = False):
        """Reward before time integration, for Isaac Lab's RewardManager."""
        c, m = self.cfg, self.motion
        p, R, v, w, q, qd = S
        bp, bR, bv, bw = self.kin.forward(p, R, v, w, q, qd)
        rp, rR = m.root_p[f], m.root_R[f]
        # reference bodies re-anchored to the robot's root (xy + yaw), as in BeyondMimic
        dR = yaw_mat(R @ rR.transpose(-1, -2))
        anchor = torch.cat((p[..., :2], rp[..., 2:3]), -1)
        ref_bp = anchor.unsqueeze(-2) + (
            (dR.unsqueeze(-3) @ (m.body_p[f] - rp.unsqueeze(-2)).unsqueeze(-1)).squeeze(
                -1
            )
        )
        ref_bR = dR.unsqueeze(-3) @ m.body_R[f]
        e_pos = ((ref_bp - bp) ** 2).sum(-1).mean(-1)
        e_ori = rot_angle_sq(ref_bR, bR).mean(-1)
        e_lv = ((m.body_v[f] - bv) ** 2).sum(-1).mean(-1)
        e_av = ((m.body_w[f] - bw) ** 2).sum(-1).mean(-1)
        e_rp = ((rp - p) ** 2).sum(-1)
        e_ro = rot_angle_sq(rR, R)
        r_track = (
            c.w_body_pos * torch.exp(-e_pos / c.body_pos_std**2)
            + c.w_body_ori * torch.exp(-e_ori / c.body_ori_std**2)
            + c.w_body_linvel * torch.exp(-e_lv / c.body_linvel_std**2)
            + c.w_body_angvel * torch.exp(-e_av / c.body_angvel_std**2)
            + c.w_root_pos * torch.exp(-e_rp / c.root_pos_std**2)
            + c.w_root_ori * torch.exp(-e_ro / c.root_ori_std**2)
        )
        r_rate = ((a - a_prev) ** 2).sum(-1)
        r_lim = (torch.relu(q - self.soft_hi) + torch.relu(self.soft_lo - q)).sum(-1)
        r = r_track + c.w_action_rate * r_rate + c.w_joint_limit * r_lim
        if terms:
            return r, {
                "track": r_track * CONTROL_DT,
                "e_body_pos": e_pos,
                "e_body_ori": e_ori,
                "e_root_pos": e_rp,
                "action_rate": r_rate,
                "joint_limit": r_lim,
            }
        return r

    def termination(self, S, f) -> torch.Tensor:
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        return (e_z > self.cfg.term_height) | (e_tilt > self.cfg.term_tilt)

    def soft_termination(
        self, S, f, tz: float | None = None, tt: float | None = None
    ) -> torch.Tensor:
        """Differentiable termination probability for imagined rollouts."""
        tz = self.cfg.soft_term_height_temperature if tz is None else tz
        tt = self.cfg.soft_term_tilt_temperature if tt is None else tt
        p, R = S[0], S[1]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        e_z = (rp[..., 2] - p[..., 2]).abs()
        e_tilt = (rR[..., 2, 2] - R[..., 2, 2]).abs()
        alive = (1 - torch.sigmoid((e_z - self.cfg.term_height) / tz)) * (
            1 - torch.sigmoid((e_tilt - self.cfg.term_tilt) / tt)
        )
        return 1 - alive

    def reference_orientation(self, S, f) -> torch.Tensor:
        """Legacy policy playback: reference rotation in the robot frame."""
        return mat_to_6d(S[1].transpose(-1, -2) @ self.motion.root_R[f])

    def flatten_history(self, hist):
        """[B,H,D] -> [B,H*D], term-major (old checkpoints remain frame-major)."""
        if self.legacy_inputs:
            return hist.flatten(-2)
        return torch.cat([part.flatten(-2) for part in hist.split(self.obs_widths, -1)], -1)

    def anchor_observation(self, S, f):
        """Reference torso pose in the current robot torso frame (3+6)."""
        bp, bR, _, _ = self.kin.forward(*S)
        i = self.anchor_index
        Rt = bR[..., i, :, :].transpose(-1, -2)
        dp = (Rt @ (self.motion.body_p[f][..., i, :]-bp[..., i, :]).unsqueeze(-1)).squeeze(-1)
        ori = mat_to_6d(Rt @ self.motion.body_R[f][..., i, :, :])
        return torch.cat((dp, ori), -1)

    def goal(self, S, f, K, anchor_obs=True, noise=False):
        g = self.motion.goal(f, K)
        if not anchor_obs:
            return g
        anchor = self.anchor_observation(S, f)
        if noise and self.observation_cfg.noise_enabled:
            scale = anchor.new_tensor([0.25]*3 + [0.05]*6)
            anchor = anchor + (2*torch.rand_like(anchor)-1)*scale
        return torch.cat((g, anchor), -1)

    def termination_features(self, S, f):
        # Match reproduced signed height and projected-gravity-z errors.
        height = self.motion.root_p[f][..., 2] - S[0][..., 2]
        tilt = S[1][..., 2, 2] - self.motion.root_R[f][..., 2, 2]
        return torch.stack((height, tilt), -1)

    def root_error_features(self, S, f):
        p, R = S[:2]
        rp, rR = self.motion.root_p[f], self.motion.root_R[f]
        dp = (yaw_mat(rR).transpose(-1,-2) @ (p-rp).unsqueeze(-1)).squeeze(-1)
        dyaw = torch.atan2(R[...,1,0],R[...,0,0])-torch.atan2(rR[...,1,0],rR[...,0,0])
        dyaw = torch.remainder(dyaw+torch.pi, 2*torch.pi)-torch.pi
        return torch.cat((dp, dyaw.unsqueeze(-1)), -1)

    def critic_features(self, S, goal, xi, f) -> torch.Tensor:
        return torch.cat((invariant_features(S), goal, xi, self.root_error_features(S, f)), -1)


CRITIC_EXTRA = XI_DIM + 4


def normalize_xi(xi: torch.Tensor) -> torch.Tensor:
    base = torch.cat(
        ((xi[..., :1] - 0.75) / 0.45, xi[..., 1:30] / 0.01, xi[..., 30:33] / 0.05), -1
    )
    if xi.shape[-1] == XI_DIM:
        return base
    if xi.shape[-1] != XI_DIM + 58:
        raise ValueError("sim context must have 33 or 91 entries")
    return torch.cat((base, (xi[..., XI_DIM:] - 1.0) / 0.5), -1)
