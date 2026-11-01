"""Reproduced-compatible features and dynamics, with empirical input normalization."""
from __future__ import annotations
import torch
import torch.nn as nn
from .mdp import STATE_DIM, NUM_J, XI_DIM, invariant_features
from .motion import CONTROL_DT
from .normalization import EmpiricalNormalization
from .rotations import so3_exp, so3_log, yaw_mat


def mlp(inp, hidden, out, act=nn.ELU, layer_norm=False):
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


WM_OUT = 70
MODEL_INPUT_VERSION = 7
AMP_DTYPE: torch.dtype | None = None


def run_mlp(net, x):
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


def _mv(R, x):
    return (R @ x.unsqueeze(-1)).squeeze(-1)


class TerminationHead(nn.Module):
    """Signed height/tilt error and their magnitudes -> terminal logit."""
    def __init__(self, hidden=(64, 64), height_thr=0.25, tilt_thr=0.8):
        super().__init__()
        self.register_buffer("scale", torch.tensor([1 / height_thr, 1 / tilt_thr]))
        self.norm = EmpiricalNormalization(4)
        self.net = mlp(4, list(hidden), 1, act=nn.SiLU)

    def inputs(self, feats):
        x = feats * self.scale
        return torch.cat((x, x.abs()), -1)

    def forward(self, feats):
        return run_mlp(self.net, self.norm(self.inputs(feats))).squeeze(-1)


class WorldModel(nn.Module):
    """Heading-invariant dynamics; public state keeps world angular velocity.

    Output: dv_heading(3), dw_body(3), dqd(29), dp_heading(3),
    dtheta_body(3), dq(29). Position corrections follow explicit integration.
    """
    def __init__(self, hidden=(2048, 2048, 2048), xi_dim=XI_DIM,
                 termination_head="neural", height_thr=0.25, tilt_thr=0.8):
        super().__init__()
        if termination_head not in ("neural", "analytic"):
            raise ValueError("termination_head must be neural or analytic")
        self.termination_head = termination_head
        self.in_dim = STATE_DIM + NUM_J + xi_dim
        self.net = mlp(self.in_dim, list(hidden), WM_OUT, act=nn.SiLU)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = EmpiricalNormalization(self.in_dim)
        self.register_buffer("out_scale", torch.cat([torch.full((n,), s) for n, s in
            zip((3, 3, 29, 3, 3, 29), (0.3, 0.5, 2.0, 0.01, 0.02, 0.02))]))
        self.register_buffer("error_scale", torch.cat([torch.full((n,), s) for n, s in
            zip((3, 3, 3, 3, 29, 29), (0.05, 0.1, 0.5, 1.0, 0.1, 2.0))]))
        self.term_head = (TerminationHead(height_thr=height_thr, tilt_thr=tilt_thr)
                          if termination_head == "neural" else None)

    def inputs(self, S, a, xi_n):
        return torch.cat((invariant_features(S), a, xi_n), -1)

    @staticmethod
    def targets(S, S1):
        """Physical (before out_scale) residuals matching integrate()."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        Ht = yaw_mat(R).transpose(-1, -2)
        wb, wb1 = _mv(R.transpose(-1, -2), w), _mv(R1.transpose(-1, -2), w1)
        return torch.cat((_mv(Ht, v1-v), wb1-wb, qd1-qd,
            _mv(Ht, p1-p-CONTROL_DT*v1),
            so3_log(R.transpose(-1, -2) @ R1)-CONTROL_DT*wb1,
            q1-q-CONTROL_DT*qd1), -1)

    def integrate(self, S, delta):
        p, R, v, w, q, qd = S
        dv, dw, dqd, dp, dth, dq = (delta * self.out_scale).split((3,3,29,3,3,29), -1)
        H = yaw_mat(R)
        v1 = v + _mv(H, dv)
        wb1 = _mv(R.transpose(-1, -2), w) + dw
        qd1 = qd + dqd
        p1 = p + CONTROL_DT*v1 + _mv(H, dp)
        R1 = R @ so3_exp(CONTROL_DT*wb1 + dth)
        # Convert the reproduced body's angular velocity back to our storage convention.
        return p1, R1, v1, _mv(R1, wb1), q + CONTROL_DT*qd1 + dq, qd1

    def step(self, S, a, xi_n):
        return self.integrate(S, run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n))))

    @staticmethod
    def state_error(S_hat, S):
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        return torch.cat((ph-p, so3_log(R.transpose(-1,-2) @ Rh), vh-v,
            _mv(Rh.transpose(-1,-2), wh)-_mv(R.transpose(-1,-2), w), qh-q, qdh-qd), -1)

    def termination_probability(self, mdp, S, f):
        if self.term_head is None:
            return mdp.soft_termination(S, f)
        return torch.sigmoid(self.term_head(mdp.termination_features(S, f)))


class Policy(nn.Module):
    """Term-major observation history + 68*K reference features + optional 9D anchor."""
    def __init__(self, obs_dim, goal_dim, hist=10, z_dim=32):
        super().__init__()
        self.hist, self.goal_dim = hist, goal_dim
        self.enc = mlp(goal_dim, [2048,1024,512,256], z_dim, act=nn.SiLU)
        self.dec = mlp(z_dim+obs_dim*hist, [2048,1024,512,256], NUM_J, act=nn.SiLU)
        nn.init.orthogonal_(self.dec[-1].weight, gain=0.01)
        nn.init.zeros_(self.dec[-1].bias)
        self.obs_norm = EmpiricalNormalization(obs_dim*hist)
        self.goal_norm = EmpiricalNormalization(goal_dim)

    def forward(self, obs_hist, goal):
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        return run_mlp(self.dec, torch.cat((z, self.obs_norm(obs_hist)), -1))


class Critic(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = mlp(in_dim, [256,128], 1, act=nn.SiLU)
        self.norm = EmpiricalNormalization(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


def policy_from_checkpoint(checkpoint, obs_dim):
    from .motion import checkpoint_goal_dim
    from .legacy_policy import LegacyPolicy
    cls = Policy if checkpoint.get("model_input_version", 0) >= 7 else LegacyPolicy
    model = cls(obs_dim, checkpoint_goal_dim(checkpoint), hist=checkpoint["args"]["hist"])
    model.load_state_dict(checkpoint["pi"])
    return model
