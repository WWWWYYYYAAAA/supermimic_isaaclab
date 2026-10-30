"""SuperMimic collection, autoregressive world-model fitting and BPTT updates.

No PPO runner: optimizer order, losses, schedules and checkpoint keys are preserved.
"""

from __future__ import annotations
import json
import math
import os
import time
import torch
import torch.nn as nn
from supermimic.buffer import ReplayBuffer
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import (
    STATE_DIM,
    MDP,
    XI_DIM,
    normalize_xi,
    unflatten_state,
)
from supermimic.motion import MotionLib
from supermimic import networks
from supermimic.networks import Critic, Policy, WorldModel
from supermimic.robot_spec import default_joint_pos_list


def set_requires_grad(m: nn.Module, flag: bool):
    for p in m.parameters():
        p.requires_grad_(flag)


class Trainer:
    def __init__(self, args, device="cuda:0", env_cfg=None):
        if args.resume:
            raise ValueError(
                "Resume is unavailable because replay-buffer restoration is not implemented; start a fresh run"
            )
        self.args = args
        self.task_settings = None
        if env_cfg is not None:
            env_cfg.prepare()
            env_cfg.policy_observation.history = args.hist
            env_cfg.commands.motion.goal_frames = args.goal_frames
            from supermimic.tasks.tracking.settings import task_settings

            self.task_settings = task_settings(env_cfg)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(self.args.seed)
        networks.AMP_DTYPE = torch.bfloat16 if self.args.amp == "bf16" else None
        self.dev = device
        kin_options = {} if env_cfg is None else {"mjcf_path": env_cfg.robot_model.mjcf}
        self.kin = G1Kinematics(self.dev, **kin_options)
        self.motion = MotionLib.from_dir(self.args.motion_dir, self.kin, self.dev)
        self.mdp = MDP(
            self.kin,
            self.motion,
            torch.tensor(default_joint_pos_list(), device=self.dev),
            **(
                {}
                if env_cfg is None
                else {
                    "cfg": env_cfg.analytic_reward,
                    "observation_cfg": env_cfg.policy_observation,
                    "xi_dim": env_cfg.randomization.context_dim,
                }
            ),
        )
        (self.K, self.Hh, self.HZ) = (
            self.args.goal_frames,
            self.args.hist,
            self.args.horizon,
        )
        self.GOAL_DIM = 73 * self.K
        self.CRITIC_IN = STATE_DIM + self.mdp.critic_extra + self.GOAL_DIM
        self.xi_dim = XI_DIM if env_cfg is None else env_cfg.randomization.context_dim
        self.wm = WorldModel(
            **({} if self.xi_dim == XI_DIM else {"xi_dim": self.xi_dim})
        ).to(self.dev)
        self.pi = Policy(self.mdp.obs_dim, self.GOAL_DIM, hist=self.Hh).to(self.dev)
        self.critic = Critic(self.CRITIC_IN).to(self.dev)
        self.opt_wm = torch.optim.Adam(self.wm.parameters(), lr=self.args.lr)
        self.opt_pi = torch.optim.Adam(
            self.pi.parameters(), lr=self.args.lr, betas=(0.9, 0.95)
        )
        self.opt_v = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr)
        (self.start_it, self.wall_offset, self.sim_steps) = (0, 0.0, 0)
        os.makedirs(self.args.log_dir, exist_ok=True)
        with open(os.path.join(self.args.log_dir, "args.json"), "w") as stream:
            json.dump(vars(self.args), stream, indent=2)
        if self.task_settings is not None:
            from supermimic.tasks.tracking.settings import save_task_settings

            save_task_settings(self.args.log_dir, self.task_settings)
        self.ERR_FLOOR = torch.cat(
            (
                torch.full((3,), 0.003),
                torch.full((3,), 0.005),
                torch.full((3,), 0.02),
                torch.full((3,), 0.05),
                torch.full((29,), 0.003),
                torch.full((29,), 0.05),
            )
        ).to(self.dev)
        self.wm_rollout = torch.compile(self._wm_rollout)
        self.wm_err = torch.compile(self._wm_err)
        self.imag_step = torch.compile(self._imag_step)
        self.reward_term = torch.compile(self._reward_term)
        self.critic_input_c = torch.compile(self._critic_input)
        self.PROFILE = bool(int(os.environ.get("SM_PROFILE", "0")))

    def _wm_rollout(self, s, a, xi):
        S = unflatten_state(s[:, 0])
        preds = []
        for j in range(self.args.wm_window):
            S = self.wm.step(S, a[:, j], normalize_xi(xi[:, j]))
            preds.append(S)
        return tuple((torch.stack([P[i] for P in preds], 1) for i in range(6)))

    def _wm_err(self, S_hat, s1, mask):
        std = torch.maximum(self.wm.out_norm.std(), self.ERR_FLOOR)
        e = (self.wm.state_error(S_hat, unflatten_state(s1)) / std) ** 2
        e = e.mean(-1)
        return ((e * mask).sum() / s1.shape[0], e[:, -1])

    def _imag_step(self, S, o_hist, a_prev, f, xi, xi_n, g):
        a = self.pi(o_hist.flatten(-2), g)
        S1 = self.wm.step(S, a, xi_n)
        f1 = f + 1
        o1 = self.mdp.add_noise(self.mdp.obs(S1, a, xi, f1))
        o_hist = torch.cat((o_hist[:, 1:], o1.unsqueeze(1)), 1)
        return (S1, a, o_hist, f1, self.motion.goal(f1, self.K))

    def _reward_term(self, S, f, a, a_prev):
        return (self.mdp.reward(S, f, a, a_prev), self.mdp.soft_termination(S, f))

    def _critic_input(self, S, f, xi_n, g):
        return self.mdp.critic_features(S, g, xi_n)

    def lr_at(self, it: int) -> float:
        if it >= self.args.lr_decay_iters:
            return self.args.lr * 0.03
        return self.args.lr * (
            0.03 + 0.97 * 0.5 * (1 + math.cos(math.pi * it / self.args.lr_decay_iters))
        )

    def wm_update(self):
        w = self.buffer.sample_windows(self.args.wm_batch, self.args.wm_window)
        S_hat = self.wm_rollout(w["s"], w["a"], w["xi"])
        (loss, e_last) = self.wm_err(S_hat, w["s1"], w["mask"])
        self.opt_wm.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), 1.0)
        self.opt_wm.step()
        m = w["mask"][:, -1]
        return (
            loss.detach() / self.args.wm_window,
            (e_last.detach() * m).sum() / m.sum().clamp(min=1),
        )

    def sample_valid_starts(self, B: int):
        (starts, hists, n) = ([], [], 0)
        while n < B:
            (s, h) = self.buffer.sample_starts(2 * B, self.Hh)
            f = s["f"]
            ok = f + self.HZ <= self.motion.clip_end[self.motion.clip_id[f]]
            idx = ok.nonzero(as_tuple=False).squeeze(-1)[: B - n]
            (
                starts.append({k: v[idx] for (k, v) in s.items()}),
                hists.append({k: v[idx] for (k, v) in h.items()}),
            )
            n += len(idx)
        cat = lambda ds: {k: torch.cat([d[k] for d in ds]) for k in ds[0]}
        return (cat(starts), cat(hists))

    def policy_update(self):
        if self.PROFILE:
            torch.cuda.synchronize()
            tq0 = time.time()
        (s, h) = self.sample_valid_starts(self.args.pi_batch)
        S = unflatten_state(s["s"])
        (xi, f, a_prev) = (s["xi"], s["f"], s["a_prev"])
        xi_n = normalize_xi(xi)
        with torch.no_grad():
            o_hist = self.mdp.obs(unflatten_state(h["s"]), h["a_prev"], h["xi"], h["f"])
        o_hist = self.mdp.add_noise(o_hist)
        set_requires_grad(self.wm, False)
        set_requires_grad(self.critic, False)
        B = self.args.pi_batch
        g0 = self.motion.goal(f, self.K)
        g = g0
        S0 = S
        (Ss, As, Aps, Fs, Gs) = ([], [], [], [], [])
        for t in range(self.HZ):
            Aps.append(a_prev)
            (S, a_prev, o_hist, f, g) = self.imag_step(
                S, o_hist, a_prev, f, xi, xi_n, g
            )
            (Ss.append(S), As.append(a_prev), Fs.append(f), Gs.append(g))
        flat = lambda xs: torch.stack(xs).flatten(0, 1)
        S_all = tuple((flat([St[i] for St in Ss]) for i in range(6)))
        f_all = flat(Fs)
        (r, d) = self.reward_term(S_all, f_all, flat(As), flat(Aps))
        (r, d) = (r.view(self.HZ, B), d.view(self.HZ, B))
        alive = torch.cumprod(1 - d, 0)
        c_t = torch.cat((torch.ones(1, B, device=self.dev), alive[:-1]), 0)
        disc = self.args.gamma ** torch.arange(
            self.HZ, device=self.dev, dtype=torch.float32
        ).unsqueeze(-1)
        J = (disc * c_t * r).sum(0)
        c = alive[-1]
        r_sum = (c_t * r).detach().sum(0)
        (rews, dones) = (r.detach(), d.detach())
        if not self.args.no_critic:
            J = J + self.args.gamma**self.HZ * c * self.critic(
                self._critic_input(S, f, xi_n, g)
            )
        loss = -J.mean()
        if self.PROFILE:
            torch.cuda.synchronize()
            tq1 = time.time()
        self.opt_pi.zero_grad(set_to_none=True)
        loss.backward()
        if self.PROFILE:
            torch.cuda.synchronize()
            tq2 = time.time()
        gn = nn.utils.clip_grad_norm_(self.pi.parameters(), 1.0)
        nn.utils.clip_grad_value_(self.pi.parameters(), 0.1)
        if torch.isfinite(gn):
            self.opt_pi.step()
        set_requires_grad(self.wm, True)
        set_requires_grad(self.critic, True)
        v_loss = torch.zeros(())
        if not self.args.no_critic:
            with torch.no_grad():
                S_c = tuple(
                    (
                        torch.cat(
                            (
                                S0[i].unsqueeze(0),
                                torch.stack([St[i].detach() for St in Ss]),
                            ),
                            0,
                        ).flatten(0, 1)
                        for i in range(6)
                    )
                )
                f_c = torch.cat((s["f"].unsqueeze(0), torch.stack(Fs)), 0).flatten()
                g_c = torch.cat((g0.unsqueeze(0), torch.stack(Gs)), 0).flatten(0, 1)
                xi_c = xi_n.unsqueeze(0).expand(self.HZ + 1, -1, -1).flatten(0, 1)
                X = self.critic_input_c(S_c, f_c, xi_c, g_c).view(self.HZ + 1, B, -1)
            self.critic.norm.update(X)
            with torch.no_grad():
                V = self.critic(X)
                R = V[-1]
                targets = []
                for t in range(self.HZ - 1, -1, -1):
                    R = rews[t] + self.args.gamma * (1 - dones[t]) * (
                        (1 - self.args.lam) * V[t + 1] + self.args.lam * R
                    )
                    targets.append(R)
                targets = torch.stack(targets[::-1])
            (Xf, Yf) = (X[:-1].reshape(-1, X.shape[-1]), targets.reshape(-1))
            n = Xf.shape[0]
            mb = n // self.args.critic_minibatches
            for _ in range(self.args.critic_epochs):
                perm = torch.randperm(n, device=self.dev)
                for i in range(self.args.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    v_loss = ((self.critic(Xf[idx]) - Yf[idx]) ** 2).mean()
                    self.opt_v.zero_grad(set_to_none=True)
                    v_loss.backward()
                    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                    self.opt_v.step()
        if self.PROFILE:
            torch.cuda.synchronize()
            print(
                f"[prof] imagine fwd {1000.0 * (tq1 - tq0):.0f} ms | bwd {1000.0 * (tq2 - tq1):.0f} ms | critic {1000.0 * (time.time() - tq2):.0f} ms",
                flush=True,
            )
        return (
            loss.detach(),
            v_loss.detach(),
            (r_sum / self.HZ / self.mdp.max_step_reward).mean(),
            gn.detach(),
            c.mean().detach(),
        )

    @torch.no_grad()
    def collect(self, n_steps: int, stats: dict):
        for _ in range(n_steps):
            S = self.env.state()
            o = self.mdp.add_noise(
                self.mdp.obs(S, self.env.a_prev, self.env.xi, self.env.f)
            )
            fresh = self.env.ep_step == 0
            self.obs_hist = torch.where(
                fresh[:, None, None],
                o.unsqueeze(1).expand_as(self.obs_hist),
                torch.cat((self.obs_hist[:, 1:], o.unsqueeze(1)), 1),
            )
            g = self.motion.goal(self.env.f, self.K)
            self.pi.obs_norm.update(o)
            self.pi.goal_norm.update(g)
            a = self.pi(self.obs_hist.flatten(-2), g)
            a = torch.clamp(a + self.args.action_noise * torch.randn_like(a), -20, 20)
            out = self.env.step(a)
            self.buffer.add(out)
            (S0, S1) = (unflatten_state(out["s"]), unflatten_state(out["s1"]))
            self.wm.in_norm.update(
                self.wm.inputs(S0, out["a"], normalize_xi(out["xi"]))
            )
            self.wm.out_norm.update(self.wm.targets(S0, S1))
            self.sim_steps += self.args.num_envs
            stats["r"].append(out["r"].mean() / self.mdp.max_step_reward)
            if "ep_len" in out:
                stats["ep_len"].append(out["ep_len"].mean())
                stats["ep_ret"].append(out["ep_ret_norm"].mean())
                stats["term"].append(out["term"].float().sum())
                stats["done"].append(out["done"].float().sum())

    def run(self, env):
        self.env = env
        self.buffer = ReplayBuffer(
            self.args.buffer_steps, self.args.num_envs, self.dev, xi_dim=self.xi_dim
        )
        self.obs_hist = torch.zeros(
            self.args.num_envs, self.Hh, self.mdp.obs_dim, device=self.dev
        )
        stats = {"r": [], "ep_len": [], "ep_ret": [], "term": [], "done": []}
        if self.start_it == 0:
            self.collect(self.args.warmup_steps, stats)
        log_f = open(os.path.join(self.args.log_dir, "train_log.csv"), "a")
        if self.start_it == 0:
            log_f.write(
                "iteration,wall_h,sim_steps,wm_loss,wm_err8,pi_loss,v_loss,imag_rew,pi_gradnorm,imag_survival,sim_rew,ep_len,ep_ret_norm,term_frac,it_per_s,lr\n"
            )
        t_start = time.time() - self.wall_offset
        t_log = time.time()
        step_acc = 0.0
        agg = {"wm": [], "wm8": [], "pi": [], "v": [], "ir": [], "gn": [], "surv": []}
        for it in range(self.start_it, self.args.iterations):
            lr = self.lr_at(it)
            for o in (self.opt_wm, self.opt_pi, self.opt_v):
                for gparam in o.param_groups:
                    gparam["lr"] = lr
            step_acc += 1.0 / self.args.utd
            n = int(step_acc)
            step_acc -= n
            prof = self.PROFILE and it % 20 == 0 and (it > self.args.wm_only_iters + 5)
            if prof:
                torch.cuda.synchronize()
                tp0 = time.time()
            self.collect(n, stats)
            if prof:
                torch.cuda.synchronize()
                tp1 = time.time()
            for _ in range(self.args.wm_batches):
                (l, e8) = self.wm_update()
            if prof:
                torch.cuda.synchronize()
                tp2 = time.time()
            (agg["wm"].append(l), agg["wm8"].append(e8))
            if it >= self.args.wm_only_iters:
                (pl, vl, ir, gn, surv) = self.policy_update()
                if prof:
                    torch.cuda.synchronize()
                    tp3 = time.time()
                    print(
                        f"[prof] collect {1000.0 * (tp1 - tp0):.0f} ms ({n} steps) | wm {1000.0 * (tp2 - tp1):.0f} ms | policy+critic {1000.0 * (tp3 - tp2):.0f} ms",
                        flush=True,
                    )
                (
                    agg["pi"].append(pl),
                    agg["v"].append(vl),
                    agg["ir"].append(ir),
                    agg["gn"].append(gn),
                    agg["surv"].append(surv),
                )
            if (it + 1) % self.args.log_every == 0:
                mean = lambda x: float(sum((float(v) for v in x)) / max(len(x), 1))
                ips = self.args.log_every / (time.time() - t_log)
                t_log = time.time()
                wall_h = (time.time() - t_start) / 3600
                term_frac = float(
                    sum((float(v) for v in stats["term"]))
                    / max(sum((float(v) for v in stats["done"])), 1)
                )
                row = [
                    it + 1,
                    wall_h,
                    self.sim_steps,
                    mean(agg["wm"]),
                    mean(agg["wm8"]),
                    mean(agg["pi"]),
                    mean(agg["v"]),
                    mean(agg["ir"]),
                    mean(agg["gn"]),
                    mean(agg["surv"]),
                    mean(stats["r"]),
                    mean(stats["ep_len"]),
                    mean(stats["ep_ret"]),
                    term_frac,
                    ips,
                    lr,
                ]
                log_f.write(
                    ",".join(
                        (f"{x:.6g}" if isinstance(x, float) else str(x) for x in row)
                    )
                    + "\n"
                )
                log_f.flush()
                print(
                    f"it {it + 1} | {wall_h:.3f} h | wm {row[3]:.4f} e8 {row[4]:.4f} | pi {row[5]:.3f} v {row[6]:.4f} imag_r {row[7]:.3f} surv {row[9]:.3f} | sim_r {row[10]:.3f} ep_len {row[11]:.1f} ep_ret {row[12]:.3f} term {term_frac:.2f} | {ips:.2f} it/s",
                    flush=True,
                )
                stats = {k: [] for k in stats}
                agg = {k: [] for k in agg}
            final_iteration = it + 1 == self.args.iterations
            save_ckpt = (it + 1) % self.args.ckpt_every == 0 or final_iteration
            save_last = (it + 1) % self.args.last_every == 0 or final_iteration
            if save_ckpt or save_last:
                ck = {
                    "wm": self.wm.state_dict(),
                    "pi": self.pi.state_dict(),
                    "critic": self.critic.state_dict(),
                    "opt_wm": self.opt_wm.state_dict(),
                    "opt_pi": self.opt_pi.state_dict(),
                    "opt_v": self.opt_v.state_dict(),
                    "iteration": it + 1,
                    "wall_time": time.time() - t_start,
                    "sim_steps": self.sim_steps,
                    "args": vars(self.args),
                    "model_input_version": networks.MODEL_INPUT_VERSION,
                    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                if self.task_settings is not None:
                    ck["task_settings"] = self.task_settings
                if save_last:
                    last = os.path.join(self.args.log_dir, "last.pt")
                    torch.save(ck, last + ".tmp")
                    os.replace(last + ".tmp", last)
                if save_ckpt:
                    small = {
                        k: ck[k]
                        for k in (
                            "pi",
                            "wm",
                            "iteration",
                            "wall_time",
                            "sim_steps",
                            "args",
                            "model_input_version",
                            "saved_at",
                        )
                    }
                    if self.task_settings is not None:
                        small["task_settings"] = self.task_settings
                    path = os.path.join(self.args.log_dir, f"ckpt_{it + 1:06d}.pt")
                    torch.save(small, path + ".tmp")
                    os.replace(path + ".tmp", path)
        log_f.close()
