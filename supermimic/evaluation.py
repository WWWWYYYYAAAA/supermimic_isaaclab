"""Deterministic full-clip evaluation, including checkpoint watching and NPZ recording."""

from __future__ import annotations
import glob
import json
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from supermimic.env import TrackingSim
import numpy as np
import torch
from supermimic import networks
from supermimic.mdp import MDP, flatten_state
from supermimic.motion import MotionLib, checkpoint_goal_dim, motion_goal_dim, update_orientation_history
from supermimic.kinematics import G1Kinematics
from supermimic.networks import Policy
from supermimic.robot_spec import default_joint_pos_list


class Evaluator:
    def __init__(self, motion_dir, device="cuda:0", env_cfg=None):
        self.task_settings = None
        if env_cfg is not None:
            env_cfg.prepare()
            from supermimic.tasks.tracking.settings import task_settings

            self.task_settings = task_settings(env_cfg)
        torch.backends.cuda.matmul.allow_tf32 = True
        self.dev = device
        kin_options = {} if env_cfg is None else {"mjcf_path": env_cfg.robot_model.mjcf}
        kin = G1Kinematics(device, **kin_options)
        self.motion = MotionLib.from_dir(motion_dir, kin, device)
        self.mdp = MDP(
            kin,
            self.motion,
            torch.tensor(default_joint_pos_list(), device=device),
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
        self.C = len(self.motion.names)

    @torch.no_grad()
    def evaluate(self, env: TrackingSim, ck: dict, record: str | None = None) -> dict:
        saved = ck.get("task_settings")
        if saved is not None:
            from supermimic.tasks.tracking.settings import same_settings

            if not same_settings(saved, getattr(self, "task_settings", None)):
                raise ValueError(
                    "Checkpoint task settings differ from evaluation; reload via evaluate.py"
                )
        a_ = ck["args"]
        (K, Hh) = (a_["goal_frames"], a_["hist"])
        networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
        goal_dim = checkpoint_goal_dim(ck)
        self.mdp.legacy_inputs = ck.get("model_input_version", 0) < 7
        pi = networks.policy_from_checkpoint(ck, self.mdp.obs_dim).to(self.dev)
        pi.load_state_dict(ck["pi"])
        pi.eval()
        self.motion.validate_goal_window(K)
        env.reference.cfg.goal_frames = K
        ids = torch.arange(self.C, device=self.dev)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, self.motion.clip_start.clone())
        clip_end = self.motion.tracking_end(K)
        tracking_steps = clip_end - self.motion.clip_start
        ret = torch.zeros(self.C, device=self.dev)
        steps = torch.zeros(self.C, device=self.dev)
        hist = None
        orientation_history = None
        traj = {name: [] for name in (
            "root_p", "root_quat", "q", "f", "alive",
            "obs_history", "motion_goal", "action", "state", "xi",
        )}
        while not bool(env.done_eval.all()):
            S = env.state()
            o = self.mdp.obs(S, env.a_prev, env.xi, env.f)
            hist = (
                o.unsqueeze(1).repeat(1, Hh, 1)
                if hist is None
                else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
            )
            if self.mdp.legacy_inputs and goal_dim > 73*K:
                orientation_history = update_orientation_history(
                    orientation_history, self.mdp.reference_orientation(S, env.f), Hh
                )
            goal = (self.motion.policy_goal(env.f, K, S[1], orientation_history, goal_dim)
                    if self.mdp.legacy_inputs else self.mdp.goal(S, env.f, K, a_.get("anchor_obs", True)))
            a = pi(self.mdp.flatten_history(hist), goal)
            alive = ~env.done_eval.clone()
            if record:
                from supermimic.rotations import mat_to_quat

                traj["obs_history"].append(self.mdp.flatten_history(hist).cpu().clone())
                traj["motion_goal"].append(goal.cpu().clone())
                traj["action"].append(a.cpu().clone())
                traj["state"].append(flatten_state(S).cpu())
                traj["xi"].append(env.xi.cpu().clone())
                (
                    traj["root_p"].append(S[0].cpu()),
                    traj["root_quat"].append(mat_to_quat(S[1]).cpu()),
                )
                (
                    traj["q"].append(S[4].cpu()),
                    traj["f"].append(env.f.cpu()),
                    traj["alive"].append(alive.cpu()),
                )
            out = env.step(a)
            ret += torch.where(alive, out["r"], 0.0)
            steps += alive.float()
            env.f = torch.minimum(env.f, clip_end - 1)
        norm = ret / (tracking_steps.float() * self.mdp.max_step_reward)
        res = {
            "goal_frames": K,
            "clip_end_rule": "current_frame + goal_frames - 1 >= last_reference_frame",
            "norm_return": norm.mean().item(),
            "norm_return_framew": (
                ret.sum()
                / (tracking_steps.sum() * self.mdp.max_step_reward)
            ).item(),
            "n_failed": int(env.failed_eval.sum().item()),
            "failed": [
                self.motion.names[i]
                for i in env.failed_eval.nonzero().flatten().tolist()
            ],
            "per_clip": {
                self.motion.names[i]: round(norm[i].item(), 4) for i in range(self.C)
            },
        }
        if record:
            np.savez_compressed(
                record,
                names=np.array(self.motion.names),
                **{k: torch.stack(v).numpy() for (k, v) in traj.items()},
                ref_root_p=self.motion.root_p.cpu().numpy(),
                ref_root_R=self.motion.root_R.cpu().numpy(),
                ref_q=self.motion.q.cpu().numpy(),
                clip_start=self.motion.clip_start.cpu().numpy(),
                clip_len=self.motion.clip_len.cpu().numpy(),
            )
        return res

    def run(self, env, args):
        if args.ckpt:
            ck = torch.load(args.ckpt, map_location=self.dev)
            res = self.evaluate(env, ck, args.record)
            print(
                json.dumps(
                    {k: v for (k, v) in res.items() if k != "per_clip"}, indent=1
                )
            )
            return
        out_csv = os.path.join(args.run_dir, "eval.csv")
        done = set()
        if os.path.exists(out_csv):
            with open(out_csv) as stream:
                done = {
                    int(line.split(",")[0])
                    for line in stream.read().strip().splitlines()[1:]
                }
        else:
            with open(out_csv, "w") as stream:
                stream.write(
                    "iteration,train_wall_h,saved_at,eval_at,norm_return,norm_return_framew,n_failed,failed\n"
                )
        while True:
            ckpts = sorted(glob.glob(os.path.join(args.run_dir, "ckpt_*.pt")))
            todo = [c for c in ckpts if int(os.path.basename(c)[5:11]) not in done]
            todo = [
                c
                for c in todo
                if int(os.path.basename(c)[5:11]) // 2000 % args.every == 0
            ]
            for c in todo:
                it = int(os.path.basename(c)[5:11])
                ck = torch.load(c, map_location=self.dev)
                t0 = time.time()
                res = self.evaluate(env, ck)
                line = f"{it},{ck['wall_time'] / 3600:.4f},{ck['saved_at']},{time.strftime('%Y-%m-%d %H:%M:%S')},{res['norm_return']:.4f},{res['norm_return_framew']:.4f},{res['n_failed']},{'|'.join(res['failed'])}\n"
                with open(out_csv, "a") as stream:
                    stream.write(line)
                with open(c.replace(".pt", "_eval.json"), "w") as stream:
                    json.dump(res, stream, indent=1)
                print(
                    f"[eval] it {it} ({ck['wall_time'] / 3600:.2f} h): norm return {res['norm_return']:.4f}, failed {res['n_failed']}/{self.C} ({time.time() - t0:.0f}s)",
                    flush=True,
                )
                done.add(it)
            if not args.watch:
                break
            time.sleep(60)
