"""CLI contracts kept compatible with the original training/evaluation commands."""

import argparse
import os
from supermimic.paths import REPO


def _positive_iteration(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("iteration must be a positive integer")
    return value


def train_parser(include_launcher=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=320_000)
    parser.add_argument("--buffer_steps", type=int, default=1000)
    parser.add_argument(
        "--utd", type=float, default=0.4, help="updates per vectorized simulator step"
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--goal_frames", type=int, default=10)
    parser.add_argument("--anchor_obs", "--anchor-obs", action=argparse.BooleanOptionalAction, default=True,
                        help="Append reference torso pose in robot torso frame to motion goal")
    parser.add_argument("--termination_head", "--termination-head", choices=("neural", "analytic"), default="neural")
    parser.add_argument("--term_loss_weight", type=float, default=1.0)
    parser.add_argument("--hist", type=int, default=10)
    parser.add_argument("--wm_window", type=int, default=8)
    parser.add_argument("--wm_batches", type=int, default=10)
    parser.add_argument("--wm_batch", type=int, default=2048)
    parser.add_argument("--pi_batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--action_noise", type=float, default=0.1)
    parser.add_argument("--critic_epochs", type=int, default=5)
    parser.add_argument("--critic_minibatches", type=int, default=4)
    parser.add_argument("--no_critic", action="store_true")
    parser.add_argument(
        "--warmup_steps", type=int, default=64, help="sim steps before any update"
    )
    parser.add_argument(
        "--wm_only_iters",
        type=int,
        default=500,
        help="world-model-only iterations at the start",
    )
    parser.add_argument("--lr_decay_iters", type=int, default=250_000)
    parser.add_argument("--ckpt_every", type=int, default=2000)
    parser.add_argument("--last_every", type=int, default=200)
    parser.add_argument("--freeze_policy_norm_at", "--freeze-policy-norm-at", type=_positive_iteration,
                        default=None, help="Freeze actor empirical statistics after N completed iterations")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1")
    )
    parser.add_argument(
        "--log_dir", type=str, default=os.path.join(REPO, "logs", "run")
    )
    parser.add_argument(
        "--timestamp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a unique Beijing-time suffix to --log_dir (default: enabled)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Unsupported: replay-buffer restoration is not implemented; start a fresh run",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="bf16",
        choices=["none", "bf16"],
        help="autocast dtype for MLP matmuls",
    )
    if include_launcher:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)
    return parser
