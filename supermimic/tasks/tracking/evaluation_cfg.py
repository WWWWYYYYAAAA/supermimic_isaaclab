"""Evaluation CLI contract; independent of the learning algorithm."""

import argparse
import os
from supermimic.paths import REPO


def eval_parser(include_launcher=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument(
        "--ckpt", type=str, default=None, help="evaluate one checkpoint"
    )
    parser.add_argument(
        "--record", type=str, default=None, help="save trajectories (npz) for rendering"
    )
    parser.add_argument(
        "--motion_dir", type=str, default=os.path.join(REPO, "data", "lafan1_g1")
    )
    parser.add_argument(
        "--every", type=int, default=1, help="evaluate every n-th checkpoint"
    )
    if include_launcher:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)
    return parser
