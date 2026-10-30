"""CPU evidence for the paper audit; does not change training defaults or repair behavior."""

from pathlib import Path
import json
import math
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from supermimic.config import train_parser
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import OBS_DIM
from supermimic.paths import REPO
from supermimic.training.trainer import Trainer


def evidence():
    args = train_parser(False).parse_args([])
    # Exactly the first collection on resume: replay is fresh and no warmup runs.
    replay = ReplayBuffer(16, 1, "cpu")
    for _ in range(int(1.0 / args.utd)):
        replay.add({name: value[0].clone() for name, value in replay.data.items()})
    try:
        replay.sample_windows(1, args.wm_window)
    except RuntimeError as exc:
        resume = str(exc)
    else:
        raise AssertionError(
            "Resume probe no longer reproduces; re-audit the implementation"
        )
    # Probe actual loss reduction with a controlled unit-error state head.
    learner = Trainer.__new__(Trainer)
    learner.wm = SimpleNamespace(
        out_norm=SimpleNamespace(std=lambda: torch.ones(70)),
        state_error=lambda predicted, target: torch.ones(2, 8, 70),
    )
    learner.ERR_FLOOR = torch.zeros(70)
    loss, _ = learner._wm_err(None, torch.zeros(2, 8, 71), torch.ones(2, 8))
    # Multiple completed environments can address the same bin in a single update.
    bins = torch.tensor([0, 0])
    estimates = torch.zeros(1)
    returns = torch.tensor([1.0, 3.0])
    estimates.index_put_((bins,), 0.8 * estimates[bins] + 0.2 * returns)
    terminal_bins = {}
    for name in ("lafan1_g1",):
        affected = []
        files = sorted((REPO / "data" / name).glob("*.csv"))
        for file in files:
            with file.open() as stream:
                source_frames = sum(bool(line.strip()) for line in stream)
            frames = math.floor(((source_frames - 1) / 30) / 0.02) + 1
            if frames % 50 == 1:
                affected.append(file.name)
        terminal_bins[name] = {"clips_checked": len(files), "affected": affected}
    return {
        "observation": {
            "paper_table_A1": 93,
            "implementation": OBS_DIM,
            "extra_reference_orientation": OBS_DIM - 93,
        },
        "world_model_loss_unit_error": {
            "code": loss.item(),
            "paper_raw_sum_over_70_components": 8 * 70,
            "note": "Controlled unit std; production additionally divides by running std and floors.",
        },
        "default_schedule": {
            "wm_only_iters": args.wm_only_iters,
            "warmup_steps": args.warmup_steps,
            "amp": args.amp,
            "iterations": args.iterations,
        },
        "resume": {
            "fresh_replay_size": replay.size,
            "required_window": args.wm_window,
            "error": resume,
        },
        "duplicate_bin_update": {
            "CPU_observed": estimates.item(),
            "batch_mean_EMA": 0.4,
            "sequential_EMA": 0.76,
            "torch_contract": "undefined"
            if "undefined" in torch.Tensor.index_put_.__doc__
            else "inspect docstring",
        },
        "terminal_only_last_bins": terminal_bins,
    }


if __name__ == "__main__":
    print(json.dumps(evidence(), indent=2, ensure_ascii=False))
