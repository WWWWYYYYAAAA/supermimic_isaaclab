import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import torch

from supermimic.env import TrackingSim
from supermimic.evaluation import Evaluator
from supermimic.mdp import OBS_DIM
from supermimic.networks import Policy
from tests.support import make_motion, manager_env


class EvaluationTest(unittest.TestCase):
    def test_evaluation_entry_uses_one_device_for_policy_and_simulation(self):
        from supermimic.evaluate import main
        from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

        for flags, configured, expected in (
            (["--device", "cuda:1"], "cuda:0", "cuda:1"),
            ([], "cuda:2", "cuda:2"),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                cfg = G1FlatEnvCfg()
                cfg.sim.device = configured
                evaluator, env = Mock(), Mock()
                with (
                    patch("supermimic.tasks.tracking.settings.load_task_settings", return_value=None),
                    patch("supermimic.evaluation.Evaluator", return_value=evaluator) as learner,
                    patch("supermimic.env.TrackingSim", return_value=env) as physics,
                    patch("isaaclab.app.launch_simulation", return_value=contextlib.nullcontext()) as launch,
                ):
                    main(["--run_dir", temporary, *flags], env_cfg=cfg)
                self.assertEqual(learner.call_args.kwargs["device"], expected)
                self.assertEqual(physics.call_args.args[0].device, expected)
                self.assertEqual(launch.call_args.args[0].device, expected)
                self.assertEqual(launch.call_args.args[1].device, expected)
                self.assertEqual(evaluator.run.call_args.args[1].device, expected)
                env.close.assert_called_once()

    def test_evaluation_recording_and_watcher_deduplication(self):
        torch.set_num_threads(2)
        _, motion, mdp = make_motion()
        evaluator = Evaluator.__new__(Evaluator)
        evaluator.dev, evaluator.motion, evaluator.mdp, evaluator.C = (
            "cpu",
            motion,
            mdp,
            2,
        )
        adapter = TrackingSim.__new__(TrackingSim)
        adapter.env = manager_env(motion, mdp, eval_mode=True)
        ck = {
            "model_input_version": 7,
            "args": {"goal_frames": 2, "hist": 2, "amp": "none"},
            "pi": Policy(OBS_DIM, 145, hist=2).state_dict(),
            "wall_time": 3600.0,
            "saved_at": "2026-10-29 12:00:00",
            "iteration": 2000,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = evaluator.evaluate(adapter, ck, str(root / "trajectory.npz"))
            self.assertTrue(np.isfinite(result["norm_return"]))
            self.assertEqual(set(result["per_clip"]), set(motion.names))
            with np.load(root / "trajectory.npz") as trajectory:
                np.testing.assert_array_equal(trajectory["alive"].sum(axis=0), (motion.clip_len-2).numpy())
                self.assertEqual(result["goal_frames"], 2)
                self.assertEqual(result["clip_end_rule"], "current_frame + goal_frames - 1 >= last_reference_frame")
                self.assertEqual(
                    set(trajectory.files),
                    {
                        "names",
                        "root_p",
                        "root_quat",
                        "q",
                        "f",
                        "alive",
                        "state", "motion_goal", "obs_history", "xi", "action",
                        "ref_root_p",
                        "ref_root_R",
                        "ref_q",
                        "clip_start",
                        "clip_len",
                    },
                )
                self.assertEqual(trajectory["q"].shape[1:], (2, 29))
                self.assertEqual(trajectory["root_quat"].shape[-1], 4)
            torch.save(ck, root / "ckpt_002000.pt")
            args = SimpleNamespace(ckpt=None, run_dir=tmp, every=1, watch=False)
            evaluator.run(adapter, args)
            first = (root / "eval.csv").read_text()
            self.assertEqual(len(first.splitlines()), 2)
            self.assertTrue((root / "ckpt_002000_eval.json").exists())
            evaluator.run(adapter, args)
            self.assertEqual((root / "eval.csv").read_text(), first)


if __name__ == "__main__":
    unittest.main()
