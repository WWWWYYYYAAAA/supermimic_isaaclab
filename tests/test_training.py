"""Exercise collection, world-model optimization, BPTT and checkpoint output on CPU."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch

from supermimic.config import train_parser
from supermimic.networks import WorldModel
from supermimic.training.trainer import Trainer
from tests.support import make_motion, manager_env


class TrainingTest(unittest.TestCase):
    def test_last_and_numbered_checkpoint_schedules_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer.__new__(Trainer)
            trainer.args = train_parser(False).parse_args([
                "--iterations", "2001", "--num_envs", "1", "--buffer_steps", "1",
                "--wm_only_iters", "3000", "--wm_batches", "1",
                "--log_every", "3000", "--log_dir", tmp,
            ])
            trainer.dev, trainer.Hh, trainer.xi_dim = "cpu", 1, 1
            trainer.mdp = SimpleNamespace(obs_dim=1)
            trainer.start_it = trainer.wall_offset = trainer.sim_steps = 0
            trainer.PROFILE, trainer.task_settings = False, None
            trainer.wm = trainer.pi = trainer.critic = Mock()
            trainer.wm.state_dict.return_value = {}
            trainer.opt_wm = trainer.opt_pi = trainer.opt_v = Mock(param_groups=[])
            trainer.opt_wm.state_dict.return_value = {}
            trainer.collect = Mock()
            trainer.wm_update = Mock(return_value=(0.0, 0.0))
            trainer.lr_at = Mock(return_value=0.001)
            saved = []
            original_save = torch.save

            def record_save(checkpoint, path):
                saved.append((Path(path).name, checkpoint["iteration"]))
                original_save(checkpoint, path)

            with patch("supermimic.training.trainer.torch.save", side_effect=record_save):
                trainer.run(None)
            self.assertEqual(
                [it for name, it in saved if name == "last.pt.tmp"],
                [*range(200, 2001, 200), 2001],
            )
            self.assertEqual(
                [name for name, _ in saved if name.startswith("ckpt_")],
                ["ckpt_002000.pt.tmp", "ckpt_002001.pt.tmp"],
            )
            self.assertEqual(torch.load(Path(tmp) / "last.pt")["iteration"], 2001)

    def test_world_model_bptt_and_checkpoint_contract(self):
        torch.set_num_threads(2)
        kin, motion, mdp = make_motion()
        with tempfile.TemporaryDirectory() as tmp:
            args = train_parser(False).parse_args(
                [
                    "--num_envs",
                    "2",
                    "--iterations",
                    "1",
                    "--buffer_steps",
                    "16",
                    "--hist",
                    "2",
                    "--goal_frames",
                    "2",
                    "--horizon",
                    "2",
                    "--wm_window",
                    "2",
                    "--wm_batches",
                    "1",
                    "--wm_batch",
                    "2",
                    "--pi_batch",
                    "2",
                    "--warmup_steps",
                    "4",
                    "--wm_only_iters",
                    "0",
                    "--critic_epochs",
                    "1",
                    "--critic_minibatches",
                    "1",
                    "--amp",
                    "none",
                    "--log_every",
                    "1",
                    "--log_dir",
                    tmp,
                ]
            )
            with (
                patch("supermimic.training.trainer.G1Kinematics", return_value=kin),
                patch(
                    "supermimic.training.trainer.MotionLib.from_dir",
                    return_value=motion,
                ),
                patch(
                    "supermimic.training.trainer.WorldModel",
                    side_effect=lambda: WorldModel(hidden=(16, 16)),
                ),
                patch("torch.compile", side_effect=lambda fn: fn),
            ):
                trainer = Trainer(args, device="cpu")
            env = manager_env(motion, mdp, reset_noise=False)

            class Adapter:
                def __getattr__(self, name):
                    if hasattr(env, name):
                        return getattr(env, name)
                    return getattr(env.reference, name)

                def step(self, action):
                    return env.step(action)[4]["transition"]

            before_pi = [p.detach().clone() for p in trainer.pi.parameters()]
            before_wm = [p.detach().clone() for p in trainer.wm.parameters()]
            before_v = [p.detach().clone() for p in trainer.critic.parameters()]
            trainer.run(Adapter())
            for before, model in (
                (before_pi, trainer.pi),
                (before_wm, trainer.wm),
                (before_v, trainer.critic),
            ):
                self.assertTrue(
                    any(
                        not torch.equal(a, b)
                        for a, b in zip(before, model.parameters())
                    )
                )
                self.assertTrue(
                    all(bool(torch.isfinite(p).all()) for p in model.parameters())
                )
            full = torch.load(Path(tmp) / "last.pt", weights_only=False)
            small = torch.load(Path(tmp) / "ckpt_000001.pt", weights_only=False)
            self.assertEqual(
                set(full),
                {
                    "wm",
                    "pi",
                    "critic",
                    "opt_wm",
                    "opt_pi",
                    "opt_v",
                    "iteration",
                    "wall_time",
                    "sim_steps",
                    "args",
                    "model_input_version",
                    "saved_at",
                },
            )
            self.assertEqual(
                set(small),
                {"wm", "pi", "iteration", "wall_time", "sim_steps", "args", "saved_at", "model_input_version"},
            )
            self.assertEqual(full["iteration"], 1)
            self.assertEqual(full["sim_steps"], 12)
            self.assertEqual(
                json.loads((Path(tmp) / "args.json").read_text()), vars(args)
            )
            self.assertEqual(
                len((Path(tmp) / "train_log.csv").read_text().splitlines()), 2
            )


if __name__ == "__main__":
    unittest.main()
