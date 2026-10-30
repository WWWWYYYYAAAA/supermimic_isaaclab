"""Nondefault task settings must survive collection, BPTT and checkpoint evaluation."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch

from supermimic.config import train_parser
from supermimic.env import TrackingSim
from supermimic.evaluation import Evaluator
from supermimic.networks import WorldModel
from supermimic.training.trainer import Trainer
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.tasks.tracking.settings import (
    task_settings,
    apply_task_settings,
    same_settings,
    load_task_settings,
)
from tests.support import make_motion, manager_env


class ConfiguredPipelineTest(unittest.TestCase):
    def test_pd_and_observation_change_train_and_evaluate_saved_configuration(self):
        torch.set_num_threads(2)
        kin, motion, _ = make_motion()
        cfg = G1FlatEnvCfg()
        cfg.self_collisions = True
        cfg.control.stiffness_scale = 1.15
        cfg.randomization.pd_gains.enabled = True
        cfg.randomization.reset.enabled = False
        cfg.policy_observation.reference_orientation = None
        cfg.policy_observation.joint_vel.scale = 0.1
        cfg.analytic_reward.w_root_pos = 0.75
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
                    side_effect=lambda **kw: WorldModel(hidden=(16, 16), **kw),
                ),
                patch("torch.compile", side_effect=lambda fn: fn),
            ):
                trainer = Trainer(args, device="cpu", env_cfg=cfg)
            self.assertEqual(trainer.mdp.obs_dim, 93)
            self.assertEqual(trainer.xi_dim, 91)
            adapter = TrackingSim.__new__(TrackingSim)
            adapter.env = manager_env(motion, trainer.mdp, env_cfg=cfg)
            ref = adapter.env.reference
            self.assertEqual(ref.xi.shape, (2, 91))
            torch.testing.assert_close(
                adapter.env.scene["robot"].stiffness, ref.nominal_kp * ref.xi[:, 33:62]
            )
            before = trainer.pi.dec[-1].weight.detach().clone()
            trainer.run(adapter)
            self.assertFalse(torch.equal(before, trainer.pi.dec[-1].weight))
            self.assertEqual(trainer.buffer.data["xi"].shape[-1], 91)
            self.assertEqual(trainer.obs_hist.shape[-1], 93)
            ckpt_path = Path(tmp) / "ckpt_000001.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            self.assertIn("task_settings", ckpt)
            saved = load_task_settings(tmp, ckpt_path)
            self.assertTrue(
                same_settings(
                    saved, json.loads((Path(tmp) / "task_settings.json").read_text())
                )
            )
            # Changed source defaults must not alter the observation/physics used for saved weights.
            restored = G1FlatEnvCfg()
            restored.control.stiffness_scale = 0.5
            apply_task_settings(restored, saved)
            self.assertTrue(same_settings(saved, task_settings(restored)))
            self.assertTrue(restored.self_collisions)
            with (
                patch("supermimic.evaluation.G1Kinematics", return_value=kin),
                patch("supermimic.evaluation.MotionLib.from_dir", return_value=motion),
            ):
                evaluator = Evaluator("fixture", device="cpu", env_cfg=restored)
            adapter.env = manager_env(
                motion, evaluator.mdp, env_cfg=restored, eval_mode=True
            )
            torch.testing.assert_close(
                adapter.env.reference.xi[:, 33:], torch.ones(2, 58)
            )
            result = evaluator.evaluate(adapter, ckpt)
            self.assertTrue(torch.isfinite(torch.tensor(result["norm_return"])))
            self.assertAlmostEqual(evaluator.mdp.max_step_reward, 0.105)

    def test_override_order_is_part_of_checkpoint_compatibility(self):
        first = G1FlatEnvCfg()
        second = G1FlatEnvCfg()
        first.control.stiffness_overrides = {".*_joint": 20.0, ".*_knee_joint": 100.0}
        second.control.stiffness_overrides = {".*_knee_joint": 100.0, ".*_joint": 20.0}
        self.assertFalse(same_settings(task_settings(first), task_settings(second)))

    def test_invalid_task_cannot_silently_desynchronize_manager_reward_or_noise(self):
        cfg = G1FlatEnvCfg()
        cfg.rewards.tracking.weight = 2.0
        with self.assertRaisesRegex(ValueError, "analytic_reward"):
            cfg.prepare()
        cfg = G1FlatEnvCfg()
        cfg.observations.policy.enable_corruption = True
        with self.assertRaisesRegex(ValueError, "policy_observation"):
            cfg.prepare()


if __name__ == "__main__":
    unittest.main()
