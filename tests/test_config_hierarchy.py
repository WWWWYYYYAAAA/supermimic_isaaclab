"""Config inheritance must preserve defaults and pass one reward/model to both paths."""

import tempfile
import unittest
from unittest.mock import patch
import torch
from supermimic.config import train_parser
from supermimic.paths import MJCF_PATH, USD_PATH
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from supermimic.training.trainer import Trainer
from supermimic.evaluation import Evaluator
from supermimic.networks import WorldModel
from tests.support import make_motion


class ConfigHierarchyTest(unittest.TestCase):
    def test_presets_are_independent_and_select_matching_assets(self):
        from isaaclab.utils import validate

        standard, customized = G1FlatEnvCfg(), G1FlatEnvCfg()
        self.assertIsInstance(standard, TrackingEnvCfg)
        self.assertIsInstance(customized, G1FlatEnvCfg)
        for cfg in (standard, customized):
            validate(cfg)
            self.assertEqual(cfg.scene.robot.spawn.usd_path, cfg.robot_model.usd)
            self.assertEqual(cfg.sim.dt * cfg.decimation, 0.02)
        self.assertEqual(standard.robot_model.mjcf, MJCF_PATH)
        self.assertEqual(standard.scene.robot.spawn.usd_path, USD_PATH)
        self.assertEqual(customized.robot_model.mjcf, MJCF_PATH)
        self.assertEqual(standard.analytic_reward, customized.analytic_reward)
        customized.analytic_reward.w_action_rate = -0.2
        customized.scene.num_envs = 8
        self.assertEqual(standard.analytic_reward.w_action_rate, -0.1)
        self.assertEqual(G1FlatEnvCfg().analytic_reward.w_action_rate, -0.1)
        self.assertEqual(standard.scene.num_envs, 4096)

    def test_trainer_and_evaluator_use_task_model_and_shared_reward(self):
        torch.set_num_threads(2)
        kin, motion, _ = make_motion()
        cfg = G1FlatEnvCfg()
        cfg.analytic_reward.w_action_rate = -0.2
        with tempfile.TemporaryDirectory() as tmp:
            args = train_parser(False).parse_args(["--log_dir", tmp, "--amp", "none"])
            with (
                patch(
                    "supermimic.training.trainer.G1Kinematics", return_value=kin
                ) as factory,
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
            factory.assert_called_once_with("cpu", mjcf_path=cfg.robot_model.mjcf)
            self.assertIs(trainer.mdp.cfg, cfg.analytic_reward)
            with (
                patch(
                    "supermimic.evaluation.G1Kinematics", return_value=kin
                ) as factory,
                patch("supermimic.evaluation.MotionLib.from_dir", return_value=motion),
            ):
                evaluator = Evaluator(cfg.motion_dir, device="cpu", env_cfg=cfg)
            factory.assert_called_once_with("cpu", mjcf_path=cfg.robot_model.mjcf)
            self.assertIs(evaluator.mdp.cfg, cfg.analytic_reward)

    def test_customized_config_does_not_rebind_global_robot_defaults(self):
        from supermimic import paths, robot

        before = (paths.MJCF_PATH, robot.G1_CFG.spawn.usd_path)
        cfg = G1FlatEnvCfg()
        cfg.self_collisions = True
        cfg.prepare()
        props = cfg.scene.robot.spawn.articulation_props
        self.assertTrue(
            next(
                prop for prop in props if hasattr(prop, "self_collision_enabled")
            ).self_collision_enabled
        )
        self.assertTrue(
            next(
                prop for prop in props if hasattr(prop, "enabled_self_collisions")
            ).enabled_self_collisions
        )
        self.assertEqual((paths.MJCF_PATH, robot.G1_CFG.spawn.usd_path), before)
        self.assertFalse(G1FlatEnvCfg().self_collisions)


if __name__ == "__main__":
    unittest.main()
