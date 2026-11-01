"""CPU checks for standalone live playback configuration and viewer lifecycle."""

import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch

from supermimic import play
from supermimic.env import TrackingSim
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.tasks.tracking.settings import task_settings
from tests.support import make_motion, manager_env


class PlayTest(unittest.TestCase):
    def test_saved_collision_and_policy_settings_restore_with_explicit_override(self):
        cfg = G1FlatEnvCfg()
        cfg.self_collisions = False
        cfg.policy_observation.reference_orientation = None
        cfg.policy_observation.history = 2
        cfg.commands.motion.goal_frames = 3
        cfg.randomization.pd_gains.enabled = True
        cfg.control.stiffness_scale = 1.2
        cfg.prepare()
        saved = task_settings(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            for index in (1, 2, 3):
                (motion_dir / f"walk{index}_subject1.csv").touch()
            checkpoint = root / "ckpt_000012.pt"
            torch.save(
                {
                    "pi": {},
                    "args": {"hist": 2, "goal_frames": 3},
                    "iteration": 12,
                    "task_settings": saved,
                },
                checkpoint,
            )
            args = play.parse_args(
                ["--ckpt", str(checkpoint), "--motion_dir", str(motion_dir)], False
            )
            result = play.prepare_playback(args)
            self.assertFalse(result.env_cfg.self_collisions)
            self.assertFalse(result.checkpoint_self_collisions)
            self.assertEqual(result.env_cfg.policy_observation.dim, 93)
            self.assertEqual(result.env_cfg.randomization.context_dim, 91)
            self.assertEqual(result.env_cfg.control.stiffness_scale, 1.2)
            self.assertEqual(
                result.clips,
                [f"walk{i}_subject1" for i in (1, 2, 3)],
            )
            props = result.env_cfg.scene.robot.spawn.articulation_props
            self.assertTrue(
                any(getattr(p, "self_collision_enabled", None) is False for p in props)
            )
            self.assertTrue(
                any(getattr(p, "enabled_self_collisions", None) is False for p in props)
            )
            self.assertFalse(result.checkpoint["task_settings"]["self_collisions"])
            args.self_collisions = True
            enabled = play.prepare_playback(args).env_cfg
            self.assertTrue(enabled.self_collisions)
            self.assertTrue(
                any(
                    getattr(p, "self_collision_enabled", None) is True
                    for p in enabled.scene.robot.spawn.articulation_props
                )
            )
            self.assertTrue(
                any(
                    getattr(p, "enabled_self_collisions", None) is True
                    for p in enabled.scene.robot.spawn.articulation_props
                )
            )
            self.assertFalse(
                torch.load(checkpoint, weights_only=True)["task_settings"][
                    "self_collisions"
                ]
            )

    def test_latest_legacy_checkpoint_needs_no_task_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            motion_dir = root / "motions"
            motion_dir.mkdir()
            (motion_dir / "clip.csv").touch()
            for iteration in (2, 10):
                torch.save(
                    {
                        "pi": {},
                        "args": {"hist": 2, "goal_frames": 2},
                        "iteration": iteration,
                    },
                    root / f"ckpt_{iteration:06d}.pt",
                )
            args = play.parse_args(
                ["--run_dir", tmp, "--motion_dir", str(motion_dir)], False
            )
            result = play.prepare_playback(args)
            self.assertEqual(result.checkpoint["iteration"], 10)
            self.assertFalse(result.env_cfg.self_collisions)
            self.assertIsNone(result.checkpoint_self_collisions)
            self.assertEqual(result.clips, ["clip"])
            args.clips = ["missing"]
            with self.assertRaisesRegex(ValueError, "Unknown or missing"):
                play.prepare_playback(args)

    def test_companion_settings_restore_when_checkpoint_has_no_snapshot(self):
        cfg = G1FlatEnvCfg()
        cfg.self_collisions = False
        cfg.policy_observation.history = 2
        cfg.commands.motion.goal_frames = 2
        cfg.analytic_reward.w_root_pos = 0.9
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "clip.csv").touch()
            (root / "task_settings.json").write_text(json.dumps(task_settings(cfg)))
            checkpoint = root / "last.pt"
            torch.save({"pi": {}, "args": {"hist": 2, "goal_frames": 2}}, checkpoint)
            args = play.parse_args(
                ["--checkpoint", str(checkpoint), "--motion_dir", tmp], False
            )
            result = play.prepare_playback(args)
            self.assertEqual(result.env_cfg.analytic_reward.w_root_pos, 0.9)
            self.assertFalse(result.env_cfg.self_collisions)

    def test_incompatible_policy_observation_shape_has_clear_error(self):
        from supermimic.legacy_policy import LegacyPolicy as Policy

        torch.set_num_threads(2)
        kin, motion, _ = make_motion()
        cfg = G1FlatEnvCfg()
        cfg.policy_observation.reference_orientation = None
        cfg.policy_observation.history = 2
        cfg.commands.motion.goal_frames = 2
        ck = {
            "pi": Policy(99, 146, hist=2).state_dict(),
            "args": {"hist": 2, "goal_frames": 2, "amp": "none"},
        }
        playback = play.Playback(
            Path("fixture.pt"), ck, cfg, Path("."), ["clip"], [], None
        )
        with (
            patch("supermimic.kinematics.G1Kinematics", return_value=kin),
            patch("supermimic.motion.MotionLib", return_value=motion),
        ):
            with self.assertRaisesRegex(ValueError, "policy shapes"):
                play.build_runtime(playback, "cpu")

    def test_invalid_headless_or_speed_options_fail_before_loading(self):
        for argv in (
            ["--headless"],
            ["--headless", "--steps", "2", "--paused"],
            ["--speed", "0"],
            ["--speed", "nan"],
            ["--steps", "-1"],
        ):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    play.parse_args(argv, False)

    def test_preview_pose_mapping_and_ghost_never_collides(self):
        cfg = G1FlatEnvCfg()
        model, ghosts = play.preview_model(cfg.robot_model.mjcf)
        self.assertEqual(model.nq, 72)
        self.assertGreater(len(ghosts), 0)
        self.assertTrue(np.all(model.geom_contype[ghosts] == 0))
        self.assertTrue(np.all(model.geom_conaffinity[ghosts] == 0))
        self.assertTrue(
            any("rubber_hand" in model.body(i).name for i in range(model.nbody))
        )
        data = mujoco.MjData(model)
        p, ref_p = np.array([2.0, 3.0, 0.8]), np.array([4.0, 5.0, 0.9])
        q = np.linspace(-0.1, 0.1, 29)
        R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        play.set_preview_state(model, data, p, [1, 0, 0, 0], q, ref_p, R, -q)
        np.testing.assert_allclose(data.qpos[:3], p)
        np.testing.assert_allclose(data.qpos[7:36], q)
        np.testing.assert_allclose(data.qpos[36:39], [2, 3, 0.9])
        np.testing.assert_allclose(
            data.xmat[model.body("ref_pelvis").id].reshape(3, 3), R, atol=1e-12
        )
        play.set_preview_state(model, data, p, [1, 0, 0, 0], q, ref_p, R, -q, True)
        np.testing.assert_allclose(data.qpos[36:39], ref_p)

    def test_finite_headless_loop_and_viewer_cleanup(self):
        torch.set_num_threads(2)
        _, motion, mdp = make_motion()
        adapter = TrackingSim.__new__(TrackingSim)
        adapter.env = manager_env(motion, mdp, eval_mode=True, n=1)
        args = play.parse_args(["--headless", "--steps", "3"], False)
        args.history_length = args.goal_frames = 2
        policy = lambda hist, goal: torch.zeros(1, 29)
        self.assertEqual(play.play_loop(adapter, policy, motion, mdp, args, ""), 3)
        # A display error after opening the viewer must release it.
        viewer = NS(
            lock=contextlib.nullcontext,
            cam=NS(lookat=np.zeros(3)),
            sync=Mock(side_effect=RuntimeError("display failed")),
            close=Mock(),
        )
        args.headless = False
        with self.assertRaisesRegex(RuntimeError, "display failed"):
            play.play_loop(
                adapter,
                policy,
                motion,
                mdp,
                args,
                G1FlatEnvCfg().robot_model.mjcf,
                viewer_factory=lambda *a, **kw: viewer,
            )
        viewer.close.assert_called_once()

    def test_main_closes_simulator_after_loop_error(self):
        args = play.parse_args(["--headless", "--steps", "3"], False)
        cfg = G1FlatEnvCfg()
        playback = play.Playback(
            Path("fixture.pt"),
            {"args": {"hist": 2, "goal_frames": 2}},
            cfg,
            Path("."),
            ["clip"],
            ["clip.csv"],
            False,
        )
        env = Mock()
        with (
            patch.object(play, "parse_args", return_value=args),
            patch.object(play, "prepare_playback", return_value=playback),
            patch.object(play, "build_runtime", return_value=(Mock(), Mock(), Mock())),
            patch.object(play, "play_loop", side_effect=RuntimeError("loop failed")),
            patch("torch.cuda.is_available", return_value=True),
            patch(
                "isaaclab.app.launch_simulation", return_value=contextlib.nullcontext()
            ),
            patch("supermimic.env.TrackingSim", return_value=env),
        ):
            with self.assertRaisesRegex(RuntimeError, "loop failed"):
                play.main([])
        env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
