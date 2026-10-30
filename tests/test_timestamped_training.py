"""Training logs use unique paths and never overwrite an existing experiment."""

import contextlib
from datetime import datetime
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from supermimic.config import train_parser
from supermimic.training import run_directory


class TimestampedTrainingTest(unittest.TestCase):
    def test_timestamp_preserves_existing_runs_and_uses_beijing_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "training"
            base.mkdir()
            sentinel = base / "args.json"
            sentinel.write_text('{"task": "old experiment"}')
            fixed_time = datetime(2026, 10, 29, 21, 30, 0, 123456,
                                  tzinfo=ZoneInfo("Asia/Shanghai"))
            occupied = base.with_name("training_20261029_213000_123456")
            occupied.mkdir()
            previous = occupied / "args.json"
            previous.write_text('{"task": "previous run"}')
            with patch.object(run_directory, "datetime") as clock:
                clock.now.return_value = fixed_time
                actual = run_directory.create_run_directory(base)
                clock.now.assert_called_once_with(ZoneInfo("Asia/Shanghai"))
            self.assertEqual(actual, occupied.with_name(occupied.name + "_01"))
            self.assertTrue(actual.is_dir())
            self.assertEqual(sentinel.read_text(), '{"task": "old experiment"}')
            self.assertEqual(previous.read_text(), '{"task": "previous run"}')

    def test_exact_path_accepts_empty_directory_but_rejects_existing_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "run"
            self.assertEqual(run_directory.create_run_directory(target, timestamp=False), target)
            self.assertEqual(run_directory.create_run_directory(target, timestamp=False), target)
            sentinel = target / "last.pt"
            sentinel.write_bytes(b"existing checkpoint")
            with self.assertRaises((ValueError, FileExistsError)):
                run_directory.create_run_directory(target, timestamp=False)
            self.assertEqual(sentinel.read_bytes(), b"existing checkpoint")

    def test_train_parser_timestamp_defaults_and_explicit_override(self):
        self.assertTrue(train_parser(False).parse_args([]).timestamp)
        self.assertFalse(train_parser(False).parse_args(["--no-timestamp"]).timestamp)
        self.assertTrue(train_parser(False).parse_args(["--timestamp"]).timestamp)

    def test_train_entry_passes_resolved_directory_and_closes_environment(self):
        from supermimic.train import main
        from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

        with tempfile.TemporaryDirectory() as temporary:
            resolved = Path(temporary) / "run_20261029_213000_123456"
            resolved.mkdir()
            trainer = Mock()
            env = Mock()
            with (
                patch.object(run_directory, "create_run_directory", return_value=resolved),
                patch("supermimic.training.trainer.Trainer", return_value=trainer) as factory,
                patch("supermimic.env.TrackingSim", return_value=env),
                patch("isaaclab.app.launch_simulation", return_value=contextlib.nullcontext()),
            ):
                main(["--log_dir", str(Path(temporary) / "run"), "--num_envs", "2"],
                     env_cfg=G1FlatEnvCfg())
            args = factory.call_args.args[0]
            self.assertEqual(args.log_dir, str(resolved))
            self.assertEqual(args.num_envs, 2)
            trainer.run.assert_called_once_with(env)
            env.close.assert_called_once()

    def test_train_entry_uses_one_device_for_learning_and_simulation(self):
        from supermimic.train import main
        from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

        for flags, configured, expected in (
            (["--device", "cuda:1"], "cuda:0", "cuda:1"),
            ([], "cuda:2", "cuda:2"),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as temporary:
                cfg = G1FlatEnvCfg()
                cfg.sim.device = configured
                trainer, env = Mock(), Mock()
                with (
                    patch.object(run_directory, "create_run_directory", return_value=Path(temporary)),
                    patch("supermimic.training.trainer.Trainer", return_value=trainer) as learner,
                    patch("supermimic.env.TrackingSim", return_value=env) as physics,
                    patch("isaaclab.app.launch_simulation", return_value=contextlib.nullcontext()) as launch,
                ):
                    main(flags, env_cfg=cfg)
                self.assertEqual(learner.call_args.kwargs["device"], expected)
                self.assertEqual(learner.call_args.args[0].device, expected)
                self.assertEqual(physics.call_args.args[0].device, expected)
                self.assertEqual(launch.call_args.args[0].device, expected)
                self.assertEqual(launch.call_args.args[1].device, expected)
                env.close.assert_called_once()

    def test_train_entry_rejects_resume_before_allocating_trainer(self):
        from supermimic.train import main

        with (
            patch("supermimic.training.trainer.Trainer") as factory,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises((ValueError, SystemExit)):
                main(["--resume", "previous/last.pt"])
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
