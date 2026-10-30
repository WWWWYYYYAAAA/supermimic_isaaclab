"""Exported artifacts must preserve policy outputs and carry their input/control contract."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from supermimic import networks
from supermimic.export_policy import export_policy, export_all, load_export_inputs
from supermimic.paths import REPO
from supermimic.tasks.tracking.control_cfg import PDControlCfg, joint_pd_parameters
from supermimic.tasks.tracking.observation_cfg import ObservationCfg, SensorTermCfg
from supermimic.tasks.tracking.randomization_cfg import RandomizationCfg
from supermimic.tasks.tracking.reward_cfg import RewardCfg
from supermimic.robot_spec import G1_JOINT_NAMES


def checkpoint(directory, *, configured=False):
    observation = ObservationCfg(history=2)
    control = PDControlCfg()
    if configured:
        observation.reference_orientation = SensorTermCfg(noise=0.05)
        observation.joint_vel.scale = 0.1
        control.stiffness_overrides = {".*_knee_joint": 100.0}
    torch.manual_seed(381)
    policy = networks.Policy(observation.dim, 146, hist=2).eval()
    policy.obs_norm.update(torch.randn(10, observation.dim) * 2 + 1)
    policy.goal_norm.update(torch.randn(10, 146) * 0.3 - 0.1)
    saved = {
        "args": {"hist": 2, "goal_frames": 2, "amp": "bf16"},
        "pi": policy.state_dict(),
        "iteration": 2000,
    }
    if configured:
        saved["task_settings"] = {
            "version": 1,
            "control": asdict(control),
            "observations": asdict(observation),
            "randomization": asdict(RandomizationCfg()),
            "reward": asdict(RewardCfg()),
            "model": {
                "urdf": "data/g1_description/g1_29dof_rev_1_0.urdf",
                "mjcf": "data/g1_description/g1_29dof.xml",
                "usd": "data/g1_usd/g1_29dof/g1_29dof.usda",
            },
            "self_collisions": False,
            "episode_length_s": 10.0,
            "goal_frames": 2,
        }
    path = Path(directory) / "checkpoint.pt"
    torch.save(saved, path)
    return path, policy, saved, control


class ExportPolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_torchscript_matches_policy_and_leaves_amp_and_checkpoint_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, policy, _, _ = checkpoint(temporary)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            previous_amp = networks.AMP_DTYPE
            networks.AMP_DTYPE = torch.bfloat16
            try:
                output, metadata_path = export_policy(
                    source, Path(temporary) / "policy.ts"
                )
                self.assertEqual(networks.AMP_DTYPE, torch.bfloat16)
            finally:
                networks.AMP_DTYPE = previous_amp
            loaded = torch.jit.load(str(output))
            for batch in (1, 3):
                obs, goal = torch.randn(batch, 2, 93), torch.randn(batch, 146)
                with torch.no_grad():
                    latent = torch.tanh(policy.enc(policy.goal_norm(goal)))
                    expected = policy.dec(
                        torch.cat((latent, policy.obs_norm(obs.flatten(-2))), dim=-1)
                    )
                    actual = loaded(obs.flatten(-2), goal)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                self.assertEqual(actual.shape, (batch, 29))
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["settings_source"], "legacy_defaults")
            self.assertEqual(
                metadata["inputs"]["obs_history"]["shape"], ["batch", 186]
            )
            self.assertEqual(metadata["joint_names"], G1_JOINT_NAMES)
            self.assertIsNone(metadata["checkpoint_self_collisions"])
            self.assertEqual(metadata["validation"]["batch_sizes"], [1, 3])
            self.assertEqual(
                metadata["artifact_sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )
            with self.assertRaises((torch.jit.Error, RuntimeError)):
                loaded(torch.randn(1, 3, 93), torch.randn(1, 146))

    def test_configured_layout_and_pd_metadata_are_restored(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, saved, control = checkpoint(temporary, configured=True)
            output, metadata_path = export_policy(source, Path(temporary) / "custom.ts")
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(
                metadata["inputs"]["obs_history"]["shape"], ["batch", 198]
            )
            self.assertEqual(len(metadata["observation_layout"]), 6)
            self.assertFalse(metadata["checkpoint_self_collisions"])
            self.assertEqual(metadata["observation_layout"][1]["scale"], 0.1)
            gains = joint_pd_parameters(control)
            knee = G1_JOINT_NAMES.index("left_knee_joint")
            self.assertEqual(metadata["nominal_kp"][knee], 100.0)
            self.assertEqual(
                metadata["action_scale"][knee], gains["left_knee_joint"][2]
            )
            loaded = torch.jit.load(str(output))
            self.assertEqual(
                loaded(torch.randn(3, 198), torch.randn(3, 146)).shape, (3, 29)
            )
            saved["task_settings"]["observations"]["history"] = 3
            torch.save(saved, source)
            with self.assertRaisesRegex(ValueError, "history/goal"):
                load_export_inputs(source)

    def test_existing_outputs_are_replaced_but_source_is_protected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _, _ = checkpoint(temporary)
            with self.assertRaisesRegex(ValueError, "source checkpoint"):
                export_policy(source, source)
            output = Path(temporary) / "existing.ts"
            output.write_text("old export")
            metadata_path = output.with_suffix(".ts.json")
            metadata_path.write_text("old metadata")
            export_policy(source, output)
            loaded = torch.jit.load(str(output))
            self.assertEqual(loaded(torch.randn(1, 186), torch.randn(1, 146)).shape, (1, 29))
            self.assertEqual(json.loads(metadata_path.read_text())["artifact_sha256"],
                             hashlib.sha256(output.read_bytes()).hexdigest())

    def test_split_components_compose_like_full_policy_with_pt_suffix(self):
        for configured in (False, True):
            with (
                self.subTest(configured=configured),
                tempfile.TemporaryDirectory() as temporary,
            ):
                source, policy, _, _ = checkpoint(temporary, configured=configured)
                before = hashlib.sha256(source.read_bytes()).hexdigest()
                outputs = export_all(source, Path(temporary) / "policy.pt")
                self.assertEqual(set(outputs), {"policy", "encoder", "decoder"})
                loaded = {
                    name: torch.jit.load(str(paths[0]))
                    for name, paths in outputs.items()
                }
                reports = {
                    name: json.loads(paths[1].read_text())
                    for name, paths in outputs.items()
                }
                self.assertEqual(
                    len({report["policy_fingerprint"] for report in reports.values()}),
                    1,
                )
                self.assertEqual(set(reports["encoder"]["inputs"]), {"motion_goal"})
                self.assertEqual(
                    set(reports["decoder"]["inputs"]), {"obs_history", "latent"}
                )
                self.assertEqual(reports["encoder"]["output"]["shape"], ["batch", 32])
                self.assertFalse(
                    any(
                        key.startswith("dec.") for key in loaded["encoder"].state_dict()
                    )
                )
                self.assertFalse(
                    any(
                        key.startswith("enc.") for key in loaded["decoder"].state_dict()
                    )
                )
                width = 99 if configured else 93
                for batch in (1, 3):
                    obs, goal = torch.randn(batch, 2, width), torch.randn(batch, 146)
                    with torch.no_grad():
                        expected_z = torch.tanh(policy.enc(policy.goal_norm(goal)))
                        expected_a = policy.dec(
                            torch.cat(
                                (expected_z, policy.obs_norm(obs.flatten(-2))), -1
                            )
                        )
                        z = loaded["encoder"](goal)
                        action = loaded["decoder"](obs.flatten(-2), z)
                    self.assertTrue(bool((z.abs() <= 1).all()))
                    torch.testing.assert_close(z, expected_z, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(action, expected_a, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(
                        action, loaded["policy"](obs.flatten(-2), goal), rtol=1e-5, atol=1e-6
                    )
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), before
                )
                for name, (artifact, _) in outputs.items():
                    graph = str(loaded[name].inlined_graph)
                    for operation in (
                        "prim::If", "prim::RaiseException", "aten::dim",
                        "aten::size", "aten::ne(", "aten::to(", "aten::_to_copy(",
                        "aten::reshape", "aten::repeat",
                    ):
                        self.assertNotIn(operation, graph)
                    self.assertFalse(reports[name]["input_shape_checks"])
                    self.assertEqual(graph.count("aten::flatten("), 0)
                    self.assertIn("aten::batch_norm", graph)
                    self.assertNotIn("aten::sqrt", graph)
                    self.assertEqual(reports[name]["normalization_type"], "BatchNorm1d")
                    self.assertEqual(reports[name]["normalizer_epsilon"], 1e-4)
                    self.assertEqual(reports[name]["component"], name)
                    self.assertEqual(
                        reports[name]["artifact_sha256"],
                        hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    )
                    self.assertIn(
                        "composition_max_absolute_error", reports[name]["validation"]
                    )

    def test_individual_exports_and_decoder_does_not_tanh_latent_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, policy, _, _ = checkpoint(temporary)
            encoder_path, _ = export_policy(
                source, component="encoder",
                check_inputs=True,
            )
            decoder_path, _ = export_policy(
                source, component="decoder",
                check_inputs=True,
            )
            self.assertFalse((Path(temporary) / "policy.pt").exists())
            self.assertEqual(encoder_path, source.parent / "export/policy_checkpoint_encoder.pt")
            self.assertEqual(decoder_path, source.parent / "export/policy_checkpoint_decoder.pt")
            encoder, decoder = (
                torch.jit.load(str(encoder_path)),
                torch.jit.load(str(decoder_path)),
            )
            obs = torch.randn(3, 2, 93)
            latent = torch.linspace(-0.99, 0.99, 32).expand(3, -1)
            with torch.no_grad():
                expected = policy.dec(
                    torch.cat((latent, policy.obs_norm(obs.flatten(-2))), -1)
                )
            torch.testing.assert_close(
                decoder(obs.flatten(-2), latent), expected, rtol=1e-5, atol=1e-6
            )
            for bad_goal in (torch.randn(1, 145), torch.randn(1, 2, 146)):
                with self.assertRaises((torch.jit.Error, RuntimeError)):
                    encoder(bad_goal)
            for bad_obs, bad_z in (
                (obs.flatten(-2), torch.randn(3, 31)),
                (obs.flatten(-2), torch.randn(1, 32)),
                (torch.randn(3, 3, 93), latent),
            ):
                with self.assertRaises((torch.jit.Error, RuntimeError)):
                    decoder(bad_obs, bad_z)

    def test_checked_bundle_preserves_shape_errors_and_matches_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _, _ = checkpoint(temporary)
            plain = export_all(source, Path(temporary) / "plain.pt")
            checked = export_all(
                source, Path(temporary) / "checked.pt", check_inputs=True
            )
            obs = torch.randn(3, 186)
            goal, latent = torch.randn(3, 146), torch.randn(3, 32)
            inputs = {
                "policy": (obs, goal), "encoder": (goal,), "decoder": (obs, latent)
            }
            for name, (artifact, metadata) in checked.items():
                model = torch.jit.load(str(artifact))
                baseline = torch.jit.load(str(plain[name][0]))
                self.assertTrue(json.loads(metadata.read_text())["input_shape_checks"])
                self.assertIn("prim::If", str(model.inlined_graph))
                torch.testing.assert_close(
                    model(*inputs[name]), baseline(*inputs[name]), rtol=0, atol=0
                )
            policy = torch.jit.load(str(checked["policy"][0]))
            for bad_obs, bad_goal, message in (
                (torch.randn(3, 2, 93), goal, "obs_history must have shape"),
                (torch.randn(3, 93), goal, "wrong flattened observation dimension"),
                (obs, torch.randn(3, 1, 146), "motion_goal must have shape"),
                (obs, torch.randn(1, 146), "Input batch sizes must match"),
            ):
                with self.assertRaisesRegex(torch.jit.Error, message):
                    policy(bad_obs, bad_goal)

    def test_bundle_overwrites_and_validation_failure_keeps_previous_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _, _ = checkpoint(temporary)
            outputs = export_all(source)
            paths = [path for pair in outputs.values() for path in pair]
            for path in paths:
                path.write_text("old export")
            export_all(source)
            for artifact, metadata in outputs.values():
                torch.jit.load(str(artifact))
                self.assertEqual(json.loads(metadata.read_text())["artifact_sha256"],
                                 hashlib.sha256(artifact.read_bytes()).hexdigest())
            before = {path: path.read_bytes() for path in paths}
            with patch("torch.testing.assert_close", side_effect=AssertionError("validation failed")):
                with self.assertRaisesRegex(AssertionError, "validation failed"):
                    export_all(source)
            self.assertTrue(all(path.read_bytes() == data for path, data in before.items()))
            with self.assertRaisesRegex(ValueError, "component"):
                export_policy(source, component="unknown")

    def test_export_cli_has_no_simulator_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, _, _ = checkpoint(temporary)
            output = Path(temporary) / "export/policy_checkpoint.pt"
            code = """
import builtins, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in ('isaaclab','isaaclab_newton','isaaclab_physx','mujoco','warp','newton'):
        raise AssertionError('Unexpected simulator dependency: '+name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from supermimic.export_policy import main
main(['--ckpt',sys.argv[1]])
"""
            result = subprocess.run(
                [sys.executable, "-c", code, str(source)],
                cwd=REPO,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_name("policy_checkpoint_encoder.pt").is_file())
            self.assertTrue(output.with_name("policy_checkpoint_decoder.pt").is_file())
            self.assertEqual(list(Path(temporary).glob("ckpt_*.pt")), [])


if __name__ == "__main__":
    unittest.main()
