"""Export a SuperMimic policy, encoder and/or decoder as FP32 TorchScript.

The artifact takes prepared observation history and motion goals, applies the
saved normalizers, and returns normalized joint actions. It does not contain
physics, motion loading, sensor construction, history management or a PD loop.
No Isaac Lab, MuJoCo, CUDA or ONNX installation is needed for this exporter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from supermimic.normalization import FeatureBatchNorm, EmpiricalNormalization
from supermimic.motion import LEGACY_GOAL_FRAME_DIM as GOAL_FRAME_DIM, checkpoint_goal_dim
from supermimic.robot_spec import G1_JOINT_NAMES, default_joint_pos_list
from supermimic.tasks.tracking.control_cfg import PDControlCfg, joint_pd_parameters
from supermimic.tasks.tracking.observation_cfg import ObservationCfg, SensorTermCfg
from supermimic.tasks.tracking.config.g1.model_cfg import G1ModelCfg


def _batch_norm_from_state(state: dict, prefix: str, history: int = 1) -> FeatureBatchNorm:
    norm = FeatureBatchNorm(state[f"{prefix}.running_mean"].numel() * history)
    buffers = {
        name: state[f"{prefix}.{name}"].detach().cpu()
        for name in norm.state_dict()
    }
    for name in ("running_mean", "running_var"):
        buffers[name] = buffers[name].repeat(history)
    norm.load_state_dict(buffers)
    return norm.eval()


def _normalizer_from_state(state, prefix, history=1):
    if f"{prefix}._mean" not in state:
        return _batch_norm_from_state(state, prefix, history)
    norm = EmpiricalNormalization(state[f"{prefix}._mean"].shape[-1])
    norm.load_state_dict({name: state[f"{prefix}.{name}"].detach().cpu() for name in norm.state_dict()})
    return norm.eval()


def _mlp_from_state(state: dict, prefix: str) -> nn.Sequential:
    indices = sorted(
        int(match.group(1))
        for key in state
        if (match := re.fullmatch(re.escape(prefix) + r"\.(\d+)\.weight", key))
    )
    if not indices or indices != list(range(0, len(indices) * 2, 2)):
        raise ValueError(f"Unsupported {prefix} layout: expected alternating Linear/activation MLP")
    layers = []
    previous_width = None
    for position, index in enumerate(indices):
        weight = state[f"{prefix}.{index}.weight"]
        bias = state.get(f"{prefix}.{index}.bias")
        if weight.ndim != 2 or bias is None or bias.shape != (weight.shape[0],):
            raise ValueError(f"Invalid {prefix}.{index} weight/bias shape")
        if previous_width is not None and weight.shape[1] != previous_width:
            raise ValueError(f"Inconsistent layer widths in {prefix}")
        layer = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.bias.copy_(bias)
        layers.append(layer)
        if position != len(indices) - 1:
            layers.append(nn.SiLU() if "goal_norm._mean" in state else nn.ELU())
        previous_width = weight.shape[0]
    return nn.Sequential(*layers)


class ExportedPolicy(nn.Module):
    """Dynamic batch, fixed history/feature widths; normalizers are included."""

    __constants__ = ["history", "obs_dim", "goal_dim", "check_inputs"]

    def __init__(self, state: dict, history: int, obs_dim: int, goal_dim: int):
        super().__init__()
        self.check_inputs = False
        self.history, self.obs_dim, self.goal_dim = history, obs_dim, goal_dim
        self.obs_norm = _normalizer_from_state(state, "obs_norm", history)
        self.goal_norm = _normalizer_from_state(state, "goal_norm")
        self.enc = _mlp_from_state(state, "enc")
        self.dec = _mlp_from_state(state, "dec")
        if self.enc[0].in_features != goal_dim:
            raise ValueError("Motion goal width does not match encoder")
        if self.dec[0].in_features != self.enc[-1].out_features + history * obs_dim:
            raise ValueError("Observation history width does not match decoder")
        if self.dec[-1].out_features != len(G1_JOINT_NAMES):
            raise ValueError("Expected 29 normalized G1 joint actions")
        expected = {f"{prefix}.{name}" for prefix, norm in
                    (("obs_norm", self.obs_norm), ("goal_norm", self.goal_norm)) for name in norm.state_dict()}
        for prefix, model in (("enc", self.enc), ("dec", self.dec)):
            expected.update(f"{prefix}.{name}" for name in model.state_dict())
        if set(state) != expected:
            raise ValueError(
                f"Unexpected policy state keys: missing={sorted(expected - set(state))}, extra={sorted(set(state) - expected)}"
            )

    def forward(
        self, obs_history: torch.Tensor, motion_goal: torch.Tensor
    ) -> torch.Tensor:
        if self.check_inputs:
            if obs_history.dim() != 2:
                raise ValueError(
                    "obs_history must have shape [batch, history * observation_dim]"
                )
            if obs_history.size(1) != self.history * self.obs_dim:
                raise ValueError(
                    "obs_history has the wrong flattened observation dimension"
                )
            if motion_goal.dim() != 2 or motion_goal.size(1) != self.goal_dim:
                raise ValueError("motion_goal must have shape [batch, goal_dim]")
            if motion_goal.size(0) != obs_history.size(0):
                raise ValueError("Input batch sizes must match")
        goal = self.goal_norm(motion_goal)
        latent = torch.tanh(self.enc(goal))
        observation = self.obs_norm(obs_history)
        return self.dec(torch.cat((latent, observation), dim=-1))


class ExportedEncoder(nn.Module):
    """Motion goal -> post-tanh latent, including the saved goal normalizer."""

    __constants__ = ["goal_dim", "latent_dim", "check_inputs"]

    def __init__(self, policy: ExportedPolicy):
        super().__init__()
        self.check_inputs = policy.check_inputs
        self.goal_dim = policy.goal_dim
        self.latent_dim = policy.enc[-1].out_features
        self.goal_norm = deepcopy(policy.goal_norm)
        self.enc = deepcopy(policy.enc)

    def forward(self, motion_goal: torch.Tensor) -> torch.Tensor:
        if self.check_inputs:
            if motion_goal.dim() != 2 or motion_goal.size(1) != self.goal_dim:
                raise ValueError("motion_goal must have shape [batch, goal_dim]")
        return torch.tanh(self.enc(self.goal_norm(motion_goal)))


class ExportedDecoder(nn.Module):
    """Prepared observation history + post-tanh latent -> normalized action."""

    __constants__ = ["history", "obs_dim", "latent_dim", "check_inputs"]

    def __init__(self, policy: ExportedPolicy):
        super().__init__()
        self.check_inputs = policy.check_inputs
        self.history, self.obs_dim = policy.history, policy.obs_dim
        self.latent_dim = policy.enc[-1].out_features
        self.obs_norm = deepcopy(policy.obs_norm)
        self.dec = deepcopy(policy.dec)

    def forward(self, obs_history: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        if self.check_inputs:
            if obs_history.dim() != 2:
                raise ValueError(
                    "obs_history must have shape [batch, history * observation_dim]"
                )
            if obs_history.size(1) != self.history * self.obs_dim:
                raise ValueError(
                    "obs_history has the wrong flattened observation dimension"
                )
            if latent.dim() != 2 or latent.size(1) != self.latent_dim:
                raise ValueError("latent must have shape [batch, latent_dim]")
            if latent.size(0) != obs_history.size(0):
                raise ValueError("Input batch sizes must match")
        observation = self.obs_norm(obs_history)
        return self.dec(torch.cat((latent, observation), dim=-1))


def _positive_int(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _observations_from_saved(values):
    values = deepcopy(values)
    expected = {field.name for field in fields(ObservationCfg)}
    if set(values) - expected:
        raise ValueError("Unknown saved observation settings")
    for name, _, _ in ObservationCfg().terms():
        if name in values and values[name] is not None:
            values[name] = SensorTermCfg(**values[name])
    config = ObservationCfg(**values)
    config.validate()
    return config


def load_export_inputs(
    checkpoint_path: Path, task: str = "g1", *, checkpoint_data: dict | None = None
):
    """Read a checkpoint and its portable semantics without loading simulator code."""
    if task not in ("volleyball", "g1"):
        raise ValueError("task must be volleyball or g1")
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint_data is None else checkpoint_data
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("pi"), dict):
        raise ValueError(
            "Expected a SuperMimic training checkpoint containing pi and args"
        )
    arguments = checkpoint.get("args", {})
    history = _positive_int(arguments.get("hist"), "checkpoint args.hist")
    goal_frames = _positive_int(
        arguments.get("goal_frames"), "checkpoint args.goal_frames"
    )
    state = checkpoint["pi"]
    if "obs_norm.mean" in state or "goal_norm.mean" in state:
        raise ValueError(
            "Legacy RunningNorm checkpoint is unsupported by this exporter; use its matching exporter or train a v7 policy"
        )
    empirical = checkpoint.get("model_input_version", 0) >= 7
    mean_key, var_key = ("_mean", "_var") if empirical else ("running_mean", "running_var")
    for prefix in ("obs_norm", "goal_norm"):
        mean, var = state.get(f"{prefix}.{mean_key}"), state.get(f"{prefix}.{var_key}")
        if not isinstance(mean, torch.Tensor) or mean.ndim != (2 if empirical else 1):
            raise ValueError(f"Invalid or missing {prefix}.{mean_key}")
        if not isinstance(var, torch.Tensor) or var.shape != mean.shape or (var < 0).any():
            raise ValueError(f"Invalid {prefix} variance")
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Policy tensor is missing or nonfinite: {key}")
    obs_width = state[f"obs_norm.{mean_key}"].numel()
    if empirical and obs_width % history:
        raise ValueError("Observation normalizer width must be divisible by history")
    obs_dim = obs_width // history if empirical else obs_width
    goal_dim = checkpoint_goal_dim(checkpoint)
    orientation_frames = 0 if empirical else (goal_dim - GOAL_FRAME_DIM * goal_frames) // 6
    includes_current_orientation = orientation_frames > 0
    if state[f"goal_norm.{mean_key}"].numel() != goal_dim:
        raise ValueError("Checkpoint goal normalizer disagrees with args.goal_frames")

    saved = checkpoint.get("task_settings")
    source = "checkpoint"
    companion = checkpoint_path.parent / "task_settings.json"
    if saved is None and companion.is_file():
        saved = json.loads(companion.read_text())
        source = "task_settings.json"
    if saved is not None:
        if not isinstance(saved, dict) or saved.get("version") != 1:
            raise ValueError("Unsupported task_settings version")
        observation = _observations_from_saved(saved["observations"])
        control = PDControlCfg(**saved["control"])
        if observation.history != history or saved["goal_frames"] != goal_frames:
            raise ValueError(
                "Saved task history/goal frames disagree with checkpoint args"
            )
        model = deepcopy(saved["model"])
        self_collisions = saved.get("self_collisions")
        joint_offset = (
            saved.get("randomization", {}).get("joint_offsets", {}).get("nominal", 0.0)
        )
    else:
        source = "legacy_defaults"
        observation = ObservationCfg(history=history)
        # Metadata-free old checkpoints used the original 99-feature layout.
        if obs_dim == 99:
            observation.reference_orientation = SensorTermCfg(noise=0.05)
        control = PDControlCfg()
        model = asdict(G1ModelCfg())
        # An old checkpoint does not identify its collision setting reliably.
        self_collisions = None
        joint_offset = 0.0
    if observation.dim != obs_dim:
        raise ValueError(
            "Saved observation layout disagrees with policy width; checkpoints without settings must use 93 or 99 features"
        )
    gains = joint_pd_parameters(control)
    with torch.random.fork_rng(devices=[]):
        model_nn = ExportedPolicy(state, history, obs_dim, goal_dim).eval()
    for parameter in model_nn.parameters():
        parameter.requires_grad_(False)
    sources = {
        "joint_pos": "joint_position - default_joint_position - joint_calibration_offset",
        "joint_vel": "joint_velocity",
        "base_ang_vel": "transpose(robot_root_rotation) @ world_root_angular_velocity",
        "projected_gravity": "-robot_root_rotation[2, :]",
        "previous_action": "previous executed normalized joint action",
        "reference_orientation": "mat_to_6d(transpose(robot_root_rotation) @ reference_root_rotation[current_frame])",
    }
    layout = []
    offset = 0
    for name, width, term in observation.terms():
        if term is not None and term.enabled:
            layout.append(
                {
                    "name": name,
                    "offset": offset,
                    "width": width,
                    "scale": term.scale,
                    "raw_noise_half_width": term.noise,
                    "unscaled_source": sources[name],
                }
            )
            offset += width
    metadata = {
        "schema_version": 4,
        "format": "torchscript",
        "task": task,
        "checkpoint": str(checkpoint_path),
        "checkpoint_iteration": checkpoint.get("iteration"),
        "checkpoint_model_input_version": checkpoint.get("model_input_version"),
        "settings_source": source,
        "precision": "float32",
        "normalization_type": "BatchNorm1d",
        "normalizer_epsilon": 1e-4,
        "normalizer_epsilon_convention": "sqrt(running_var + epsilon)",
        "normalizer_momentum": 0.1,
        "checkpoint_amp": arguments.get("amp", "none"),
        "control_hz": 50,
        "control_dt": 0.02,
        "inputs": {
            "obs_history": {
                "shape": ["batch", history * obs_dim],
                "history": history,
                "observation_dim": obs_dim,
                "flatten_order": "by_frames; index = frame * observation_dim + feature",
                "dtype": "float32",
                "order": "oldest_to_newest",
                "normalization": "included; pass MDP.obs-scaled features without additional normalization",
            },
            "motion_goal": {
                "shape": ["batch", goal_dim],
                "dtype": "float32",
                "goal_frames": goal_frames,
                "future_frame_dim": GOAL_FRAME_DIM,
                "includes_current_reference_orientation": includes_current_orientation,
                "reference_orientation_history": orientation_frames,
                "normalization": "included; pass MotionLib.goal features without additional normalization",
            },
        },
        "output": {
            "shape": ["batch", 29],
            "dtype": "float32",
            "meaning": "normalized joint action; no exploration noise or clipping",
        },
        "observation_settings": asdict(observation),
        "observation_layout": layout,
        "goal_layout_per_frame": [
            {
                "name": "future_root_displacement_in_current_reference_heading",
                "offset": 0,
                "width": 3,
            },
            {
                "name": "future_root_orientation_6d_in_current_reference_heading",
                "offset": 3,
                "width": 6,
            },
            {
                "name": "future_root_linear_velocity_in_future_root_frame",
                "offset": 9,
                "width": 3,
            },
            {
                "name": "future_root_angular_velocity_in_future_root_frame",
                "offset": 12,
                "width": 3,
            },
            {"name": "future_joint_position", "offset": 15, "width": 29},
            {"name": "future_joint_velocity_times_0.1", "offset": 44, "width": 29},
        ],
        "goal_order": "future reference frames f+1 through f+K; hold final pose and zero synthetic velocities after clip end",
        "goal_layout_suffix": ([{
            "name": "reference_orientation_history" if orientation_frames > 1 else "current_reference_orientation_in_current_robot_frame",
            "offset": GOAL_FRAME_DIM * goal_frames,
            "width": 6 * orientation_frames,
            "history": orientation_frames,
            "frame_width": 6,
            "order": "oldest_to_newest",
            "source": "For each past/current sample: mat_to_6d(transpose(robot_root_rotation_at_sample) @ reference_root_rotation[reference_frame_at_sample])",
        }] if includes_current_orientation else []),
        "rotation_6d": "first two matrix columns, row-major flatten as in rotations.mat_to_6d",
        "joint_names": list(G1_JOINT_NAMES),
        "default_joint_position": default_joint_pos_list(),
        "nominal_joint_offset": joint_offset,
        "action_scale": [gains[name][2] for name in G1_JOINT_NAMES],
        "nominal_kp": [gains[name][0] for name in G1_JOINT_NAMES],
        "nominal_kd": [gains[name][1] for name in G1_JOINT_NAMES],
        "action_to_target": "q_target = default_joint_position + joint_offset + action_scale * action",
        "model": model,
        "checkpoint_self_collisions": self_collisions,
        "task_settings": deepcopy(saved),
        "caller_responsibilities": [
            "Build the observation terms, applying configured scales before calling this module.",
            "Maintain oldest-to-newest history; on reset repeat the first observation across the history.",
            "Flatten [batch, history, observation_dim] to [batch, history * observation_dim] before calling the policy.",
            "Build the future motion goal using MotionLib.goal conventions.",
            "Apply joint targets through an external PD controller with matching joint order and gains.",
            "The export is a policy network, not a standalone simulator or complete hardware controller.",
        ],
    }
    if includes_current_orientation:
        metadata["goal_order"] += "; append reference orientation history oldest-to-newest after the complete future sequence"
        metadata["caller_responsibilities"].append(
            "Maintain goal_layout_suffix.history relative-orientation samples, each using that sample's robot/reference rotations; on reset repeat the initial sample. Flatten by time and append after the future reference sequence."
        )
    if empirical:
        anchor_obs = arguments.get("anchor_obs", True)
        metadata.update(schema_version=5, normalization_type="EmpiricalNormalization",
                        normalizer_epsilon=0.01, normalizer_epsilon_convention="sqrt(variance) + epsilon",
                        normalizer_momentum=None, anchor_obs=anchor_obs, activation="SiLU",
                        goal_order="frames f through f+K-1; stop tracking when f+K-1 reaches the last reference frame",
                        clip_end_rule="current_frame + goal_frames - 1 >= last_reference_frame")
        metadata["inputs"]["obs_history"]["flatten_order"] = "term-major: all history of joint_pos, then joint_vel, base_ang_vel, gravity, previous_action; oldest first within each term"
        metadata["inputs"]["motion_goal"].update(future_frame_dim=68, anchor_obs=anchor_obs)
        metadata["goal_layout_per_frame"] = [dict(name=name, offset=offset, width=width) for name,offset,width in (
            ("joint_position_minus_default",0,29), ("joint_velocity",29,29),
            ("projected_gravity_body",58,3), ("angular_velocity_body",61,3),
            ("linear_velocity_heading",64,3), ("root_height",67,1))]
        metadata["goal_layout_suffix"] = ([dict(name="anchor_obs", offset=68*goal_frames, width=9,
            anchor_body="torso_link", source="reference torso position (3) and orientation columns (6) in current robot torso frame")]
            if anchor_obs else [])
        for term in metadata["observation_layout"]:
            if term["name"] == "joint_pos":
                term["unscaled_source"] = "joint_position + joint_calibration_offset - default_joint_position"
        metadata["caller_responsibilities"][2] = "Flatten history by observation term using MDP.flatten_history()."
        metadata["caller_responsibilities"][3] = "Build 68*K reference features using MotionLib.goal(); append MDP.anchor_observation() when anchor_obs is true."
    return model_nn, metadata


def _component_metadata(metadata, name, policy, fingerprint):
    result = deepcopy(metadata)
    latent = {
        "shape": ["batch", policy.enc[-1].out_features],
        "dtype": "float32",
        "meaning": "post-tanh latent z; do not normalize or apply tanh again",
    }
    result["component"] = name
    result["policy_fingerprint"] = fingerprint
    result["latent_dim"] = policy.enc[-1].out_features
    if name == "encoder":
        result["inputs"] = {"motion_goal": deepcopy(metadata["inputs"]["motion_goal"])}
        result["output"] = latent
        result["call_signature"] = "latent = encoder(motion_goal)"
        result["included_operations"] = ["goal normalization", "motion encoder", "tanh"]
        result["caller_responsibilities"] = [
            "Build future motion goals using MotionLib.goal conventions.",
            "Include goal_layout_suffix when present; each history sample uses its own past/current robot and reference, not future reference frames.",
            "Pass the returned latent directly to a matching decoder; do not normalize or tanh again.",
            "Use components with the same policy_fingerprint.",
        ]
    elif name == "decoder":
        result["inputs"] = {
            "obs_history": deepcopy(metadata["inputs"]["obs_history"]),
            "latent": latent,
        }
        result["call_signature"] = "actions = decoder(obs_history, latent)"
        result["included_operations"] = [
            "observation normalization",
            "decoder",
        ]
        result["caller_responsibilities"] = [
            "Build scaled observation terms and flatten oldest-to-newest history by frames before calling the decoder.",
            "Supply the post-tanh latent from a matching encoder (same policy_fingerprint).",
            "Do not repeat observation normalization; it is included.",
            "Apply normalized actions using the external PD controller contract in this metadata.",
        ]
    else:
        result["call_signature"] = "actions = policy(obs_history, motion_goal)"
        result["included_operations"] = [
            "goal normalization",
            "motion encoder",
            "tanh",
            "observation normalization",
            "decoder",
        ]
    return result


def _fingerprint(policy):
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _export_components(checkpoint_path, outputs, task, check_inputs=False, checkpoint_data=None):
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    paths = {}
    for name, value in outputs.items():
        if name not in ("policy", "encoder", "decoder"):
            raise ValueError("component must be policy, encoder or decoder")
        output = Path(value).expanduser().resolve()
        paths[name] = (output, output.with_suffix(output.suffix + ".json"))
    destinations = [path for pair in paths.values() for path in pair]
    if len(set(destinations)) != len(destinations):
        raise ValueError("Export destinations must be distinct")
    if checkpoint_path in destinations:
        raise ValueError("Export output must not overwrite its source checkpoint")
    if any(path.is_dir() for path in destinations):
        raise IsADirectoryError("Export destinations must be files")
    policy, metadata = load_export_inputs(checkpoint_path, task, checkpoint_data=checkpoint_data)
    # A TorchScript constant removes the whole validation branch when disabled.
    policy.check_inputs = check_inputs
    metadata["input_shape_checks"] = check_inputs
    modules = {"policy": policy}
    if "encoder" in paths:
        modules["encoder"] = ExportedEncoder(policy).eval()
    if "decoder" in paths:
        modules["decoder"] = ExportedDecoder(policy).eval()
    fingerprint = _fingerprint(policy)
    parent = next(iter(paths.values()))[0].parent
    if any(path.parent != parent for path in destinations):
        raise ValueError("A component bundle must use one output directory")
    parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu").manual_seed(2718)
    cases = [
        (
            torch.randn(batch, policy.history * policy.obs_dim, generator=generator),
            torch.randn(batch, policy.goal_dim, generator=generator),
            torch.rand(batch, policy.enc[-1].out_features, generator=generator) * 2 - 1,
        )
        for batch in (1, 3)
    ]
    with tempfile.TemporaryDirectory(prefix=".policy-export-", dir=parent) as temporary:
        staged_files, loaded, reports = {}, {}, {}
        for name, (output, metadata_path) in paths.items():
            staged = Path(temporary) / output.name
            torch.jit.save(torch.jit.script(modules[name]), str(staged))
            loaded[name] = torch.jit.load(str(staged), map_location="cpu").eval()
            errors = []
            for obs, goal, arbitrary_latent in cases:
                if name == "encoder":
                    expected = torch.tanh(policy.enc(policy.goal_norm(goal)))
                    actual = loaded[name](goal)
                elif name == "decoder":
                    expected = policy.dec(
                        torch.cat(
                            (arbitrary_latent, policy.obs_norm(obs)), -1
                        )
                    )
                    actual = loaded[name](obs, arbitrary_latent)
                else:
                    expected, actual = policy(obs, goal), loaded[name](obs, goal)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                errors.append(float((actual - expected).abs().max()))
            report = _component_metadata(metadata, name, policy, fingerprint)
            report["validation"] = {
                "batch_sizes": [1, 3],
                "max_absolute_error": max(errors),
                "reference": "FP32 eager inference, not BF16 CUDA output",
            }
            report["artifact_sha256"] = hashlib.sha256(staged.read_bytes()).hexdigest()
            reports[name] = report
            staged_files[output] = staged
        if "encoder" in loaded and "decoder" in loaded:
            errors = []
            for obs, goal, _ in cases:
                actual = loaded["decoder"](obs, loaded["encoder"](goal))
                expected = (
                    loaded["policy"](obs, goal)
                    if "policy" in loaded
                    else policy(obs, goal)
                )
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                errors.append(float((actual - expected).abs().max()))
            for report in reports.values():
                report["validation"]["composition_max_absolute_error"] = max(errors)
        for name, (_, metadata_path) in paths.items():
            staged = Path(temporary) / metadata_path.name
            staged.write_text(
                json.dumps(reports[name], indent=2, ensure_ascii=False) + "\n"
            )
            staged_files[metadata_path] = staged
        # Validate the entire bundle before replacing any file. Keep backups so
        # a failed replacement can restore the previous bundle.
        backups = {}
        for index, destination in enumerate(staged_files):
            if destination.exists():
                backup = Path(temporary) / f".backup-{index}"
                os.link(destination, backup)
                backups[destination] = backup
        replaced = []
        try:
            for destination, staged in staged_files.items():
                os.replace(staged, destination)
                replaced.append(destination)
        except OSError:
            for destination in reversed(replaced):
                if destination in backups:
                    os.replace(backups[destination], destination)
                else:
                    destination.unlink()
            raise
    return paths


def _default_output(checkpoint_path: Path) -> Path:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    # Keep exports outside the ckpt_*.pt pattern used by play/evaluation watchers.
    return checkpoint.parent / "export" / f"policy_{checkpoint.stem}.pt"


def export_policy(
    checkpoint_path: Path,
    output_path: Path | None = None,
    task: str = "g1",
    component: str = "policy",
    *,
    check_inputs: bool = False,
):
    """Export one component, omitting runtime shape checks unless requested."""
    if output_path is None:
        output_path = _default_output(checkpoint_path)
        if component != "policy":
            output_path = output_path.with_name(
                output_path.stem + f"_{component}" + output_path.suffix
            )
    return _export_components(
        checkpoint_path, {component: output_path}, task, check_inputs
    )[component]


def export_all(
    checkpoint_path: Path,
    output_path: Path | None = None,
    task: str = "g1",
    *,
    check_inputs: bool = False,
    checkpoint_data: dict | None = None,
):
    """Export a bundle; an optional loaded snapshot keeps exports aligned with play."""
    output = _default_output(checkpoint_path) if output_path is None else Path(output_path)
    return _export_components(
        checkpoint_path,
        {
            "policy": output,
            "encoder": output.with_name(output.stem + "_encoder" + output.suffix),
            "decoder": output.with_name(output.stem + "_decoder" + output.suffix),
        },
        task,
        check_inputs,
        checkpoint_data,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", "--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="TorchScript file, overwritten if present (default: export/policy_<checkpoint stem>.pt beside the checkpoint)",
    )
    parser.add_argument(
        "--task",
        choices=("volleyball", "g1"),
        default="g1",
        help="Model metadata fallback for legacy checkpoints without task_settings",
    )
    parser.add_argument(
        "--component",
        choices=("policy", "encoder", "decoder", "all"),
        default="all",
        help="Export the full policy, one standalone component, or all three (default: all)",
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Include runtime input shape checks and If branches (default: omitted)",
    )
    args = parser.parse_args(argv)
    outputs = (
        export_all(args.ckpt, args.out, args.task, check_inputs=args.check_inputs)
        if args.component == "all"
        else {
            args.component: export_policy(
                args.ckpt,
                args.out,
                args.task,
                args.component,
                check_inputs=args.check_inputs,
            )
        }
    )
    for name, (output, metadata) in outputs.items():
        print(f"TorchScript {name}: {output}\nInput/PD metadata: {metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
