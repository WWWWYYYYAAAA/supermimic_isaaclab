"""Plain-data task snapshots so evaluation uses the configuration that trained a policy."""

from dataclasses import asdict, fields, is_dataclass
from copy import deepcopy
import json
from pathlib import Path

from supermimic.paths import REPO
from .control_cfg import PDControlCfg
from .randomization_cfg import RandomizationCfg
from .reward_cfg import RewardCfg
from .observation_cfg import ObservationCfg, SensorTermCfg, StateObservationCfg


SECTIONS = {
    "control": ("control", PDControlCfg),
    "randomization": ("randomization", RandomizationCfg),
    "reward": ("analytic_reward", RewardCfg),
    "observations": ("policy_observation", ObservationCfg),
    "critic_observations": ("critic_observation", StateObservationCfg),
    "world_model_observations": ("world_model_observation", StateObservationCfg),
}


def _portable(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def task_settings(cfg):
    """Serialize only supported knobs; no callables or simulator objects in checkpoints."""
    return {
        "version": 1,
        **{name: asdict(getattr(cfg, attr)) for name, (attr, _) in SECTIONS.items()},
        "model": {
            name: _portable(value) for name, value in asdict(cfg.robot_model).items()
        },
        "self_collisions": cfg.self_collisions,
        "episode_length_s": cfg.episode_length_s,
        "goal_frames": cfg.commands.motion.goal_frames,
    }


def _restore(instance, values):
    expected = {field.name for field in fields(instance)}
    unknown = set(values) - expected
    if unknown:
        raise ValueError(
            f"Unknown {type(instance).__name__} settings: {sorted(unknown)}"
        )

    def tuple_value(value):
        return (
            tuple(tuple_value(v) for v in value)
            if isinstance(value, (list, tuple))
            else value
        )

    for name, value in values.items():
        current = getattr(instance, name)
        if value is None:
            setattr(instance, name, None)
        elif isinstance(instance, ObservationCfg) and name == "reference_orientation":
            setattr(instance, name, _restore(SensorTermCfg(), value))
        elif is_dataclass(current):
            _restore(current, value)
        else:
            setattr(
                instance,
                name,
                tuple_value(value) if isinstance(current, tuple) else deepcopy(value),
            )
    return instance


def apply_task_settings(cfg, saved):
    if saved.get("version") != 1:
        raise ValueError(f"Unsupported task settings version: {saved.get('version')}")
    for name, (attr, factory) in SECTIONS.items():
        values = saved.get(name, {}) if name in ("critic_observations", "world_model_observations") else saved[name]
        setattr(cfg, attr, _restore(factory(), values))
    for name, value in saved["model"].items():
        if name not in ("urdf", "mjcf", "usd"):
            raise ValueError(f"Unknown model asset {name}")
        path = Path(value)
        setattr(cfg.robot_model, name, str(path if path.is_absolute() else REPO / path))
    cfg.self_collisions = saved["self_collisions"]
    cfg.episode_length_s = saved["episode_length_s"]
    cfg.commands.motion.goal_frames = saved["goal_frames"]
    return cfg.prepare()


def save_task_settings(log_dir, settings):
    path = Path(log_dir) / "task_settings.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(settings, indent=2) + "\n")
    temporary.replace(path)


def load_task_settings(run_dir, checkpoint=None):
    """Checkpoint snapshot takes precedence over the run's companion JSON."""
    if checkpoint:
        import torch

        saved = torch.load(checkpoint, map_location="cpu").get("task_settings")
        if saved is not None:
            return saved
    path = Path(run_dir) / "task_settings.json"
    return json.loads(path.read_text()) if path.is_file() else None


def same_settings(left, right):
    # Ranges are tuple/list-equivalent; regex override order is meaningful.
    def canonical(value):
        value = deepcopy(value)
        if isinstance(value, dict) and "control" in value:
            # Older policy-only evaluation snapshots did not describe these groups.
            for name in ("critic_observations", "world_model_observations"):
                value.setdefault(name, asdict(StateObservationCfg()))
            for name in ("stiffness_overrides", "damping_overrides"):
                overrides = value["control"].get(name, {})
                value["control"][name] = list(overrides.items())
        return json.dumps(value, sort_keys=True)

    return canonical(left) == canonical(right)
