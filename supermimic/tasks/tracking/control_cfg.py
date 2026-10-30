"""Nominal G1 PD gains and action scaling, independent of simulator imports."""

from dataclasses import dataclass, field
import math
import re

from supermimic.robot_spec import G1_JOINT_NAMES, motor_of


@dataclass
class PDControlCfg:
    natural_frequency_hz: float = 10.0
    damping_ratio: float = 2.0
    stiffness_scale: float = 1.0
    damping_scale: float = 1.0
    action_scale_multiplier: float = 0.25
    # Absolute gains after nominal scaling. Regexes use fullmatch; last match wins.
    stiffness_overrides: dict[str, float] = field(default_factory=dict)
    damping_overrides: dict[str, float] = field(default_factory=dict)

    def validate(self):
        for name in (
            "natural_frequency_hz",
            "stiffness_scale",
            "action_scale_multiplier",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"control.{name} must be finite and positive")
        for name in ("damping_ratio", "damping_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"control.{name} must be finite and nonnegative")
        for name in ("stiffness_overrides", "damping_overrides"):
            for pattern, value in getattr(self, name).items():
                if not any(re.fullmatch(pattern, joint) for joint in G1_JOINT_NAMES):
                    raise ValueError(
                        f"control.{name} regex matches no G1 joints: {pattern!r}"
                    )
                minimum_ok = value > 0 if name == "stiffness_overrides" else value >= 0
                if not math.isfinite(value) or not minimum_ok:
                    raise ValueError(
                        f"Invalid control.{name} gain for {pattern!r}: {value}"
                    )


def joint_pd_parameters(cfg: PDControlCfg) -> dict[str, tuple[float, float, float]]:
    """Map each SDK-order joint to (Kp, Kd, action scale).

    Action scales use the configured nominal Kp, never the per-episode random Kp.
    Keep the baseline pi constant and arithmetic order for bitwise default parity.
    """
    cfg.validate()
    omega = cfg.natural_frequency_hz * 2.0 * 3.1415926535
    result = {}
    for name in G1_JOINT_NAMES:
        effort, _, armature = motor_of(name)
        kp = armature * omega**2 * cfg.stiffness_scale
        kd = 2.0 * cfg.damping_ratio * armature * omega * cfg.damping_scale
        for pattern, gain in cfg.stiffness_overrides.items():
            if re.fullmatch(pattern, name):
                kp = gain
        for pattern, gain in cfg.damping_overrides.items():
            if re.fullmatch(pattern, name):
                kd = gain
        result[name] = kp, kd, cfg.action_scale_multiplier * effort / kp
    return result
