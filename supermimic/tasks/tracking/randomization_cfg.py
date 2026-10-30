"""Physical context and reset distributions shared by configuration and runtime."""

from dataclasses import dataclass, field
import math
from typing import Literal

Mode = Literal["startup", "reset"]
Range = tuple[float, float]
VectorRanges = tuple[Range, Range, Range]


def _range(value, name, *, positive=False, nonnegative=False):
    if (
        len(value) != 2
        or not all(math.isfinite(v) for v in value)
        or value[0] > value[1]
    ):
        raise ValueError(f"{name} must be a finite ordered (lower, upper) pair")
    if positive and value[0] <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value[0] < 0:
        raise ValueError(f"{name} must be nonnegative")


def _mode(value, name):
    if value not in ("startup", "reset"):
        raise ValueError(f"{name} must be startup or reset")


@dataclass
class ScalarRandomizationCfg:
    enabled: bool = True
    range: Range = (0.0, 0.0)
    mode: Mode = "startup"
    nominal: float = 0.0

    def validate(self, name, *, nonnegative=False):
        _range(self.range, f"{name}.range", nonnegative=nonnegative)
        _mode(self.mode, f"{name}.mode")
        if not math.isfinite(self.nominal) or (nonnegative and self.nominal < 0):
            raise ValueError(f"{name}.nominal is invalid")


@dataclass
class VectorRandomizationCfg:
    enabled: bool = True
    ranges: VectorRanges = ((0.0, 0.0),) * 3
    mode: Mode = "startup"
    nominal: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self, name):
        if len(self.ranges) != 3 or len(self.nominal) != 3:
            raise ValueError(f"{name} requires three XYZ ranges and nominal values")
        for axis, bounds in zip("xyz", self.ranges):
            _range(bounds, f"{name}.ranges.{axis}")
        _mode(self.mode, f"{name}.mode")
        if not all(math.isfinite(v) for v in self.nominal):
            raise ValueError(f"{name}.nominal must be finite")


@dataclass
class ResetNoiseCfg:
    enabled: bool = True
    root_position: VectorRanges = ((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01))
    root_rotation: VectorRanges = ((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2))
    root_linear_velocity: VectorRanges = ((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2))
    root_angular_velocity: VectorRanges = ((-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78))
    joint_position: Range = (-0.1, 0.1)

    def validate(self):
        for name in (
            "root_position",
            "root_rotation",
            "root_linear_velocity",
            "root_angular_velocity",
        ):
            ranges = getattr(self, name)
            if len(ranges) != 3:
                raise ValueError(
                    f"randomization.reset.{name} requires three XYZ ranges"
                )
            for axis, bounds in zip("xyz", ranges):
                _range(bounds, f"randomization.reset.{name}.{axis}")
        _range(self.joint_position, "randomization.reset.joint_position")


@dataclass
class PDGainRandomizationCfg:
    enabled: bool = False
    kp_scale: Range = (0.8, 1.2)
    kd_scale: Range = (0.8, 1.2)
    mode: Mode = "reset"

    def validate(self):
        _range(self.kp_scale, "randomization.pd_gains.kp_scale", positive=True)
        _range(self.kd_scale, "randomization.pd_gains.kd_scale", nonnegative=True)
        _mode(self.mode, "randomization.pd_gains.mode")


@dataclass
class RandomizationCfg:
    friction: ScalarRandomizationCfg = field(
        default_factory=lambda: ScalarRandomizationCfg(range=(0.3, 1.2), nominal=1.0)
    )
    joint_offsets: ScalarRandomizationCfg = field(
        default_factory=lambda: ScalarRandomizationCfg(
            range=(-0.01, 0.01), mode="reset"
        )
    )
    torso_com: VectorRandomizationCfg = field(
        default_factory=lambda: VectorRandomizationCfg(
            ranges=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05))
        )
    )
    reset: ResetNoiseCfg = field(default_factory=ResetNoiseCfg)
    pd_gains: PDGainRandomizationCfg = field(default_factory=PDGainRandomizationCfg)

    @property
    def context_dim(self):
        # 33 baseline + 29 Kp factors + 29 Kd factors when gain randomization is on.
        return 91 if self.pd_gains.enabled else 33

    def validate(self):
        self.friction.validate("randomization.friction", nonnegative=True)
        self.joint_offsets.validate("randomization.joint_offsets")
        self.torso_com.validate("randomization.torso_com")
        self.reset.validate()
        self.pd_gains.validate()
