"""Policy sensor layout shared by simulation and imagined rollouts.

Noise is a uniform half-width in the sensor's raw units. Scaling is applied to
both the clean signal and its noise. Defaults match the paper's 93 values;
the legacy OBS_SCALE constant was not applied and is not enabled here.
"""

from dataclasses import dataclass, field
import math


@dataclass
class SensorTermCfg:
    enabled: bool = True
    scale: float = 1.0
    noise: float = 0.0

    def validate(self, name: str) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"observation.{name}.enabled must be bool")
        if not math.isfinite(self.scale):
            raise ValueError(f"observation.{name}.scale must be finite")
        if not math.isfinite(self.noise) or self.noise < 0:
            raise ValueError(f"observation.{name}.noise must be finite and nonnegative")


@dataclass
class ObservationCfg:
    joint_pos: SensorTermCfg = field(default_factory=lambda: SensorTermCfg(noise=0.01))
    joint_vel: SensorTermCfg = field(default_factory=lambda: SensorTermCfg(noise=0.5))
    base_ang_vel: SensorTermCfg = field(
        default_factory=lambda: SensorTermCfg(noise=0.2)
    )
    projected_gravity: SensorTermCfg = field(
        default_factory=lambda: SensorTermCfg(noise=0.05)
    )
    previous_action: SensorTermCfg = field(default_factory=SensorTermCfg)
    # Optional only for reading historical actor checkpoints; new tasks use None.
    reference_orientation: SensorTermCfg | None = None
    noise_enabled: bool = True
    history: int = 10

    def terms(self):
        """Stable concatenation order; changing enabled terms changes policy width."""
        return (
            ("joint_pos", 29, self.joint_pos),
            ("joint_vel", 29, self.joint_vel),
            ("base_ang_vel", 3, self.base_ang_vel),
            ("projected_gravity", 3, self.projected_gravity),
            ("previous_action", 29, self.previous_action),
            ("reference_orientation", 6, self.reference_orientation),
        )

    @property
    def dim(self) -> int:
        return sum(width for _, width, term in self.terms() if term is not None and term.enabled)

    def validate(self) -> None:
        if not isinstance(self.noise_enabled, bool):
            raise ValueError("observation.noise_enabled must be bool")
        if (
            not isinstance(self.history, int)
            or isinstance(self.history, bool)
            or self.history < 1
        ):
            raise ValueError("observation.history must be a positive integer")
        for name, _, term in self.terms():
            if name == "reference_orientation" and term is None:
                continue
            if not isinstance(term, SensorTermCfg):
                raise ValueError(f"observation.{name} must be SensorTermCfg")
            term.validate(name)
        if self.dim == 0:
            raise ValueError("at least one observation term must be enabled")


@dataclass
class StateObservationCfg:
    """Privileged networks use full state, without reference-derived observations."""

    reference_orientation: None = None

    def validate(self):
        if self.reference_orientation is not None:
            raise ValueError("reference_orientation must be None for critic/world-model inputs")
