"""Analytic reward/termination parameters shared by real and imagined steps."""

from dataclasses import dataclass
import math


@dataclass
class RewardCfg:
    # Tracking terms: weight * exp(-squared_error / std**2).
    w_body_pos: float = 1.0
    w_body_ori: float = 1.0
    w_body_linvel: float = 1.0
    w_body_angvel: float = 1.0
    w_root_pos: float = 0.5
    w_root_ori: float = 0.5
    body_pos_std: float = 0.3
    body_ori_std: float = 0.4
    body_linvel_std: float = 1.0
    body_angvel_std: float = 3.14
    root_pos_std: float = 0.3
    root_ori_std: float = 0.4
    # Regularizers are penalties. Set a weight to zero to disable its contribution.
    w_action_rate: float = -0.1
    w_joint_limit: float = -2.0
    soft_limit_factor: float = 0.9
    # Hard termination thresholds and differentiable survival temperatures.
    term_height: float = 0.25
    term_tilt: float = 0.8
    soft_term_height_temperature: float = 0.02
    soft_term_tilt_temperature: float = 0.04

    @property
    def max_reward_rate(self) -> float:
        """Ideal tracking upper bound, before control-dt integration."""
        return (
            self.w_body_pos
            + self.w_body_ori
            + self.w_body_linvel
            + self.w_body_angvel
            + self.w_root_pos
            + self.w_root_ori
        )

    def validate(self) -> None:
        for name in (
            "w_body_pos",
            "w_body_ori",
            "w_body_linvel",
            "w_body_angvel",
            "w_root_pos",
            "w_root_ori",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"reward.{name} must be finite and nonnegative")
        if self.max_reward_rate <= 0:
            raise ValueError("at least one tracking reward weight must be positive")
        for name in ("w_action_rate", "w_joint_limit"):
            value = getattr(self, name)
            if not math.isfinite(value) or value > 0:
                raise ValueError(f"reward.{name} must be finite and nonpositive")
        for name in (
            "body_pos_std",
            "body_ori_std",
            "body_linvel_std",
            "body_angvel_std",
            "root_pos_std",
            "root_ori_std",
            "term_height",
            "term_tilt",
            "soft_term_height_temperature",
            "soft_term_tilt_temperature",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"reward.{name} must be finite and positive")
        if (
            not math.isfinite(self.soft_limit_factor)
            or not 0 < self.soft_limit_factor <= 1
        ):
            raise ValueError("reward.soft_limit_factor must be in (0, 1]")
