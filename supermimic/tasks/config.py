"""Compatibility exports; edit the task hierarchy under tasks/tracking instead."""

from .tracking.tracking_env_cfg import (
    ActionsCfg as ActionsCfg,
    CommandsCfg as CommandsCfg,
    ObservationsCfg as ObservationsCfg,
    RewardsCfg as RewardsCfg,
    TerminationsCfg as TerminationsCfg,
    EventsCfg as EventsCfg,
    make_sim as make_sim,
)
from .tracking.config.g1.flat_env_cfg import (
    G1FlatEnvCfg as TrackingEnvCfg,
    G1SceneCfg as TrackingSceneCfg,
)

__all__ = [
    "TrackingEnvCfg",
    "TrackingSceneCfg",
    "ActionsCfg",
    "CommandsCfg",
    "ObservationsCfg",
    "RewardsCfg",
    "TerminationsCfg",
    "EventsCfg",
    "make_sim",
]
