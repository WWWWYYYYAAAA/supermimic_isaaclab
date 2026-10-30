"""Manager configuration for G1 tracking on the pinned Isaac Lab 3 Newton backend."""

import isaaclab.sim as sim_utils
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)
from dataclasses import MISSING
from isaaclab.utils import configclass
from supermimic.motion import CONTROL_DT
from .mdp.actions import TrackingActionCfg
from .mdp.commands import ReferenceMotionCommandCfg
from .mdp import events, terms
from .reward_cfg import RewardCfg
from .observation_cfg import ObservationCfg, StateObservationCfg
from .control_cfg import PDControlCfg
from .randomization_cfg import RandomizationCfg


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = MISSING


def make_sim(device: str, args):
    physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=500,
            nconmax=128,
            cone="pyramidal",
            integrator="implicitfast",
            use_mujoco_contacts=True,
        ),
        num_substeps=4,
    )
    return sim_utils.SimulationCfg(dt=CONTROL_DT, device=device, physics=physics)


@configclass
class ActionsCfg:
    joints = TrackingActionCfg()


@configclass
class CommandsCfg:
    motion = ReferenceMotionCommandCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        proprioception = ObservationTermCfg(func=terms.proprioception)
        enable_corruption = False
        concatenate_terms = True

    policy = PolicyCfg()


@configclass
class RewardsCfg:
    # MDP.reward_rate is unscaled; RewardManager applies CONTROL_DT exactly once.
    tracking = RewardTermCfg(func=terms.reward_rate, weight=1.0)


@configclass
class TerminationsCfg:
    tracking_failure = TerminationTermCfg(func=terms.tracking_failure)
    clip_or_time_limit = TerminationTermCfg(
        func=terms.clip_or_time_limit, time_out=True
    )


@configclass
class EventsCfg:
    context = EventTermCfg(func=events.startup_context, mode="startup")
    reference_state = EventTermCfg(func=events.reset_reference, mode="reset")


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    class_type = "supermimic.tasks.tracking.tracking_env:TrackingEnv"
    sim = make_sim("cuda:0", None)
    scene = TrackingSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)
    # Newton already owns four 5 ms substeps within a 20 ms SimulationContext step.
    decimation = 1
    episode_length_s = 10.0
    seed = None  # Trainer seeds once before model construction, as in the baseline.
    analytic_reward = RewardCfg()
    policy_observation = ObservationCfg()
    critic_observation = StateObservationCfg(reference_orientation=None)
    world_model_observation = StateObservationCfg(reference_orientation=None)
    control = PDControlCfg()
    randomization = RandomizationCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    observations = ObservationsCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventsCfg()
    curriculum = None
    compute_final_obs = False  # Raw terminal state is captured explicitly for replay.
