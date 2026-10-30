"""Standard G1 tracking preset. All values retain SuperMimic's defaults.

Only the configuration hierarchy follows the familiar task/config/robot layout;
no BeyondMimic robot, physics, reward or PPO settings are imported.
"""

from isaaclab.utils import configclass, replace
from supermimic import robot
from supermimic.robot import G1_CFG
from supermimic.paths import REPO
from ...tracking_env_cfg import TrackingEnvCfg, TrackingSceneCfg
from .model_cfg import G1ModelCfg, build_robot_cfg
from ...control_cfg import PDControlCfg
from ...randomization_cfg import (
    RandomizationCfg,
    ScalarRandomizationCfg,
    VectorRandomizationCfg,
    PDGainRandomizationCfg,
    ResetNoiseCfg,
)
from ...reward_cfg import RewardCfg
from ...observation_cfg import ObservationCfg, SensorTermCfg, StateObservationCfg


@configclass
class G1SceneCfg(TrackingSceneCfg):
    robot = replace(G1_CFG, prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
    scene = G1SceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)
    robot_model = G1ModelCfg(
        urdf=robot.URDF_PATH, mjcf=robot.MJCF_PATH, usd=G1_CFG.spawn.usd_path
    )
    motion_dir = str(REPO / "data/lafan1_g1")
    log_dir = str(REPO / "logs/run")

    self_collisions: bool = False

    # Nominal PD gains; regex overrides are absolute Kp/Kd, last match wins.
    control = PDControlCfg(
        natural_frequency_hz=10.0,
        damping_ratio=2.0,
        stiffness_scale=1.0,
        damping_scale=1.0,
        action_scale_multiplier=0.25,
        stiffness_overrides={},  # e.g. {".*_knee_joint": 100.0}
        damping_overrides={},
    )

    # Randomization modes: "startup" or "reset". Evaluation uses nominal values.
    randomization = RandomizationCfg(
        friction=ScalarRandomizationCfg(
            enabled=True, range=(0.3, 1.2), mode="startup", nominal=1.0
        ),
        joint_offsets=ScalarRandomizationCfg(
            enabled=True, range=(-0.01, 0.01), mode="reset", nominal=0.0
        ),
        torso_com=VectorRandomizationCfg(
            enabled=True,
            ranges=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
            mode="startup",
            nominal=(0.0, 0.0, 0.0),
        ),
        pd_gains=PDGainRandomizationCfg(
            enabled=False,
            kp_scale=(0.8, 1.2),
            kd_scale=(0.8, 1.2),
            mode="reset",
        ),
        reset=ResetNoiseCfg(
            enabled=True,
            root_position=((-0.05, 0.05), (-0.05, 0.05), (-0.01, 0.01)),
            root_rotation=((-0.1, 0.1), (-0.1, 0.1), (-0.2, 0.2)),
            root_linear_velocity=((-0.5, 0.5), (-0.5, 0.5), (-0.2, 0.2)),
            root_angular_velocity=((-0.52, 0.52), (-0.52, 0.52), (-0.78, 0.78)),
            joint_position=(-0.1, 0.1),
        ),
    )

    # Shared by the simulator AND imagined BPTT rollouts; zero disables a reward.
    analytic_reward = RewardCfg(
        w_body_pos=1.0,
        w_body_ori=1.0,
        w_body_linvel=1.0,
        w_body_angvel=1.0,
        w_root_pos=0.5,
        w_root_ori=0.5,
        w_action_rate=-0.1,
        w_joint_limit=-2.0,
        body_pos_std=0.3,
        body_ori_std=0.4,
        body_linvel_std=1.0,
        body_angvel_std=3.14,
        root_pos_std=0.3,
        root_ori_std=0.4,
        soft_limit_factor=0.9,
        term_height=0.25,
        term_tilt=0.8,
        soft_term_height_temperature=0.02,
        soft_term_tilt_temperature=0.04,
    )

    # Noise is raw-unit uniform half-width; scaling also scales noise.
    # Disabling a term changes the policy input size: train a new checkpoint.
    policy_observation = ObservationCfg(
        history=10,
        noise_enabled=True,
        joint_pos=SensorTermCfg(enabled=True, scale=1.0, noise=0.01),
        joint_vel=SensorTermCfg(enabled=True, scale=1.0, noise=0.5),
        base_ang_vel=SensorTermCfg(enabled=True, scale=1.0, noise=0.2),
        projected_gravity=SensorTermCfg(enabled=True, scale=1.0, noise=0.05),
        previous_action=SensorTermCfg(enabled=True, scale=1.0, noise=0.0),
        reference_orientation=None,
    )
    critic_observation = StateObservationCfg(reference_orientation=None)
    world_model_observation = StateObservationCfg(reference_orientation=None)

    def __post_init__(self):
        self.prepare()

    def prepare(self):
        """Validate task edits and rebuild actuator/spawn settings before use."""
        if self.sim.dt != 0.02 or self.decimation != 1:
            raise ValueError(
                "Tracking uses a fixed 50 Hz motion/MDP clock: sim.dt=0.02 and decimation=1 are required"
            )
        self.control.validate()
        self.randomization.validate()
        self.analytic_reward.validate()
        self.policy_observation.validate()
        self.critic_observation.validate()
        self.world_model_observation.validate()
        if self.rewards.tracking.weight != 1.0:
            raise ValueError(
                "Change analytic_reward weights; rewards.tracking.weight must stay 1 to match imagination"
            )
        if self.observations.policy.enable_corruption:
            raise ValueError(
                "Configure noise in policy_observation; collector owns noise for real and imagined steps"
            )
        self.scene.robot = build_robot_cfg(
            self.robot_model, self.control, self.self_collisions
        )
        return self
