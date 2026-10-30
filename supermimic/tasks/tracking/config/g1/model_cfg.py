"""Local G1 asset presets; values come from SuperMimic's existing assets."""

from dataclasses import dataclass
from supermimic.paths import URDF_PATH, MJCF_PATH, USD_PATH


@dataclass
class G1ModelCfg:
    urdf: str = URDF_PATH
    mjcf: str = MJCF_PATH
    usd: str = USD_PATH


def build_robot_cfg(model, control, self_collisions=False):
    """Build one fresh articulation from the task's nominal gains and assets."""
    from isaaclab.utils import replace
    from isaaclab_newton.sim.schemas import NewtonArticulationCfg
    from isaaclab_physx.sim.schemas import PhysxArticulationCfg
    from supermimic.robot import G1_CFG
    from ...control_cfg import joint_pd_parameters

    gains = joint_pd_parameters(control)
    actuators = {
        name.replace("_joint", ""): replace(
            G1_CFG.actuators[name.replace("_joint", "")], stiffness=kp, damping=kd
        )
        for name, (kp, kd, _) in gains.items()
    }
    spawn = replace(
        G1_CFG.spawn,
        usd_path=model.usd,
        articulation_props=[
            PhysxArticulationCfg(enabled_self_collisions=self_collisions),
            NewtonArticulationCfg(self_collision_enabled=self_collisions),
        ],
    )
    return replace(
        G1_CFG, prim_path="{ENV_REGEX_NS}/Robot", spawn=spawn, actuators=actuators
    )
