"""Minimal Isaac Lab articulation adapter for the byte-preserved TASLAB UAV."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

TASLAB_USD_PATH = (Path(__file__).resolve().parent / "taslab_uav.usd").as_posix()
ROTOR_JOINT_EXPRESSION = r"rotor_[0-3]_joint"

TASLAB_ARTICULATION_CFG = ArticulationCfg(
    prim_path="/World/TASLAB",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TASLAB_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        joint_pos={ROTOR_JOINT_EXPRESSION: 0.0},
        joint_vel={ROTOR_JOINT_EXPRESSION: 0.0},
    ),
    actuators={
        "free_rotor_joints": ImplicitActuatorCfg(
            joint_names_expr=[ROTOR_JOINT_EXPRESSION],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)

__all__ = ["ROTOR_JOINT_EXPRESSION", "TASLAB_ARTICULATION_CFG", "TASLAB_USD_PATH"]
