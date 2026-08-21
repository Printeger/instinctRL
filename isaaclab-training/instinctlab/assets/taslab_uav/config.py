"""Pure declarative TASLAB UAV contract copied from the frozen active YAML.

This module deliberately uses only the Python standard library. It does not
register a robot, construct a simulator object, or resolve runtime semantics
reserved for later migration orders.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InertiaContract:
    xx: float
    xy: float
    xz: float
    yy: float
    yz: float
    zz: float
    unit: str = "kg*m^2"


@dataclass(frozen=True)
class RotorContract:
    arm_lengths: tuple[float, ...]
    angles: tuple[float, ...]
    directions: tuple[float, ...]
    force_constants: tuple[float, ...]
    moment_constants: tuple[float, ...]
    max_angular_velocities: tuple[float, ...]
    usd_prim_paths: tuple[str, ...]
    list_order: tuple[int, ...] = (0, 1, 2, 3)
    arm_length_unit: str = "m"
    angle_unit: str = "rad"
    direction_unit: str = "dimensionless sign"
    force_constant_unit: str = "N/(rad/s)^2"
    moment_constant_unit: str = "N*m/(rad/s)^2"
    max_angular_velocity_unit: str = "rad/s"


@dataclass(frozen=True)
class LeeGainContract:
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    attitude: tuple[float, float, float]
    angular_rate: tuple[float, float, float]
    units: str = "legacy controller gains; dimensional semantics unresolved"


@dataclass(frozen=True)
class TaslabUavContract:
    name: str
    mass: float
    inertia: InertiaContract
    length: float
    drag_coefficient: float
    rotor_count: int
    rotors: RotorContract
    lee_gains: LeeGainContract
    usd_path: Path
    vehicle_yaml_path: Path
    lee_yaml_path: Path
    mass_unit: str = "kg"
    length_unit: str = "m"
    drag_coefficient_unit: str = "legacy active value; dimensional semantics unresolved"


ASSET_DIRECTORY = Path(__file__).resolve().parent

TASLAB_UAV = TaslabUavContract(
    name="taslab_uav",
    mass=1.1583,
    inertia=InertiaContract(
        xx=0.0051350687,
        xy=0.0000000049,
        xz=0.0002842395,
        yy=0.0056075235,
        yz=0.0000000307,
        zz=0.0052743931,
    ),
    length=0.24042,
    drag_coefficient=0.2,
    rotor_count=4,
    rotors=RotorContract(
        arm_lengths=(0.12021, 0.12021, 0.12021, 0.12021),
        angles=(5.49778714, 0.78539816, 2.35619449, 3.92699082),
        directions=(1.0, -1.0, 1.0, -1.0),
        force_constants=(1.55e-06, 1.55e-06, 1.55e-06, 1.55e-06),
        moment_constants=(2.39e-08, 2.39e-08, 2.39e-08, 2.39e-08),
        max_angular_velocities=(2261.0, 2261.0, 2261.0, 2261.0),
        usd_prim_paths=(
            "/taslab_uav/rotor_0",
            "/taslab_uav/rotor_1",
            "/taslab_uav/rotor_2",
            "/taslab_uav/rotor_3",
        ),
    ),
    lee_gains=LeeGainContract(
        position=(4.0, 4.0, 4.0),
        velocity=(2.2, 2.2, 2.2),
        attitude=(0.7, 0.7, 0.035),
        angular_rate=(0.1, 0.1, 0.025),
    ),
    usd_path=ASSET_DIRECTORY / "taslab_uav.usd",
    vehicle_yaml_path=ASSET_DIRECTORY / "taslab_uav.yaml",
    lee_yaml_path=ASSET_DIRECTORY / "lee_controller_taslab_uav.yaml",
)

UNRESOLVED_RUNTIME_SEMANTICS = (
    "Isaac Lab articulation and actuator configuration",
    "rotor force application and angular-velocity convention",
    "drag-coefficient dimensional interpretation",
    "Lee-gain dimensional interpretation and controller execution",
    "body/world frame command conversion",
)
