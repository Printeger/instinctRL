"""Pure CPU geometry and readback contracts for the 32-copy scene candidate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


MESH_PRIM_PATH = "/World/MID360StaticScene/Scale003CombinedMesh"
WALL_COLLIDER_ROOT = "/World/Scale003WallColliders"
GLOBAL_GROUND_PATH = "/World/ground"
NUM_ENVS = 32
ENV_SPACING_M = 100.0
MID360_MAX_RANGE_M = 40.0


Point = tuple[float, float, float]
Triangle = tuple[int, int, int]


@dataclass(frozen=True)
class WallCollider:
    environment_index: int
    surface: str
    prim_path: str
    points: tuple[Point, Point, Point, Point]
    triangles: tuple[Triangle, Triangle] = ((0, 1, 2), (0, 2, 3))


@dataclass(frozen=True)
class CombinedGeometry:
    mesh_prim_path: str
    env_origins: tuple[Point, ...]
    points: tuple[Point, ...]
    triangles: tuple[Triangle, ...]
    triangle_surfaces: tuple[str, ...]
    wall_colliders: tuple[WallCollider, ...]
    sensed_mesh_collision: bool = False
    global_ground_path: str = GLOBAL_GROUND_PATH


def _point(value: Sequence[float]) -> Point:
    if len(value) != 3:
        raise ValueError("environment origins must have three coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("environment origins must be finite")
    return result  # type: ignore[return-value]


def fixture_grid_origins() -> tuple[Point, ...]:
    """CPU-test fixture only; production must use live ``scene.env_origins``."""
    return tuple((float(column * 100), float(row * 100), 0.0) for row in range(4) for column in range(8))


def _translate(origin: Point, local: Point) -> Point:
    return tuple(origin[index] + local[index] for index in range(3))  # type: ignore[return-value]


def _local_geometry() -> tuple[list[Point], list[Triangle], list[str], tuple[tuple[Point, ...], tuple[Point, ...]]]:
    coordinates = [float(value) for value in range(-42, 43, 2)]
    points: list[Point] = [(x, y, 0.0) for y in coordinates for x in coordinates]
    triangles: list[Triangle] = []
    surfaces: list[str] = []
    for row in range(42):
        for column in range(42):
            lower_left = row * 43 + column
            triangles.extend(((lower_left, lower_left + 1, lower_left + 44), (lower_left, lower_left + 44, lower_left + 43)))
            surfaces.extend(("floor", "floor"))
    x_wall: tuple[Point, ...] = ((6.0, -42.0, 0.0), (6.0, -42.0, 8.0), (6.0, 6.0, 8.0), (6.0, 6.0, 0.0))
    y_wall: tuple[Point, ...] = ((-42.0, 6.0, 0.0), (6.0, 6.0, 0.0), (6.0, 6.0, 8.0), (-42.0, 6.0, 8.0))
    for name, wall in (("x_wall", x_wall), ("y_wall", y_wall)):
        base = len(points)
        points.extend(wall)
        triangles.extend(((base, base + 1, base + 2), (base, base + 2, base + 3)))
        surfaces.extend((name, name))
    return points, triangles, surfaces, (x_wall, y_wall)


def _assemble(origins: tuple[Point, ...]) -> CombinedGeometry:
    local_points, local_triangles, local_surfaces, local_walls = _local_geometry()
    points: list[Point] = []
    triangles: list[Triangle] = []
    surfaces: list[str] = []
    colliders: list[WallCollider] = []
    for environment_index, origin in enumerate(origins):
        base = len(points)
        points.extend(_translate(origin, point) for point in local_points)
        triangles.extend(tuple(base + index for index in triangle) for triangle in local_triangles)  # type: ignore[arg-type]
        surfaces.extend(f"env_{environment_index:02d}:{surface}" for surface in local_surfaces)
        for surface, wall in zip(("x_wall", "y_wall"), local_walls, strict=True):
            colliders.append(WallCollider(
                environment_index=environment_index,
                surface=surface,
                prim_path=f"{WALL_COLLIDER_ROOT}/env_{environment_index:02d}_{surface}",
                points=tuple(_translate(origin, point) for point in wall),  # type: ignore[arg-type]
            ))
    return CombinedGeometry(MESH_PRIM_PATH, origins, tuple(points), tuple(triangles), tuple(surfaces), tuple(colliders))


def build_combined_geometry(env_origins: Iterable[Sequence[float]]) -> CombinedGeometry:
    origins = tuple(_point(value) for value in env_origins)
    if len(origins) != NUM_ENVS:
        raise ValueError("exactly 32 live environment origins are required")
    geometry = _assemble(origins)
    validate_geometry(geometry, origins)
    return geometry


def _normal(points: tuple[Point, ...], triangle: Triangle) -> Point:
    a, b, c = (points[index] for index in triangle)
    u = tuple(b[index] - a[index] for index in range(3))
    v = tuple(c[index] - a[index] for index in range(3))
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    norm = math.sqrt(sum(item * item for item in cross))
    return tuple(item / norm for item in cross)  # type: ignore[return-value]


def validate_geometry(geometry: CombinedGeometry, actual_env_origins: Iterable[Sequence[float]]) -> dict[str, object]:
    origins = tuple(_point(value) for value in actual_env_origins)
    if geometry.mesh_prim_path != MESH_PRIM_PATH:
        raise ValueError("wrong explicit RayCaster mesh prim")
    if geometry.env_origins != origins or len(origins) != NUM_ENVS:
        raise ValueError("geometry translations do not match all live env_origins")
    if geometry.sensed_mesh_collision or geometry.global_ground_path != GLOBAL_GROUND_PATH:
        raise ValueError("sensed mesh/floor collision boundary mismatch")
    if geometry != _assemble(origins):
        raise ValueError("combined geometry differs from exact actual-origin translations")
    if len(geometry.points) != 59424 or len(geometry.triangles) != 113024 or len(geometry.triangle_surfaces) != 113024:
        raise ValueError("combined mesh topology count mismatch")
    if len(geometry.wall_colliders) != 64 or len({item.prim_path for item in geometry.wall_colliders}) != 64:
        raise ValueError("64 unique wall collider prims are required")
    expected_normals = {"floor": (0.0, 0.0, 1.0), "x_wall": (-1.0, 0.0, 0.0), "y_wall": (0.0, -1.0, 0.0)}
    for index, surface in enumerate(geometry.triangle_surfaces):
        label = surface.split(":", 1)[1]
        if _normal(geometry.points, geometry.triangles[index]) != expected_normals[label]:
            raise ValueError("triangle normal/winding mismatch")
    minimum_origin_distance = min(
        math.dist(origins[left], origins[right])
        for left in range(len(origins)) for right in range(left + 1, len(origins))
    )
    if minimum_origin_distance < ENV_SPACING_M:
        raise ValueError("environment origins violate 100m minimum spacing")
    return {
        "points": len(geometry.points), "triangles": len(geometry.triangles),
        "wall_colliders": len(geometry.wall_colliders),
        "minimum_origin_distance_m": minimum_origin_distance,
        "mid360_max_range_m": MID360_MAX_RANGE_M,
        "initial_neighbor_origin_beyond_range": minimum_origin_distance > MID360_MAX_RANGE_M,
        "coverage_limit": "local floor ends at 42m and walls span -42m..6m; visibility after drift outside each local fixture is not guaranteed",
    }


def validate_readback(
    *,
    expected: CombinedGeometry,
    mesh_prim_path: str,
    points: Sequence[Sequence[float]],
    triangle_indices: Sequence[int],
    collider_rows: Sequence[dict[str, object]],
    ground_collision_present: bool,
    sensed_mesh_collision: bool,
) -> None:
    if mesh_prim_path != MESH_PRIM_PATH or mesh_prim_path != expected.mesh_prim_path:
        raise ValueError("RayCaster resolved a wrong mesh prim")
    actual_points = tuple(_point(value) for value in points)
    flattened = tuple(index for triangle in expected.triangles for index in triangle)
    if actual_points != expected.points or tuple(triangle_indices) != flattened:
        raise ValueError("live Warp/USD mesh differs from actual-origin geometry")
    if sensed_mesh_collision or not ground_collision_present:
        raise ValueError("duplicate sensed-floor collision or missing global ground")
    expected_colliders = {item.prim_path: item for item in expected.wall_colliders}
    if len(collider_rows) != 64 or {str(row.get("prim_path")) for row in collider_rows} != set(expected_colliders):
        raise ValueError("wall collider inventory mismatch")
    for row in collider_rows:
        collider = expected_colliders[str(row["prim_path"])]
        if tuple(_point(point) for point in row.get("points", ())) != collider.points or row.get("collision") is not True:
            raise ValueError("wall collider readback mismatch")


def validate_exact_observability_gate(
    *, surface_hits: dict[str, Sequence[int]], numerical_rank: Sequence[int], sigma_min: Sequence[float], num_envs: int = 32,
) -> dict[str, object]:
    if set(surface_hits) != {"floor", "x_wall", "y_wall"}:
        raise ValueError("surface hit categories mismatch")
    if any(len(values) != num_envs or any(type(value) is not int or value <= 0 for value in values) for values in surface_hits.values()):
        raise ValueError("every environment must hit each exact surface")
    if list(numerical_rank) != [3] * num_envs:
        raise ValueError("normal-Jacobian numerical rank must be 3 for every environment")
    if len(sigma_min) != num_envs or any(not isinstance(value, float) or not math.isfinite(value) or value <= 0.0 for value in sigma_min):
        raise ValueError("sigma_min must be finite and positive for every environment")
    return {"status": "PASS", "algorithm": "exact_range_normal_jacobian", "environments": num_envs}
