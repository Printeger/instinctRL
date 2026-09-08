"""Pure finite combined-scene geometry shared by live authoring and CPU checks."""

from __future__ import annotations

import torch

H1_RAY_SCENE_PATH = "/World/MID360StaticScene"
H1_RAY_MESH_PATH = f"{H1_RAY_SCENE_PATH}/CombinedFloorXWallYWall"
H1_WALL_LIMIT_M = 6.0
H1_WALL_HEIGHT_M = 12.0
H1_FLOOR_HALF_EXTENT_M = 42.0


def combined_scene_triangles(env_origins: torch.Tensor) -> tuple[list[tuple[float, float, float]], list[int], tuple[str, ...]]:
    if not isinstance(env_origins, torch.Tensor) or env_origins.ndim != 2 or env_origins.shape[1] != 3:
        raise TypeError("env_origins must be [N,3]")
    if not bool(torch.isfinite(env_origins).all().item()): raise ValueError("env_origins must be finite")
    points: list[tuple[float, float, float]] = []; indices: list[int] = []; identities: list[str] = []
    f, w, h = H1_FLOOR_HALF_EXTENT_M, H1_WALL_LIMIT_M, H1_WALL_HEIGHT_M
    surfaces = (("floor", ((-f,-f,0.),(f,-f,0.),(f,f,0.),(-f,f,0.))),
                ("x_wall", ((w,-w,0.),(w,-w,h),(w,w,h),(w,w,0.))),
                ("y_wall", ((-w,w,0.),(w,w,0.),(w,w,h),(-w,w,h))))
    for env_index, origin in enumerate(env_origins.detach().cpu().to(torch.float64).tolist()):
        for surface, vertices in surfaces:
            start = len(points)
            points.extend(tuple(float(vertex[axis] + origin[axis]) for axis in range(3)) for vertex in vertices)
            indices.extend((start,start+1,start+2,start,start+2,start+3)); identities.extend((f"env{env_index}:{surface}",)*2)
    return points, indices, tuple(identities)


__all__ = ["H1_FLOOR_HALF_EXTENT_M", "H1_RAY_MESH_PATH", "H1_RAY_SCENE_PATH", "H1_WALL_HEIGHT_M", "H1_WALL_LIMIT_M", "combined_scene_triangles"]
