"""Training-only axial approaches, with coincident ray and collision geometry."""

import math

from instinctlab.task.scene_h1 import combined_scene_triangles

from .robustness import DIRECTIONS


TUBE_LANES = (0, 2, 5, 9, 16, 19, 22, 26)


def geometry(origins):
    points, faces, _labels = combined_scene_triangles(origins)
    walls, wall_faces = [], []
    for offset in range(0, len(points), 12):
        for surface in (4, 8):
            first = len(walls)
            walls.extend(points[offset+surface:offset+surface+4])
            wall_faces.extend((first, first+1, first+2, first, first+2, first+3))

    for index, lane in enumerate(TUBE_LANES):
        x, y, z = origins[lane].tolist()
        dx, dy, _ = DIRECTIONS[lane % 8]
        radius = 1.38 + .04*(index % 4)
        rings = []
        for axial in (-12., 12.):
            rings.append([(x + axial*dx - radius*dy * math.cos(a),
                           y + axial*dy + radius*dx * math.cos(a),
                           z + 2. + radius * math.sin(a))
                          for a in (i*math.tau/16 for i in range(16))])
        for segment in range(16):
            adjacent = (segment+1) % 16
            vertices = (rings[0][segment], rings[1][segment], rings[1][adjacent], rings[0][adjacent])
            for mesh_points, mesh_faces in ((points, faces), (walls, wall_faces)):
                first = len(mesh_points)
                mesh_points.extend(vertices)
                mesh_faces.extend((first, first+1, first+2, first, first+2, first+3))
    return origins.new_tensor(points).tolist(), faces, origins.new_tensor(walls).tolist(), wall_faces
