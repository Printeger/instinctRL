"""Held-out geometric fixtures. Used by scene authoring and preflight only."""

import math
import random


LAYOUTS = ('room_door', 'trunk_aisle', 'street_gap', 'open_tube')
COMMAND_LAYOUTS = ('room_corner', 'trunk_aisle', 'street_gap', 'corridor', 'open_tube')


def command_layout(lane):
    return COMMAND_LAYOUTS[lane//8 if lane < 28 else 4]


def scene_geometry(origins, seed, *, mirrored_pipe_pairs=False, mirrored_trunk_pairs=False, command_routes=False):
    """Construct matching sensed and collision geometry in each lane."""
    points, faces, solids, solid_faces = [], [], [], []
    rng = random.Random(seed)
    pipe_radius = None
    trunk_pair = None
    for lane, origin in enumerate(origins.tolist()):
        layout = command_layout(lane) if command_routes else LAYOUTS[lane//8]

        def face(vertices, *, solid=True):
            translated = [tuple(a+b for a, b in zip(v, origin)) for v in vertices]
            for vertices_out, faces_out in ((points, faces), (solids, solid_faces)) if solid else ((points, faces),):
                n = len(vertices_out)
                vertices_out.extend(translated)
                faces_out.extend((n, n+1, n+2, n, n+2, n+3))

        def partition(x0, x1, y0, y1):
            face(((x0, y0, 0.), (x1, y1, 0.), (x1, y1, 5.), (x0, y0, 5.)))

        def building(x0, x1, y0, y1):
            corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
            for (x, y), (xx, yy) in zip(corners, corners[1:] + corners[:1]):
                partition(x, xx, y, yy)

        floor = 42.
        face(((-floor, -floor, 0.), (floor, -floor, 0.), (floor, floor, 0.), (-floor, floor, 0.)), solid=False)
        if layout in ('room_door', 'room_corner'):
            half = rng.uniform(2.8, 3.2)
            door = rng.uniform(.88, 1.12)
            partition(-7., 7., -half, -half)
            partition(7., -7., half, half)
            partition(2.1, 2.1, -half, -door)
            partition(2.1, 2.1, door, half)
            if layout == 'room_corner':
                partition(4.1, 4.1, -half, .55)
                partition(4.1, 5.9, .55, .55)
        elif layout == 'trunk_aisle':
            trunks = [(x, side * rng.uniform(1.15, 1.65), rng.uniform(.22, .36))
                      for x,side in ((-2.7, 1), (-.7, -1), (2.8, 1), (3.9, -1), (4.8, 1))]
            if mirrored_trunk_pairs and lane % 2:
                trunks = [(x,-y,r) for x,y,r in trunk_pair]
            trunk_pair = trunks
            for x, cy, r in trunks:
                ring = [(x+r*math.cos(j*math.tau/8), cy+r*math.sin(j*math.tau/8)) for j in range(8)]
                for (px, py), (qx, qy) in zip(ring, ring[1:] + ring[:1]):
                    partition(px, qx, py, qy)
        elif layout == 'street_gap':
            half = rng.uniform(1.9, 2.5)
            for x0, x1 in ((-6., .15), (2.45, 7.)):
                building(x0, x1, half, half+2.)
                building(x0, x1, -half-2., -half)
        elif layout == 'corridor':
            half = rng.uniform(.95, 1.12)
            partition(-42., 42., -half, -half)
            partition(42., -42., half, half)
        else:
            radius = rng.uniform(1.35, 1.56)
            if mirrored_pipe_pairs and lane % 2:
                radius = pipe_radius
            pipe_radius = radius
            ring = [(radius*math.cos(j*math.tau/16), 2.+radius*math.sin(j*math.tau/16)) for j in range(16)]
            for (y, z), (yy, zz) in zip(ring, ring[1:] + ring[:1]):
                face(((-42., y, z), (42., y, z), (42., yy, zz), (-42., yy, zz)))
    return origins.new_tensor(points).tolist(), faces, origins.new_tensor(solids).tolist(), solid_faces
