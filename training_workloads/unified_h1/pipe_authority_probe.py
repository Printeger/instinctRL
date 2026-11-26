"""Retain delivered observations for a fixed-policy pipe diagnostic."""

import io
from pathlib import Path

from tools.r6_evidence import checked
from .lossless_evidence import ChunkReader, EvidencePool
from .robustness_scenarios import rotate_yaw


PIPE_PROBE_RUN = 'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-U1792-TWO-PIPE-PROBE-01'
PIPE_SWITCH_RUN = 'R6-DIRECT-INSTINCT-GPU-R5-RECOVERY-U1792-TWO-PIPE-SWITCH-01'
MIRRORED_PIPE_RUNS = (PIPE_PROBE_RUN, PIPE_SWITCH_RUN)
PIPE_LANES = slice(24, 32)
FRAME_BYTES = (3 * 59 * 360 + 52) * 8 * 4
FRAME_CAPACITY = FRAME_BYTES * 120


def mirror_reset(spec, other):
    """Reflect the accepted local position and yaw without changing lane identity."""
    import math

    x, y, z = spec['sensor_position_local_m']
    sensor = [x, -y, z]
    yaw = -spec['yaw_degrees']
    angle = math.radians(yaw)
    mount = other['mount_position_m']
    return dict(other, sensor_position_local_m=sensor, yaw_degrees=yaw,
                root_position_local_m=[a-b for a,b in zip(sensor, rotate_yaw(mount, angle), strict=True)],
                root_quaternion_wxyz=[math.cos(angle/2), 0., 0., math.sin(angle/2)])


def frame_name(index):
    return f'group-{index}/delivered-latest.f32'


def open_pool(output):
    return EvidencePool(Path(output)/'approved_measurements', run_id=PIPE_PROBE_RUN,
                        maximum_bytes=800*2**20, maximum_raw_file_bytes=FRAME_CAPACITY)


def capture(stream, policy):
    import torch

    latest = policy['lidar_grid'][PIPE_LANES, 9:].flatten(1)
    state = policy['state_vec'][PIPE_LANES]
    frame = torch.cat((latest, state), dim=1).detach().cpu().contiguous().numpy().astype('<f4', copy=False)
    if frame.nbytes != FRAME_BYTES:
        raise ValueError('PIPE_AUTHORITY_APPROVED_SENSOR_SHAPE')
    stream.write(frame.tobytes())


def replay_frames(pin, index):
    import numpy as np

    with ChunkReader(checked(pin), run_id=PIPE_PROBE_RUN, logical_name=frame_name(index),
                     maximum_raw_bytes=FRAME_CAPACITY) as archive:
        if archive.size != FRAME_CAPACITY:
            raise ValueError('PIPE_AUTHORITY_INCOMPLETE_FRAMES')
        with io.BufferedReader(archive) as reader:
            for _ in range(120):
                row = np.frombuffer(reader.read(FRAME_BYTES), dtype='<f4').reshape(8,-1)
                yield row[:, :-52].reshape(8,3,59,360), row[:, -52:]
