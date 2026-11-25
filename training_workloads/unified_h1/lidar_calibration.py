"""Retain and independently validate the actual body-frame ray table."""

import math
from pathlib import Path

from tools.r6_evidence import checked, file_pin


def record(env, output, label):
    from training_workloads.scale_003.persistence import reserve_output_bytes

    pitch = env.lidar_mount_pitch_degrees
    rays = env._ray_caster.ray_directions[0].detach().cpu().contiguous().numpy()
    data = rays.astype('<f4', copy=False).tobytes()
    path = Path(output) / f'{label}-lidar-rays.f32'
    reserve_output_bytes(path, len(data))
    with path.open('xb') as stream:
        stream.write(data)
    return dict(pitch_degrees=pitch, rotation_wxyz=list(env._ray_caster.cfg.offset.rot), body_rays=file_pin(path))


def verify(value, output, label, pitch):
    import numpy as np
    from instinctlab.control.ics import RAY_COUNT, mounted_mid360_rays_body

    path = checked(value['body_rays'])
    if path != Path(output) / f'{label}-lidar-rays.f32' or value['pitch_degrees'] != pitch:
        raise ValueError('LIDAR_CALIBRATION_IDENTITY')
    half_angle = math.radians(pitch)/2
    if not np.allclose(value['rotation_wxyz'], [math.cos(half_angle),0.,math.sin(half_angle),0.],
                       atol=1.e-8, rtol=0.):
        raise ValueError('LIDAR_CALIBRATION_QUATERNION')
    observed = np.fromfile(path, dtype='<f4').reshape(RAY_COUNT,3)
    nominal = mounted_mid360_rays_body(device='cpu').numpy()
    delta = math.radians(pitch - 30)
    expected = nominal.copy()
    expected[:,0] = math.cos(delta)*nominal[:,0] + math.sin(delta)*nominal[:,2]
    expected[:,2] = -math.sin(delta)*nominal[:,0] + math.cos(delta)*nominal[:,2]
    if not np.allclose(observed, expected, atol=2.e-6, rtol=0.):
        raise ValueError('LIDAR_CALIBRATION_RAY_READBACK')
    return value['body_rays']
