"""Pure CPU R6 reset and force specifications. Privileged setup/evaluation only.

No generated position, lane ID, stream seed, force plan or curriculum clock is
an actor input. Runtime must verify actual reset and applied-force observations.
"""
import hashlib
import json
import math
from training_workloads.unified_h1.robustness import lane_role,vector

YAW_DEGREES=(0,-15,15,-30,30)


def uniform_signed(seed,split,block_id,lane,field):
    if type(seed)is not int or not isinstance(split,str) or not split:
        raise ValueError('EXPLICIT_STREAM_REQUIRED')
    raw=json.dumps(['r6-reset-v1',seed,split,block_id,lane,field],separators=(',',':')).encode()
    # Exact53-bit mapping avoids global RNG mutation and iterator-order coupling.
    integer=int.from_bytes(hashlib.sha256(raw).digest()[:8],'big')>>11
    return 2*(integer/2**53)-1


def rotate_yaw(v,yaw):
    x,y,z=vector(v);c=math.cos(yaw);s=math.sin(yaw)
    return (c*x-s*y,s*x+c*y,z)


def reset_spec(block,*,seed,split,lane,mount_position_m):
    group,direction=lane_role(block,lane)
    mount=vector(mount_position_m)
    yaw_degrees=YAW_DEGREES[(lane//8+lane%8)%5]
    yaw=math.radians(yaw_degrees)
    xy=[.2*uniform_signed(seed,split,block['id'],lane,axis) for axis in ('x','y')]
    sensor=[*xy,2.]
    if block['task']=='wall_modes':sensor[0 if lane<16 else 1]=6-block['sensor_to_wall_distance_m']
    offset=rotate_yaw(mount,yaw)
    root=[a-b for a,b in zip(sensor,offset)]
    return dict(schema_version='r6-reset-spec/1',block_id=block['id'],lane=lane,split=split,
                seed=seed,subgroup=group,sensor_position_local_m=sensor,root_position_local_m=root,
                root_quaternion_wxyz=[math.cos(yaw/2),0.,0.,math.sin(yaw/2)],
                root_linear_velocity_w_m_s=[0.,0.,0.],root_angular_velocity_w_rad_s=[0.,0.,0.],
                yaw_degrees=yaw_degrees,direction_world=list(direction),mount_position_m=list(mount))


def verify_reset(spec,actual,*,position_atol=1e-5,quaternion_atol=1e-6,velocity_atol=1e-6):
    for field in ('sensor_position_local_m','root_position_local_m'):
        if max(abs(a-b) for a,b in zip(vector(actual[field]),vector(spec[field])))>position_atol:
            raise ValueError('RESET_POSITION_MISMATCH:'+field)
    q=actual['root_quaternion_wxyz'];expected=spec['root_quaternion_wxyz']
    if len(q)!=4 or any(type(x)not in (int,float) or not math.isfinite(x) for x in q):raise ValueError('RESET_QUATERNION')
    # q and -q describe the same attitude; compare both without normalizing data.
    if min(max(abs(a-b) for a,b in zip(q,expected)),max(abs(a+b) for a,b in zip(q,expected)))>quaternion_atol:
        raise ValueError('RESET_QUATERNION_MISMATCH')
    for field in ('root_linear_velocity_w_m_s','root_angular_velocity_w_rad_s'):
        if max(abs(x) for x in vector(actual[field]))>velocity_atol:raise ValueError('RESET_VELOCITY_MISMATCH')
    return True


def pulse_force(block,*,lane,physics_tick,mass_kg):
    if type(physics_tick)is not int or physics_tick<0:raise ValueError('PHYSICS_TICK')
    if type(mass_kg)not in (int,float) or not math.isfinite(mass_kg) or mass_kg<=0:raise ValueError('MASS')
    _,direction=lane_role(block,lane)
    active=block['task']=='pulse_hold' and 1000<=physics_tick<1040
    acceleration=block.get('pulse_acceleration_m_s2',0.) if active else 0.
    return tuple(mass_kg*acceleration*x for x in direction)
