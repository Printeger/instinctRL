"""CPU numeric reconstruction of recorded wrench staging and reset evidence.

Uses NumPy quaternion algebra, not production distributed_pulse. Passing does
not prove that Isaac integrated the staged forces or that sensor truth is valid.
"""
import math
import ast
from pathlib import Path
import numpy as np
from .robustness import lane_role
from .robustness_scenarios import reset_spec,verify_reset
from .robustness_audit import read_audit
from .robustness_analysis import pin


def verified_mount(protocol):
    binding=protocol['physics_replay']['mount_source'];path=Path(binding['path'])
    actual=pin(path)
    if {k:actual[k] for k in ('path','sha256')}!=binding:raise ValueError('PHYSICS_MOUNT_SOURCE_HASH')
    values=[ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MOUNT_POSITION_M' for t in node.targets)]
    if len(values)!=1 or list(values[0])!=protocol['physics_replay']['mount_position_m']:
        raise ValueError('PHYSICS_MOUNT_SOURCE_VALUE')
    return actual


def array(value,shape):
    result=np.asarray(value,dtype=np.float64)
    if result.shape!=shape or not np.isfinite(result).all():raise ValueError('PHYSICS_ARRAY_SHAPE_OR_FINITE')
    return result


def epsilon(row):
    if row.get('dtype') not in ('torch.float32','torch.float64'):raise ValueError('PHYSICS_DTYPE')
    return np.finfo(np.float32 if row['dtype']=='torch.float32' else np.float64).eps


def close(actual,expected,eps,name):
    # Arithmetic roundoff only, never a relaxation of motion/safety thresholds.
    bound=16*eps*max(1.,float(np.max(np.abs(expected))))
    if np.max(np.abs(actual-expected))>bound:raise ValueError('PHYSICS_MISMATCH:'+name)


def verify_force(row,block,*,expected_tick,previous_sim_tick,expected_mass_kg):
    if type(expected_mass_kg) not in (int,float) or not math.isfinite(expected_mass_kg) or expected_mass_kg<=0:
        raise ValueError('PHYSICS_EXPECTED_MASS')
    eps=epsilon(row)
    if type(row['block_tick']) is not int or row['block_tick']!=expected_tick:
        raise ValueError('PHYSICS_BLOCK_TICK')
    sim=row['sim_tick']
    if type(sim) is not int or sim<0 or (previous_sim_tick is not None and sim!=previous_sim_tick+1):
        raise ValueError('PHYSICS_SIM_TICK')
    dt=row['physics_dt_s']
    if type(dt) not in (int,float) or not math.isfinite(dt) or abs(dt-.005)>1e-12:raise ValueError('PHYSICS_DT')
    active=block['task']=='pulse_hold' and 1000<=expected_tick<1040
    if type(row['active']) is not bool or row['active']!=active:raise ValueError('PHYSICS_PULSE_WINDOW')
    masses=array(row['mass_kg'],(32,5))
    if (masses<=0).any() or (np.abs(masses.sum(axis=1)-expected_mass_kg)>1e-5).any():raise ValueError('PHYSICS_MASS')
    quats=array(row['body_quaternion_wxyz'],(32,5,4));norm=np.linalg.norm(quats,axis=-1)
    if (np.abs(norm-1)>1e-5).any():raise ValueError('PHYSICS_QUATERNION')
    quats=quats/norm[:,:,None];w,x,y,z=np.moveaxis(quats,-1,0)
    rotation=np.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),axis=-1).reshape(32,5,3,3)
    directions=np.asarray([lane_role(block,i)[1] for i in range(32)])
    acceleration=directions*(block.get('pulse_acceleration_m_s2',0.) if active else 0.)
    world=masses[:,:,None]*acceleration[:,None,:]
    local=np.einsum('nbji,nbj->nbi',rotation,world)
    recorded_world=array(row['additional_world_n'],(32,5,3));recorded_local=array(row['additional_local_n'],(32,5,3))
    rotor=array(row['rotor_forces_local_n'],(32,5,3));torque=array(row['rotor_torques_local_nm'],(32,5,3))
    staged=array(row['staged_forces_local_n'],(32,5,3));staged_torque=array(row['staged_torques_local_nm'],(32,5,3))
    close(recorded_world,world,eps,'world_force');close(recorded_local,local,eps,'local_force')
    close(staged,rotor+recorded_local,eps,'rotor_addition')
    if not np.array_equal(staged_torque,torque):raise ValueError('PHYSICS_ROTOR_TORQUE_CHANGED')
    if not active and (np.any(recorded_world) or np.any(recorded_local) or not np.array_equal(staged,rotor)):
        raise ValueError('PHYSICS_FORCE_NOT_WITHDRAWN')
    return sim,recorded_world.sum(axis=1)*dt


def verify_reset_record(row,block,*,seed,split,mount_position_m):
    lane=row['lane'];eps=epsilon(row)
    if type(lane) is not int or not 0<=lane<32:raise ValueError('PHYSICS_RESET_LANE')
    expected=reset_spec(block,seed=seed,split=split,lane=lane,mount_position_m=mount_position_m)
    if row['expected']!=expected:raise ValueError('PHYSICS_RESET_SPEC')
    origin=array(row['env_origin_w_m'],(3,));root=array(row['root_position_w_m'],(3,));sensor=array(row['sensor_position_w_m'],(3,))
    tolerance=max(1e-5,2*eps*max(1.,float(np.max(np.abs(root))),float(np.max(np.abs(sensor)))))
    if row['position_atol_m']!=tolerance:raise ValueError('PHYSICS_RESET_TOLERANCE')
    actual=row['actual']
    dtype=np.float32 if row['dtype']=='torch.float32' else np.float64
    for name,world in [('root_position_local_m',root),('sensor_position_local_m',sensor)]:
        local=(world.astype(dtype)-origin.astype(dtype)).astype(np.float64)
        if not np.array_equal(array(actual[name],(3,)),local):raise ValueError('PHYSICS_RESET_WORLD_LOCAL')
    verify_reset(expected,actual,position_atol=tolerance)
    return lane


def replay_staging(path,block,*,run_id,seed,split,expected_records,expected_mass_kg,mount_position_m,completion):
    before=pin(path);ticks=0;last=None;resets=0;initial=set();impulse=np.zeros((32,3))
    if type(expected_records) is not int or not 0<expected_records<=34+block['steps']*54:
        raise ValueError('PHYSICS_RECORD_BOUND')
    for record in read_audit(path,run_id=run_id,block_id=block['id'],expected_records=expected_records):
        row=record['payload']
        if record['kind']=='reset':
            lane=verify_reset_record(row,block,seed=seed,split=split,mount_position_m=mount_position_m)
            if row['block_tick']!=ticks:raise ValueError('PHYSICS_RESET_TICK')
            if ticks==0:
                if lane in initial:raise ValueError('PHYSICS_DUPLICATE_INITIAL_RESET')
                initial.add(lane)
            resets+=1
        elif record['kind']=='force':
            if initial!=set(range(32)):raise ValueError('PHYSICS_INITIAL_RESET_COVERAGE')
            last,delta=verify_force(row,block,expected_tick=ticks,previous_sim_tick=last,expected_mass_kg=expected_mass_kg)
            impulse+=delta;ticks+=1
    if ticks!=block['steps']*20:raise ValueError('PHYSICS_FORCE_COVERAGE')
    if (completion.get('block_ticks')!=ticks or completion.get('last_staged_sim_tick')!=last
            or completion.get('completed_sim_tick')!=last+1 or completion.get('physics_dt_s')!=.005):
        raise ValueError('PHYSICS_FINAL_TICK_NOT_COMPLETED')
    if pin(path)!=before:raise ValueError('PHYSICS_ARCHIVE_CHANGED')
    return dict(schema_version='r6-staging-replay/1',audit=before,force_records=ticks,reset_records=resets,
        staged_impulse_world_ns=impulse.tolist(),staging_reconstructed=True,physical_integration_verified=False,
        scientific_acceptance=False,numeric_tolerance='16*dtype_epsilon*max(1,max(abs(expected))); exact preserved torque/withdrawal')
