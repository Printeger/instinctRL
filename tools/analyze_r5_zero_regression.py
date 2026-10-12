"""Recompute full repaired zero-target response by case/lane/reset episode.

Coverage/numeric consistency are distinct from independent control acceptance.
No signature ranking or new acceptance threshold is introduced.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from training_workloads.unified_h1.b0_enhanced import TraceValidator
from training_workloads.unified_h1.b0_causal_analyzer import segment_key

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve existing regression analysis')
    manifest = json.loads(args.manifest.read_text())
    validator = TraceValidator()
    digest = hashlib.sha256()
    states = {}
    for line in args.raw.open('rb'):
        digest.update(line)
        row = json.loads(line)
        validator.append(row)
        if row['controller']['target_velocity_b_m_s'] != [0.,0.,0.]:
            raise ValueError('this condition is zero target throughout')
        key = segment_key(row)
        pos = row['response']['position_world_m']
        velocity = row['response']['velocity_world_m_s']
        state = states.setdefault(key,{'n':0,'root_start':pos,'root_last':pos,
            'mass':row['system']['total_mass_kg'],'velocity_sum':[0.,0.,0.],
            'speed_squared_sum':0.,'max_speed':0.,'max_horizontal_from_first_row':0.,'max_height_from_first_row':0.,
            'startup_n':0,'startup_velocity_sum':[0.,0.,0.],
            'late_n':0,'late_velocity_sum':[0.,0.,0.],'late_collective_sum':0.,
            'late_rotor_torque_sum':[0.,0.,0.],'last_velocity':velocity,
            'first_collective':row['controller']['collective_n']})
        if state['mass'] != row['system']['total_mass_kg']:
            raise ValueError('measured mass changed within an episode')
        state['n'] += 1
        state['root_last'] = pos
        state['last_velocity'] = velocity
        speed2 = sum(v*v for v in velocity)
        state['speed_squared_sum'] += speed2
        state['max_speed'] = max(state['max_speed'],math.sqrt(speed2))
        state['max_horizontal_from_first_row'] = max(state['max_horizontal_from_first_row'],
            math.hypot(pos[0]-state['root_start'][0],pos[1]-state['root_start'][1]))
        state['max_height_from_first_row'] = max(state['max_height_from_first_row'],abs(pos[2]-state['root_start'][2]))
        for i in range(3):
            state['velocity_sum'][i] += velocity[i]
        if state['n'] <= 200:
            state['startup_n'] += 1
            for i in range(3):
                state['startup_velocity_sum'][i] += velocity[i]
        if state['n'] > 1000:
            state['late_n'] += 1
            state['late_collective_sum'] += row['controller']['collective_n']
            for i in range(3):
                state['late_velocity_sum'][i] += velocity[i]
                state['late_rotor_torque_sum'][i] += row['wrenches']['rotor']['torque_at_com_nm'][i]
    coverage = validator.finalize()
    if digest.hexdigest() != manifest['raw']['sha256'] or coverage['physics_rows'] != manifest['raw']['written_rows']:
        raise ValueError('complete source stream differs from manifest')
    segments = []
    for key,s in states.items():
        late = s['late_n']
        segments.append({'segment':list(key),'physics_rows':s['n'],'duration_s':s['n']*.005,
            'mass_kg':s['mass'],'root_start_first_physics_row_m':s['root_start'],'root_end_m':s['root_last'],
            'all_physics_speed_rms_m_s':math.sqrt(s['speed_squared_sum']/s['n']),
            'max_speed_m_s':s['max_speed'],'max_horizontal_from_first_physics_row_m':s['max_horizontal_from_first_row'],
            'max_height_from_first_physics_row_m':s['max_height_from_first_row'],
            'all_velocity_mean_m_s':[v/s['n'] for v in s['velocity_sum']],
            'first_1s_velocity_mean_m_s':[v/s['startup_n'] for v in s['startup_velocity_sum']],
            'after_5s_velocity_mean_m_s':[v/late for v in s['late_velocity_sum']] if late else None,
            'after_5s_rotor_torque_at_com_mean_nm':[v/late for v in s['late_rotor_torque_sum']] if late else None,
            'first_collective_n':s['first_collective'],
            'after_5s_collective_mean_n':s['late_collective_sum']/late if late else None,
            'actual_mass_times_gravity_n':s['mass']*9.81,'last_velocity_m_s':s['last_velocity']})
    result = {'schema_version':'instinctrl-r5-zero-target-regression/1',
        'status':'COMPLETE_NUMERICALLY_VALID_INDEPENDENT_CONTROL_ACCEPTANCE_REQUIRED',
        'condition':'ENHANCED_B0_ZERO_TARGET','coverage':coverage,'source_identity':manifest['source_identity'],
        'raw_path':str(args.raw.resolve()),'raw_sha256':digest.hexdigest(),'segments':segments,
        'statistics_reference':'root link; diagnostic displacement starts at first physics row, not original metric sensor T0',
        'window_semantics':'all physics retained; first1s and after5s are supplemental perepisode statistics only',
        'formal_ZTS_gate':'separate command passthrough must use unchanged original sensor-origin evaluate_episode',
        'mechanism_verdict':'not inferred from heuristic scores','unavailable_wrenches':['drag_aero','contact','joint_constraint','other'],
        'analysis_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'coverage':coverage,'segments':len(segments),
        'max_all_physics_speed_rms':max(s['all_physics_speed_rms_m_s'] for s in segments),
        'max_horizontal_from_first_physics_row':max(s['max_horizontal_from_first_physics_row_m'] for s in segments)}))

if __name__ == '__main__':
    main()
