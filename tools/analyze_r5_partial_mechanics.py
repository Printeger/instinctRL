"""Offline bounded-prefix mechanics analysis; never grants complete coverage."""
import argparse,hashlib,json,math,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--raw',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError('preserve existing analysis')
    sys.path[:0]=[str(args.source_root),str(args.source_root/'isaaclab-training')]
    from training_workloads.unified_h1.b0_enhanced import TraceValidator,rotor_wrench,quaternion_rotate
    from training_workloads.unified_h1.b0_causal_analyzer import segment_key
    from instinctlab.assets.taslab_uav.config import TASLAB_UAV
    from instinctlab.control.lee_velocity_controller import taslab_velocity_control
    import torch
    manifest=json.loads(args.manifest.read_text());validator=TraceValidator();digest=hashlib.sha256()
    segments={};steady_examples={}
    with args.raw.open('rb') as stream:
        for line in stream:
            digest.update(line);row=json.loads(line);validator.append(row)
            key=segment_key(row)
            state=segments.setdefault(key,{'count':0,'first':row['response']['position_world_m'],
                'last':None,'mass':row['system']['total_mass_kg'],'late_count':0,'late_velocity_sum':[0.,0.,0.],
                'late_inferred_force_sum':[0.,0.,0.],'late_rotor_torque_sum':[0.,0.,0.],
                'minimum_root_height_m':float('inf')})
            state['count']+=1;state['last']=row['response']['position_world_m']
            state['minimum_root_height_m']=min(state['minimum_root_height_m'],state['last'][2])
            if state['count']>1000: # first five seconds of each episode kept separately
                state['late_count']+=1
                for i in range(3):
                    state['late_velocity_sum'][i]+=row['response']['velocity_world_m_s'][i]
                    state['late_inferred_force_sum'][i]+=row['momentum']['inferred_wrench']['force_n'][i]
                    state['late_rotor_torque_sum'][i]+=row['wrenches']['rotor']['torque_at_com_nm'][i]
                if row['lane']==0 and row['case']=='Z' and row['policy_step']==100 and row['physics_step']==19:
                    steady_examples['Z_lane0_policy100']=row
    assert validator.count==manifest['raw']['written_rows'] and digest.hexdigest()==manifest['raw']['sha256']
    summaries=[]
    for key,state in segments.items():
        count=state['late_count']
        summaries.append({'segment':list(key),'physics_rows':state['count'],'duration_s':state['count']*.005,
            'root_start_m':state['first'],'root_end_m':state['last'],'minimum_root_height_m':state['minimum_root_height_m'],
            'mass_kg':state['mass'],'late_window':'within this episode only, after first5sec; formal startup samples remain in raw',
            **{name:[x/count for x in state[field]] if count else None for name,field in
                [('late_root_velocity_mean_m_s','late_velocity_sum'),('late_inferred_net_force_mean_n','late_inferred_force_sum'),('late_rotor_torque_at_com_mean_nm','late_rotor_torque_sum')]}})
    predictions=[]
    maximum=torch.tensor([kf*w*w for kf,w in zip(TASLAB_UAV.rotors.force_constants,TASLAB_UAV.rotors.max_angular_velocities,strict=True)],dtype=torch.float64)
    for name,row in steady_examples.items():
        root=torch.tensor([row['root_state']['pose']+[0.]*6],dtype=torch.float64)
        mass=row['system']['total_mass_kg'];com=row['system']['combined_com_world_m']
        rotors=row['rotors'];columns=[]
        for j in range(4):
            basis=[dict(r,thrust_n=0.,reaction_moment_nm=0.) for r in rotors]
            basis[j]['thrust_n']=1.
            basis[j]['reaction_moment_nm']=-TASLAB_UAV.rotors.directions[j]*TASLAB_UAV.rotors.moment_constants[j]/TASLAB_UAV.rotors.force_constants[j]
            columns.append(rotor_wrench(basis,com))
        # Solve original controller's steady net torque x/y and vertical force
        # equations using measured application geometry. Starts at zero velocity;
        # no gain fitting, optimizer or parameter update is involved.
        def residual(velocity):
            probe=root.clone();probe[:,7:10]=velocity
            result=taslab_velocity_control(probe,torch.zeros(1,3,dtype=torch.float64))
            forces=(result.normalized_rotor_commands[0]+1)*maximum/2
            if not bool(((forces>0)&(forces<maximum)).all()):raise ValueError('equilibrium prediction enters actuator saturation')
            torque=torch.tensor([c['torque_at_com_nm'] for c in columns],dtype=torch.float64).T@forces
            force=torch.tensor([c['force_n'] for c in columns],dtype=torch.float64).T@forces
            return torch.stack((torque[0],torque[1],force[2]-mass*9.81))
        velocity=torch.zeros(3,dtype=torch.float64)
        for _ in range(6):
            h=1e-4;basis=torch.eye(3,dtype=torch.float64)*h
            jacobian=torch.stack([(residual(velocity+b)-residual(velocity-b))/(2*h) for b in basis],dim=1)
            velocity-=torch.linalg.solve(jacobian,residual(velocity))
        predictions.append({'example':name,'scope':'source-and-measured-geometry equilibrium prediction; not new simulation',
            'predicted_velocity_world_m_s':velocity.tolist(),'observed_velocity_world_m_s':row['response']['velocity_world_m_s'],
            'equation_residual_torque_xy_nm_and_vertical_force_n':residual(velocity).tolist(),
            'mass_only_vertical_prediction_m_s':-(mass/TASLAB_UAV.mass-1)*9.81/TASLAB_UAV.lee_gains.velocity[2],
            'nominal_rotor_origin_is_not_actual_system_com':True})
    result={'status':'VALIDATED_PARTIAL_PREFIX_NOT_COMPLETE_PROTOCOL','source_identity':manifest['source_identity'],
        'raw_path':str(args.raw.resolve()),'raw_sha256':digest.hexdigest(),'validated_physics_rows':validator.count,
        'expected_full_protocol_rows':403200,'formal_control_acceptance':'NOT_GRANTED',
        'segments':summaries,'equilibrium_predictions':predictions,
        'mechanism_support':'Source mass/actual wrench geometry can be quantitatively checked by predictions; real repaired regression still required',
        'unavailable_wrenches':['drag_aero','contact','joint_constraint','other'],
        'analysis_tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'rows':validator.count,'predictions':predictions},indent=2))

if __name__=='__main__':main()
