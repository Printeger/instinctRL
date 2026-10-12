"""Generate reviewable future pilot proposals, without authorization or launch.

Evaluation checkpoint hashes must come from the future accepted training output.
"""
import argparse
import hashlib
import json
from pathlib import Path
from training_workloads.unified_h1.budget import required_mode_budget

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline-input-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve existing proposals')
    baseline = json.loads(args.baseline_input_manifest.read_text())
    snapshot = Path(baseline['snapshot'])
    config_path = snapshot / 'training_workloads/unified_h1/candidate_config.json'
    config = json.loads(config_path.read_text())
    def pin(path):
        return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    pins = {'dispatcher':pin(snapshot/'training_workloads/unified_h1/local_dispatch.py'),
            'executable':pin(config['execution']['python_path']),
            'product':pin(snapshot/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py'),
            'workload':pin(snapshot/'training_workloads/unified_h1/formal_unified_h1.py'),
            'runtime':pin(config['execution']['python_path'])}
    rows = []
    for group in ('P0','PLR'):
        for mode, suffix in (('pilot-training','TRAIN'),('pilot-evaluation','EVAL')):
            order_id = f'UNIFIED-H1-R5-{group}-PILOT-{suffix}'
            attempt = ROOT/'coordination/research/attempts'/(order_id+'-BASELINE-'+baseline['snapshot_commit'][:12]+'-20261012')
            parameters = {'group':group}
            if suffix == 'EVAL':
                train = ROOT/'coordination/research/attempts'/(f'UNIFIED-H1-R5-{group}-PILOT-TRAIN-BASELINE-'+baseline['snapshot_commit'][:12]+'-20261012')
                parameters['checkpoints'] = [{'path':str(train/'results/pilot-training/checkpoints'/f'u{u}.pt'),
                                              'sha256':None,'update':u} for u in (16,64)]
            budget = required_mode_budget(mode,parameters)
            authorization_path = attempt/'commands/AUTHORIZATION.json'
            authorization = {'schema_version':'unified-h1-r3-authorization.v2',
                'status':'UNISSUED_REQUIRES_COMMANDER_APPROVAL','order_id':order_id,
                'method_sha256':config['method']['proposal_sha256'],
                'config_sha256':pin(config_path)['sha256'],
                'product_commit':baseline['snapshot_commit'],'product_tree':baseline['snapshot_tree'],
                'workload_commit':baseline['snapshot_commit'],'workload_tree':baseline['snapshot_tree'],
                'attempt':str(attempt),'mode':mode,'budgets':budget,'parameters':parameters}
            spec = {'id':order_id+'-BASELINE-'+baseline['snapshot_commit'][:12]+'-20261012',
                'stage_id':'R5-PAIRED-PILOT-NEXT-AUTHORIZATION','backend':'real',
                'argv':[config['execution']['python_path'],'-B','-m','training_workloads.unified_h1.formal_unified_h1',
                        '--authorization',str(authorization_path),'--attempt',str(attempt),'--mode',mode],
                'cwd':str(snapshot),'pins':dict(pins,authorization={'path':str(authorization_path),'sha256':None}),
                'method':config['method']['version'],'evaluation':config['versions']['metric'],
                'runtime':config['execution']['frozen_runtime_root'],
                'seed':config['training']['seed' if suffix=='TRAIN' else 'evaluation_seed'],
                'resources':{'starts':1,'wall_seconds':budget['wall_seconds']+10,
                    'gpu_seconds':budget['wall_seconds']+10,
                    'transitions':budget['training_transitions']+budget['nontraining_transitions']},
                'conditions':[], 'observations':['status','counters','process_identity','process_start','cleanup',
                    'pilot_diagnostic' if suffix=='TRAIN' else 'evaluation_raw'],
                'h1':{'workload_root':str(snapshot),'attempt':str(attempt),'mode':mode,
                    'config_sha256':pin(config_path)['sha256'],'gpu_devices':['0'],
                    'gpu_accounting':'allocated_device_wall_seconds'}}
            rows.append({'group':group,'mode':mode,'authorization_issued':False,
                'authorization_proposal':authorization,'experiment_proposal':spec,
                'output':str(attempt/'results'/mode),
                'runtime_resource_checks_maximum':1225 if suffix=='TRAIN' else 403,
                'prelaunch_nvidia_smi_calls':6,'automatic_retries':0,
                'missing_bindings':['new user training/evaluation envelope and Commander approval',
                    'accepted final readiness evidence predicate','actual authorization bytes/hash']+
                    (['accepted future u16/u64 checkpoint bytes/hashes/source identity'] if suffix=='EVAL' else [])})
    output = {'schema_version':'instinctrl-r5-future-pilot-proposals/1',
        'status':'PROPOSALS_ONLY_NOT_AUTHORIZED','baseline':baseline,
        'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'current_goal_execution_authority':{'training_transitions':0,'optimizer_updates':0,'training_checkpoint_writes':0},
        'future_total':{'process_starts':4,'simulator_initializations':4,'training_transitions':65536,
            'nontraining_evaluation_transitions':25600,'updates':128,'training_checkpoint_writes':14,
            'outer_wall_seconds_including_cleanup':10840,'gpu_allocation_seconds':10840},
        'groups':rows,
        'issue_sequence':['Obtain new explicit training/evaluation envelope; preserve this Goal zero-training boundary.',
            'Verify exact shared readonly baseline/config/tools and accepted final readiness evidence.',
            'Create fresh train attempt commands/results directories; never reuse prior run.',
            'Only Commander after approval sets authorization status AUTHORIZED and writes actual byte file; bind its SHA256 to proposal.',
            'Add accepted independent evidence predicates; experiments.validate/propose; policy.show_proposal/decide; experiments.submit once.',
            'Entry checks STOP/ownership and six coexistence probes before AppLauncher; first real rollout probability gate must pass before first optimizer step.',
            'After accepted training, mechanically SHA256 u16/u64 and bind checkpoint source identities; separately approve each evaluation.',
            'Preserve partial failures and settle owned cleanup and actual resources before another release.']}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'proposals':len(rows),'training_launched':False}))

if __name__ == '__main__':
    main()
