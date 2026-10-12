"""Bind a separate fixed-command B0 attempt to the exact repaired snapshot.

CPU preparation only; expected authorization is not an issued authorization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from training_workloads.unified_h1.control_diagnostic import COMMAND_PASSTHROUGH_ID, validate_parameters
from training_workloads.unified_h1.budget import required_mode_budget
from rlflow.policy import stopped

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enhanced-input-manifest', type=Path, required=True)
    parser.add_argument('--prefix', default='passthrough_r1')
    args = parser.parse_args()
    if stopped(ROOT / 'coordination/research'):
        raise RuntimeError('STOP forbids new preparation')
    if not args.prefix.replace('_', '').isalnum():
        raise ValueError('invalid prefix')
    prior = json.loads(args.enhanced_input_manifest.read_text())
    snapshot = Path(prior['snapshot'])
    config = snapshot / 'training_workloads/unified_h1/candidate_config.json'
    enhanced_auth_path = args.enhanced_input_manifest.with_name(args.enhanced_input_manifest.name.replace('_input_manifest', '_authorization.expected'))
    enhanced = json.loads(enhanced_auth_path.read_text())
    if sha(config) != enhanced['config_sha256']:
        raise ValueError('shared baseline config changed')
    attempt = ROOT / 'coordination/research/attempts' / COMMAND_PASSTHROUGH_ID
    if attempt.exists() and any(attempt.rglob('*')):
        raise ValueError('never reuse an existing attempt')
    ledger = json.loads((ROOT / 'docs/diagnostics/r5-readiness/goal_budget.json').read_text())
    remaining = {k: ledger['caps'][k] - ledger['used'][k] - ledger.get('reserved', {}).get(k, 0)
                 for k in ('live_starts', 'simulator_initializations', 'nontraining_transitions', 'wall_seconds', 'gpu_seconds', 'output_bytes')}
    # This pair is only feasible when both still fit; neither draft reserves or releases a run.
    limits = {'live_starts':2,'simulator_initializations':2,'nontraining_transitions':40320,
              'wall_seconds':1820,'gpu_seconds':1820,'output_bytes':32*1024**3}
    if any(remaining[k] < v for k, v in limits.items()):
        raise ValueError('both distinct conditions must fit remaining envelope')
    outputs = ROOT / 'docs/diagnostics/r5-readiness/inputs'
    def write(name, value):
        path = outputs / (args.prefix + '_' + name + '.json')
        if path.exists():
            raise ValueError('preserve existing input')
        path.write_text(json.dumps(value, indent=2) + '\n')
        return path
    parameters = validate_parameters({'condition':'B0','experiment_id':COMMAND_PASSTHROUGH_ID,
        'checkpoint':None,'checkpoint_sha256':None,'checkpoint_source_identity':None,
        'cases':['Z','T','S'],'transitions':20160})
    (attempt / 'commands').mkdir(parents=True)
    (attempt / 'results').mkdir()
    auth = dict(enhanced, order_id=COMMAND_PASSTHROUGH_ID, attempt=str(attempt),
                mode='control-diagnostic', parameters=parameters,
                budgets=required_mode_budget('control-diagnostic', parameters))
    expected = write('authorization.expected', auth)
    spec_path = args.enhanced_input_manifest.with_name(args.enhanced_input_manifest.name.replace('_input_manifest', '_experiment_spec.draft'))
    spec = json.loads(spec_path.read_text())
    actual_auth = attempt / 'commands/AUTHORIZATION.json'
    spec.update(id=COMMAND_PASSTHROUGH_ID)
    spec['argv'] = [spec['argv'][0], '-B', '-m', 'training_workloads.unified_h1.formal_unified_h1',
                    '--authorization',str(actual_auth),'--attempt',str(attempt),'--mode','control-diagnostic']
    spec['pins']['authorization'] = {'path':str(actual_auth),'sha256':sha(expected)}
    spec['h1'].update(attempt=str(attempt),mode='control-diagnostic')
    spec['observations'] = ['status','counters','process_identity','process_start','control_trace','evaluation_raw','cleanup','partial_failure']
    draft = write('experiment_spec.draft', spec)
    write('input_manifest', {'schema_version':'instinctrl-r5-passthrough-inputs/1',
        'status':'PREPARED_NOT_RELEASED','experiment_id':COMMAND_PASSTHROUGH_ID,
        'snapshot':str(snapshot),'snapshot_commit':prior['snapshot_commit'],'snapshot_tree':prior['snapshot_tree'],
        'snapshot_diff_sha256':prior['snapshot_diff_sha256'],'config_sha256':sha(config),
        'enhanced_input_manifest':{'path':str(args.enhanced_input_manifest.resolve()),'sha256':sha(args.enhanced_input_manifest)},
        'authorization_issued':False,'approval_issued':False,'read_only':True,
        'condition':'B0_COMMAND_PASSTHROUGH','action':[1.,.5,.5,.5],
        'policy_forward_calls':0,'training_transitions':0,'optimizer_updates':0,'checkpoint_writes':0,
        'expected_authorization_sha256':sha(expected),'draft_spec_sha256':sha(draft),
        'supplemental_limits':prior['supplemental_limits'],'remaining_goal_budget':remaining,
        'release_dependency':'independent review of completed enhanced regression; separate Commander approval',
        'free_disk_bytes_observed':shutil.disk_usage(attempt).free,'generator_sha256':sha(__file__)})
    print(json.dumps({'attempt':str(attempt),'snapshot':str(snapshot),'status':'PREPARED_NOT_RELEASED'}))

if __name__ == '__main__':
    main()
