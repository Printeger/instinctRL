"""Build an exact R6 evaluation or training Store proposal from a validated frozen authority.

CPU-only, no Store mutation/approval, process launch or runtime import here.
The existing H1 inspector validates all source/config/authorization bindings.
"""
import argparse
import json
from pathlib import Path

from rlflow import experiments
from tools.r6_recovery import checked,file_pin,RecoveryError

from tools.r6_job_resources import MODES, resources as job_resources


def build(config_pin,authorization_pin,attempt,stage_id,conditions,store_root):
    config_path=checked(config_pin)
    config=json.loads(config_path.read_text())
    authority=json.loads(checked(authorization_pin).read_text())
    mode=authority.get('mode')
    replay=mode=='research-r6-video-replay'
    if mode not in MODES and not replay:raise RecoveryError('SPEC_R6_MODE')
    work=Path(config['workload']['path']);attempt=Path(attempt)
    if (not work.is_absolute() or work.resolve()!=work or work.is_symlink()
            or config_path!=work/'training_workloads/unified_h1/candidate_config.json'
            or not attempt.is_absolute() or attempt.resolve()!=attempt or attempt.is_symlink()):
        raise RecoveryError('SPEC_EXACT_PATHS')
    parameters=authority['parameters'];budget=authority['budgets']
    if mode!=MODES[1] and budget['training_transitions']!=0:raise RecoveryError('SPEC_BASELINE_NO_TRAINING')
    executable=file_pin(config['execution']['python_path'])
    if executable['sha256']!=config['execution']['python_sha256']:
        raise RecoveryError('SPEC_INTERPRETER_DRIFT')
    limit=budget['wall_seconds']+10
    module='tools.run_r6_render' if replay else 'training_workloads.unified_h1.formal_unified_h1'
    argv=[executable['path'],'-B','-m',module,'--authorization',authorization_pin['path'],'--attempt',str(attempt)]
    if not replay:argv+=['--mode',mode]
    spec=dict(id=parameters['experiment_id'],stage_id=stage_id,backend='real',
        argv=argv,
        cwd=str(work),pins=dict(dispatcher=file_pin(work/'training_workloads/unified_h1/local_dispatch.py'),
            executable=executable,product=file_pin(Path(config['product']['path'])/('isaaclab-training/instinctlab/task/scene_h1.py' if replay else 'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py')),
            workload=file_pin(work/('tools/run_r6_render.py' if replay else 'training_workloads/unified_h1/formal_unified_h1.py')),
            runtime=dict(executable),authorization=dict(authorization_pin)),
        method='r6-recorded-trajectory-replay-v1' if replay else config['method']['version'],
        evaluation='NO_NEW_PHYSICS' if replay else config['versions']['metric'],
        runtime=config['execution']['frozen_runtime_root'],seed=0 if replay else parameters['training_seed' if mode==MODES[1] else 'evaluation_seed'],
        resources=dict(starts=1,wall_seconds=limit,gpu_seconds=limit,transitions=budget['nontraining_transitions']+budget['training_transitions']),
        conditions=[dict(c) for c in conditions],
        observations=['status','counters','process_identity','process_start'],
        h1=dict(workload_root=str(work),attempt=str(attempt),mode=mode,
            config_sha256=config_pin['sha256'],gpu_devices=['0'],gpu_accounting='allocated_device_wall_seconds'))
    job_resources(spec,authority)
    experiments.validate(spec,Path(store_root))
    return spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs',type=Path)
    p.add_argument('--inputs-sha256',required=True)
    args=p.parse_args()
    inputs=json.loads(checked(dict(path=str(args.inputs.resolve()),sha256=args.inputs_sha256)).read_text())
    if set(inputs)!={'config_pin','authorization_pin','attempt','stage_id','conditions','store_root'}:
        raise RecoveryError('SPEC_INPUT_FIELDS')
    print(json.dumps(build(**inputs),sort_keys=True))


if __name__=='__main__':main()
