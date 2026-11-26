"""CPU verification of a closed, exact-input CUDA probability replay."""
import json
import math
from pathlib import Path
from tools.r6_evidence import checked
from tools.r6_direct.import_legacy import ROOT, process_identity


TRUSTED_SOURCES = [{'path': '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-04/source.json', 'sha256': '412ce757777d6ec9e4d09b897eac61af0a06fef220f126b737e9e541a19d6c6e'}, {'path': '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/train-02b/source.json', 'sha256': 'cd4025bffd1747fc45aca27136c0d74983d87012647c8564af6ee6b79d93e070'}]

TRUSTED_SOURCES.append({'path': '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-06/source.json', 'sha256': 'e1e5de0a7b193631c63a1de8ae117bc98df3bcddb517b57e4199eb5ee6b62734'})

_NEXT_LONG_DOCS = ROOT / 'docs/experiments/instinct-post-r4-next-long'
_NEXT_LONG_SNAPSHOT = Path('/home/mint/instinctrl-runs/INSTINCT-POST-R4-NEXT-LONG-SOURCE-02')
_NEXT_LONG_HELPER_SOURCE = dict(path=str(_NEXT_LONG_DOCS/'source-bf677e94115e.json'),
    sha256='6d8800c2492e63ddef332e549805cba50db627df69497a2e1ee32d24fd2dafef')
_NEXT_LONG_SOURCE_TRACE = dict(path=str(_NEXT_LONG_DOCS/'PROBABILITY_SOURCE_TRACE.json'),
    sha256='a99df07bbd605fed2e196b5a928355636acf97380252fb269bee245f966f5328')
_SCRIPT = 'tools/r6_direct/cuda_diagnostic.py'


def next_long_helper():
    """Use the independently pinned child that can validate the u768 fork."""
    manifest = json.loads(checked(_NEXT_LONG_HELPER_SOURCE).read_text())
    checked(_NEXT_LONG_SOURCE_TRACE)
    script = dict(path=str(_NEXT_LONG_SNAPSHOT/_SCRIPT),sha256=manifest['sources'][_SCRIPT])
    checked(script)
    return dict(snapshot=str(_NEXT_LONG_SNAPSHOT),source_commit=manifest['source_commit'],
                helper_source=dict(_NEXT_LONG_HELPER_SOURCE),script=script,
                source_trace=dict(_NEXT_LONG_SOURCE_TRACE))


def verify(diagnostic_pin, contract_pin, retained_pin):
    from training_workloads.unified_h1.dynamic_assay import NEXT_LONG_PREFIX

    pins=[]
    def read(pin):
        value=json.loads(checked(pin).read_text());pins.append(pin);return value
    def require(ok,reason):
        if not ok:raise ValueError('DIRECT_PROBABILITY_'+reason)
    envelope=read(diagnostic_pin)
    require(envelope['schema_version']=='r6-probability-sealed-evidence/1','SEALED_SCHEMA')
    plan=read(envelope['plan'])
    request=plan['probability_replay']
    require(request==dict(contract=contract_pin,retained=retained_pin,backend='isaaclab_seeded_reset'),'EXACT_INPUTS')
    contract=read(contract_pin);checked(retained_pin);pins.append(retained_pin)
    training=contract['parameters']['r6_training_plan']
    origin=training['initialization']['checkpoint']
    require(envelope['origin']==origin,'ORIGIN')
    checked(origin);pins.append(origin)
    root=Path(plan['output_root']);native=Path(plan['native_dir'])
    for key,path in dict(run_inputs=root/'RUN_INPUTS.json',intent=root/'intent.json',receipt=root/'receipt.json',result=native/'probability_result.json').items():
        require(envelope[key]['path']==str(path),'SEALED_PATH')
    run=read(envelope['run_inputs']);intent=read(envelope['intent'])
    receipt=read(envelope['receipt']);result=read(envelope['result'])
    require(run['plan']==plan and intent['plan']==plan and intent['parent']==receipt['parent'],'RUN_BINDING')
    require(receipt['status']=='CLOSED' and receipt['exit_code']==0 and receipt['error'] is None
            and receipt['remaining']==[],'CLEAN_EXIT')
    require(all(process_identity(x['pid'])!=x for x in [receipt['parent'],*receipt['owned_processes']]),'LIVE')
    source=read(plan['helper_source']);require(source['source_commit']==plan['source_commit'],'SOURCE_COMMIT')
    run_id=training.get('experiment_id','')
    if run_id.startswith(NEXT_LONG_PREFIX):
        require(all(plan[key]==value for key,value in next_long_helper().items()),'AUDITED_IMPLEMENTATION')
        require(plan['run_id'].startswith(run_id+'-DIAGNOSTIC'),'DIAGNOSTIC_ID')
        launcher=plan['launcher_script'];checked(launcher);pins.append(launcher)
        tail=[source['execution']['python_path'],'-B',plan['script']['path'],
              '--plan',envelope['plan']['path'],'--child']
        snapshot=plan['snapshot']
        require(run['argv'][-len(tail):]==tail
            and run['environment']['PYTHONPATH']==snapshot+':'+snapshot+'/isaaclab-training','CHILD_EXECUTION')
    else:
        trusted=[read(p) for p in TRUSTED_SOURCES]
        require(any(source==known for known in trusted),'AUDITED_IMPLEMENTATION')
    for name,digest in source['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'SOURCE_PATH')
        source_pin=dict(path=str(Path(plan['snapshot'])/name),sha256=digest)
        checked(source_pin);pins.append(source_pin)
    require(envelope['source_trace']==plan['source_trace'],'SOURCE_TRACE')
    trace=read(envelope['source_trace'])
    for entry in trace['source_pins']:checked(entry);pins.append(entry)
    require(plan['script']==dict(path=str(Path(plan['snapshot'])/_SCRIPT),sha256=source['sources'][_SCRIPT]),'SCRIPT')
    require(result['inputs']==request and result['training_transitions']==result['optimizer_updates']==0,'RESULT_BINDING')
    summary=result['result']
    require(summary['status']=='PASS' and summary['replay_device']=='cuda:0' and summary['sample_count']==512
            and summary['forward_batch_size']==512 and summary['float32_atol']==1e-6
            and summary['float32_rtol']==1e-5,'ORIGINAL_GATE')
    require(summary['starting_update']==training['start_update'],'START_UPDATE')
    require(len(result['observations'])==1,'OBSERVATIONS')
    observation=result['observations'][0]
    require(observation['backend']['cudnn_benchmark'] is True
            and observation['backend']['cudnn_deterministic'] is False
            and all(observation['metrics'][k]['failed_count']==0 for k in ('alpha','beta','log_probability')),'METRICS')
    require(observation['replay_device']==summary['replay_device']
            and observation['forward_batch_size']==summary['forward_batch_size']
            and observation['atol']==summary['float32_atol'] and observation['rtol']==summary['float32_rtol'],'OBSERVATION_IDENTITY')
    for key in ('alpha','beta','log_probability'):
        error=summary['maximum_absolute_errors'][key]
        require(type(error) in (int,float) and math.isfinite(error) and error>=0
                and error==observation['metrics'][key]['maximum_absolute_error'],'ERROR_CONSISTENCY')
    return dict(summary,verification='closed exact-input CUDA diagnostic',diagnostic=diagnostic_pin),pins
