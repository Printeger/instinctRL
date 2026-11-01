"""CPU verification of a closed, exact-input CUDA probability replay."""
import json
import math
from pathlib import Path
from tools.r6_evidence import checked, file_pin
from tools.r6_direct.import_legacy import process_identity


TRUSTED_SOURCES = [{'path': '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-04/source.json', 'sha256': '412ce757777d6ec9e4d09b897eac61af0a06fef220f126b737e9e541a19d6c6e'}, {'path': '/home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/train-02b/source.json', 'sha256': 'cd4025bffd1747fc45aca27136c0d74983d87012647c8564af6ee6b79d93e070'}]

def verify(diagnostic_pin, contract_pin, retained_pin):
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
    origin=contract['parameters']['r6_training_plan']['initialization']['checkpoint']
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
    trusted=[read(p) for p in TRUSTED_SOURCES]
    require(any(source==known for known in trusted),'AUDITED_IMPLEMENTATION')
    for name,digest in source['sources'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'SOURCE_PATH')
        source_pin=dict(path=str(Path(plan['snapshot'])/name),sha256=digest)
        checked(source_pin);pins.append(source_pin)
    require(envelope['source_trace']==plan['source_trace'],'SOURCE_TRACE')
    trace=read(envelope['source_trace'])
    for entry in trace['source_pins']:checked(entry);pins.append(entry)
    require(plan['script']==dict(path=str(Path(plan['snapshot'])/'tools/r6_direct/cuda_diagnostic.py'),
                                sha256=source['sources']['tools/r6_direct/cuda_diagnostic.py']),'SCRIPT')
    require(result['inputs']==request and result['training_transitions']==result['optimizer_updates']==0,'RESULT_BINDING')
    summary=result['result']
    require(summary['status']=='PASS' and summary['replay_device']=='cuda:0' and summary['sample_count']==512
            and summary['forward_batch_size']==512 and summary['float32_atol']==1e-6
            and summary['float32_rtol']==1e-5,'ORIGINAL_GATE')
    require(summary['starting_update']==contract['parameters']['r6_training_plan']['start_update'],'START_UPDATE')
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
