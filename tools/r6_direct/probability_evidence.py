"""CPU verification of a closed, exact-input CUDA probability replay."""
import json
from pathlib import Path
from tools.r6_evidence import checked, file_pin
from tools.r6_direct.import_legacy import process_identity


def verify(diagnostic_pin, contract_pin, retained_pin):
    pins=[]
    def read(pin):
        value=json.loads(checked(pin).read_text());pins.append(pin);return value
    def require(ok,reason):
        if not ok:raise ValueError('DIRECT_PROBABILITY_'+reason)
    plan=read(diagnostic_pin)
    request=plan['probability_replay']
    require(request==dict(contract=contract_pin,retained=retained_pin,backend='isaaclab_seeded_reset'),'EXACT_INPUTS')
    read(contract_pin);checked(retained_pin);pins.append(retained_pin)
    root=Path(plan['output_root']);native=Path(plan['native_dir'])
    run=read(file_pin(root/'RUN_INPUTS.json'));intent=read(file_pin(root/'intent.json'))
    receipt=read(file_pin(root/'receipt.json'));result=read(file_pin(native/'probability_result.json'))
    require(run['plan']==plan and intent['plan']==plan and intent['parent']==receipt['parent'],'RUN_BINDING')
    require(receipt['status']=='CLOSED' and receipt['exit_code']==0 and receipt['error'] is None
            and receipt['remaining']==[],'CLEAN_EXIT')
    require(all(process_identity(x['pid'])!=x for x in [receipt['parent'],*receipt['owned_processes']]),'LIVE')
    source=read(plan['helper_source']);require(source['source_commit']==plan['source_commit'],'SOURCE_COMMIT')
    for name in ('tools/r6_training_probability.py','tools/r6_direct/probability_replay.py','tools/r6_direct/cuda_diagnostic.py'):
        source_pin=dict(path=str(Path(plan['snapshot'])/name),sha256=source['sources'][name])
        checked(source_pin);pins.append(source_pin)
        require(file_pin(Path(__file__).resolve().parents[2]/name)['sha256']==source_pin['sha256'],'AUDITED_IMPLEMENTATION')
    require(plan['script']==dict(path=str(Path(plan['snapshot'])/'tools/r6_direct/cuda_diagnostic.py'),
                                sha256=source['sources']['tools/r6_direct/cuda_diagnostic.py']),'SCRIPT')
    require(result['inputs']==request and result['training_transitions']==result['optimizer_updates']==0,'RESULT_BINDING')
    summary=result['result']
    require(summary['status']=='PASS' and summary['replay_device']=='cuda:0' and summary['sample_count']==512
            and summary['forward_batch_size']==512 and summary['float32_atol']==1e-6
            and summary['float32_rtol']==1e-5,'ORIGINAL_GATE')
    require(len(result['observations'])==1,'OBSERVATIONS')
    observation=result['observations'][0]
    require(observation['backend']['cudnn_benchmark'] is True
            and observation['backend']['cudnn_deterministic'] is False
            and all(observation['metrics'][k]['failed_count']==0 for k in ('alpha','beta','log_probability')),'METRICS')
    return dict(summary,verification='closed exact-input CUDA diagnostic',diagnostic=diagnostic_pin),pins
