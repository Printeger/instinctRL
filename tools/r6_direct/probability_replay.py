"""Read-only saved first-rollout CUDA replay inside the direct diagnostic child."""
import json
from tools.r6_evidence import checked


def replay(request):
    import torch
    from training_workloads.unified_h1.lossless_evidence import ChunkReader
    from tools.r6_training_probability import recompute
    from training_workloads.unified_h1.direct_contract import validate_runtime_contract
    contract=validate_runtime_contract(json.loads(checked(request['contract']).read_text()))
    if contract['operation']!='train':raise ValueError('PROBABILITY_TRAIN_CONTRACT')
    plan=contract['parameters']['r6_training_plan']
    origin=torch.load(checked(plan['initialization']['checkpoint']),map_location='cpu',weights_only=True)
    name=f"pilot_diagnostic/update_{plan['start_update']+1}_full_input.pt"
    with ChunkReader(checked(request['retained']),run_id=plan['experiment_id'],logical_name=name,
                     maximum_raw_bytes=plan['storage']['maximum_raw_file_bytes']) as reader:
        data=torch.load(reader,map_location='cpu',weights_only=True)
    observations=[]
    backend=request.get('backend','default')
    if backend not in ('default','isaaclab_seeded_reset'):raise ValueError('PROBABILITY_BACKEND')
    before=dict(benchmark=torch.backends.cudnn.benchmark,deterministic=torch.backends.cudnn.deterministic)
    try:
        # Mirror configure_seed's backend flags without reseeding any RNG or
        # importing the simulator. All flags are restored even on gate failure.
        with torch.backends.cudnn.flags(enabled=torch.backends.cudnn.enabled,
                benchmark=True if backend=='isaaclab_seeded_reset' else before['benchmark'],
                deterministic=False if backend=='isaaclab_seeded_reset' else before['deterministic'],
                allow_tf32=torch.backends.cudnn.allow_tf32):
            result=recompute(origin,data,plan,device='cuda:0',diagnostic=observations.append)
    except (ValueError,RuntimeError) as error:
        result=dict(status='FAIL',error=repr(error))
    return dict(result=result,observations=observations,backend_before=before,inputs=request,torch_version=torch.__version__,
                device_name=torch.cuda.get_device_name(0),training_transitions=0,optimizer_updates=0,
                scientific_acceptance=False)
