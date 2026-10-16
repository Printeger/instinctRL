"""Explicit CPU reanalysis of retained evidence; never backfill a runtime binding.

Only this frozen case may use the alternative evidence route. CPU checks propose
candidates; independent review must decide whether original substantive criteria
are met. The ordinary analyzer still requires the original binding file.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from tools.analyze_r5_formal_segment import (ROOT, BASE, CASE, training_analysis,
    verify_probe, validate_update_logs, read, jsonl, digest)

SOURCE = 'feb1e96f97e44c7d872cec44268fc2d8b352bc93'
FILES = ['training_workloads/unified_h1/formal_segment.py',
         'training_workloads/unified_h1/mean_prior.py',
         'training_workloads/unified_h1/driver.py',
         'training_workloads/unified_h1/logging.py',
         'training_workloads/unified_h1/backend.py',
         'isaaclab-training/instinctlab/training/ppo.py']


def pin(p): return {'path':str(Path(p).resolve()),'sha256':digest(Path(p))}


def check_scalars(updates, mini, weight, config):
    validate_update_logs(updates, mini, weight)
    assert all(type(r['actor_auxiliary_loss']) in (int,float) for r in mini)
    if weight == 10: assert all(r['actor_auxiliary_loss'] > 0 for r in mini)
    errors=[]
    for r in mini:
        # Original float32 arithmetic and unchanged original comparison tolerance.
        f=lambda k:torch.tensor(r[k],dtype=torch.float32)
        expected=f('actor_loss')+config['value_loss_coefficient']*f('value_loss')-config['entropy_coefficient']*f('entropy')+f('actor_auxiliary_loss')
        assert torch.isclose(expected,f('total_loss'),atol=1e-6,rtol=1e-5), 'auxiliary not included in recorded total loss'
        errors.append(abs(float(expected)-r['total_loss']))
    return max(errors)


def original_log_binding(directory, group, out, auth):
    assert directory.resolve()==(BASE/'experiments'/CASE).resolve()
    assert auth['product_commit']==auth['workload_commit']==SOURCE
    p=auth['parameters'];assert p['group']==group and p['mean_prior_weight']=={'R0':0.,'R1':10.}[group]
    assert not (out/'actor_mean_prior_binding.json').exists(), 'only the documented absent-file case'
    spec=read(directory/'inputs'/f'{group}-research-formal-training.authorized.json')
    snap=Path(spec['cwd']);sources={}
    for name in FILES:
        actual=(snap/name).read_bytes()
        committed=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)
        assert actual==committed, 'frozen source differs from exact commit: '+name
        sources[name]=actual.decode()
    # Exact frozen control flow is separately reviewed, not inferred from scores.
    ppo=ast.parse(sources[FILES[-1]])
    conditional=[n for n in ast.walk(ppo) if isinstance(n,ast.If) and ast.unparse(n.test)=='auxiliary is not None']
    assert len(conditional)==1
    assert 'records[-1][\'actor_auxiliary_loss\'] = float(auxiliary.detach().item())' in ast.unparse(conditional[0])
    text=sources[FILES[-1]]
    assert text.index('auxiliary = self.actor_auxiliary(policy, output)') < text.index('total_loss.backward()') < text.index('self.actor_optimizer.step()')
    entry=sources[FILES[0]]
    assert entry.index('meta = backend._load_evaluation_checkpoint') < entry.index('configure_trainer(backend.trainer, p)') < entry.index('backend.loop.run_update(')
    updates=jsonl(out/'learning_log/updates.jsonl');mini=jsonl(out/'learning_log/minibatches.jsonl')
    origin_path=Path(p['initialization_checkpoint']['path'])
    assert digest(origin_path)==p['initialization_checkpoint']['sha256']
    origin=torch.load(origin_path,map_location='cpu',weights_only=True)
    error=check_scalars(updates,mini,p['mean_prior_weight'],origin['ppo_config'])
    data_path=out/'pilot_diagnostic/update_257_full_input.pt';probe_path=out/'pilot_diagnostic/first_update_probe.pt'
    data=torch.load(data_path,map_location='cpu',weights_only=True)
    probe=torch.load(probe_path,map_location='cpu',weights_only=True)
    verify_probe(data,probe,mini,p,origin)
    # Preserve the second actually retained rollout and establish its finite shape.
    second=out/'pilot_diagnostic/update_258_full_input.pt'
    second_data=torch.load(second,map_location='cpu',weights_only=True)
    assert second_data['update']==258 and second_data['actions'].shape==(512,4)
    from instinctlab.training.checkpoint_h1 import _finite_tree
    _finite_tree(second_data,'retained258')
    return {'kind':'ORIGINAL_MINIBATCH_CALLBACK_EXECUTION_AND_SAVED_PROBE',
      'missing_runtime_binding_file':True,'runtime_binding_backfilled':False,
      'independent_adjudication_required':True,
      'claimed_weight':p['mean_prior_weight'],'original_runtime_callback_records':len(mini),
      'original_total_loss_formula_maximum_absolute_error':error,
      'first8_original_actor_outputs_formula_verified':True,
      'source_commit':SOURCE,'sources':[pin(snap/f) for f in FILES],
      'original_inputs':[pin(out/'learning_log/minibatches.jsonl'),pin(data_path),pin(probe_path),pin(second),pin(origin_path)],
      'scope':'Actual callback execution logged in all2048 minibatches; first8 coefficient formula rechecked; actor-only/no action override from exact frozen source. Not a new runtime observation.',
      'limits':'Later complete per-minibatch actor outputs were not retained; no claim of independently recomputing all2048 auxiliary formulas. Existing missing-file STOP preserved.'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('group',choices=['R0','R1']);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    assert not a.output.exists()
    result=training_analysis(BASE/'experiments'/CASE,a.group,binding_verifier=original_log_binding)
    result['status']='CPU_RECOVERY_CANDIDATE_REQUIRES_INDEPENDENT_ADJUDICATION'
    result['scientific_acceptance_granted']=False
    result['script']=pin(__file__)
    from rlflow.store import atomic
    atomic(a.output,(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps({'status':result['status'],'output':pin(a.output)}))

if __name__=='__main__':main()
