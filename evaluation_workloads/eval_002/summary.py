"""Report-only reductions from accepted saved M7 inputs; CPU replayable."""
from collections import OrderedDict
from dataclasses import asdict
import torch
from instinctlab.evaluation.metrics import compute_step_metrics, aggregate_metrics


def metric_rows(inputs):
    return [asdict(m) for m in aggregate_metrics(compute_step_metrics(inputs)).metrics]


def summarize(runtime, rows):
    inputs = runtime['evaluation_inputs']
    t,n = inputs['commanded_velocity_b_m_s'].shape[:2]
    summaries=[]
    def append(label,indices,envs):
        selected=OrderedDict((k,v[indices][:,envs].detach().cpu().clone()) for k,v in inputs.items())
        cmd=torch.linalg.vector_norm(selected['commanded_velocity_b_m_s'],dim=-1)
        nonzero=cmd>1e-3
        ratio=torch.linalg.vector_norm(selected['issued_velocity_b_m_s'],dim=-1)[nonzero]/cmd[nonzero]
        summaries.append(dict(label=label,metrics=metric_rows(selected),
            issued_amplification_nonzero=dict(available=bool(ratio.numel()),
                value=float(ratio.mean()) if ratio.numel() else None,denominator=ratio.numel(),
                reason=None if ratio.numel() else 'no nonzero command samples'),
            rank3_count=int((runtime['observability_details']['numerical_rank'][indices][:,envs]==3).sum()),
            sample_count=len(indices)*len(envs)))
    append('all',list(range(t)),list(range(n)))
    for block in range((t+31)//32):
        append(f'block_{block+1}',list(range(block*32,min((block+1)*32,t))),list(range(n)))
    for env in range(n):
        start=0; episode=0
        for i in range(t):
            if bool(runtime['reset_masks'][i,env]) or i==t-1:
                append(f'env_{env}_episode_{episode}',list(range(start,i+1)),[env]); start=i+1; episode+=1
    return dict(schema_version='eval-002-summary.v1',performance_gate=False,reductions=summaries,
        terminal_samples=[dict(global_index=i,env=e,reason=int(inputs['termination_reason'][i,e]))
                          for i in range(t) for e in range(n) if bool(runtime['reset_masks'][i,e])],
        visibility=[r['visibility'] for r in rows])


def main():
    import json
    import subprocess
    from pathlib import Path
    from attempt_binding import binding_from_environment, bound_path
    from constants import COORDINATION_ROOT, PRODUCT_ROOT, PRODUCT_COMMIT, PRODUCT_TREE, CHECKPOINT_PATH, CHECKPOINT_SHA256
    from io_utils import atomic_canonical_replace,sha256_file
    from semantic_validator import validate_files
    binding=binding_from_environment()
    cases={}
    for case in ('station','tracking'):
        root=binding.directory/'results'/case
        validation=validate_files(binding,root/'semantic.pt',root/'semantic.json',root/'receipt.json',root/'status.json',root/'kit.log',run_id=case)
        semantic=torch.load(root/'semantic.pt',map_location='cpu',weights_only=True)
        cases[case]=dict(validation=validation,summary=semantic['diagnostic_summary'],initial_observability=semantic['initial_observability'],
                         physics_accounting=semantic['physics_accounting'],canonical_record=json.loads(semantic['runtime_evaluation']['canonical_record']))
    if cases['station']['validation']['pid']==cases['tracking']['validation']['pid']:
        raise ValueError('diagnostic cases require distinct processes')
    starts=[]
    for path in sorted(binding.attempts_root.glob('EVAL-002-D000[1-4]/results/*/process_start.json')):
        entry=json.loads(path.read_text());entry['source_path']=str(path);entry['source_sha256']=sha256_file(path)
        progress=path.parent/'progress.json'
        entry['attempted_transitions']=json.loads(progress.read_text())['attempted_environment_transitions'] if progress.exists() else 0
        starts.append(entry)
    if len(starts)>4 or sum(x['attempted_transitions'] for x in starts)>2048:
        raise ValueError('campaign budget exceeded')
    commands=[]
    def git(root,*args):
        result=subprocess.run(['git',*args],cwd=root,text=True,capture_output=True)
        commands.append(dict(argv=['git',*args],cwd=str(root),stdout=result.stdout,stderr=result.stderr,exit=result.returncode))
        if result.returncode:raise ValueError('Git identity/diff check failed')
        return result.stdout.strip()
    if (git(PRODUCT_ROOT,'rev-parse','HEAD'),git(PRODUCT_ROOT,'rev-parse','HEAD^{tree}'),git(PRODUCT_ROOT,'status','--porcelain')) != (PRODUCT_COMMIT,PRODUCT_TREE,''):
        raise ValueError('product identity drift')
    if sha256_file(CHECKPOINT_PATH)!=CHECKPOINT_SHA256:
        raise ValueError('checkpoint mutated')
    base='16751691bad95a223c20e72eed4abd034ceb9fdc'
    head=git(COORDINATION_ROOT,'rev-parse','HEAD')
    names=git(COORDINATION_ROOT,'diff','--name-only',base+'..HEAD').splitlines()
    allowed={'evaluation_workloads/eval_002/'+n for n in ('__init__.py','constants.py','attempt_binding.py','io_utils.py','runtime_support.py','gpu_gate.py','checkpoint_audit.py','scene.py','diagnostic_evaluator.py','semantic_validator.py','summary.py','fixture.json','README.md')}
    if any(n not in allowed and not (n.startswith('tests/evaluation_workloads/test_eval_002') and n.endswith('.py')) for n in names):
        raise ValueError('candidate allowlist violation')
    git(COORDINATION_ROOT,'diff','--check',base+'..HEAD')
    diff_names=git(COORDINATION_ROOT,'diff','--name-status',base+'..HEAD')
    diff_stat=git(COORDINATION_ROOT,'diff','--stat',base+'..HEAD')
    chain=git(COORDINATION_ROOT,'log','--format=%H %P %s',base+'..HEAD')
    result=dict(status='PASS',order_id='EVAL-002',base_commit=base,head_commit=head,chain=chain,
                allowed_path_ledger=[dict(path=n,allowed=True,sha256=sha256_file(COORDINATION_ROOT/n)) for n in names],
                name_status=diff_names,stat=diff_stat,read_only_commands=commands,cases=cases,process_starts=starts,
                total_attempted_environment_transitions=sum(x['attempted_transitions'] for x in starts),
                training=False,optimizer_updates=0,checkpoint_writes=0,product_changes=False,
                recommendation='READY_TO_REQUEST_1M',
                limitations=['Fixed plane-only training and corner evaluation differ in distribution; scene/curriculum for 1M remains a Commander decision.','One deterministic process per case is diagnostic coverage, not policy convergence or statistical generalization.'])
    output=bound_path(binding,binding.directory/'results/consolidated.json','results/consolidated.json',must_exist=False)
    atomic_canonical_replace(output,result)
    print(json.dumps(dict(status='PASS',process_starts=len(starts),transitions=result['total_attempted_environment_transitions'])))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
