"""Non-authorizing projection of the CPU integer-delta probe into V5 capacity."""
import hashlib
import json
import math
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]/'docs/experiments/r6-robustness-auto'


def assess():
    paths=[BASE/'INTEGER_DELTA_CPU_V2.json',BASE/'CAPACITY_COMPOSED_DRAFT_V5.json']
    probe,design=[json.loads(p.read_text()) for p in paths]
    source=probe['source']
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
    assert [b['block_id'] for b in probe['blocks']]==[b['block_id'] for b in design['blocks']]
    mib=1024**2;gib=1024**3
    savings=[old['range_quota_bytes']-math.ceil(new['estimated_measurement_bytes']*1.15/(16*mib))*16*mib
             for old,new in zip(design['blocks'],probe['blocks'])]
    saved=sum(savings)
    final=math.ceil((design['final_total_bytes']-4*saved)/gib)*gib
    development_limit=128*gib-final
    baseline=design['baseline_only_total_bytes']-saved
    # Optimistic lower bound: credit an entire evaluation's saving to the one
    # affected-block followup as well. If this still fails, exact smaller credit
    # cannot make the full development example fit.
    optimistic_development=design['development_example_total_bytes']-2*saved
    return dict(status='PROBE_NOT_ADOPTED_FULL_GOAL_CAPACITY_STILL_INSUFFICIENT',capacity_pass=False,
        inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
        range_quota_savings_per_full_evaluation_bytes=saved,
        hypothetical_final_storage_protection_bytes=final,
        hypothetical_development_limit_bytes=development_limit,
        hypothetical_baseline_only_bytes=baseline,
        hypothetical_baseline_only_fits=baseline<=development_limit,
        optimistic_full_development_deficit_bytes=max(0,optimistic_development-development_limit),
        production_changed=False,ledger_changed=False,coverage_reduced=False,
        limitations=['Codec/header integration and independent acceptance absent',
            'Sample estimates are not bounds; native aggregate output remains unresolved',
            'Baseline-only fit is not permission to start or evidence that the complete goal fits',
            'No reduction of original/final coverage, native allowance, old snapshot reservations or repair margins'],
        initial_failure='INTEGER_DELTA_CPU.stderr preserves initial numeric case-label mismatch; corrected to float normalization before V2.')


if __name__=='__main__':print(json.dumps(assess(),indent=2))
