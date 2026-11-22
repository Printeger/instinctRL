"""Aggregate the frozen third-training holdout without changing R6 scoring."""
from pathlib import Path
import random
from math import sqrt
from statistics import mean

from tools.r6_direct.anchor_endpoint import BLOCKS, from_report
from tools.r6_direct.evidence import read


def score_model(job):
    report=read(job['report'])
    original=report['summary']['original']
    if (report['run_id'] != job['run_id'] or report['candidate_binding_issues']
            or not report['protocol_frozen'] or not original['replay_complete']
            or not original['saved_result_equal'] or not original['saved_wall_equal']):
        raise ValueError(f'Incomplete original-suite evidence {job["run_id"]}')
    tracking={}
    counts={}
    for block in report['blocks']:
        key=block['block_id']
        if (not block['replay_complete'] or not block['saved_result_equal'] or block['total'] != 32
                or not block['measurement_replay']['bit_exact_delivery']
                or not block['physics_replay']['staging_reconstructed']):
            raise ValueError(f'Incomplete evaluation evidence {job["run_id"]} {key}')
        score=block['scoring']
        counts[key]=dict(successes=score['successes'],total=32,
                         failed_lanes=[lane for lane in score['lanes'] if not lane['motion_pass']])
        errors=[metric[field] for lane in score['lanes'] for metric in (lane['metrics'],)
                for field in ('incoming_stable_rmse','stable_rmse') if field in metric]
        if errors:
            tracking[key]=sqrt(mean(error**2 for error in errors))
    endpoints={key:from_report(Path(job['report']['path']),key) for key in BLOCKS}
    if len(counts)!=16 or len(original['cases'])!=5:
        raise ValueError('Study evaluation denominators')
    return dict(arm=job['arm'],pair=job['pair'],run_id=job['run_id'],report=job['report'],seed=report['seed'],
                checkpoint=report['candidate']['checkpoint'], endpoints=endpoints,
                tracking_rmse_mps=tracking, original=original, registered_blocks=counts,
                safety_pass=original['all_gates_pass'] and original['wall_motion']['all32']
                            and all(row['successes']==32 for row in counts.values()))


def bootstrap(deltas, *, seed, draws):
    """Resample the eight directions with four paired lanes per direction."""
    groups=[mean(d[lane] for d in deltas for lane in range(direction,32,8)) for direction in range(8)]
    generator=random.Random(seed)
    values=sorted(mean(groups[generator.randrange(8)] for _ in groups) for _ in range(draws))
    return [values[int(q*(draws-1))] for q in (.025,.975)]


def difference(trial, reference, key):
    x,y=(m['endpoints'][key] for m in (trial,reference))
    if x['block']!=y['block'] or x['seed']!=y['seed']:
        raise ValueError('Unpaired third-training evaluation')
    a,b=([r['anchor_mae_mm'] for r in p['lane_scores']] for p in (x,y))
    return [one-two for one,two in zip(a,b)] if None not in a+b else None


def compare(freeze, trials, references):
    main, clean=BLOCKS
    deltas=[]
    pairs=[]
    for trial, reference in zip(trials,references):
        primary=difference(trial,reference,main)
        clear=difference(trial,reference,clean)
        if trial['tracking_rmse_mps'].keys()!=reference['tracking_rmse_mps'].keys():
            raise ValueError('Missing paired tracking metric')
        tracking={k:v-reference['tracking_rmse_mps'][k] for k,v in trial['tracking_rmse_mps'].items()}
        coverage=[mean(lane['mean_usable_fraction'] for lane in m['endpoints'][main]['lane_scores'])
                  for m in (trial,reference)]
        primary_mean=None if primary is None else mean(primary)
        clean_mean=None if clear is None else mean(clear)
        deltas.append(primary)
        pairs.append(dict(pair=trial['pair'],mean_delta_mm=primary_mean,lane_deltas_mm=primary,
            clean_delta_mm=clean_mean,coverage_delta=coverage[0]-coverage[1],tracking_delta_mps=tracking,
            guardrails_pass=(trial['safety_pass'] and reference['safety_pass']
                and clean_mean is not None and clean_mean<=freeze['clean_increase_mm']
                and coverage[0]-coverage[1]>=-freeze['coverage_drop']
                and bool(tracking) and max(tracking.values())<=freeze['tracking_rmse_increase_mps'])))
    complete=all(row is not None for row in deltas)
    pooled=mean(row['mean_delta_mm'] for row in pairs) if complete else None
    interval=bootstrap(deltas,seed=freeze['bootstrap_seed'],draws=freeze['bootstrap_draws']) if complete else None
    return dict(pairs=pairs,mean_delta_mm=pooled,conditional_95_ci_mm=interval,
                meets_5mm=(complete and pooled<=-freeze['threshold_mm'] and interval[1]<0
                           and all(row['mean_delta_mm']<0 for row in pairs)),
                guardrails_pass=all(row['guardrails_pass'] for row in pairs))


def summarize(freeze, evaluations):
    models=[score_model(job) for job in evaluations]
    if len(models)!=7 or len({m['seed'] for m in models})!=1:
        raise ValueError('Study evaluation stream identity')
    baseline,=[m for m in models if (m['arm'],m['pair'])==('baseline',0)]
    candidate,control=([next(m for m in models if (m['arm'],m['pair'])==(arm,i))
                        for i in (1,2,3)] for arm in ('candidate','control'))
    compared={'against_baseline':compare(freeze,candidate,[baseline]*3),
              'against_equal_training':compare(freeze,candidate,control)}
    passed=all(row['meets_5mm'] and row['guardrails_pass'] for row in compared.values())
    return dict(schema_version='instinctrl-third-evaluation-results/1', models=models,
                comparisons=compared, accepted=passed, holdout_seed=baseline['seed'],
                condition='S-delay-drop', endpoint='valid range-anchor MAE, mm, [1,11) seconds after stop')
