"""Prepare one fresh L1 identity after audited prelaunch FINAL-zero failure.

No release, GPU query, snapshot mutation, or scientific parameter change.
"""
from copy import deepcopy
from pathlib import Path
from tools.r5_w_commander import BASE, ROOT, read, save, pin, sha, checked


def main():
    old = BASE / 'experiments/R5-W-L1-NEARWALL-01'
    new = BASE / 'experiments/R5-W-L1-NEARWALL-RECOVERY1'
    stage = BASE / 'stages/W-L1-002.json'
    packet = deepcopy(read(old / 'launch_inputs.proposed.json'))
    summary = read(old / 'R1-research-formal-training-execution-summary.json')
    assert summary['formal_counter_completeness'] == 'FINAL'
    assert summary['formal_transitions'] == 0 and summary['settled']
    assert summary['postrun_queries_used'] == 2
    assert all(v == 0 for v in summary['observed'].values())
    failure = read(old / 'live/R1-research-formal-training/attempt/results/prelaunch/prelaunch_resource_gate_failure.json')
    assert failure['message'] == 'GPU utilization exceeds 5 percent'
    assert not new.exists()
    compat = new / 'checkpoint-compatibility.json'
    save(compat, read(old / 'checkpoint-compatibility.json'), new=True)
    card = read(old / 'EXPERIMENT_CARD.json')
    card['experiment_id'] = new.name
    card['profiles']['R1'].update(experiment_id=new.name, checkpoint_compatibility=pin(compat))
    card['recovery'] = dict(previous_experiment=old.name,
        previous_execution=pin(old / 'R1-research-formal-training-execution-summary.json'),
        decision=pin(BASE / 'L1_RECOVERY_DECISION.json'),
        scope='One fresh reviewed entry; same scientific candidate; no automatic retry')
    cardpath = new / 'EXPERIMENT_CARD.json'
    save(cardpath, card, new=True)
    row = packet['entries'][0]
    auth = row['authorization_proposal']
    spec = row['experiment_proposal']
    mode = row['mode']
    attempt = ROOT / 'coordination/research/attempts' / f'{new.name}-R1-{mode.upper()}-{packet["baseline"]["snapshot_commit"][:12]}'
    assert not attempt.exists()
    auth.update(order_id=new.name+'-R1-'+mode.upper(), attempt=str(attempt))
    auth['parameters'].update(experiment_id=new.name, checkpoint_compatibility=pin(compat), experiment_card=pin(cardpath))
    from training_workloads.unified_h1.w_closure import validate_parameters, required_budget
    validate_parameters(mode, auth['parameters'])
    assert required_budget(mode, auth['parameters']) == auth['budgets']
    for p in spec['pins'].values():
        checked(p)
    proposed = new / 'inputs' / f'R1-{mode}-AUTH.proposed.json'
    save(proposed, auth, new=True)
    actual = attempt / 'commands/AUTHORIZATION.json'
    spec.update(id=attempt.name, conditions=[])
    spec['argv'][spec['argv'].index('--authorization')+1] = str(actual)
    spec['argv'][spec['argv'].index('--attempt')+1] = str(attempt)
    spec['h1']['attempt'] = str(attempt)
    spec['pins']['authorization'] = dict(path=str(actual), sha256=sha(proposed))
    row['expected_authorization'] = pin(proposed)
    packet.update(experiment_id=new.name, stage_allocation=pin(stage), experiment_card=pin(cardpath), generator=pin(Path(__file__)))
    save(new / 'launch_inputs.proposed.json', packet, new=True)


if __name__ == '__main__':
    main()
