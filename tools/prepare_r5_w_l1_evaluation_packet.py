"""Prepare both actual L1 nodes for the frozen primary evaluation; no release."""
from copy import deepcopy
from pathlib import Path
from tools.r5_w_commander import BASE, ROOT, read, save, pin, sha, checked


def main():
    from training_workloads.unified_h1.w_closure import validate_parameters, required_budget
    old = BASE / 'experiments/R5-W-L1-NEARWALL-RECOVERY1'
    new = BASE / 'experiments/R5-W-L1-EVAL-01'
    assert not new.exists()
    summary = read(old / 'R1-research-formal-training-execution-summary.json')
    assert summary['settled'] and summary['postrun_queries_used'] == 2
    assert summary['observed'] == dict(training_transitions=131072, nontraining_transitions=0,
        optimizer_updates=256, training_checkpoint_writes=2,
        actor_optimizer_steps=2048, critic_optimizer_steps=2048)
    packet = deepcopy(read(old / 'launch_inputs.proposed.json'))
    inventory = read(old / 'live/R1-research-formal-training/INVENTORY.json')
    checkpoints = []
    for update in (640, 768):
        row = next(r for r in inventory['files'] if r['relative'] ==
            f'results/research-formal-training/checkpoints/u{update}.pt')
        cp = dict(path=row['original'], sha256=row['sha256'])
        checked(cp)
        checkpoints.append(dict(cp, update=update))
    compat = new / 'checkpoint-compatibility.json'
    save(compat, read(old / 'checkpoint-compatibility.json'), new=True)
    card = read(old / 'EXPERIMENT_CARD.json')
    card.pop('recovery', None)
    card.update(experiment_id=new.name, RL_training_transitions=0,
        primary_variable='FULL_ORIGINAL_PRIMARY_SEED_EVALUATION_OF_BOTH_ACTUAL_L1_NODES',
        training_experiment=old.name, training_inventory=pin(old / 'live/R1-research-formal-training/INVENTORY.json'),
        actual_checkpoints=checkpoints,
        evaluation_status='NOT_EXECUTED',
        performance_claim='None; E1 engineering gain remains distinct from L1 learning')
    card['profiles']['R1'].update(experiment_id=new.name, checkpoint_compatibility=pin(compat))
    cardpath = new / 'EXPERIMENT_CARD.json'
    save(cardpath, card, new=True)
    row = packet['entries'][0]
    auth, spec = row['authorization_proposal'], row['experiment_proposal']
    mode = 'research-formal-evaluation'
    attempt = ROOT / 'coordination/research/attempts' / f'{new.name}-R1-{mode.upper()}-{packet["baseline"]["snapshot_commit"][:12]}'
    assert not attempt.exists()
    auth.update(order_id=new.name+'-R1-'+mode.upper(), attempt=str(attempt), mode=mode)
    auth['parameters'].update(experiment_id=new.name, checkpoint_compatibility=pin(compat),
        experiment_card=pin(cardpath), checkpoints=checkpoints)
    validate_parameters(mode, auth['parameters'])
    auth['budgets'] = required_budget(mode, auth['parameters'])
    proposed = new / 'inputs' / f'R1-{mode}-AUTH.proposed.json'
    save(proposed, auth, new=True)
    actual = attempt / 'commands/AUTHORIZATION.json'
    spec.update(id=attempt.name, seed=20260909, conditions=[],
        resources=dict(starts=1, wall_seconds=1810, gpu_seconds=1810, transitions=65920))
    for flag, value in (('--authorization', str(actual)), ('--attempt', str(attempt)), ('--mode', mode)):
        spec['argv'][spec['argv'].index(flag)+1] = value
    spec['h1'].update(attempt=str(attempt), mode=mode)
    spec['pins']['authorization'] = dict(path=str(actual), sha256=sha(proposed))
    row.update(mode=mode, expected_authorization=pin(proposed))
    packet.update(experiment_id=new.name, experiment_card=pin(cardpath), generator=pin(Path(__file__)))
    save(new / 'launch_inputs.proposed.json', packet, new=True)


if __name__ == '__main__':
    main()
