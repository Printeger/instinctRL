"""Validate a complete materialized final batch before its single plan review.

Reads original Journal receipts and frozen specifications. Does not reserve,
issue or submit jobs; prepaid storage is not charged a second time.
"""
import json
from pathlib import Path
from rlflow import experiments
from rlflow.store import sha
from tools.r6_recovery import checked, RecoveryError
from tools.r6_budget import reserve
from tools import r6_final_storage as storage
from tools import r6_materialize_job as materializer
from tools import r6_job_resources as charges
from tools import r6_profile_preparation as profile


def require(value, reason):
    if not value:
        raise RecoveryError('FINAL_PLANS_' + reason)


def read(pin):
    return json.loads(checked(pin).read_text())


def registered(journal, key, role, field, value, cap):
    state = journal.read()
    entry = state['events'][key]
    identity = entry['identity']
    require(entry['kind'] == 'cpu' and identity['role'] == role
            and identity['goal_id'] == state['goal_id'] and sha(identity) == key
            and identity['input_sha256'] == sha(entry['request']), 'EVENT')
    require(read(entry[field]) == value, 'RECEIPT')
    profile.reservation(state, key, 'cpu', {charges.DISK: cap}, 'confirmation_delivery')
    for pin in entry['request']['implementation']:
        checked(pin)
    return entry['request']


def assignment(journal, aggregate, index, store):
    """Recheck one unsubmitted assignment; previous jobs may already be closed."""
    require(type(index) is int and 0 <= index < len(aggregate['assignments']), 'INDEX')
    row = aggregate['assignments'][index]
    require(row['assignment_index'] == index, 'ORDER')
    prepared, context, draft = storage.prepared(journal, aggregate['preparation'], index)
    require(prepared['batch'] == aggregate['batch'], 'BATCH')
    planned = row['storage']
    request = registered(journal, planned['event_key'], storage.ROLE, 'result', planned, storage.CAP)
    require(request['preparation'] == aggregate['preparation'] and request['assignment_index'] == index
            and planned['preparation'] == aggregate['preparation'] and planned['assignment_index'] == index
            and planned['dispatch_authorized'] is False, 'STORAGE')
    inputs = read(planned['inputs'])
    require(inputs == dict(snapshot_receipt=context['snapshot_receipt'], profile_draft=context['baseline_profile'],
            storage_key=planned['storage_key'], stage_id=context['stage_id'],
            conditions=context['conditions'], runner=context['runner']), 'INPUTS')
    material = row['materialization']
    request = registered(journal, material['event_key'], materializer.ROLE, 'materialization', material, materializer.CAP)
    require(request['inputs'] == planned['inputs'] and request['store_root'] == str(store.root.resolve())
            and material['inputs'] == planned['inputs'] and material['storage_key'] == planned['storage_key']
            and material['status'] == 'EXACT_INPUTS_MATERIALIZED_NOT_AUTHORIZED'
            and material['dispatch_authorized'] is False and material['scientific_acceptance'] is False, 'MATERIALIZATION')
    artifacts = material['artifacts']
    for pin in artifacts.values():
        checked(pin)
    params = read(artifacts['parameters'])
    require(set(params) == set(draft) | materializer.MANAGED
            and {k: params[k] for k in draft} == draft, 'PARAMETERS')
    require(params['experiment_card'] == artifacts['card'] and read(artifacts['card']) ==
            dict(experiment_id=draft['experiment_id'], profiles={'R1':{k:v for k,v in params.items() if k != 'experiment_card'}}), 'CARD')
    auth = read(artifacts['authorization'])
    require(auth['parameters'] == params and auth['mode'] == charges.MODES[3], 'AUTHORIZATION')
    require(read(artifacts['authorization_proposal'])['authorization_proposal'] == auth, 'PROPOSAL')
    plan = read(artifacts['plan'])
    require(plan == dict(context['runner'], spec_pin=artifacts['spec'])
            and plan['purpose'] == 'confirmation_delivery', 'PLAN')
    spec = read(artifacts['spec'])
    snapshot = read(context['snapshot_receipt'])
    require(spec['id'] == draft['experiment_id'] and spec['pins']['authorization'] == artifacts['authorization']
            and spec['stage_id'] == context['stage_id'] and spec['cwd'] == snapshot['snapshot']
            and spec['conditions'] == context['conditions'] and spec['h1']['mode'] == charges.MODES[3], 'SPEC')
    experiments.validate(spec, store.root)
    require(storage.storage.for_spec(journal, artifacts['spec'], purpose=plan['purpose'], preview=True) is not None, 'PREPAID')
    jid = 'J-' + sha(dict(id=spec['id'], spec=spec))[:16]
    require(jid == material['job_id'] and jid not in store.replay()['jobs'], 'JOB')
    return dict(assignment_index=index, job_id=jid, attempt=spec['h1']['attempt'],
                plan=artifacts['plan'], spec=artifacts['spec'], authorization=artifacts['authorization'],
                resources=charges.resources(spec, auth, prepaid=True))


def describe(journal, aggregate_pin, store, *, free_disk_bytes):
    profile.guard(journal)
    aggregate = read(aggregate_pin)
    require(set(aggregate) == {'status','preparation','batch','assignments','missing','scientific_acceptance','dispatch_authorized'}
            and aggregate['status'] == 'FINAL_BATCH_PLANS_READY_NOT_AUTHORIZED' and aggregate['missing'] == []
            and aggregate['scientific_acceptance'] is False and aggregate['dispatch_authorized'] is False, 'COMPLETE')
    prepared = read(aggregate['preparation'])
    require(len(aggregate['assignments']) == len(prepared['contexts']) and len(prepared['contexts']) in (3,4), 'COUNT')
    rows = [assignment(journal, aggregate, i, store) for i in range(len(aggregate['assignments']))]
    require(len({r['job_id'] for r in rows}) == len(rows)
            and len({str(Path(r['attempt']).resolve()) for r in rows}) == len(rows), 'UNIQUE')
    resources = {}
    for row in rows:
        for name, value in row['resources'].items():
            resources[name] = resources.get(name, 0) + value
    state = journal.read()
    reserve(state['budget'], event_key=dict(goal_id=state['goal_id'], event_id='FINAL_PLANS_CHECK:'+aggregate_pin['sha256'],
            job_id='FINAL-BATCH', input_sha256=sha(resources), role='budget-check-only'), resources=resources,
            purpose='confirmation_delivery', free_disk_bytes=free_disk_bytes)
    return dict(status='FINAL_EXACT_PLANS_VALIDATED_NOT_AUTHORIZED', materializations=aggregate_pin,
                batch=aggregate['batch'], assignments=rows, planned_job_resources=resources,
                resource_scope='PREPAID_FINAL_JOBS_ONLY_NOT_REVIEW_DELIVERY_OR_REPAIR',
                scientific_acceptance=False, dispatch_authorized=False)
