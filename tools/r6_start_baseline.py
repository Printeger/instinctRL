"""First baseline: registered review -> exact issuance -> existing job runner.

The issuance event is the durable handoff. Once a submission claim exists, the
original runner owns recovery/STOP cleanup; no new review or approval is made.
"""
import argparse
import json
from pathlib import Path
import shutil

from rlflow.store import sha
from tools.r6_recovery import Journal, checked, RecoveryError
from tools.r6_batch import model_handoff
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_run_job import run as run_job
from tools import r6_initial_baseline as initial

ROLE = 'initial-baseline-authorization'


def require(condition, reason):
    if not condition:
        raise RecoveryError('START_BASELINE_'+reason)


def start(journal, plan_pin, master_pin, review_executable, store, *, free_disk_bytes):
    journal.require_writer()
    state = journal.read()
    edges = [(k, e) for k, e in state['events'].items() if e['identity']['role'] == ROLE]
    require(len(edges) <= 1, 'ISSUANCE_ALIAS')
    if edges:
        request = edges[0][1]['request']
        require(request['plan'] == plan_pin and request['master'] == master_pin
            and request['store_root'] == str(store.root.resolve()), 'ISSUANCE_DRIFT')
    if not edges or not edges[0][1].get('authorization_result'):
        event_id = 'INITIAL_BASELINE_REVIEW:'+plan_pin['sha256']
        reviews = [(k, e) for k, e in state['events'].items()
                   if e['identity']['role'] == initial.ROLE and e['identity']['event_id'] == event_id]
        require(len(reviews) <= 1, 'REVIEW_ALIAS')
        if reviews:
            key, review = reviews[0]
            require(review['request']['executable'] == review_executable, 'REVIEW_EXECUTABLE_DRIFT')
            if not review.get('dispatch_claim') and not review.get('receipt'):
                recovered = initial.register_review(journal, plan_pin, store,
                    executable=review_executable, free_disk_bytes=free_disk_bytes)
                require(recovered == key, 'REVIEW_RECOVERY_DRIFT')
        else:
            key = initial.register_review(journal, plan_pin, store,
                executable=review_executable, free_disk_bytes=free_disk_bytes)
        model_handoff(journal, key)
        receipt = json.loads(checked(journal.read()['events'][key]['receipt']).read_text())
        result = initial.issue(journal, plan_pin, key, receipt['last_message'], master_pin,
            store, free_disk_bytes=free_disk_bytes)
        edges = [(result['event_key'], journal.read()['events'][result['event_key']])]
    edge_key, edge = edges[0]; request = edge['request']
    require(request['parent_key'] == 'INITIAL_BASELINE' and request['plan'] == plan_pin
        and request['master'] == master_pin and request['store_root'] == str(store.root.resolve()), 'ISSUANCE_DRIFT')
    state = journal.read(); review = state['events'][request['review_key']]
    require(review['request']['executable'] == review_executable, 'REVIEW_EXECUTABLE_DRIFT')
    plan = json.loads(checked(plan_pin).read_text())
    require(set(plan) == initial.FIELDS and plan['purpose'] == 'development'
        and plan['sandbox'] in {'read-only', 'danger-full-access'}, 'PLAN_FIELDS')
    spec = json.loads(checked(plan['spec_pin']).read_text())
    jid = 'J-'+sha({'id': spec['id'], 'spec': spec})[:16]
    expected = dict(goal_id=state['goal_id'], event_id='AUTHORIZE_NEXT:INITIAL_BASELINE',
                    job_id=jid, input_sha256=sha(request), role=ROLE)
    rows = [r for r in state['budget']['reservations'] if r['identity'] == expected]
    require(spec['id'] == initial.EXPERIMENT and edge['kind'] == 'cpu'
        and edge['identity'] == expected and sha(expected) == edge_key
        and edge['resources'] == {'output_actual_plus_unreleased_reservations_bytes': initial.CAP}
        and edge['purpose'] == 'development' and len(rows) == 1
        and rows[0]['resources'] == edge['resources'] and rows[0]['purpose'] == edge['purpose'], 'ISSUANCE_IDENTITY')
    issued = edge['authorization_result']
    require(issued['status'] == 'EXACT_INITIAL_BASELINE_AUTHORIZED_NOT_SUBMITTED'
        and issued['event_key'] == edge_key and issued['job_id'] == jid
        and issued['spec_sha256'] == sha(spec), 'ISSUANCE_RESULT')
    if edge.get('initial_job_result'):
        result = edge['initial_job_result']
        require(result['status'] == 'DECISION_PROPOSAL_RETURNED'
            and result['handoff_event_key'] == edge_key and result['scientific_acceptance'] is False,
            'CACHED_RESULT')
        require(journal.canonical_job(result['event_key'])['identity']['job_id'] == jid, 'CACHED_JOB')
        return result
    jobs = [e for e in state['events'].values() if e['kind'] == 'job' and e['identity']['job_id'] == jid]
    require(len(jobs) <= 1, 'JOB_ALIAS')
    if not jobs or not jobs[0].get('submission_claim'):
        journal.guard(); require(not stop_paths(journal), 'STOP'); no_model_workers(journal)
        require(state.get('current_job') is None and not any(
            e['kind'] == 'job' and e['identity']['job_id'] != jid for e in state['events'].values()),
            'OTHER_JOB')
        initial.delegation(journal, master_pin, store)
        initial.verify_review(journal, plan_pin, request['review_key'], request['review'], plan, jid)
    outcome = run_job(journal, store=store, free_disk_bytes=free_disk_bytes, **plan)
    require(outcome['status'] in {'RESOURCE_OBSERVATION_REJECTED', 'DECISION_PROPOSAL_RETURNED'}, 'RUNNER_STATUS')
    result = dict(status=outcome['status'], handoff_event_key=edge_key,
                  event_key=outcome.get('event_key'), scientific_acceptance=False)
    if result['status'] == 'DECISION_PROPOSAL_RETURNED':
        require(journal.canonical_job(result['event_key'])['identity']['job_id'] == jid, 'RUNNER_JOB')
        state = journal.read(); state['events'][edge_key]['initial_job_result'] = result; journal.save(state)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path); parser.add_argument('request', type=Path)
    parser.add_argument('--request-sha256', required=True); parser.add_argument('--store', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(checked(dict(path=str(args.request.resolve()), sha256=args.request_sha256)).read_text())
    require(set(request) == {'plan_pin', 'master_pin', 'review_executable'}, 'REQUEST_FIELDS')
    root = args.store.resolve(strict=True); require((root/'runtime/research.db').is_file(), 'STORE_REQUIRED')
    from rlflow.store import Store
    journal = Journal(args.directory.resolve())
    with journal.writer():
        result = start(journal, store=Store(root), free_disk_bytes=shutil.disk_usage(journal.directory).free, **request)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
