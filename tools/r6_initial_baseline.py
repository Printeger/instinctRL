"""Exact first-baseline review and Store issuance, without a fictional parent job.

Only the initial baseline is eligible. Existing next-job transactions own later
attempts. This module never starts a model or submits a simulator process.
"""
import hashlib
import json
from pathlib import Path

from rlflow import experiments
from rlflow.store import sha
from tools.r6_recovery import checked, file_pin, RecoveryError
from tools.r6_batch import verified_decision
from tools.r6_job_handoff import no_model_workers
from tools.r6_postrun import stop_paths
from tools.r6_next_job import FIELDS, CAP
from tools.r6_issue_next import _issue_reviewed, delegation

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT/'docs/experiments/r6-robustness-auto/prompts/initial-baseline-review.schema.json'
EXPERIMENT = 'R6-BASELINE-ROBUSTNESS-01'
ROLE = 'initial-baseline-review'


def require(value, reason):
    if not value:
        raise RecoveryError('INITIAL_BASELINE_'+reason)


def first_plan(journal, plan_pin, store):
    state = journal.read()
    require(state.get('current_job') is None, 'ACTIVE_JOB')
    require(state['budget']['capacity_pass'] is True, 'CAPACITY')
    require(not any(e['kind'] == 'job' for e in state['events'].values()), 'JOB_ALREADY_REGISTERED')
    require(not any(j.get('spec', {}).get('id', '').startswith('R6-')
                    for j in store.replay()['jobs'].values()), 'R6_JOB_ALREADY_EXISTS')
    plan = json.loads(checked(plan_pin).read_text())
    require(set(plan) == FIELDS and plan['purpose'] == 'development'
            and plan['sandbox'] in {'read-only', 'danger-full-access'}, 'PLAN_FIELDS')
    spec = json.loads(checked(plan['spec_pin']).read_text())
    require(spec['id'] == EXPERIMENT and spec['backend'] == 'real'
            and spec.get('h1', {}).get('mode') == 'research-r6-evaluation', 'EXACT_BASELINE')
    experiments.validate(spec, store.root)
    return plan, spec, 'J-'+sha({'id': spec['id'], 'spec': spec})[:16]


def register_review(journal, plan_pin, store, *, executable, free_disk_bytes):
    from tools.r6_decision import CLI, CLI_SHA, MODEL_CAP
    from tools.r6_decision_packet import persist_exact
    from tools.r6_model_worker import validate_request
    journal.require_writer(); journal.guard()
    require(not stop_paths(journal), 'STOP'); no_model_workers(journal)
    plan, spec, jid = first_plan(journal, plan_pin, store)
    path = checked(executable)
    state = journal.read()
    if state['goal_id'] == 'R6-CPU-TEST-FIXTURE':
        require(path.is_relative_to(journal.directory.parent.resolve()), 'FIXTURE_EXECUTABLE')
    else:
        require(executable == {'path': str(CLI), 'sha256': CLI_SHA}, 'PINNED_CLI')
    packet = dict(plan=plan_pin, spec=plan['spec_pin'], store_root=str(store.root))
    prompt = ('Independently review this exact initial R6 baseline plan, read-only, then exit. '
        'There is no parent R6 job. Read the pinned plan/spec and original supporting evidence. '
        'Verify fixed R5 u640 identity and ICS .15, all original gates and all16 frozen blocks, '
        'protocol/split acceptance before results, full-project time/storage/final reserves, '
        'actual production volume capacity and ownership, immutable execution snapshot and '
        'compatibility, actor/runtime boundaries, exact Store conditions, STOP/recovery and '
        'controller handoff readiness. Absence of final Store approval is expected: issuance '
        'follows this review. Missing supporting evidence is MUST_FIX. Do not query GPUs, '
        'launch jobs, change files or grant authority. Return PASS_R6_INITIAL_BASELINE only '
        'with empty must_fix and changed_files; include exact plan_sha256 and plan/spec pins '
        'in evidence_pins.\n'+json.dumps(packet, sort_keys=True)+'\n').encode()
    require(len(prompt) <= CAP, 'CONTEXT_CAP')
    target = journal.directory/'initial-baseline-reviews'/plan_pin['sha256']/'prompt.txt'
    request = dict(prompt={'path': str(target.resolve()), 'sha256': hashlib.sha256(prompt).hexdigest()},
        schema=file_pin(SCHEMA), executable=executable, cwd=str(ROOT), sandbox='read-only',
        timeout_seconds=3600, output_cap_bytes=MODEL_CAP,
        stop_paths=[str(p.resolve()) for p in (journal.directory/'STOP',
            journal.directory.parent/'STOP', ROOT/'STOP', ROOT/'coordination/STOP',
            ROOT/'coordination/commander/STOP', ROOT/'coordination/executor/STOP')])
    identity = dict(goal_id=state['goal_id'], event_id='INITIAL_BASELINE_REVIEW:'+plan_pin['sha256'],
        job_id=jid, input_sha256=sha(request), role=ROLE)
    prior = [e for e in state['events'].values() if e['identity']['role'] == ROLE
             and e['identity']['event_id'] == identity['event_id']]
    require(len(prior) <= 1 and all(e['request'] == request for e in prior), 'REVIEW_ALREADY_BOUND')
    key, _ = journal.intent(identity, kind='model', request=request,
        resources={'automatic_model_sessions': 1,
                   'output_actual_plus_unreleased_reservations_bytes': MODEL_CAP+CAP},
        purpose='development', free_disk_bytes=free_disk_bytes)
    persist_exact(target, prompt); validate_request(request)
    return key


def reviewed_plan(journal, plan_pin, review_key, review_pin, store):
    plan, spec, jid = first_plan(journal, plan_pin, store)
    state = journal.read(); entry = state['events'][review_key]; identity = entry['identity']
    require(entry['kind'] == 'model' and identity == dict(goal_id=state['goal_id'],
        event_id='INITIAL_BASELINE_REVIEW:'+plan_pin['sha256'], job_id=jid,
        input_sha256=sha(entry['request']), role=ROLE) and sha(identity) == review_key, 'REVIEW_IDENTITY')
    require(entry['request'].get('schema') == file_pin(SCHEMA)
            and entry['request'].get('sandbox') == 'read-only', 'REVIEW_CONTRACT')
    receipt = json.loads(checked(entry['receipt']).read_text())
    require(entry['status'] == 'RECEIPT_RECORDED' and receipt['event_key'] == review_key
        and receipt['request_sha256'] == sha(entry['request'])
        and receipt['process'] == entry.get('process')
        and receipt['last_message']['sha256'] == review_pin['sha256'], 'REVIEW_RECEIPT')
    rows = [r for r in state['budget']['reservations'] if r['identity'] == identity]
    require(len(rows) == 1 and rows[0]['resources'] == entry['resources']
        and rows[0]['purpose'] == entry['purpose'] == 'development'
        and entry['resources'].get('automatic_model_sessions') == 1, 'REVIEW_RESERVATION')
    result = verified_decision(journal, review_key)
    require(result == json.loads(checked(review_pin).read_text())
        and result['decision'] == 'PASS_R6_INITIAL_BASELINE' and result['must_fix'] == []
        and result['changed_files'] == [] and result['plan_sha256'] == plan_pin['sha256'], 'REVIEW_BINDING')
    require(plan_pin in result['evidence_pins'] and plan['spec_pin'] in result['evidence_pins'], 'REVIEW_EVIDENCE')
    for pin in result['evidence_pins']:
        checked(pin)
    return plan, spec, jid


def issue(journal, plan_pin, review_key, review_pin, master_pin, store, *, free_disk_bytes):
    journal.require_writer(); journal.guard()
    require(not stop_paths(journal), 'STOP'); no_model_workers(journal)
    delegation(journal, master_pin, store)
    plan, spec, jid = reviewed_plan(journal, plan_pin, review_key, review_pin, store)
    return _issue_reviewed(journal, 'INITIAL_BASELINE', plan_pin, review_key, review_pin,
        master_pin, store, plan, spec, jid, free_disk_bytes=free_disk_bytes,
        role='initial-baseline-authorization')
