"""CPU-only integration of the real-mode control-plane submission contract.

Store, H1 validation, queue and journal are real. Runtime authority inspection,
review, resource samples and release/Popen are explicit fixtures, not approvals.
No simulator, GPU query, interpreter subprocess or model worker is launched.
"""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rlflow import experiments, h1, policy
from rlflow.store import Store, Conflict, sha
from tools import r6_resource_queue as q
from tools.r6_recovery import Journal, RecoveryError, file_pin
from tools.r6_submit import submit_once
from training_workloads.scale_003.resources import GPU_IDENTITY

MINIMUM = dict(live_starts=1, simulator_initializations=1,
    live_wall_seconds=250, allocated_gpu_seconds=250,
    nontraining_transitions=512, output_actual_plus_unreleased_reservations_bytes=1024**3)


@pytest.fixture(params=['research-r6-evaluation','research-r6-training'])
def bound(tmp_path, monkeypatch, request):
    return make_bound(tmp_path, monkeypatch, request.param)


@pytest.fixture
def baseline_bound(tmp_path, monkeypatch):
    return make_bound(tmp_path, monkeypatch, 'research-r6-evaluation')


def make_bound(tmp_path, monkeypatch, mode):
    minimum = dict(MINIMUM)
    if mode == 'research-r6-training':
        minimum['training_transitions'] = minimum.pop('nontraining_transitions')
        minimum['training_checkpoint_writes'] = 2
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return file_pin(path)
    store = Store(tmp_path/'store')
    work = tmp_path/'work'; entry = work/'training_workloads/unified_h1'
    entry.mkdir(parents=True)
    dispatcher = entry/'local_dispatch.py'
    dispatcher.write_bytes(experiments.DISPATCHER.read_bytes())
    workload = write(entry/'formal_unified_h1.py', {'fixture':'never execute'})
    product = write(work/'isaaclab-training/instinctlab/envs/unified_h1_direct_env.py', {})
    config = dict(training=dict(seed=17, evaluation_seed=19), method=dict(version='fixture'),
        versions=dict(metric='fixture'), execution=dict(frozen_runtime_root='CPU-NOT-RUNTIME'),
        product=dict(path=str(work)))
    config_pin = write(entry/'candidate_config.json', config)
    auth = dict(parameters=dict(evaluation_seed=19, output_cap_bytes=1024**3),
                budgets=dict(wall_seconds=60, training_transitions=0, nontraining_transitions=512))
    if mode == 'research-r6-training':
        auth['mode'] = mode
        auth['parameters']['training_seed'] = 17
        auth['budgets'].update(training_transitions=512, nontraining_transitions=0, checkpoint_writes=2)
    auth_pin = write(tmp_path/'fixture-auth.json', auth)
    observed = dict(config=config, authorization=auth)
    monkeypatch.setattr(h1, 'inspect_authority', lambda spec:copy.deepcopy(observed))
    attempt = tmp_path/'attempt'; (attempt/'results').mkdir(parents=True)
    executable = file_pin(sys.executable)
    spec = dict(id='R6-REAL-CONTRACT-CPU-FIXTURE', stage_id='CPU', backend='real',
        argv=[executable['path'], '-B', '-m', 'training_workloads.unified_h1.formal_unified_h1',
              '--authorization', auth_pin['path'], '--attempt', str(attempt), '--mode', mode],
        cwd=str(work), pins=dict(dispatcher=file_pin(dispatcher), executable=executable,
            product=product, workload=workload, runtime=executable, authorization=auth_pin),
        method='fixture', evaluation='fixture', runtime='CPU-NOT-RUNTIME', seed=17 if mode=='research-r6-training' else 19,
        resources=dict(starts=1, wall_seconds=70, gpu_seconds=70, transitions=512),
        conditions=[dict(evidence_id='cpu', sha256='1'*64)],
        observations=['status','counters','process_identity','process_start'],
        h1=dict(workload_root=str(work), attempt=str(attempt), mode=mode,
            config_sha256=config_pin['sha256'], gpu_devices=['0'],
            gpu_accounting='allocated_device_wall_seconds'))
    store.transact('FIXTURE_CONDITION', 'cpu', lambda s:s['evidence'].update(
        cpu=dict(id='cpu', sha256='1'*64, accepted=True)))
    proposal = experiments.propose(store, spec)
    shown = policy.show_proposal(store, proposal)
    policy.decide(store, proposal, shown['sha256'], 'APPROVED', 'temporary fixture only')
    spec_pin = write(tmp_path/'spec.json', spec)
    budget = json.loads((Path(__file__).parent/'fixtures/r6_budget.json').read_text())
    budget['capacity_pass'] = True
    journal = Journal(tmp_path/'batch')
    review = write(tmp_path/'fixture-review.json', dict(decision='PASS_R6_RESOURCE_QUEUE',
        policy=q.POLICY, implementation=file_pin(Path(q.__file__)),
        validator=file_pin(Path(q.v2.__file__)), must_fix=[], fixture_only=True))
    clock = dict(boot_id='CPU', monotonic_ns=10**12)
    monkeypatch.setattr(q, 'now', lambda:dict(clock))
    with journal.writer():
        journal.initialize(budget)
        queue_key = q.begin(journal, spec_pin, ordinal=0, free_disk_bytes=500*1024**3, review=review)
        for i in range(13):
            if i:clock['monotonic_ns'] += 5*10**9
            q.append_sample(journal, queue_key, i, dict(monotonic_ns=clock['monotonic_ns'],
                gpu=dict(GPU_IDENTITY, memory_used_mib=560, utilization_percent=3,
                    free_memory_mib=GPU_IDENTITY['memory_total_mib']-560), compute_apps=[]))
        q.finish(journal, queue_key)
        jid = 'J-'+sha(dict(id=spec['id'], spec=spec))[:16]
        request = dict(spec=spec_pin, wait_deadline_unix=time.time()+70)
        event = dict(goal_id=budget['goal_id'], event_id=spec['id'], job_id=jid,
                     input_sha256=sha(request), role='job')
        key, _ = journal.intent(event, kind='job', request=request,
            resources=dict(minimum), purpose='development', free_disk_bytes=500*1024**3)
    popen = Mock(side_effect=OSError('CPU stop before Popen'))
    monkeypatch.setattr(experiments.subprocess, 'Popen', popen)
    monkeypatch.setattr('rlflow.releases.freeze', lambda root:store.root)
    return SimpleNamespace(journal=journal, store=store, key=key, queue_key=queue_key,
        jid=jid, spec=spec, observed=observed, popen=popen, minimum=minimum)


def invoke(b):
    with b.journal.writer():return submit_once(b.journal, b.key, b.store, queue_key=b.queue_key)


@pytest.mark.parametrize('resource', list(MINIMUM))
@pytest.mark.parametrize('missing', [True, False])
def test_missing_or_underreserved_real_dimension_never_claims(bound, resource, missing):
    b = bound
    if resource=='nontraining_transitions' and 'training_transitions' in b.minimum:
        resource='training_transitions'
    with b.journal.writer():
        state = b.journal.read(); event = state['events'][b.key]
        reserved = dict(b.minimum)
        if missing:reserved.pop(resource)
        else:reserved[resource] -= 1
        event['resources'] = reserved
        next(r for r in state['budget']['reservations'] if r['identity']==event['identity'])['resources'] = reserved
        b.journal.save(state)
    with pytest.raises(RecoveryError, match='RESERVATION_INCOMPLETE'):invoke(b)
    assert 'submission_claim' not in b.journal.read()['events'][b.key]
    assert not b.store.replay()['jobs']
    b.popen.assert_not_called()


def test_complete_real_reservation_reaches_store_then_recovers(bound):
    b = bound; budget = b.journal.read()['budget']
    with pytest.raises(OSError, match='CPU stop before Popen'):invoke(b)
    assert b.store.replay()['jobs'][b.jid]['status']=='START_INTENT'
    assert (b.store.root/'jobs'/b.jid/'h1-authority.json').is_file()
    assert invoke(b)['job_id']==b.jid
    assert b.popen.call_count==1 and b.journal.read()['budget']==budget


@pytest.mark.parametrize('field', ['evaluation_seed', 'wall_seconds'])
def test_real_h1_validation_still_rejects_authority_drift(bound, field):
    b = bound
    if field=='evaluation_seed' and b.spec['h1']['mode']=='research-r6-training':
        field='training_seed'
    section = 'parameters' if field.endswith('_seed') else 'budgets'
    b.observed['authorization'][section][field] += 1
    with pytest.raises(Conflict, match='H1'):invoke(b)
    assert 'submission_claim' not in b.journal.read()['events'][b.key]
    b.popen.assert_not_called()

@pytest.mark.parametrize('missing', [True, False])
def test_training_checkpoint_allowance_required(bound, missing):
    b = bound
    if b.spec['h1']['mode'] != 'research-r6-training':
        return
    with b.journal.writer():
        state=b.journal.read(); event=state['events'][b.key]
        if missing: event['resources'].pop('training_checkpoint_writes')
        else: event['resources']['training_checkpoint_writes']-=1
        next(r for r in state['budget']['reservations'] if r['identity']==event['identity'])['resources']=dict(event['resources'])
        b.journal.save(state)
    with pytest.raises(RecoveryError, match='RESERVATION_INCOMPLETE'): invoke(b)
    assert 'submission_claim' not in b.journal.read()['events'][b.key]
    assert not b.store.replay()['jobs']
    b.popen.assert_not_called()
