"""Real Store/H1 spec checks; runtime authority inspection remains a CPU fixture."""
import json
from pathlib import Path
import pytest
from tools.test_r6_submit_real_contract import bound
from tools.r6_recovery import file_pin,RecoveryError
from tools.r6_build_spec import build
from rlflow import experiments
from rlflow.store import Conflict


def inputs(b):
    c=b.observed['config'];a=b.observed['authorization']
    c['workload']={'path':b.spec['cwd']}
    exe=b.spec['pins']['executable']
    c['execution'].update(python_path=exe['path'],python_sha256=exe['sha256'])
    a.update(mode=b.spec['h1']['mode']);a['parameters']['experiment_id']='R6-BUILD-SPEC-CPU'
    cp=Path(b.spec['cwd'])/'training_workloads/unified_h1/candidate_config.json'
    cp.write_text(json.dumps(c));ap=Path(b.spec['pins']['authorization']['path']);ap.write_text(json.dumps(a))
    return dict(config_pin=file_pin(cp),authorization_pin=file_pin(ap),attempt=b.spec['h1']['attempt'],
        stage_id=b.spec['stage_id'],conditions=b.spec['conditions'],store_root=b.store.root)


def test_build_exact_resources_without_store_mutation_or_permission(bound):
    b=bound;args=inputs(b);before=b.store.replay();spec=build(**args)
    assert spec['resources']==dict(starts=1,wall_seconds=70,gpu_seconds=70,transitions=512)
    assert spec['seed']==b.spec['seed'] and spec['h1']['config_sha256']==args['config_pin']['sha256']
    assert b.store.replay()==before;b.popen.assert_not_called()
    with pytest.raises(Conflict,match='no unique exact experiment approval'):
        experiments.authorized(b.store.replay(),spec,b.store.root)


def test_authority_failure_is_not_hidden(bound,monkeypatch):
    b=bound;args=inputs(b)
    def reject(spec):raise Conflict('invalid authority fixture')
    monkeypatch.setattr('rlflow.h1.inspect_authority',reject)
    with pytest.raises(Conflict,match='invalid authority'):build(**args)
    b.popen.assert_not_called()


def test_drift_and_consumed_attempt_rejected(bound):
    b=bound;args=inputs(b)
    Path(args['authorization_pin']['path']).write_text('{}')
    with pytest.raises((RecoveryError,ValueError)):build(**args)
    args=inputs(b)
    (Path(args['attempt'])/'results'/b.spec['h1']['mode']).mkdir()
    with pytest.raises(Conflict,match='already consumed'):build(**args)


def test_cannot_build_training_as_baseline(bound):
    b=bound;args=inputs(b);a=b.observed['authorization'];a['mode']='research-r6-evaluation';a['budgets']['training_transitions']=512
    path=Path(args['authorization_pin']['path']);path.write_text(json.dumps(a));args['authorization_pin']=file_pin(path)
    with pytest.raises(RecoveryError,match='NO_TRAINING'):build(**args)


def test_materializer_uses_exact_stage_builder_and_replays(bound,monkeypatch):
    """Real materializer, builder, Store/H1; kernel/frozen authority are fixtures."""
    from unittest.mock import Mock
    from tools import r6_materialize_job as m
    b=bound;args=inputs(b);root=b.store.root.parent
    def pin(name,value):
        path=root/name;path.write_text(json.dumps(value));return file_pin(path)
    auth=b.observed['authorization'];params=dict(auth['parameters'],group='R1')
    params.pop('output_cap_bytes')
    draft=pin('materializer-profile.json',params)
    config=json.loads(Path(args['config_pin']['path']).read_text())
    snapshot=pin('snapshot-receipt.json',dict(status='CPU_VERIFIED_READ_ONLY_NOT_LIVE_AUTHORITY',
        snapshot=b.spec['cwd'],config=args['config_pin']))
    native=pin('native-fixture.json',{});result=pin('result-fixture.json',{})
    monkeypatch.setattr(m.storage,'validate',lambda *a:(
        dict(request=dict(experiment_id=params['experiment_id']),purpose='development'),{},
        dict(target=str(Path(args['attempt'])/'results')),1024**3))
    monkeypatch.setattr(m.volumes,'inspect',lambda *a:dict(status='VOLUMES_VERIFIED_NOT_ACCEPTED',
        manifests=dict(native_volume=native,result_volume=result)))
    monkeypatch.setattr(m.storage,'for_spec',lambda *a,**k:None)
    monkeypatch.setattr(m.authority,'verify_frozen',lambda *a:None)
    def authority(snapshot_pin,parameters_pin,attempt):
        value=json.loads(m.checked(parameters_pin).read_text())
        exact=dict(auth,parameters=value,attempt=attempt)
        b.observed['authorization']=exact
        return dict(status='PROPOSED_NOT_ISSUED',store_permission_issued=False,
            job_submitted=False,authorization_proposal=exact)
    prepare=Mock(side_effect=authority);monkeypatch.setattr(m.authority,'prepare',prepare)
    runner={k:args['config_pin'] for k in ('query_executable','decision_executable','queue_review','sampler_review')}
    runner.update(purpose='development',sandbox='read-only')
    request=pin('materializer-input.json',dict(snapshot_receipt=snapshot,profile_draft=draft,
        storage_key='fixture',stage_id=args['stage_id'],conditions=args['conditions'],runner=runner))
    before=b.store.replay()
    with b.journal.writer():
        first=m.materialize(b.journal,request,b.store,free_disk_bytes=500*1024**3)
        budget=b.journal.read()['budget']
        assert m.materialize(b.journal,request,b.store,free_disk_bytes=500*1024**3)==first
    spec=json.loads(m.checked(first['artifacts']['spec']).read_text())
    assert spec['h1']['mode']==b.spec['h1']['mode'] and spec['seed']==b.spec['seed']
    assert spec['resources']['transitions']==512
    assert b.journal.read()['budget']==budget and b.store.replay()==before
    assert not first['dispatch_authorized']
    prepare.assert_called_once();b.popen.assert_not_called()
