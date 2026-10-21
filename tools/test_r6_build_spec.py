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
    assert spec['seed']==19 and spec['h1']['config_sha256']==args['config_pin']['sha256']
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
    (Path(args['attempt'])/'results'/'research-r6-evaluation').mkdir()
    with pytest.raises(Conflict,match='already consumed'):build(**args)


def test_cannot_build_training_as_baseline(bound):
    b=bound;args=inputs(b);a=b.observed['authorization'];a['budgets']['training_transitions']=512
    path=Path(args['authorization_pin']['path']);path.write_text(json.dumps(a));args['authorization_pin']=file_pin(path)
    with pytest.raises(RecoveryError,match='NO_TRAINING'):build(**args)
