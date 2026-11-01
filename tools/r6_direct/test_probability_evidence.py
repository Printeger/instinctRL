"""Read-only integration checks against sealed diagnostic04; skip absent originals."""
import copy
import json
from pathlib import Path
import pytest
from tools.r6_evidence import file_pin
from tools.r6_direct import probability_evidence as m

BASE=Path('docs/experiments/r6-direct-training/continuation-20261101/probability-cuda-04')

@pytest.fixture
def evidence():
    envelope=BASE/'SEALED_EVIDENCE.json'
    data=json.loads(envelope.read_text())
    if not Path(data['result']['path']).exists():pytest.skip('original diagnostic not mounted')
    request=json.loads((BASE/'plan.json').read_text())['probability_replay']
    return envelope,data,request


def test_actual_sealed_replay(evidence):
    path,_,r=evidence
    assert m.verify(file_pin(path),r['contract'],r['retained'])[0]['status']=='PASS'


def test_wrong_sealed_digest_rejected(evidence,tmp_path):
    _,data,r=evidence;data['result']['sha256']='0'*64
    p=tmp_path/'bad.json';p.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='PIN_DRIFT'):m.verify(file_pin(p),r['contract'],r['retained'])


@pytest.mark.parametrize('fault',['starting_update','error','device','tolerance','source'])
def test_semantic_counterexamples_after_decode(evidence,monkeypatch,fault):
    path,_,r=evidence;original=json.loads
    def mutated(*args,**kwargs):
        value=original(*args,**kwargs)
        if isinstance(value,dict) and 'backend_before' in value and 'observations' in value:
            if fault=='starting_update':value['result']['starting_update']=999
            if fault=='error':value['result']['maximum_absolute_errors']['alpha']=999
            if fault=='device':value['observations'][0]['replay_device']='cpu'
            if fault=='tolerance':value['observations'][0]['atol']=1
        if fault=='source' and isinstance(value,dict) and value.get('run_id')=='R6-DIRECT-PROBABILITY-DIAGNOSTIC-04':
            value['source_commit']='0'*40
        return value
    monkeypatch.setattr(m.json,'loads',mutated)
    with pytest.raises(ValueError):m.verify(file_pin(path),r['contract'],r['retained'])
