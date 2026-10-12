"""Execute backend collection code on CPU with injected protocol/validator faults.

This exercises real JSON writes and failure manifests; protocol coverage is mocked.
No simulator or backend construction is performed.
"""
import ast
import hashlib
import json
import multiprocessing
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any,Mapping

import pytest
from training_workloads.scale_003.lifecycle import exception_record
from training_workloads.scale_003.persistence import atomic_json_replace
from training_workloads.unified_h1 import b0_enhanced,b0_observer,b0_causal_analyzer


def backend_fixture(tmp_path,monkeypatch,fault):
    source=Path('training_workloads/unified_h1/backend.py')
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='run_enhanced_b0_diagnostic')
    tree=ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='Backend',bases=[],keywords=[],decorator_list=[],body=[method])],type_ignores=[]))
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    ns={'__name__':'training_workloads.unified_h1._cpu_extracted_backend','__package__':'training_workloads.unified_h1','Path':Path,'Mapping':Mapping,'Any':Any,'os':os,'hashlib':hashlib,'atomic_json_replace':atomic_json_replace,'exception_record':exception_record,'sha256_file':digest}
    exec(compile(tree,str(source),'exec'),ns)
    class Validator:
        def __init__(self):self.count=0
        def append(self,row):
            if fault=='first_row' or (fault=='mid_validation' and self.count==7):raise ValueError('injected validation')
            self.count+=1
        def finalize(self):
            if fault=='finalization':raise ValueError('injected finalization')
            return {'status':'MOCK_32_ROWS_ONLY','physics_rows':self.count}
    monkeypatch.setattr(b0_enhanced,'TraceValidator',Validator)
    monkeypatch.setattr(b0_observer.ReadOnlyObserver,'finalize',lambda self:{'fixture_only':True,'physics_rows':self.rows})
    captured={}
    def register(callback,**binding):
        captured['callback']=callback
        return {'boundary':'after_physics_before_reset','ordering_unchanged':True,'read_only':True,'exactly_once':True,'lanes':32}
    def run(**kwargs):
        if fault=='measurement':
            error=b0_enhanced.EvidenceError('injected before writer')
            error.measurement_failure={'status':'UNVALIDATED','normal_physics_row':False,
                'failure':{'body_index':0,'body_name':'base','failure_branch':'INERTIA_SHAPE_OR_NONFINITE'},
                'bodies':[{'inertia_at_com_body_kg_m2':[[float('nan'),0.,0.],[0.,1.,0.],[0.,0.,1.]]}]}
            raise error
        rows=[{'case':'Z','policy_step':0,'physics_step':0,'lane':i,'episode_id':0} for i in range(32)]
        if fault=='encoding':rows[7]['bad']=float('nan')
        captured['callback'](rows)
        if fault=='environment':raise RuntimeError('injected environment')
        # Mock protocol handshake only; Validator/observer manifest marks 32 fixture rows.
        return {'physics_callback_batches':12600,'physics_rows':b0_observer.PHYSICS_ROWS,'control_physics_reset_observation_order':'UNCHANGED'}
    def protocol():
        if fault=='setup':raise RuntimeError('injected setup')
        return {'register':register,'state_probe':lambda:OrderedDict((k,{'fixture':0}) for k in b0_observer.PROTECTED_STATE),'accounting':lambda callbacks:None,'run':run,'identity':'post-physics-before-reset/v1'}
    def analyze(path):
        if fault=='analysis':raise RuntimeError('injected analysis')
        return {'fixture_only':True}
    monkeypatch.setattr(b0_causal_analyzer,'analyze_jsonl',analyze)
    value=ns['Backend']();value.environment=SimpleNamespace(enhanced_b0_read_only_protocol=protocol);value.output=tmp_path
    value.resource_check=lambda **kw:None;value.counters=SimpleNamespace(snapshot=lambda:{'training_transitions':0,'real_optimizer_updates':0,'real_checkpoint_writes':0})
    value.authorization={'budgets':{},'authorization_sha256':'a'*64,**{k:'b'*40 for k in ('product_commit','product_tree','workload_commit','workload_tree')},'config_sha256':'c'*64}
    return value,ns


@pytest.mark.parametrize('fault,written,attempted,validated,phase,calls',[
    ('setup',0,0,0,'PREPARED',None),('first_row',0,1,0,'COLLECTION',0),
    ('mid_validation',7,8,7,'COLLECTION',0),('encoding',7,8,8,'COLLECTION',0),
    ('environment',32,32,32,'COLLECTION',1),('finalization',32,32,32,'FINALIZATION',1),
    ('analysis',32,32,32,'ANALYSIS',1)])
def test_failure_retains_actual_raw_progress_and_primary_error(tmp_path,monkeypatch,fault,written,attempted,validated,phase,calls):
    value,_=backend_fixture(tmp_path,monkeypatch,fault)
    with pytest.raises((RuntimeError,ValueError)) as caught:value.run_enhanced_b0_diagnostic({})
    manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text());raw=(tmp_path/'enhanced_b0_physics.jsonl').read_bytes()
    assert manifest['status']=='PARTIAL_FAILURE' and manifest['phase']==phase
    assert manifest['primary_error']['message']==str(caught.value)
    assert manifest['raw']['written_rows']==written==len(raw.splitlines())
    assert manifest['raw']['attempted_rows']==attempted and manifest['raw']['validated_rows']==validated
    assert manifest['raw']['stream_accepted_rows']==written and not manifest['raw']['trailing_partial_row']
    assert manifest['raw']['encoded_bytes']==manifest['raw']['file_bytes']==len(raw)
    assert manifest['raw']['sha256']==hashlib.sha256(raw).hexdigest() and manifest['raw']['stream_closed']
    if calls is None:assert manifest['observer_progress'] is None
    else:assert manifest['observer_progress']['completed_callback_calls']==calls
    if phase in ('PREPARED','COLLECTION','FINALIZATION'):assert manifest['coverage'] is None
    if written:assert manifest['raw']['last_written_coordinate']['lane']==written-1
    assert manifest['execution_counters']['real_optimizer_updates']==0


def test_manifest_save_failure_does_not_replace_collection_error(tmp_path,monkeypatch):
    value,namespace=backend_fixture(tmp_path,monkeypatch,'environment')
    def fail_terminal(output,name,contents):
        if name=='enhanced_b0_manifest.json' and contents['status']=='PARTIAL_FAILURE':raise OSError('injected evidence error')
        atomic_json_replace(output,name,contents)
    namespace['atomic_json_replace']=fail_terminal
    with pytest.raises(RuntimeError,match='injected environment') as caught:value.run_enhanced_b0_diagnostic({})
    assert any('injected evidence error' in note for note in caught.value.__notes__)
    assert len((tmp_path/'enhanced_b0_physics.jsonl').read_bytes().splitlines())==32


def test_complete_manifest_has_finalized_coverage_and_no_error(tmp_path,monkeypatch):
    value,_=backend_fixture(tmp_path,monkeypatch,None);value.run_enhanced_b0_diagnostic({})
    manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text())
    assert manifest['status']=='COMPLETE' and manifest['phase']=='COMPLETE'
    assert manifest['coverage']['status']=='MOCK_32_ROWS_ONLY' and manifest['observation']['fixture_only']
    assert manifest['raw']['written_rows']==32 and manifest['primary_error'] is None


def test_analyzer_process_death_retains_durable_collection_without_close_claim(tmp_path,monkeypatch):
    value,_=backend_fixture(tmp_path,monkeypatch,None)
    observed={'completed_policy_steps':1,'attempted_policy_steps':1,
              'nontraining_transitions':32,'training_transitions':0,
              'real_optimizer_updates':0,'real_checkpoint_writes':0}
    value.counters=SimpleNamespace(snapshot=lambda:dict(observed))
    def abrupt_analyzer(path):
        manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text())
        assert manifest['status']=='COLLECTION_COMPLETE_ANALYSIS_PENDING'
        assert manifest['raw']['stream_closed'] and manifest['execution_counters']==observed
        os._exit(137)  # No exception/finally can save a terminal manifest.
    monkeypatch.setattr(b0_causal_analyzer,'analyze_jsonl',abrupt_analyzer)
    process=multiprocessing.get_context('fork').Process(target=value.run_enhanced_b0_diagnostic,args=({},))
    process.start();process.join(timeout=5)
    if process.is_alive():process.kill();process.join();pytest.fail('CPU substitute did not exit')
    assert process.exitcode==137
    manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text())
    raw=(tmp_path/'enhanced_b0_physics.jsonl').read_bytes()
    assert manifest['status']=='COLLECTION_COMPLETE_ANALYSIS_PENDING' and manifest['phase']=='ANALYSIS'
    assert manifest['coverage']['status']=='MOCK_32_ROWS_ONLY'
    assert manifest['raw']['written_rows']==manifest['raw']['validated_rows']==manifest['raw']['attempted_rows']==32
    assert manifest['raw']['sha256']==hashlib.sha256(raw).hexdigest()
    assert manifest['raw']['last_written_coordinate']['lane']==31
    assert manifest['execution_counters']==observed and manifest['primary_error'] is None
    assert not (tmp_path/'status.json').exists() and not (tmp_path/'cleanup_evidence.json').exists()


def test_failed_flush_distinguishes_stream_acceptance_from_complete_file_rows(tmp_path,monkeypatch):
    value,_=backend_fixture(tmp_path,monkeypatch,'environment')
    class Output(type(tmp_path)):
        def open(self,mode='r',*args,**kwargs):
            stream=super().open(mode,*args,**kwargs)
            if mode!='x' or self.name!='enhanced_b0_physics.jsonl':return stream
            class PartialClose:
                @property
                def closed(self):return stream.closed
                def write(self,text):return stream.write(text)
                def close(inner):
                    stream.close()
                    raw=Path(self).read_bytes();Path(self).write_bytes(raw[:-5])
                    raise OSError('injected partial flush')
            return PartialClose()
    value.output=Output(tmp_path)
    with pytest.raises(RuntimeError,match='injected environment') as caught:value.run_enhanced_b0_diagnostic({})
    manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text());raw=(tmp_path/'enhanced_b0_physics.jsonl').read_bytes()
    assert manifest['raw']['stream_accepted_rows']==32 and manifest['raw']['written_rows']==31
    assert manifest['raw']['file_bytes']==len(raw)<manifest['raw']['encoded_bytes']
    assert manifest['raw']['trailing_partial_row'] and manifest['raw']['last_written_coordinate'] is None
    assert manifest['raw']['sha256']==hashlib.sha256(raw).hexdigest()
    assert any('injected partial flush' in note for note in caught.value.__notes__)


def test_measurement_before_writer_saves_separate_unvalidated_artifact(tmp_path,monkeypatch):
    value,_=backend_fixture(tmp_path,monkeypatch,'measurement')
    with pytest.raises(b0_enhanced.EvidenceError,match='before writer'):value.run_enhanced_b0_diagnostic({})
    artifact=tmp_path/'enhanced_b0_measurement_failure.json'
    raw=artifact.read_bytes();context=json.loads(raw)
    assert context['status']=='UNVALIDATED' and not context['normal_physics_row']
    assert context['bodies'][0]['inertia_at_com_body_kg_m2'][0][0]=={'availability':'NONFINITE','representation':'nan'}
    manifest=json.loads((tmp_path/'enhanced_b0_manifest.json').read_text())
    assert manifest['primary_error']['message']=='injected before writer'
    assert manifest['measurement_failure']['sha256']==hashlib.sha256(raw).hexdigest()
    assert manifest['raw']['written_rows']==manifest['raw']['attempted_rows']==0
    assert manifest['coverage'] is None and (tmp_path/'enhanced_b0_physics.jsonl').read_bytes()==b''
