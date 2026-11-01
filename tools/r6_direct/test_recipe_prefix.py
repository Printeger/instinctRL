"""Registered-node provenance tests; no live execution or scientific acceptance."""
import copy
import json
from pathlib import Path
import pytest
from tools.r6_evidence import file_pin
from training_workloads.unified_h1.robustness_confirmation import validate_registered_prefix


@pytest.fixture
def prefix(tmp_path):
    source=Path('docs/experiments/r6-direct-training/continuation-20261101/train-02b/recipe.json')
    old=json.loads(source.read_text());new=copy.deepcopy(old)
    curriculum=json.loads(Path(old['curriculum']['path']).read_text())
    curriculum['phases']=curriculum['phases'][:2]
    path=tmp_path/'curriculum.json';path.write_text(json.dumps(curriculum))
    new.update(branch_updates=128,checkpoint_offsets=[128],curriculum=file_pin(path))
    lineage=dict(kind='REGISTERED_DEVELOPMENT_PREFIX',development_recipe=file_pin(source))
    plan=dict(schema_version='r6-local-training-plan/1',recipe=file_pin(source),phase='development',
        start_update=640,target_update=768,checkpoint_updates=[768],
        initialization={'kind':'COMMON_U640_NEW_STREAM'})
    return lineage,new,dict(selection={'training_plan':plan},model={'update':768})


def test_exact_registered_prefix(prefix):
    validate_registered_prefix(*prefix)


@pytest.mark.parametrize('fault',['method','source','seeds','unregistered','checkpoints','model',
    'unfinished','resume','recipe_pin','curriculum'])
def test_rejects_changed_method_or_unevaluated_endpoint(prefix,fault,tmp_path):
    lineage,new,trained=prefix;plan=trained['selection']['training_plan']
    if fault=='method':new['method']=new['curriculum']
    elif fault=='source':new['source_identity']['product_commit']='0'*40
    elif fault=='seeds':new['confirmation_seeds'][0]+=10
    elif fault=='unregistered':new['branch_updates']=64
    elif fault=='checkpoints':new['checkpoint_offsets']=[64,128]
    elif fault=='model':trained['model']['update']=896
    elif fault=='unfinished':plan['target_update']=896
    elif fault=='resume':plan['initialization']['kind']='RESUME_R6_STREAM'
    elif fault=='recipe_pin':plan['recipe']=new['curriculum']
    else:
        data=json.loads(Path(new['curriculum']['path']).read_text())
        data['phases'][1]['noise_std_m']=0.02
        path=tmp_path/'changed.json';path.write_text(json.dumps(data));new['curriculum']=file_pin(path)
    with pytest.raises(PermissionError):validate_registered_prefix(lineage,new,trained)


@pytest.mark.parametrize('fault',[None,'distribution','holdout','original','review','duplicate'])
def test_seed_amendment_requires_exact_reviewed_override(tmp_path,fault):
    from training_workloads.unified_h1.robustness_confirmation import validate_seed_amendment,base
    def pin(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return file_pin(path)
    old=dict(confirmation_training_seeds=[1,2,3],development={'seed':10},
        final={'seed_realization_generated':False})
    old_pin=pin('old.json',old);protocol={'split_manifest':old_pin}
    protocol_pin=pin('protocol.json',protocol)
    new=copy.deepcopy(old);new['confirmation_training_seeds']=[4,5,6]
    if fault=='distribution':new['development']['seed']=11
    if fault=='holdout':new['final']['seed_realization_generated']=True
    if fault=='duplicate':new['confirmation_training_seeds']=[4,4,6]
    new_pin=pin('new.json',new)
    payload=dict(schema_version='r6-confirmation-seed-amendment/1',protocol=protocol_pin,
        original_split=old_pin,replacement_split=new_pin,justification='Synthetic conflict',evidence_pins=[old_pin])
    if fault=='original':payload['original_split']=new_pin
    review=pin('review.json',dict(decision='MUST_FIX' if fault=='review' else 'PASS_R6_CONFIRMATION_SEED_AMENDMENT',
        reviewer='independent-fixture',must_fix=[],payload_sha256=base.object_sha(payload)))
    amendment=pin('amendment.json',dict(payload,independent_review=review))
    if fault is None:assert validate_seed_amendment(amendment,protocol_pin,protocol,new_pin)==old_pin
    else:
        with pytest.raises((PermissionError,ValueError)):
            validate_seed_amendment(amendment,protocol_pin,protocol,new_pin)


@pytest.mark.parametrize('fault',[None,'method','diff','identity','review'])
def test_source_rebinding_preserves_science_and_reviewed_diff(tmp_path,fault):
    from training_workloads.unified_h1.robustness_confirmation import validate_source_rebinding,base
    from tools.r6_direct.source import training_identity
    def pin(name,value):
        path=tmp_path/name;path.write_text(json.dumps(value));return file_pin(path)
    old=json.loads(Path('docs/experiments/r6-direct-training/continuation-20261101/train-02b/source.json').read_text())
    new=copy.deepcopy(old);new['source_commit']='a'*40;new['source_tree']='b'*40
    key='tools/r6_direct/test_recipe_prefix.py';new['sources'][key]='c'*64
    changes=[dict(path=key,before=old['sources'].get(key),after='c'*64)]
    if fault=='method':new['method']={'changed':True}
    if fault=='diff':changes=[]
    old_id=training_identity(old);new_id=training_identity(new)
    if fault=='identity':new_id['config_sha256']='0'*64
    payload=dict(schema_version='r6-confirmation-source-rebinding/1',original_source=pin('old.json',old),
        replacement_source=pin('new.json',new),changes=changes)
    review=pin('review.json',dict(decision='MUST_FIX' if fault=='review' else 'PASS_R6_CONFIRMATION_SOURCE_REBIND',
        reviewer='independent-fixture',must_fix=[],payload_sha256=base.object_sha(payload)))
    value=pin('rebind.json',dict(payload,independent_review=review))
    if fault is None:validate_source_rebinding(value,old_id,new_id)
    else:
        with pytest.raises(PermissionError):validate_source_rebinding(value,old_id,new_id)
