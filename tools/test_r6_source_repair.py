"""Real temporary Git and fake source writer; snapshot/preflight remain fixtures."""
import copy
import json
from pathlib import Path
import pytest
from tools.test_r6_source_preparation import prepared,register,run
from tools import r6_source_preparation as source,r6_source_repair as repair
from tools.r6_recovery import RecoveryError,file_pin,Journal


def fail(b):
    b.builder.with_suffix('.settings').write_text('{"incomplete":true}')
    with pytest.raises(RecoveryError,match='INCOMPLETE'):run(b)
    b.oldkey=repair.active(b.j,b.parent);b.old=copy.deepcopy(b.j.read()['events'][b.oldkey])
    b.oldfiles={p:p.read_bytes() for p in Path(b.old['source_input']['output_directory']).rglob('*') if p.is_file()}
    b.builder.with_suffix('.settings').write_text('{}')


def proposal(b,reason='Complete the partially committed source preparation'):
    return b.write('source-repair-'+str(len(reason))+'.json',dict(parent_key=b.parent,
        predecessor=b.oldkey,predecessor_receipt=b.j.read()['events'][b.oldkey]['receipt'],
        reason=reason,evidence_pins=[b.report_pin],repository_state=repair.reconcile(b.j,b.oldkey)))


def replace(b,pin):
    with b.j.writer():
        return source.register(b.j,b.parent,b.store,executable=file_pin(b.builder),free_disk_bytes=500*1024**3,repair_pin=pin)


def test_partial_committed_output_is_preserved_and_new_directory_used(prepared):
    b=prepared;fail(b);pin=proposal(b);key=replace(b,pin);budget=b.j.read()['budget']
    assert replace(b,pin)==key and b.j.read()['budget']==budget
    b.j=Journal(b.j.directory);result=run(b)
    assert result['snapshot']['status']=='CPU_SNAPSHOT_FIXTURE'
    assert '/repairs/'+pin['sha256']+'/' in result['preparation']['card']['path']
    assert b.j.read()['events'][b.oldkey]==b.old
    assert all(p.read_bytes()==data for p,data in b.oldfiles.items())
    assert run(b)==result and b.j.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\ncall\n'
    assert b.j.read()['events'][key]['purpose']=='repair'
    assert 'major_scientific_candidates' not in b.j.read()['events'][key]['resources']


def test_missing_replacement_prompt_recovers_without_extra_charge(prepared):
    b=prepared;fail(b);pin=proposal(b);key=replace(b,pin)
    entry=b.j.read()['events'][key];Path(entry['request']['prompt']['path']).unlink();budget=b.j.read()['budget']
    run(b);assert b.j.read()['budget']==budget
    assert b.builder.with_suffix('.count').read_text()=='call\ncall\n'


def test_committed_reconciliation_document_can_precede_new_attempt(prepared):
    b=prepared;fail(b);pin=proposal(b)
    document=b.repo/'docs/source-reconciliation.json';document.write_bytes(Path(pin['path']).read_bytes())
    source.snapshot.git(b.repo,'add','docs/source-reconciliation.json')
    source.snapshot.git(b.repo,'-c','user.name=CPU','-c','user.email=cpu@example.invalid','commit','-qm','record reconciliation')
    key=replace(b,file_pin(document));run(b)
    assert repair.active(b.j,b.parent)==key
    assert all(p.read_bytes()==data for p,data in b.oldfiles.items())


def test_nonzero_after_git_commit_preserves_partial_writes(prepared):
    b=prepared;original=b.builder.read_text()
    b.builder.write_text(original.replace("result={'decision':'SOURCE_READY'", "sys.exit(7)\nresult={'decision':'SOURCE_READY'"))
    with pytest.raises(RecoveryError):run(b)
    b.oldkey=repair.active(b.j,b.parent);old=copy.deepcopy(b.j.read()['events'][b.oldkey])
    receipt=json.loads(Path(old['receipt']['path']).read_text());assert receipt['exit_code']==7
    b.builder=b.root/'fixed_builder';b.builder.write_text(original);b.builder.chmod(0o700)
    b.builder.with_suffix('.settings').write_text('{}')
    key=replace(b,proposal(b));result=run(b)
    assert result['snapshot']['status']=='CPU_SNAPSHOT_FIXTURE'
    assert repair.active(b.j,b.parent)==key and b.j.read()['events'][b.oldkey]==old


def test_same_candidate_repair_does_not_charge_another_candidate(prepared,monkeypatch):
    b=prepared;original=source.parent_decision
    def candidate(*args):
        value=copy.deepcopy(original(*args));value['decision']['decision']='PROPOSE_CANDIDATE';return value
    monkeypatch.setattr(source,'parent_decision',candidate)
    fail(b);assert b.old['resources']['major_scientific_candidates']==1
    key=replace(b,proposal(b))
    assert 'major_scientific_candidates' not in b.j.read()['events'][key]['resources']
    from tools.r6_budget import totals
    assert totals(b.j.read()['budget'])['major_scientific_candidates']==1


@pytest.mark.parametrize('damage',['dirty_source','untracked_source','old_output','old_output_parent_alias','head','cleanup','missing_receipt','stop'])
def test_unreconciled_effects_block_before_reservation(prepared,damage):
    b=prepared;fail(b);pin=proposal(b)
    if damage=='dirty_source':(b.repo/'training_workloads/unified_h1/candidate_config.json').write_text('{"dirty":true}')
    if damage=='untracked_source':(b.repo/'training_workloads/untracked.py').write_text('untracked')
    if damage=='old_output':next(iter(b.oldfiles)).write_text('overwritten')
    if damage=='old_output_parent_alias':
        path=Path(b.old['source_input']['output_directory']);saved=path.with_name(path.name+'-preserved')
        path.rename(saved);path.symlink_to(saved,target_is_directory=True)
    if damage=='head':
        (b.repo/'seed').write_text('unexplained');source.snapshot.git(b.repo,'add','seed')
        source.snapshot.git(b.repo,'-c','user.name=CPU','-c','user.email=cpu@example.invalid','commit','-qm','unexplained')
    if damage in ('cleanup','missing_receipt'):
        receipt=json.loads(Path(b.old['receipt']['path']).read_text());receipt['cleanup']['stable_empty_scans']=0
        with b.j.writer():
            state=b.j.read();state['events'][b.oldkey]['receipt']=None if damage=='missing_receipt' else b.write('bad-receipt.json',receipt);b.j.save(state)
        pin=proposal(b)
    if damage=='stop':(b.j.directory/'STOP').touch()
    before=b.j.read()
    with pytest.raises(RecoveryError):replace(b,pin)
    assert b.j.read()==before;b.freeze.assert_not_called()
    assert b.builder.with_suffix('.count').read_text()=='call\n'


def test_banned_committed_effect_cannot_be_adopted(prepared):
    b=prepared;b.builder.with_suffix('.settings').write_text('{"incomplete":true,"banned":true}')
    with pytest.raises(RecoveryError,match='INCOMPLETE'):run(b)
    b.oldkey=repair.active(b.j,b.parent)
    with pytest.raises(RecoveryError,match='CHANGE_SCOPE'):proposal(b)
    b.freeze.assert_not_called()


def test_accepted_source_cannot_be_replaced(prepared):
    b=prepared;run(b);b.oldkey=repair.active(b.j,b.parent);before=b.j.read()
    with pytest.raises(RecoveryError,match='PREDECESSOR'):replace(b,proposal(b))
    assert b.j.read()==before


def test_obsolete_failure_cannot_fork(prepared):
    b=prepared;fail(b);replace(b,proposal(b));before=b.j.read()
    with pytest.raises(RecoveryError,match='NOT_ACTIVE_TIP'):replace(b,proposal(b,'Alternative repair from the same obsolete failure'))
    assert b.j.read()==before
