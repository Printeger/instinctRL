from pathlib import Path
import sqlite3
import pytest
from rlflow.store import Store
from rlflow.accounting import correct_prelaunch
from rlflow.migration import upgrade_control


def test_original_92_events_upgrade_and_corrections_preserve_history(tmp_path):
    source=Path('/home/mint/rl_dev/instinctRL_orchestrator/implementation_records/orchestrator-20260914/historical-92.db')
    root=tmp_path/'history';store=Store(root)
    with sqlite3.connect(source) as src,sqlite3.connect(store.db) as dst:src.backup(dst)
    before=store.replay()
    with store.connect() as db:events=[dict(r) for r in db.execute('select * from events order by seq')]
    assert len(events)==92
    state=upgrade_control(store)
    for sid,se in before['sessions'].items():
        if se.get('prelaunch_failure'):state=correct_prelaunch(store,sid)
    assert state['paused']==before['paused'] and state['enabled']==before['enabled']
    assert state['proposals']==before['proposals'] and state['approvals']==before['approvals']
    stage='UNIFIED-H1-R3-CPU-A0007'
    assert state['stages'][stage]['used_calls']==before['stages'][stage]['used_calls']-2
    assert state['stages'][stage]['used_seconds']==before['stages'][stage]['used_seconds']
    assert state['stages'][stage]['status']=='HOLD'
    with store.connect() as db:after=[dict(r) for r in db.execute('select * from events order by seq')]
    assert after[:92]==events
    assert store.rebuild()==state
    assert upgrade_control(store)==state
