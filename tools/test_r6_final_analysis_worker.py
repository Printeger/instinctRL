"""Control interpreter event test; motion replay is an explicit missing-raw stub."""
import json
from pathlib import Path
import pytest
from tools.test_r6_final_analysis import case,closed
from training_workloads.unified_h1 import robustness_analysis as replay

@pytest.fixture(autouse=True)
def missing_original(monkeypatch):
    def missing(*a,**k):
        summary={'scientific_acceptance':False}
        return dict(inputs=[],summary=summary),summary
    monkeypatch.setattr(replay,'analyze',missing)

def test_final_cpu_event_uses_pinned_authorization_and_program(closed):
    from tools import r6_cpu_analysis as worker
    from tools.r6_recovery import Journal
    j=Journal(closed.root/'batch');budget=json.loads(Path('tools/fixtures/r6_budget.json').read_text())
    with j.writer():
        j.initialize(budget);state=j.read();state['events'][closed.key]=closed.entry;j.save(state)
        key=worker.register_analysis(j,closed.key,closed.store,free_disk_bytes=500*1024**3)
        entry,argv=worker.validate(j,key)
        assert argv[2]==str(worker.ROOT/'tools/analyze_r6_final.py')
        assert argv[3:5]==[str(closed.output),closed.ap['path']]
        assert argv[-2:]==['--authorization-sha256',closed.ap['sha256']]
        assert worker.register_analysis(j,closed.key,closed.store,free_disk_bytes=500*1024**3)==key
