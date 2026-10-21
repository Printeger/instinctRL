"""Read-only delegation checks against exact user authority, without Store writes."""
import copy,json
from types import SimpleNamespace
import pytest
from tools import r6_issue_next as issue
from tools.r6_recovery import RecoveryError,file_pin


def state():
    master=json.loads(issue.MASTER.read_text());old=master['caps']
    new=dict(old,automatic_model_sessions=512)
    return dict(goal_id=master['id'],budget=dict(caps=new,budget_amendments=[dict(
        authority=issue.MODEL_AMENDMENT,old_caps=old,new_caps=copy.deepcopy(new))]))


def check(s):
    issue.delegation(SimpleNamespace(read=lambda:s),file_pin(issue.MASTER),
        SimpleNamespace(root=issue.ROOT/'coordination/research'))


def test_exact_amendment_preserves_historical_master():
    s=state();before=copy.deepcopy(s);check(s);assert s==before
    assert json.loads(issue.MASTER.read_text())['caps']['automatic_model_sessions']==256


@pytest.mark.parametrize('damage',['missing','forged','scientific','duplicate','old_caps'])
def test_unapproved_caps_or_amendment_cannot_issue(damage):
    s=state();b=s['budget'];a=b['budget_amendments'][0]
    if damage=='missing':b['budget_amendments']=[]
    if damage=='forged':a['authority']=dict(issue.MODEL_AMENDMENT,sha256='0'*64)
    if damage=='scientific':b['caps']['live_starts']+=1;a['new_caps']['live_starts']+=1
    if damage=='duplicate':b['budget_amendments'].append(copy.deepcopy(a))
    if damage=='old_caps':a['old_caps']['automatic_model_sessions']=32
    with pytest.raises(RecoveryError):check(s)
