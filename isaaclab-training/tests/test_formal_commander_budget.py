"""No live actions: cumulative admissions must protect future obligations."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from tools.r5_formal_commander import remaining_check
B=Path(__file__).resolve().parents[2]/'docs/experiments/r5-formal-readiness-auto'
def scenario():
 master=json.loads((B/'MASTER_AUTHORITY.json').read_text());adj=json.loads((B/'experiments/R5-H128-DEV-01/BUDGET_ADJUSTMENT.json').read_text())
 b={'caps':master['master_caps'],'used':{k:0 for k in master['master_caps']},'stage_allocation':adj['new_stage'],'reservations':{'live_entries':[],'formal_initial_segment':adj['formal_reservation'],'confirmation_and_transition':adj['confirmation_reservation'],'source_snapshots':[{'reserved_bytes':2*1024**3}]}}
 budget={'training_transitions':65536,'nontraining_transitions':0,'optimizer_updates':128,'checkpoint_writes':9,'wall_seconds':3600}
 return b,budget

def test_initial_entry_fits_without_spending_future_reserves():
 b,p=scenario();before=deepcopy(b);r=remaining_check(b,p,8*1024**3);assert b==before and r['conservative_output_reserved_bytes_including_confirmation_formal_and_metadata']<b['caps']['output_peak_and_reservations_bytes']

@pytest.mark.parametrize('field',['training_transitions','nontraining_transitions','live_starts','simulator_initialization_reservations','optimizer_updates','actor_optimizer_steps','critic_optimizer_steps','training_checkpoint_writes','allocated_gpu_seconds','wall_seconds_including_cleanup','output_peak_and_reservations_bytes'])
def test_exhaustion_is_rejected_before_submission(field):
 b,p=scenario();b['caps'][field]=0
 with pytest.raises(AssertionError):remaining_check(b,p,8*1024**3)

@pytest.mark.parametrize('field',['training_transitions','optimizer_updates','training_checkpoint_writes','live_starts','simulator_initialization_reservations','output_peak_and_reservations_bytes','allocated_gpu_seconds','wall_seconds_including_cleanup'])
def test_stage_exhaustion_requires_recorded_delegated_adjustment(field):
 b,p=scenario();b['stage_allocation'][field]=0
 with pytest.raises(AssertionError):remaining_check(b,p,8*1024**3)


@pytest.mark.parametrize('field',['allocated_gpu_seconds','wall_seconds_including_cleanup'])
def test_stage_time_includes_nonzero_used_and_existing_reservations(field):
 b,p=scenario();b['used'][field]=100;b['reservations']['live_entries']=[{'reserved':{k:(200 if k==field else 0) for k in b['used']},'output_cap_bytes':0}]
 b['stage_allocation'][field]=3910
 remaining_check(b,p,8*1024**3)
 b['stage_allocation'][field]=3909
 with pytest.raises(AssertionError):remaining_check(b,p,8*1024**3)


def clean_fixture(tmp_path):
 from tools.r5_formal_commander import previous_quiescence
 d=tmp_path/'jobs/J-CPU';d.mkdir(parents=True)
 owner={'pid':1};guard={'pid':2};child={'pid':3};desc={'pid':4}
 def put(name,data):(d/name).write_text(json.dumps(data))
 put('lifeguard-ready.json',{'identity':guard});put('process-identity.json',child);put('descendant-ledger.json',{'descendants':[desc]})
 clean={'quiescent':True,'stable_scans':2,'survivors':[],'discovered':[desc],'timed_out':False}
 put('lifeguard-ack.json',{'no_survivors':True,'descendant_cleanup':clean})
 return d,{'jobs':{'J-CPU':{'identity':owner,'status':'FAILED'}}},previous_quiescence


def test_normal_prior_cleanup_is_explicitly_verified(tmp_path):
 d,state,verify=clean_fixture(tmp_path);assert verify(tmp_path,state,probe=lambda _:False)[0]['boundary']=='NORMAL_ACK'

@pytest.mark.parametrize('change',['missing_ack','alive_guard','alive_child','alive_descendant','timed_out','not_quiescent'])
def test_terminal_failed_does_not_bypass_missing_or_uncertain_cleanup(tmp_path,change):
 d,state,verify=clean_fixture(tmp_path)
 if change=='missing_ack':(d/'lifeguard-ack.json').unlink()
 if change in ['timed_out','not_quiescent']:
  a=json.loads((d/'lifeguard-ack.json').read_text());a['descendant_cleanup']['timed_out' if change=='timed_out' else 'quiescent']=change=='timed_out';(d/'lifeguard-ack.json').write_text(json.dumps(a))
 live={'alive_guard':2,'alive_child':3,'alive_descendant':4}.get(change)
 with pytest.raises(AssertionError):verify(tmp_path,state,probe=lambda i:bool(i and i['pid']==live))
