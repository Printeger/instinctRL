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

@pytest.mark.parametrize('field',['training_transitions','optimizer_updates','training_checkpoint_writes','live_starts','simulator_initialization_reservations','output_peak_and_reservations_bytes'])
def test_stage_exhaustion_requires_recorded_delegated_adjustment(field):
 b,p=scenario();b['stage_allocation'][field]=0
 with pytest.raises(AssertionError):remaining_check(b,p,8*1024**3)
