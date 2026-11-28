"""Freeze the four v60 contracts and the initial one-shot launch packet."""
import sys
from pathlib import Path

from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin, stop_paths
from tools.r6_direct.instinct_cycle import destination, load, save
from tools.r6_direct.prepare import build
from tools.r6_direct.resource_policy import COMMAND_REPLAY_SCHEMA, OUTPUT_CAP, THIRD_CAPS
from tools.r6_direct.run import resources_after_validation
from tools.r6_direct.source import ROOT as snapshot, manifest
from tools.r6_direct.storage import CONTROL_ALLOWANCE, SYSTEM_ROOT, SYSTEM_STORAGE_SCHEMA
from training_workloads.unified_h1 import command_replay as replay
from training_workloads.unified_h1.direct_contract import validate_runtime_contract


root = ROOT/'docs/experiments/instinct-continuous-command'
scope = root/'mover_passage_reproof'
assert not stop_paths() and snapshot != ROOT
source = save(scope/'SOURCE.json', manifest())
control = pin(root/'gravity_preview_renewal/RESULT.json')
collision = pin(root/'registered_mover_entry/RESULT.json')
baseline = read(control)
failed = read(collision)
_, previous, _ = validate_binding(baseline['binding'])
_, diagnostic, _ = validate_binding(failed['binding'])
params = previous['contract']['parameters']
assert baseline['execution'] == replay.GRAVITY_PREVIEW_RENEWAL_EXECUTION
assert failed['execution'] == replay.REGISTERED_MOVER_ENTRY_EXECUTION
assert failed['contacts_all'] == {'16':[731]}
assert all(v == diagnostic['contract']['parameters'][k] for k,v in params.items()
           if k not in ('experiment_id','execution'))
assert params['seed'] == 1999375583 and read(params['stream'])['generator']['seed'] == 1469625851
assert previous['pins']['receipt'] == baseline['receipt'] and diagnostic['pins']['receipt'] == failed['receipt']
assert params['models']['candidate']['sha256'] == replay.MODELS['b']
execution = replay.MOVER_PASSAGE_REPROOF_EXECUTION
model = params['models']['candidate']
first = replay.NOVEL_MOVER_PASSAGE_REPROOF_ID
second = replay.NOVEL_V60_CLOSURE_FRESH_ID
regression = replay.REGRESSION_V60_CLOSURE_ID
long_id = replay.LONG_V60_CLOSURE_ID
storage = save(scope/'STORAGE.json',dict(schema_version=SYSTEM_STORAGE_SCHEMA, root=str(SYSTEM_ROOT),
                                         new_volume_allocation_bytes=0))


def freeze(directory, compiled, count, cap=None):
    compiled = validate_runtime_contract(compiled)
    run_id = compiled['parameters']['experiment_id']
    budgets = compiled['budgets']
    assert (budgets['nontraining_transitions'],budgets['training_transitions'],budgets['checkpoint_writes']) == (count,0,0)
    conf = save(directory/(run_id+'-contract.json'), compiled)
    zero = dict.fromkeys(THIRD_CAPS,0)
    limits = dict(zero, **resources_after_validation(compiled), major_scientific_candidates=1)
    cap = cap or compiled['parameters']['output_cap_bytes']
    limits[OUTPUT_CAP] = cap+CONTROL_ALLOWANCE
    policy = save(directory/'RESOURCE_POLICY.json',dict(schema_version=COMMAND_REPLAY_SCHEMA,
        user_authority=pin(scope/'PLAN.md'),caps=limits,elapsed_seconds=86400))
    save(directory/'RESOURCE_LEDGER.json',dict(resource_policy=policy,caps=limits,historical_charged=zero,
        new_used=zero,protected_confirmation_delivery_repair=zero,status='FROZEN',new_reservations=[
            dict(id=previous['plan']['storage_reservation_id'],status='UNRELEASED',resources={OUTPUT_CAP:0},storage_plan=storage)]))
    return dict(previous['plan'], **destination(run_id), contract=conf, resource_policy=policy, source=source,
        snapshot=str(snapshot), result_payload_cap_bytes=cap)


def rescope(directory, old_opportunities, run_id):
    return save(directory/'OPPORTUNITIES.json',dict(read(old_opportunities),run_id=run_id))


main_opps = rescope(scope,pin(root/'gravity_preview_renewal/OPPORTUNITIES.json'),first)
initial = freeze(scope,replay.contract(dict(params,experiment_id=first,execution=execution)),38400)
initial['predecessor_receipt'] = diagnostic['pins']['receipt']

fresh = scope/'fresh'
planned = load(Path(sys.argv[1])/'SEEDS.json')
fresh_stream = save(fresh/'STREAMS.json',read(planned['stream']))
fresh_opps = rescope(fresh,planned['opportunities'],second)
fresh_windows = save(fresh/'RESTORATION-WINDOWS.json',read(planned['restoration']))
assert planned['scene_seed'] not in (params['seed'],696110106)
assert planned['generator_seed'] not in (read(params['stream'])['generator']['seed'],741106095)
fresh_params = dict(params,experiment_id=second,seed=planned['scene_seed'],stream=fresh_stream,
                    street_sweep=planned['street_sweep'],execution=params['execution'],candidate_execution=execution,
                    models=dict(control=model,candidate=model),wall_seconds=1800)
fresh_plan = freeze(fresh,replay.contract(fresh_params),76800)

r6 = scope/'regression'
previous_r6 = replay.REGRESSION_SUPPORTED_WALL_ID
r6_params = load(root/'regression/attempt-03'/(previous_r6+'-contract.json'))['parameters']
selection = r6_params['evaluation_selection']
assert selection['checkpoint'] == model
selection.update(comparison=save(r6/'COMPARISON.json',dict(read(selection['comparison']),id=regression)),
                 execution_override=execution)
r6_params.update(experiment_id=regression,r6=dict(r6_params['r6'],run_id=regression))
r6_plan = freeze(r6,build('contract',dict(operation='evaluate',parameters=r6_params)),193600,cap=5*2**30)

long = scope/'long'
long_input = load(Path(sys.argv[2])/'SEEDS.json')
assert long_input['scene_seed'] == 696110106 and long_input['generator_seed'] == 741106095
long_opps = rescope(long,long_input['opportunities'],long_id)
long_stream = long_input['stream']
gates = pin(root/'long/GATES.json')
assert gates['sha256'] == replay.FORMAL_GATE_SHA256
protocol = save(long/'PROTOCOL.json',dict(schema_version='instinctrl-command-long-protocol/4',
    experiment_id=long_id,scene_seed=long_input['scene_seed'],stream=long_stream,execution=execution,checkpoint=model,
    progress_opportunities=long_opps,acceptance_gates=gates,street_sweep=long_input['street_sweep'],
    pulse_acceleration_m_s2=long_input['pulse_acceleration_m_s2']))
long_params = dict(experiment_id=long_id,training_seed=params['training_seed'],seed=long_input['scene_seed'],
    models=dict(candidate=model),execution=execution,stream=long_stream,protocol=protocol,
    clearance_envelope=params['clearance_envelope'],wall_seconds=7200,output_cap_bytes=2**31)
long_plan = freeze(long,replay.contract(long_params),576000)

release = save(scope/'RELEASE.json',dict(source=source,checkpoint=model,execution=execution,
    limits=dict(same_input=1,fresh_pair=1,regression=1,long=1,training=0),control=control,collision=collision,
    same=dict(contract=initial['contract'],opportunities=main_opps,restoration=pin(scope/'RESTORATION-WINDOWS.json')),
    fresh=dict(template=save(fresh/'CONDITIONAL-INPUTS.json',fresh_plan),opportunities=fresh_opps,
               restoration=fresh_windows,scene_seed=planned['scene_seed'],generator_seed=planned['generator_seed'],
               planning=pin(Path(sys.argv[1])/'PLANNING_CHECK.json')),
    regression=dict(template=save(r6/'CONDITIONAL-INPUTS.json',r6_plan),
                    reference=pin(ROOT/'docs/experiments/instinct-post-u1792-two-long/study/R6-B.json'),
                    gates=pin(ROOT/'docs/experiments/third-training/formal/FREEZE.json')),
    long=dict(template=save(long/'CONDITIONAL-INPUTS.json',long_plan),opportunities=long_opps,gates=gates,
              original_planning=pin(Path(sys.argv[2])/'SEEDS.json'))))
print(release)
print(save(scope/(first+'-inputs.json'),initial))
