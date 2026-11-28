"""Close NOVEL-115, preserving both controls and the early and late crossing evidence."""
from collections import defaultdict
import gzip
from pathlib import Path
import sys

import numpy as np
from tools.r6_direct.evidence import read, validate_binding
from tools.r6_direct.import_legacy import ROOT, pin, stop_paths
from tools.r6_direct.instinct_cycle import load, save
from tools.r6_direct.io import atomic
from tools.r6_direct.lifecycle import writer
from tools.r6_direct.settle import settle_time
from training_workloads.unified_h1.command_replay import NOVEL_MOVER_PASSAGE_REPROOF_ID as RUN

root = ROOT/'docs/experiments/instinct-continuous-command'
scope = root/'mover_passage_reproof'
release = load(scope/'RELEASE.json')
plan = load(scope/(RUN+'-inputs.json'))
analysis = Path(plan['attempt_dir'])/'analysis'
assert not stop_paths(), stop_paths()
binding = pin(analysis/'binding.json')
bound, context, report = validate_binding(binding)
assert context['plan'] == plan and plan['source'] == release['source']
measures = pin(analysis/'clearance-and-progress.json')
assert read(measures)['binding'] == binding
score_pin = pin(Path(sys.argv[1]))
scored = read(score_pin)
assert not scored['self_check'] and measures in scored['inputs'][1]
base = read(release['control'])
failed = read(release['collision'])
lanes = report['summary']['models']['candidate']['layout_routes']
trace = Path(lanes['trace']['path'])
_, _, old = validate_binding(base['binding'])
assert pin(trace) == lanes['trace']
old_layout = old['summary']['models']['candidate']['layout_routes']
changes, registrations, passes, contacts, resets = (defaultdict(list) for _ in range(5))
with gzip.open(trace,'rt') as trial, gzip.open(old_layout['trace']['path'],'rt') as reference:
    assert next(trial) == next(reference)
    import json
    for expected,(a,b) in enumerate(zip(trial,reference,strict=True)):
        row, old_row = json.loads(a), json.loads(b)
        assert row['step'] == old_row['step'] == expected and row['raw_event'] == old_row['raw_event']
        for lane in range(32):
            for field,index in (('root_state_after_w',0),('v_final_w',1)):
                if row[field][lane] != old_row[field][lane] and index not in [x['index'] for x in changes[lane]]:
                    changes[lane].append(dict(field=field,index=index,step=expected))
            for flag, data in (('collision',contacts),('reset',resets)):
                if row[flag][lane]:
                    data[lane].append(expected)
            for field, received in (('escape_registered_mover_entry',registrations),('escape_mover_passage_reproof',passes)):
                evidence = row[field][lane]
                if evidence[0]:
                    received[lane].append(dict(step=expected,proof=evidence,issued=row['v_final_w'][lane],
                                                latest=row['raw_event'][lane]['velocity_w_m_s']))
assert expected == 1199
_, _, collision = validate_binding(failed['binding'])
versions = dict(v58=old_layout,v59=collision['summary']['models']['candidate']['layout_routes'],v60=lanes)
ranges = {}
for version, data in versions.items():
    p = data['clearance']
    assert pin(p['path']) == p
    envelope = np.memmap(p['path'],dtype='<f4',mode='r',shape=(1200,21,32,7))
    first_reset = data['lanes'][16]['first_reset_step']
    ranges[version] = {}
    for start,stop in ((390,430),(660,700),(700,781)):
        if first_reset is not None:
            stop = min(stop,first_reset+1)
        if start >= stop:
            continue
        dense = np.asarray(envelope[start:stop,:,16],dtype=float)
        net = np.linalg.norm(dense[:,:,:3]-dense[:,:,3:6],axis=-1)-.34-dense[:,:,6]
        j,k = np.unravel_index(net.argmin(),net.shape)
        ranges[version][str(start)] = dict(from_step=start,through_step=stop-1,
                             minimum_m=float(net[j,k]),at_s=round((start+j)*.1+k*.005,3))
metric = read(measures)['models']['candidate']['layout_routes']
assert sum(len(x['externally_scheduled_stops']) for x in metric) == 32
assert sum(len(x['externally_scheduled_progress']) for x in metric) == 64
restoration = scored['matched_restoration']
per_layout = {}
for layout, worst in scored['layout_worst'][1].items():
    events = [x for x in restoration['matched_rows'] if x['window']['layout'] == layout]
    summary = {}
    for version, arm in (('v58','baseline'),('v60','candidate')):
        ticks = [x[arm]['ticks'] for x in events]
        summary[version] = dict(total=len(ticks),missing=ticks.count(None),
                               timeout=sum(x is not None and x>15 for x in ticks),
                               longest_confirmed_s=max((x/10 for x in ticks if x is not None),default=None))
    per_layout[layout] = dict(worst,matched_restoration=summary)
worst_lane = min(metric,key=lambda m:m['physics_tick_clearance']['minimum_clearance_lower_bound_m'])['lane']
passed = scored['bounded_pass']
result = save(scope/'RESULT.json',dict(schema='instinctrl-v60-bounded-closure/1',
    status='SAME_INPUT_PASSED' if passed else 'UNMET_SAME_INPUT',
    stages=dict(same='PASSED' if passed else 'FAILED',fresh='NOT_RUN',regression='NOT_RUN',long='NOT_RUN'),
    release=pin(scope/'RELEASE.json'),binding=binding,clearance=measures,paired_scoring=score_pin,
    execution=release['execution'],checkpoint=release['checkpoint'],counters=report['summary']['counters'],
    native_receipt=bound['receipt'],native_exit=bound['native_exit_code'],wall_seconds=context['receipt']['wall_seconds'],
    contacts=dict(contacts),resets=dict(resets),n113_first_differences={k:v for k,v in changes.items() if v},
    registered_mover_packets=dict(registrations),mover_passage_packets=dict(passes),street_lane16_mover_clearance=ranges,
    worst_clearance_lane=worst_lane,worst_clearance=metric[worst_lane]['physics_tick_clearance'],
    worst_clearance_tracking=metric[worst_lane]['tracking_in_feasible_moving_windows'],
    worst_clearance_stops=metric[worst_lane]['externally_scheduled_stops'],
    worst_clearance_restoration=[e for e in restoration['matched_rows'] if e['window']['lane']==worst_lane],
    by_layout=per_layout,elapsed_lane_seconds=sum(f['elapsed_s'] for f in lanes['lanes'])))
ledger = scope/'RESOURCE_LEDGER.json'
with writer():
    original = load(ledger)
    assert original['status'] == 'WAITING_FOR_ANALYSIS'
    closed = settle_time(original,RUN)
    closed['status'] = 'CLOSED_SAME_INPUT_PASSED' if passed else 'CLOSED_SAME_INPUT_UNMET'
    atomic(ledger,closed)
print(result)
