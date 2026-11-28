"""Locate delayed restoration in a closed replay on its frozen recovery clock."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

from instinctlab.control.governor import decode_governor
from tools.r6_direct.import_legacy import ROOT, stop_paths
from tools.r6_direct.instinct_cycle import save
from tools.r6_direct.replay_clearance import confirmed_starts, projected_clearance

scope = ROOT/'docs/experiments/instinct-continuous-command/mover_passage_reproof'
sys.path.insert(0, str(scope))
import score_matched_restoration as registered


def first_confirmation(mask):
    starts = confirmed_starts(mask)
    return int(starts[0])+5 if starts.size else None


def on_time(ticks):
    return ticks is not None and ticks <= registered.RESTORE_TICKS


def along_fraction(velocity, request):
    return np.sum(velocity*request,axis=-1)/np.maximum(np.sum(request*request,axis=-1),1.e-6)


def trace_paths(raw):
    rows = raw['rows']
    state = torch.as_tensor([row['actor_state'] for row in rows], dtype=torch.float32)
    req_b = state[:,:,-1,6:9].numpy()
    state = state.reshape(-1,52)
    decoded = []
    for key in ('action', 'selected_action'):
        action = torch.as_tensor([row[key] for row in rows], dtype=state.dtype)
        decoded.append(decode_governor(action.reshape(-1,4),state).v_gov_b.reshape(*req_b.shape).numpy())
    governed = np.asarray([row['v_gov_b'] for row in rows])
    difference = np.max(np.abs(decoded[1]-governed),axis=-1)
    extra = [(int(step),int(lane),float(difference[step,lane])) for step,lane in zip(*np.where(difference>=1.e-6))]
    assert all(raw['rows'][step]['action'][lane] != raw['rows'][step]['selected_action'][lane]
               for step,lane,_ in extra)
    assert np.max(np.abs(np.linalg.norm(req_b,axis=-1)-np.linalg.norm(raw['request'],axis=-1))) < 1.e-6
    issued = np.asarray([row['v_final_w'] for row in rows])
    return dict(actor=along_fraction(decoded[0],req_b),selected=along_fraction(governed,req_b),
                issued=along_fraction(issued,raw['request']),actual=along_fraction(raw['after'][:,:,7:10],raw['request']))


def sweep(archive, comparison):
    raw = registered.ingest(archive)
    fractions = trace_paths(raw)
    mesh = registered.triangles(archive['geometry'])
    windows, by_layout = [], {}
    for m in archive['metrics']:
        lane = m['lane']
        layout = m['layout']
        before = raw['before'][:,lane]
        request = raw['request'][:,lane]
        mover = raw['mover'][:,lane] if layout == 'street_gap' else None
        radius = m['physics_tick_clearance']['maximum_measured_link_envelope_m']
        nominal,_ = projected_clearance(before[:,:3],before[:,7:10],request,mover,mesh,radius)
        feasible = (np.linalg.norm(request,axis=-1)>.03)&(nominal>=.15)
        for e in comparison['matched_rows']:
            window = e['window']
            if window['lane'] != lane:
                continue
            s,t = window['anchor'],window['end']
            confirmations = dict(feasible=first_confirmation(feasible[s:t]))
            for label,v in fractions.items():
                confirmations[label] = first_confirmation(feasible[s:t]&(v[s:t,lane]>=.60))
            assert confirmations['actual'] == e['candidate']['ticks'], (lane,s,confirmations,e)
            if on_time(confirmations['actual']):
                continue
            if not on_time(confirmations['feasible']):
                stage = 'no_sustained_feasible_request'
            elif on_time(confirmations['issued']):
                stage = 'downstream_of_issued'
            elif on_time(confirmations['selected']):
                stage = 'limited_after_selection'
            elif on_time(confirmations['actor']):
                stage = 'action_selection'
            else:
                stage = 'raw_actor_below_progress'
            stop = min(t,s+registered.RESTORE_TICKS)
            details = dict(stage=stage,window=window,confirmation_ticks=confirmations,
                settings=dict(ics_beta_below_point9=sum(raw['rows'][j]['ics_beta'][lane]<.9 for j in range(s,stop)),
                              escape_selected_differs=sum(raw['rows'][j]['action'][lane]!=raw['rows'][j]['selected_action'][lane] for j in range(s,stop)),
                              feasible_ticks=int(feasible[s:stop].sum())),
                by_step=[dict(tick=j,feasible=bool(feasible[j]),ics_beta=raw['rows'][j]['ics_beta'][lane],
                              **{name:float(values[j,lane]) for name,values in fractions.items()}) for j in range(s,stop)])
            windows.append(details)
            by_layout.setdefault(layout,Counter())[stage] += 1
    assert len(windows) == comparison['arms'][1]['bad']
    return dict(stage_counts=Counter(w['stage'] for w in windows),by_layout=by_layout,windows=windows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    assert not stop_paths(), stop_paths()
    archive = registered.report(registered.CANDIDATE)
    result = registered.read(scope/'RESULT.json')
    comparison_pin = result['paired_scoring']
    paired = registered.read(registered.checked(comparison_pin))['matched_restoration']
    diagnostic = sweep(archive,paired)
    assert not stop_paths(), stop_paths()
    print(save(args.output,dict(method='first five consecutive feasible packets at 60% of latest request; counts checked against frozen v60 recovery result; stage classification is per-window timing, not a causal counterfactual',
                                source=registered.pin(Path(__file__)),paired_scoring=comparison_pin,inputs=archive['pins'],**diagnostic)))
