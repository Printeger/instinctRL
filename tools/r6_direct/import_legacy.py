"""Read-only legacy budget import. Never releases historical reservations."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

from tools.r6_budget import charged, totals

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'docs/experiments/r6-robustness-auto'
NEW = ROOT / 'docs/experiments/r6-direct-training'


def pin(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def stop_paths():
    paths = [ROOT / p for p in ('STOP', 'coordination/STOP', 'coordination/research/STOP',
             'coordination/commander/STOP', 'coordination/executor/STOP')]
    paths += [OLD/'STOP', OLD/'batch/STOP', NEW/'STOP', ROOT.parent/'STOP']
    paths += [ROOT.parent/'instinctRL_coordination'/p for p in ('STOP','commander/STOP','executor/STOP')]
    return [str(p) for p in paths if p.exists()]


def process_identity(pid):
    try:
        raw = Path(f'/proc/{int(pid)}/stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        if fields[0] == 'Z': return None
        return dict(pid=int(pid), starttime=fields[19],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def build(state):
    budget = state['budget']
    total = totals(budget)
    mapping = []
    for i, row in enumerate(budget['reservations']):
        mapping.append(dict(source_pointer=f'/budget/reservations/{i}',
                            event_key=row['event_key'], purpose=row['purpose'],
                            original=row, charged=charged(row, budget['caps'])))
    return dict(schema_version='r6-direct-legacy-import/1',
                policy='used + sum(max(reservation, each observation)); no refunds',
                caps=budget['caps'], used=budget['used'], reservation_mapping=mapping,
                historical_charged=total,
                protected_confirmation_delivery_repair=budget['protected_confirmation_delivery_repair'],
                protected_model_sessions=budget['protected_model_sessions'],
                remaining_before_new_work={k:budget['caps'][k]-v for k,v in total.items()})


def exclusive_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--write', action='store_true')
    args=parser.parse_args()
    stops=stop_paths()
    if stops: raise RuntimeError(f'STOP: {stops}')
    # The archived controller uses this same flock. Do not truncate its file.
    with (OLD/'batch/writer.lock').open('r') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source=OLD/'batch/state.json'; before=pin(source)
        state=json.loads(source.read_text())
        if state['current_job'] is not None or state['current_model_event'] is not None:
            raise RuntimeError('Legacy job/model requires reconciliation')
        previous=state.get('controller_identity')
        observed=process_identity(previous['pid']) if previous else None
        if observed and all(observed[k]==previous[k] for k in observed):
            raise RuntimeError('Legacy controller still live')
        result=build(state)
        result['sources']=[before, pin(OLD/'BUDGET_INCREASE_USER_AUTHORITY_20261022.json'),
            pin(ROOT/'docs/goals/InstinctRL_R6_Standalone_Training_Requirements.md'),
            pin(NEW/'USER_AUTHORITY.txt')]
        result['legacy_process_check']=dict(recorded=previous, observed=observed,
            current_job=state['current_job'], current_model_event=state['current_model_event'],
            writer_lock_acquired=True, stops=stops)
        if pin(source)!=before: raise RuntimeError('Legacy ledger changed during import')
        if args.write:
            exclusive_json(NEW/'LEGACY_BUDGET_IMPORT.json', result)
            exclusive_json(NEW/'RESOURCE_LEDGER.json', dict(schema_version='r6-direct-ledger/1',
                legacy_import=pin(NEW/'LEGACY_BUDGET_IMPORT.json'), caps=result['caps'],
                historical_charged=result['historical_charged'],
                protected_confirmation_delivery_repair=result['protected_confirmation_delivery_repair'],
                protected_model_sessions=result['protected_model_sessions'], new_used={k:0 for k in result['caps']},
                new_reservations=[], status='PREPARING_NO_LIVE_RUN'))
        print(json.dumps(result['remaining_before_new_work'],indent=2))


if __name__=='__main__': main()
