"""Evaluate real fixed-command B0 using the original H1 Z/T/S evaluator.

All 32 lanes, startup samples, reset and safety failures remain included.
This is a three-case control baseline, not a five-case policy checkpoint gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
from training_workloads.unified_h1.evaluation import replay_raw_case, evaluate_episode, case_outcome

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation-raw',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('preserve existing analysis')
    cases = {}
    for case in ('Z','T','S'):
        path = args.evaluation_raw / 'command_passthrough' / (case + '.jsonl')
        traces = replay_raw_case(str(path),case=case)
        if len(traces) != 32 or [t.env_index for t in traces] != list(range(32)):
            raise ValueError('all intended lanes must remain')
        rows = [evaluate_episode(t) for t in traces]
        groups = ['d' + str(i % 8) for i in range(32)]
        expected = ['d' + str(i) for i in range(8)]
        cases[case] = {'raw_path':str(path.resolve()),'raw_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'episodes':rows,'gate':case_outcome([bool(r['success']) for r in rows],groups,expected_subgroups=expected)}
    output = {'schema_version':'instinctrl-r5-command-passthrough-evaluation/1',
              'condition':'B0_COMMAND_PASSTHROUGH','cases':cases,
              'all_original_ZTS_gates_pass':all(c['gate']['pass'] for c in cases.values()),
              'thresholds':'unchanged metric-h1-v1 evaluate_episode/case_outcome',
              'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'training_transitions':0,'optimizer_updates':0,'training_checkpoint_writes':0}
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k: v['gate'] for k, v in cases.items()}))

if __name__ == '__main__':
    main()
