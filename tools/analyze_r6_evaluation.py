"""CPU-only replay into a new directory, preserving all runtime originals."""
import argparse
from pathlib import Path
from rlflow.store import atomic
from training_workloads.unified_h1.robustness_analysis import analyze, encoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='exact research-r6-evaluation output directory')
    parser.add_argument('protocol', type=Path)
    parser.add_argument('destination', type=Path, help='new analysis directory outside runtime originals')
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--split', required=True, choices=['development', 'final'])
    args = parser.parse_args()
    output, destination = args.output.resolve(), args.destination.resolve()
    if destination == output or output in destination.parents or destination in output.parents:
        parser.error('analysis destination must be separate from runtime originals')
    report, summary = analyze(output, args.protocol, run_id=args.run_id, seed=args.seed, split=args.split)
    data = encoded(report)
    if len(data) > 16*1024**2:
        raise ValueError('REPORT_SIZE_BOUND')
    destination.mkdir(parents=True, exist_ok=False)
    summary_data=encoded(summary)
    if len(summary_data)>48*1024:raise ValueError('SUMMARY_SIZE_BOUND')
    atomic(destination/'report.json',data)
    atomic(destination/'summary.json',summary_data)
    print(str(destination/'summary.json'))


if __name__ == '__main__':
    main()
