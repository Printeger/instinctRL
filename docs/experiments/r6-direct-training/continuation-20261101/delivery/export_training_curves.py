"""Export descriptive curves from complete, analysis-pinned update streams (CPU only)."""
import argparse
import csv
import io
import json
from pathlib import Path

from tools.r6_direct.import_legacy import pin, stop_paths
from training_workloads.unified_h1.lossless_evidence import ChunkReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if stop_paths():
        raise RuntimeError('STOP')
    report = json.loads(args.report.read_text())
    sources = [p for p in report['inputs'] if p['path'].endswith('/updates.jsonl.r6z')]
    assert len(sources) == 2, 'Expected two completed curriculum phases'
    rows = []
    for source in sorted(sources, key=lambda p: p['path']):
        path = Path(source['path'])
        assert pin(path)['sha256'] == source['sha256'], 'Source hash drift'
        logical = path.as_posix().split('/compressed_evidence/', 1)[1][:-4]
        with ChunkReader(path, run_id=args.run_id, logical_name=logical,
                         maximum_raw_bytes=64 * 1024**3) as reader:
            with io.BufferedReader(reader) as stream:
                for line in stream:
                    value = json.loads(line)
                    stat = value['statistics']
                    rows.append(dict(update=value['completed_updates'],
                                     return_target_mean=stat['returns']['signed_mean'],
                                     value_mean=stat['values']['signed_mean'],
                                     advantage_mean=stat['advantages']['signed_mean'],
                                     explained_variance=stat['explained_variance']['value']))
    assert [r['update'] for r in rows] == list(range(641, 769)), 'Incomplete or unordered updates'
    args.destination.mkdir(parents=True, exist_ok=False)
    csv_path = args.destination / 'updates.csv'
    with csv_path.open('x') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for axis, key in zip(axes.flat, list(rows[0])[1:]):
        axis.plot([r['update'] for r in rows], [r[key] for r in rows], linewidth=1)
        axis.axvline(704.5, color='gray', linestyle='--', linewidth=0.8)
        axis.set(xlabel='Absolute PPO update', ylabel=key.replace('_', ' '))
        axis.grid(alpha=0.25)
    figure.suptitle(args.run_id + '\nDescriptive training statistics; no evaluation gain claim')
    figure.savefig(args.destination / 'training_curves.png', dpi=160)
    plt.close(figure)
    manifest = dict(run_id=args.run_id, report=pin(args.report), sources=sources,
                    exporter=pin(__file__), rows=len(rows), update_range=[641, 768],
                    outputs=[pin(csv_path), pin(args.destination / 'training_curves.png')],
                    limitations=['Return target is a PPO rollout target, not episodic reward or evaluation success.',
                                 'Dashed boundary separates the two registered curriculum phases.',
                                 'No smoothing, no learning-gain inference, no replacement of scientific audit.',
                                 'Report issues and native cleanup failure remain unchanged.'])
    (args.destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(run_id=args.run_id, rows=len(rows), destination=str(args.destination))))


if __name__ == '__main__':
    main()
