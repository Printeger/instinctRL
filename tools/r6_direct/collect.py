"""Collect the four local final evaluations and select recorded video sources."""
import argparse
import json
from pathlib import Path
from tools.r6_direct.dependencies import install
from tools.r6_direct.evidence import read, validate_binding, require
from tools.r6_evidence import file_pin, canonical, persist_exact


def collect(batch_pin, analyses):
    install()
    from training_workloads.unified_h1 import robustness_final_selection as selection
    from tools.r6_final_collection import gates
    from tools.r6_final_analysis import identity
    batch = read(batch_pin); protocol = read(batch['protocol'])
    require(batch['schema_version'] == 'r6-local-final-batch-inputs/1' and len(analyses) == 4, 'COMPLETE_FINAL_BATCH')
    rows = []; reports = []; contexts = []; runs = set()
    for index, analysis in enumerate(analyses):
        bound, ctx, report = validate_binding(analysis)
        selected_value = dict(schema_version='r6-local-final-selection/1', batch=batch_pin, assignment_index=index)
        selected = selection.validate(selected_value)
        require(ctx['contract']['operation'] == 'final'
                and ctx['contract']['parameters']['final_selection'] == selected_value
                and report['final'] == identity(selected) and report['final_binding_issues'] == [], 'FINAL_ASSIGNMENT')
        require(ctx['plan']['run_id'] not in runs, 'DISTINCT_FINAL_RUNS')
        runs.add(ctx['plan']['run_id'])
        counters = ctx['state']['counters']
        require(counters['nontraining_transitions'] == ctx['contract']['budgets']['nontraining_transitions']
                and counters['training_transitions'] == counters['real_optimizer_updates'] == counters['real_checkpoint_writes'] == 0,
                'FINAL_COUNTERS')
        rows.append(dict(assignment_index=index, assignment=selected['assignment'], analysis=analysis,
                         report=bound['report'], gates=gates(report, protocol),
                         native_exit_code=bound['native_exit_code'], native_error=bound['native_error']))
        reports.append(report); contexts.append(ctx)
    result = dict(schema_version='r6-direct-final-collection/1', batch=batch_pin, results=rows,
                  required_assignments=[1,2,3],
                  required_models_pass=all(r['gates']['original_pass'] and r['gates']['target_pass'] for r in rows[1:]),
                  scientific_acceptance=False, independent_review_pending=True)
    from tools.r6_video_selection import select
    video = select(result, reports, protocol)
    for clip in video['clips']:
        index = clip['assignment_index']; ctx = contexts[index]; row = rows[index]
        raw = {n: file_pin(ctx['output']/'r6'/clip['block_id']/n)
               for n in ('audit.bin', 'motion_traces.json', 'result.json')}
        indexed = {p['path']: p for p in reports[index]['inputs']}
        require(all(p['path'] in indexed and indexed[p['path']]['sha256'] == p['sha256'] for p in raw.values()), 'VIDEO_RAW_INPUTS')
        saved = read(raw['result.json'])
        require(saved['run_id'] == ctx['plan']['run_id'] and saved['block_id'] == clip['block_id']
                and saved['seed'] == row['assignment']['evaluation_seed'] and saved['split'] == 'final', 'VIDEO_IDENTITY')
        clip.update(assignment=row['assignment'], checkpoint=row['assignment']['checkpoint'], analysis=row['analysis'],
                    report=row['report'], source_snapshot=ctx['plan']['snapshot'], run_id=saved['run_id'],
                    audit_records=saved['audit_records'], raw=raw)
    video.update(schema_version='r6-video-selection/1', status='VIDEO_SOURCES_SELECTED_NOT_RENDERED', batch=batch_pin,
                 total_frames=sum(c['frame_count'] for c in video['clips']), fps=10, dispatch_authorized=False)
    return result, video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    from tools.r6_direct.import_legacy import stop_paths
    if stop_paths():
        raise RuntimeError('STOP')
    result, video = collect(**json.loads(args.request.read_text()))
    if stop_paths():
        raise RuntimeError('STOP')
    destination = args.destination.absolute()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True, exist_ok=False)
    persist_exact(destination/'results.json', canonical(result) + b'\n')
    persist_exact(destination/'video-selection.json', canonical(video) + b'\n')


if __name__ == '__main__':
    main()
