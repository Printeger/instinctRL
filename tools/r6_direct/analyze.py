"""CPU analysis of a closed direct run; writes a new, separate result directory."""
import argparse
import json
from pathlib import Path
from tools.r6_direct.dependencies import install
from tools.r6_direct.evidence import context, add_inputs, bind
from tools.r6_direct.import_legacy import pin, stop_paths
from tools.r6_direct.io import atomic


def analyze(input_pin, *, probability_diagnostic=None):
    install()
    ctx = context(input_pin)
    operation = ctx['contract']['operation']
    if probability_diagnostic is not None and operation!='train':
        raise ValueError('PROBABILITY_DIAGNOSTIC_REQUIRES_TRAIN')
    if operation == 'render':
        from tools.r6_video_presentation import verify
        manifest=pin(ctx['output']/'media/RUN_MANIFEST.json')
        source = json.loads(Path(ctx['plan']['source']['path']).read_text())
        from tools.r6_evidence import checked
        checked(ctx['plan']['source'])
        renderer = dict(path=str(Path(ctx['plan']['snapshot'])/'tools/render_r6_isaac_video.py'),
                        sha256=source['sources']['tools/render_r6_isaac_video.py'])
        summary=verify(manifest,ctx['contract']['parameters']['extraction'], renderer_pin=renderer)
        report=dict(schema_version='r6-direct-video-analysis/1',summary=summary,scientific_acceptance=False,
                    inputs=[dict(manifest,bytes=Path(manifest['path']).stat().st_size)])
        add_inputs(report,input_pin,ctx)
    elif operation == 'baseline':
        from training_workloads.unified_h1.robustness_analysis import analyze as replay
        request = ctx['contract']['parameters']['r6']
        report, summary = replay(ctx['output'], Path(request['protocol']['path']),
                                 run_id=request['run_id'], seed=request['seed'], split=request['split'])
        add_inputs(report, input_pin, ctx)
    else:
        from tools import r6_training_analysis, r6_candidate_analysis, r6_final_analysis
        module = {'train': r6_training_analysis, 'evaluate': r6_candidate_analysis,
                  'final': r6_final_analysis}[operation]
        extra=dict(probability_diagnostic=probability_diagnostic) if operation=='train' else {}
        report, summary = module.analyze(ctx['output'], input_pin, direct=True,**extra)
    return report, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--probability-diagnostic',type=Path,
                        help='Sealed evidence envelope for the exact closed CUDA diagnostic')
    args = parser.parse_args()
    install()
    if stop_paths():
        raise RuntimeError('STOP')
    input_pin = pin(args.inputs)
    ctx = context(input_pin)
    destination = args.destination.absolute()
    originals = (Path(ctx['plan']['attempt_dir']), Path(ctx['plan']['output_root']))
    if (destination.resolve() != destination or destination.exists()
            or any(destination == p or p in destination.parents or destination in p.parents for p in originals)):
        parser.error('new canonical destination separate from originals required')
    report, summary = analyze(input_pin,probability_diagnostic=None if args.probability_diagnostic is None else pin(args.probability_diagnostic))
    if len(json.dumps(report, allow_nan=False).encode()) > 16*1024**2 or len(json.dumps(summary, allow_nan=False).encode()) > 48*1024:
        raise ValueError('DIRECT_ANALYSIS_SIZE_BOUND')
    if stop_paths():
        raise RuntimeError('STOP')
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination/'report.json', report)
    atomic(destination/'summary.json', summary)
    atomic(destination/'binding.json', bind(input_pin, pin(destination/'report.json')))
    print(destination/'summary.json')


if __name__ == '__main__':
    main()
