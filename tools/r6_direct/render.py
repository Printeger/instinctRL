"""Render within the direct worker's admitted app and bounded process lifecycle."""
import faulthandler
import time
from pathlib import Path
from tools.r6_direct.io import atomic


def startup_settings(native_root):
    """Offscreen products do not need the default interactive viewport window."""
    root = Path(native_root)
    return {
        '/exts/omni.kit.viewport.window/startup/disableWindowOnLoad': True,
        # Isaac Lab's camera experience disables the Replicator tick graph for
        # training performance. Explicit orchestrator.step requires that graph.
        '/exts/omni.replicator.core/Orchestrator/enabled': True,
        '/app/tokens/cache': str(root/'render_cache'),
        '/app/tokens/omni_global_cache': str(root/'render_global_cache'),
    }


def validate_cache_tokens(settings, tokens):
    # Verify effective token expansion, not only the requested settings values.
    for name in ('cache', 'omni_global_cache'):
        if tokens.resolve('${'+name+'}') != settings['/app/tokens/'+name]:
            raise RuntimeError('DIRECT_RENDER_CACHE_TOKEN:'+name)


def run(plan, contract, source, output, app, counters, guard, *, clock=time.monotonic):
    from tools.render_r6_isaac_video import render
    started=clock(); last=None; checks=0
    def check():
        nonlocal last, checks
        now=clock()
        if now-started>=contract['budgets']['wall_seconds']:
            raise TimeoutError('DIRECT_RENDER_TIMEOUT')
        if last is None or now-last>=1:
            guard.check(stage='recorded-replay',ordinal=checks)
            last=now; checks+=1
    check()
    # A single bounded Python-stack observation distinguishes a stalled Kit
    # update from Replicator's empty capture-time queue. The native watchdog
    # still owns termination; this does not advance the timeline or physics.
    with (output/'render_python_stack.txt').open('x') as trace:
        faulthandler.dump_traceback_later(30, repeat=False, file=trace)
        try:
            manifest=render(contract['parameters']['extraction'],output/'media',
                            source_snapshot=plan['snapshot'],asset_root=source['execution']['isaacsim_asset_root'],
                            app=app,check=check)
        finally:
            faulthandler.cancel_dump_traceback_later()
    check()
    if any(manifest[k]!=0 for k in ('physics_steps','policy_inference_steps','optimizer_updates','checkpoint_writes')):
        raise RuntimeError('DIRECT_REPLAY_MUST_NOT_TRAIN')
    atomic(output/'status.json',dict(status='PASS_PENDING_CLOSE',counters=counters.snapshot(),scientific_acceptance=False))
    return manifest
