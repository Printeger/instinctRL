"""Render within the direct worker's admitted app and bounded process lifecycle."""
import time
from tools.r6_direct.io import atomic


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
    manifest=render(contract['parameters']['extraction'],output/'media',
                    source_snapshot=plan['snapshot'],asset_root=source['execution']['isaacsim_asset_root'],
                    app=app,check=check)
    check()
    if any(manifest[k]!=0 for k in ('physics_steps','policy_inference_steps','optimizer_updates','checkpoint_writes')):
        raise RuntimeError('DIRECT_REPLAY_MUST_NOT_TRAIN')
    atomic(output/'status.json',dict(status='PASS_PENDING_CLOSE',counters=counters.snapshot(),scientific_acceptance=False))
    return manifest
