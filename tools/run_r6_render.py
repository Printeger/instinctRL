"""Bounded R6 replay entry; exact Store dispatch integration is still required.

Uses the existing native capture/volume, prelaunch gate and owner-thread cleanup.
It never imports a policy, trainer or environment and preserves failed exits.
"""
import argparse
import json
from pathlib import Path
import signal
import time

from tools import r6_render_contract as contract
from tools.r6_recovery import file_pin
from training_workloads.scale_003.persistence import atomic_json_replace, install_output_write_quota
from training_workloads.unified_h1.live_adapter import _process_identity, _bounded_owner_call


def run(auth, attempt, config, *, entered=None):
    entered = time.monotonic() if entered is None else entered
    attempt = Path(attempt)
    p = contract.parameters(auth['parameters'], attempt=attempt)
    contract.require(auth['mode'] == contract.MODE, 'MODE')
    contract.exact_budget(auth['budgets'], p['wall_seconds'])
    output = attempt/'results'/contract.MODE
    contract.require((attempt/'results').is_dir() and not output.exists()
                     and not list((attempt/'results').glob('*/process_start.json')), 'FRESH_ATTEMPT')
    output.mkdir()
    install_output_write_quota(output, contract.PAYLOAD)
    counters = dict(live_process_starts=1, simulator_initializations=0, initialization_physics_ticks=0,
        training_transitions=0, nontraining_transitions=0, real_optimizer_updates=0, real_checkpoint_writes=0,
        explicit_resets=0, automatic_resets=0, attempted_policy_steps=0, completed_policy_steps=0,
        real_environment_transitions=0)
    atomic_json_replace(output, 'process_identity.json', _process_identity())
    atomic_json_replace(output, 'process_start.json', dict(mode=contract.MODE,
        authorization_sha256=auth['authorization_sha256'], budgets=auth['budgets'],
        counter_semantics='Conservative attempts; initialization charged before the gated launch can have effects.'))
    owners = {}; error = None; cleanup_error = None; rendered = None; resource_guard = None
    checks = 0; last_resource_check = -1.0
    base = Path(p['master_authority']['path']).parent; root = base.parents[2]
    stops = (base/'STOP', base/'batch/STOP', root/'STOP', root/'coordination/STOP',
             root/'coordination/commander/STOP', root/'coordination/executor/STOP')
    def check():
        nonlocal checks, last_resource_check
        now = time.monotonic()
        contract.require(not any(path.exists() for path in stops), 'STOP')
        contract.require(0 <= now-entered < p['wall_seconds'], 'DEADLINE')
        if resource_guard is not None and now-last_resource_check >= 1:
            resource_guard.check(stage='recorded-replay', ordinal=checks)
            checks += 1; last_resource_check = now
    def interrupted(*_):
        raise RuntimeError('R6_RENDER_INTERRUPTED')
    prior = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        check()
        counters['simulator_initializations'] = 1
        contract.launch(auth, attempt, config, owners)
        from training_workloads.scale_003.resources import RuntimeResourceGuard
        resource_guard = RuntimeResourceGuard(output)
        check()
        from tools.render_r6_isaac_video import render
        rendered = render(p['extraction'], output/'media', source_snapshot=p['source_snapshot'],
            asset_root=config['execution']['isaacsim_asset_root'], app=owners['app'], check=check)
        check()
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
    finally:
        # Preserve original failure before entering native shutdown.
        try:
            atomic_json_replace(output, 'status_before_cleanup.json', dict(status='FAIL' if error else 'RENDERED',
                error=error, counters=counters, manifest=None if rendered is None else file_pin(output/'media/RUN_MANIFEST.json')))
        except BaseException as exc:
            error = error or dict(type=type(exc).__name__, message=str(exc))
        try:
            if owners.get('app') is not None:
                _bounded_owner_call(owners['app'].close, 9.5, 'r6-render-app.close',
                                    hard_timeout_marker=output/'cleanup_hard_timeout.json')
        except BaseException as exc:
            cleanup_error = dict(type=type(exc).__name__, message=str(exc))
        finally:
            for sig, handler in prior.items(): signal.signal(sig, handler)
    status = dict(status='PASS' if error is None and cleanup_error is None else 'FAIL',
                  error=error, cleanup_error=cleanup_error, counters=counters,
                  elapsed_seconds=time.monotonic()-entered, scientific_acceptance=False)
    atomic_json_replace(output, 'status.json', status)
    return status


def main():
    entered = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--authorization', type=Path, required=True)
    parser.add_argument('--attempt', type=Path, required=True)
    args = parser.parse_args()
    from training_workloads.unified_h1.config import load_config
    config = load_config()
    auth = contract.authorization(file_pin(args.authorization), args.attempt, config)
    result = run(auth, args.attempt, config, entered=entered)
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
