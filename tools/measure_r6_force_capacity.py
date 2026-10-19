"""CPU-only actual force-sink serialization sizing; no simulated trajectories."""
from pathlib import Path
import hashlib
import json
import tempfile
import time
import torch
from tools.test_r6_physics_adapter import Env
from training_workloads.unified_h1.robustness_audit import AuditLog, read_audit
from training_workloads.unified_h1.logging import _jsonable

ROOT = Path(__file__).resolve().parents[1]


def run():
    generator = torch.Generator(device='cpu').manual_seed(2026102002)
    cases = []
    with tempfile.TemporaryDirectory(prefix='r6-force-capacity-') as directory:
        for amplitude in (0., .5, 1.):
            env = Env()
            env._r6_physics['block']['pulse_acceleration_m_s2'] = amplitude
            env.reset()
            path = Path(directory) / f'{amplitude}.bin'
            rows = []
            sizes = []
            seconds = []
            with AuditLog(path, run_id='CPU-FORCE-CAPACITY', block_id='synthetic', maximum_bytes=1024**2) as log:
                # Sample both pulse endpoints and withdrawal. Independent finite
                # float32 wrenches avoid the tiny all-equal tensors in unit tests.
                for tick in (0, 999, 1000, 1039, 1040, 6599):
                    q = torch.randn(32, 5, 4, generator=generator)
                    q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
                    env._robot.data.body_link_state_w[:, :, 3:7] = q
                    env._wrench.forces_body_n.copy_(torch.randn(32, 5, 3, generator=generator))
                    env._wrench.torques_body_nm.copy_(torch.randn(32, 5, 3, generator=generator) * .03)
                    env._r6_physics['ticks'] = tick
                    env._r6_physics['last_sim_tick'] = None
                    env.sim.current_time_step_index = tick + 10000
                    env.tick()
                    row = env.force_rows.pop()
                    rows.append(row)
                    before = log.written
                    start = time.monotonic()
                    log.append('force', row)
                    seconds.append(time.monotonic() - start)
                    sizes.append(log.written - before)
            decoded = list(read_audit(path, run_id='CPU-FORCE-CAPACITY', block_id='synthetic', expected_records=len(rows)))
            assert [r['payload'] for r in decoded] == [_jsonable(r) for r in rows]
            assert sum(sizes) == path.stat().st_size
            cases.append(dict(acceleration_m_s2=amplitude, ticks=[r['block_tick'] for r in rows],
                active=[r['active'] for r in rows], record_bytes=sizes,
                encode_write_fsync_seconds=seconds, exact_json_roundtrip=True))
    protocol_path = ROOT / 'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    blocks = []
    for block in protocol['blocks']:
        amplitude = block.get('pulse_acceleration_m_s2', 0.)
        case = next(c for c in cases if c['acceleration_m_s2'] == amplitude)
        active_ticks = 40 if block['task'] == 'pulse_hold' and amplitude else 0
        active_bytes = max((n for n, a in zip(case['record_bytes'], case['active']) if a), default=0)
        inactive_bytes = max(n for n, a in zip(case['record_bytes'], case['active']) if not a)
        ticks = block['steps'] * 20
        blocks.append(dict(block_id=block['id'], physics_ticks=ticks, active_ticks=active_ticks,
            estimated_force_bytes=active_ticks * active_bytes + (ticks-active_ticks) * inactive_bytes))
    total = sum(b['estimated_force_bytes'] for b in blocks)
    paths = ['tools/measure_r6_force_capacity.py', 'tools/test_r6_physics_adapter.py',
        'isaaclab-training/instinctlab/observations/robustness_physics.py',
        'training_workloads/unified_h1/robustness_audit.py', 'training_workloads/unified_h1/logging.py']
    def pin(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(status='CPU_FIXTURE_NOT_CAPACITY_ACCEPTANCE', capacity_pass=False,
        seed=2026102002, cases=cases, blocks=blocks,
        per_model_force_bytes=total, final_four_force_bytes=4*total,
        protocol=pin(protocol_path), code=[pin(ROOT/p) for p in paths],
        limitations='Synthetic independent float32 wrenches and body attitudes through actual physics mixin sink and AuditLog; fixture mass1.16kg, not baseline mass. No physics integration. Sample maxima are not bounds. Excludes reset, policy-step, motion, measurement, native, training, snapshot and video outputs. Timings include local fsync but do not predict live GPU/CPU synchronization cost.',
        gpu_queries=0, simulator_initializations=0)


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
