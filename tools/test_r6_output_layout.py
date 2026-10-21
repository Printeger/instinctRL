"""Exercise all registered blocks through the real entry with CPU fixture physics.

Checks output topology, not capacity in bytes or physical/scientific acceptance.
"""
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from tools.test_r6_evaluation import EvaluationTests
from training_workloads.scale_003.persistence import atomic_json_replace


def test_complete_protocol_has_bounded_output_inodes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads((root / 'docs/experiments/r6-robustness-auto/TEST_PROTOCOL.json').read_text())
    assert len(protocol['blocks']) == 16
    # This temporary CPU fixture never changes the real protocol's freeze state.
    protocol['frozen'] = True
    path = tmp_path / 'fixture-protocol.json'
    path.write_text(json.dumps(protocol))
    output = tmp_path / 'output'
    output.mkdir()
    backend = EvaluationTests().setup_backend(output)
    expected = sum(32 * block['steps'] for block in protocol['blocks'])
    backend.authorization = {
        'mode': 'research-r6-evaluation',
        'budgets': {'nontraining_transitions': expected, 'automatic_resets': expected,
                    'explicit_resets': 16},
        'parameters': {'r6': {
            'protocol': {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
            'block_ids': [block['id'] for block in protocol['blocks']],
            'run_id': 'CPU-OUTPUT-TOPOLOGY', 'seed': 17, 'split': 'development',
            'expected_mass_kg': 1.16,
            'output_quotas': {block['id']: {'range_bytes': 8 * 1024**2,
                                           'audit_bytes': 16 * 1024**2}
                              for block in protocol['blocks']},
        }},
    }
    def reset(seed):
        backend.counters.reserve_reset(automatic=False, maximum=16)
        return backend.environment.reset()
    backend._preserving_reset = reset
    source = root / 'training_workloads/unified_h1/backend.py'
    cls = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'UnifiedH1LiveBackend')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'evaluate_r6_blocks')
    namespace = dict(Path=Path, Mapping=Mapping, Any=Any,
                     __package__='training_workloads.unified_h1',
                     sha256_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
                     atomic_json_replace=atomic_json_replace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 str(source), 'exec'), namespace)
    results = namespace['evaluate_r6_blocks'](backend)
    assert len(results) == 16
    assert backend.counters.nontraining_transitions == expected == 160640
    assert backend.counters.explicit_resets == 16
    files = {str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
    wanted = {'r6_block_results.json'} | {
        f"r6/{block['id']}/{name}" for block in protocol['blocks']
        for name in ('audit.bin', 'measurements.bin', 'motion_traces.json', 'result.json')}
    assert files == wanted
    assert sum(p.is_dir() for p in output.rglob('*')) == 17
    assert all(not r['scientific_acceptance'] for r in results)
