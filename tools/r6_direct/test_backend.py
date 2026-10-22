"""CPU constructor/reset checks for the shared backend's explicit local contract."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from training_workloads.unified_h1.backend import UnifiedH1LiveBackend

ROOT=Path(__file__).resolve().parents[2]


class DirectBackendTests(unittest.TestCase):
    def test_local_contract_is_not_a_legacy_authorization(self):
        contract=json.loads((ROOT/'docs/experiments/r6-direct-training/BASELINE_CONTRACT.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            attempt=Path(tmp); (attempt/'results'/contract['mode']).mkdir(parents=True)
            environment=SimpleNamespace(device='cpu',reset=lambda **kw:({'policy':'CPU fixture'},{}))
            backend=UnifiedH1LiveBackend(run_contract=contract,attempt=attempt,app=None,
                environment=environment,resource_guard=SimpleNamespace())
            self.assertIsNone(backend.authorization)
            self.assertEqual(backend.execution_mode,'local-direct')
            self.assertEqual(backend._preserving_reset(seed=20260909)[0],{'policy':'CPU fixture'})
            self.assertEqual(backend.counters.explicit_resets,1)
            with self.assertRaises(ValueError):
                UnifiedH1LiveBackend(authorization=contract,run_contract=contract,
                    attempt=attempt,app=None,environment=environment,resource_guard=None)


if __name__=='__main__':unittest.main()
