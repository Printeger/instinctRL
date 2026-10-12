from __future__ import annotations
import importlib.util
from pathlib import Path
import sys, types, unittest

ROOT = Path(__file__).resolve().parents[1]
config = types.ModuleType("training_workloads.scale_003.config")
config.load_config = lambda: {}
config.sha256_file = lambda path: ""
persistence = types.ModuleType("training_workloads.scale_003.persistence")
persistence.atomic_json_replace = lambda *args, **kwargs: None
sys.modules.setdefault("training_workloads", types.ModuleType("training_workloads"))
sys.modules.setdefault("training_workloads.scale_003", types.ModuleType("training_workloads.scale_003"))
sys.modules[config.__name__] = config
sys.modules[persistence.__name__] = persistence
spec = importlib.util.spec_from_file_location("training_workloads.scale_003.resources", ROOT / "candidate/training_workloads/scale_003/resources.py")
resources = importlib.util.module_from_spec(spec); sys.modules[spec.name] = resources; spec.loader.exec_module(resources)

def sample(stamp):
    return {"monotonic_ns": stamp, "gpu": {"index": 0, "name": "NVIDIA GeForce RTX 5070", "driver": "580.173.02", "memory_total_mib": 12227, "memory_used_mib": 580, "free_memory_mib": 11647, "utilization_percent": 0}, "compute_apps": []}

class TestWindow(unittest.TestCase):
    def test_slow_first_provider_cannot_shorten_window(self):
        now=0.0; latencies=iter((3.0,1.0,1.0))
        def clock(): return now
        def sleep(seconds):
            nonlocal now; now += seconds
        def provider():
            nonlocal now; now += next(latencies); return sample(int(now*1_000_000_000))
        window=resources.collect_coexistence_window(provider,sleep,clock)
        self.assertEqual(window["span_ns"],11_000_000_000)
        self.assertEqual(resources.validate_coexistence_window(window)["status"],"PASS")

    def test_threshold_is_unchanged(self):
        window={"sample_count":3,"span_ns":9_999_999_999,"samples":[sample(1),sample(5_000_000_000),sample(10_000_000_000)]}
        with self.assertRaisesRegex(ValueError,"at least ten seconds"):
            resources.validate_coexistence_window(window)

if __name__ == "__main__": unittest.main()
