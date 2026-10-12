import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import torch

from instinctlab.evaluation import raycaster_observability as ray
from instinctlab.evaluation.raycaster_observability import RaycasterAcquisitionDiagnostics, RaycasterAcquisitionError
from training_workloads.unified_h1.preflight import validate_initial_observability

class PreflightPersistenceTests(unittest.TestCase):
    def source(self):
        starts = torch.zeros((32, 1, 3), dtype=torch.float32)
        directions = torch.zeros_like(starts); directions[..., 0] = 1
        hits = starts.clone(); hits[..., 0] = 2
        return {"ray_starts_w": starts, "ray_directions_w": directions, "accepted_ray_hits_w": hits,
                "accepted_beam_valid": torch.ones((32, 1), dtype=torch.bool)}

    def test_failure_is_atomic_and_primary_error_escapes(self):
        tensors = torch.ones((32, 1, 3), dtype=torch.float32)
        diagnostic = RaycasterAcquisitionDiagnostics(tensors, torch.ones((32, 1)), tensors,
            torch.tensor(0.0), torch.tensor(1.0), (0, 0), (4, 0))
        primary = RaycasterAcquisitionError("primary-ray-failure", diagnostic)
        with tempfile.TemporaryDirectory() as directory, patch(
            "instinctlab.evaluation.raycaster_observability.acquire_exact_observability_from_source", side_effect=primary):
            output = Path(directory)
            with self.assertRaisesRegex(RaycasterAcquisitionError, "primary-ray-failure"):
                validate_initial_observability(self.source(), output)
            record = json.loads((output / "initial_preflight/raw_acquisition_failure.json").read_text())
            saved = torch.load(output / "initial_preflight/raw_acquisition_failure.pt", weights_only=True)
            self.assertEqual(record["status"], "FAIL"); self.assertEqual(record["forward_reconstruction_failing_beam"], [4, 0])
            self.assertIn("ray_parameters", saved); self.assertIn("raw_surface_normals_w", saved)
            self.assertFalse((output / "initial_preflight/observability.json").exists())

    def test_success_writes_only_pass_record(self):
        source = self.source(); scalar = torch.tensor(0.0)
        acquisition = SimpleNamespace(ray_hits_w=source["accepted_ray_hits_w"], ray_parameters=torch.ones((32,1))*2,
            ray_distances_m=torch.ones((32,1))*2, raw_surface_normals_w=torch.ones((32,1,3)),
            beam_valid=source["accepted_beam_valid"].unsqueeze(0), hit_max_error_m=scalar,
            distance_max_error_m=scalar, reconstruction_max_error_m=scalar,
            observability=SimpleNamespace(numerical_rank=torch.ones((1,32),dtype=torch.long)*3,
                                          sigma_min=torch.ones((1,32))))
        with tempfile.TemporaryDirectory() as directory, patch(
            "instinctlab.evaluation.raycaster_observability.acquire_exact_observability_from_source", return_value=acquisition):
            output=Path(directory); result=validate_initial_observability(source, output)
            self.assertEqual(result["status"], "PASS")
            self.assertFalse((output / "initial_preflight/raw_acquisition_failure.json").exists())

    def test_actual_post_cast_failure_paths_are_persisted(self):
        source = self.source()
        good_parameters = torch.full((32, 1), 2.0, dtype=torch.float32)
        good_normals = torch.ones((32, 1, 3), dtype=torch.float32)
        good_normals[..., 1:] = 0.0
        class Mesh:
            id = 1
            device = "cpu"
        cases = (
            ("nonfinite parameter", source["accepted_ray_hits_w"],
             torch.full((32, 1), float("nan"), dtype=torch.float32), good_normals,
             "parameters must be finite", "nan"),
            ("validity mismatch", torch.full_like(source["accepted_ray_hits_w"], float("inf")),
             good_parameters, good_normals, "hit validity disagrees", "positive_infinity"),
            ("invalid normal", source["accepted_ray_hits_w"], good_parameters,
             torch.full_like(good_normals, float("nan")), "normals must be finite", 0.0),
        )
        for name, returned_hits, parameters, normals, message, reconstruction_max in cases:
            def acquire(_source):
                with patch.object(
                    ray, "_raycast_mesh_with_normals",
                    return_value=(returned_hits, parameters, normals, None),
                ):
                    return ray.acquire_exact_raycaster_observability(
                        source["ray_starts_w"], source["ray_directions_w"], Mesh(), 40.0,
                        source["accepted_ray_hits_w"], source["accepted_beam_valid"],
                    )
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory, patch(
                "instinctlab.evaluation.raycaster_observability.acquire_exact_observability_from_source",
                side_effect=acquire,
            ):
                output = Path(directory)
                with self.assertRaisesRegex(RaycasterAcquisitionError, message):
                    validate_initial_observability(source, output)
                record = json.loads((output / "initial_preflight/raw_acquisition_failure.json").read_text())
                saved = torch.load(output / "initial_preflight/raw_acquisition_failure.pt", weights_only=True)
                self.assertEqual(record["forward_reconstruction_max_error_m"], reconstruction_max)
                self.assertTrue(torch.allclose(saved["ray_parameters"], parameters, equal_nan=True))

if __name__ == "__main__": unittest.main()
