import ctypes
import math
import struct
import unittest
from unittest.mock import patch
import torch

from instinctlab.evaluation import raycaster_observability as ray

class Mesh:
    id = 1
    device = "cpu"

class RaycasterDiagnosticsTests(unittest.TestCase):
    @staticmethod
    def f32(value):
        return struct.unpack("f", struct.pack("f", value))[0]

    def inputs(self):
        starts = torch.tensor([[[300.0, 0.0, 0.0]]], dtype=torch.float32)
        directions = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        accepted = torch.tensor([[[320.0, 0.0, 0.0]]], dtype=torch.float32)
        valid = torch.tensor([[True]])
        normals = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        return starts, directions, accepted, valid, normals

    def test_large_world_wrong_parameter_reports_actual_values(self):
        starts, directions, accepted, valid, normals = self.inputs()
        parameters = torch.tensor([[21.0]], dtype=torch.float32)
        with patch.object(ray, "_raycast_mesh_with_normals", return_value=(accepted.clone(), parameters, normals, None)):
            with self.assertRaises(ray.RaycasterAcquisitionError) as caught:
                ray.acquire_exact_raycaster_observability(starts, directions, Mesh(), 40.0, accepted, valid)
        diagnostic = caught.exception.diagnostics
        self.assertTrue(torch.equal(diagnostic.ray_parameters, parameters))
        self.assertEqual(diagnostic.reconstruction_failing_beam, (0, 0))
        self.assertGreater(float(diagnostic.reconstruction_max_error_m), 1e-5)

    def test_true_fused_reference_differs_from_multiply_then_add(self):
        f32 = self.f32
        start_x = f32(280.94110107421875)
        direction_x = f32(0.23474282026290894)
        parameter = f32(20.336639404296875)
        direction_y = f32(math.sqrt(1.0 - direction_x * direction_x))
        libm = ctypes.CDLL("libm.so.6")
        libm.fmaf.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
        libm.fmaf.restype = ctypes.c_float
        fused_x = libm.fmaf(parameter, direction_x, start_x)
        fused_y = libm.fmaf(parameter, direction_y, 0.0)
        starts = torch.tensor([[[start_x, 0.0, 0.0]]], dtype=torch.float32)
        directions = torch.tensor([[[direction_x, direction_y, 0.0]]], dtype=torch.float32)
        parameters = torch.tensor([[parameter]], dtype=torch.float32)
        returned = torch.tensor([[[fused_x, fused_y, 0.0]]], dtype=torch.float32)
        valid = torch.tensor([[True]])
        normals = torch.tensor([[[1.0, 0.0, 0.0]]], dtype=torch.float32)
        unfused = starts + parameters.unsqueeze(-1) * directions
        self.assertGreater(float(torch.linalg.vector_norm(unfused - returned)), 1e-5)
        with patch.object(ray, "_raycast_mesh_with_normals", return_value=(returned, parameters, normals, None)):
            with self.assertRaises(ray.RaycasterAcquisitionError) as caught:
                ray.acquire_exact_raycaster_observability(
                    starts, directions, Mesh(), 40.0, returned.clone(), valid
                )
        self.assertEqual(caught.exception.diagnostics.reconstruction_failing_beam, (0, 0))
        self.assertGreater(float(caught.exception.diagnostics.reconstruction_max_error_m), 1e-5)

    def test_genuine_wrong_hit_control_is_distinct(self):
        starts, directions, accepted, valid, normals = self.inputs()
        returned = torch.tensor([[[321.0, 0.0, 0.0]]], dtype=torch.float32)
        parameters = torch.tensor([[21.0]], dtype=torch.float32)
        with patch.object(ray, "_raycast_mesh_with_normals", return_value=(returned, parameters, normals, None)):
            with self.assertRaisesRegex(ray.RaycasterAcquisitionError, "returned hits disagree") as caught:
                ray.acquire_exact_raycaster_observability(starts, directions, Mesh(), 40.0, accepted, valid)
        self.assertEqual(caught.exception.diagnostics.hit_failing_beam, (0, 0))
        self.assertTrue(torch.equal(caught.exception.diagnostics.ray_hits_w, returned))

    def test_every_structurally_valid_post_cast_failure_carries_diagnostics(self):
        starts, directions, accepted, valid, normals = self.inputs()
        parameters = torch.tensor([[20.0]], dtype=torch.float32)
        cases = (
            ("hit validity", torch.full_like(accepted, math.inf), parameters, normals,
             "hit validity disagrees"),
            ("nonfinite parameter", accepted, torch.tensor([[math.nan]], dtype=torch.float32), normals,
             "parameters must be finite"),
            ("negative parameter", accepted, torch.tensor([[-1.0]], dtype=torch.float32), normals,
             "parameters must be nonnegative"),
            ("nonfinite normal", accepted, parameters, torch.full_like(normals, math.nan),
             "normals must be finite"),
            ("nonunit normal", accepted, parameters, normals * 2.0,
             "normals must be unit length"),
        )
        for name, returned, returned_parameters, returned_normals, message in cases:
            with self.subTest(name=name), patch.object(
                ray, "_raycast_mesh_with_normals",
                return_value=(returned, returned_parameters, returned_normals, None),
            ):
                with self.assertRaisesRegex(ray.RaycasterAcquisitionError, message) as caught:
                    ray.acquire_exact_raycaster_observability(
                        starts, directions, Mesh(), 40.0, accepted, valid,
                    )
                self.assertTrue(torch.allclose(
                    caught.exception.diagnostics.ray_parameters, returned_parameters, equal_nan=True,
                ))

    def test_invalid_beam_miss_conventions_carry_diagnostics(self):
        starts = torch.tensor([[[300.0, 0.0, 0.0], [300.0, 0.0, 0.0]]], dtype=torch.float32)
        directions = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=torch.float32)
        accepted = torch.tensor([[[320.0, 0.0, 0.0], [math.inf, math.inf, math.inf]]], dtype=torch.float32)
        valid = torch.tensor([[True, False]])
        returned_normals = torch.tensor([[[1.0, 0.0, 0.0], [math.inf, math.inf, math.inf]]], dtype=torch.float32)
        for name, parameters, normals, message in (
            ("parameter miss", torch.tensor([[20.0, 0.0]], dtype=torch.float32), returned_normals,
             "invalid ray parameters"),
            ("normal miss", torch.tensor([[20.0, math.inf]], dtype=torch.float32),
             torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], dtype=torch.float32),
             "invalid normals"),
        ):
            with self.subTest(name=name), patch.object(
                ray, "_raycast_mesh_with_normals", return_value=(accepted, parameters, normals, None),
            ):
                with self.assertRaisesRegex(ray.RaycasterAcquisitionError, message):
                    ray.acquire_exact_raycaster_observability(
                        starts, directions, Mesh(), 40.0, accepted, valid,
                    )

if __name__ == "__main__": unittest.main()
