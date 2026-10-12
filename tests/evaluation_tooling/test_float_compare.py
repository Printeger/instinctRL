from __future__ import annotations

import math
import struct
import unittest

from evaluation_tooling.float_compare import compare_floats


def step(value: float, count: int) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0] + count
    return struct.unpack(">f", struct.pack(">I", bits))[0]


class FloatComparatorTests(unittest.TestCase):
    exact_identity = {"device": "cpu", "implementation": "reference", "algorithm": "scalar", "dtype": "float32", "execution_mode": "deterministic"}

    def compare(self, left, right, **kwargs):
        defaults = dict(shape=(1,), dtype="float32", policy_name="FLOAT32_GEOMETRY_ULP2_ABS1E5_V1", reference_backend="cpu", candidate_backend="cuda")
        defaults.update(kwargs)
        return compare_floats(left, right, **defaults)

    def test_zero_ulp(self) -> None:
        self.assertEqual(self.compare([1.0], [1.0]).status, "PASS")

    def test_one_ulp(self) -> None:
        self.assertEqual(self.compare([1.0], [step(1.0, 1)]).status, "PASS")

    def test_two_ulps(self) -> None:
        result = self.compare([1.0], [step(1.0, 2)])
        self.assertEqual((result.status, result.maximum_ulp_distance), ("PASS", 2))

    def test_more_than_two_ulps(self) -> None:
        self.assertEqual(self.compare([1.0], [step(1.0, 3)]).status, "FAIL")

    def test_absolute_bound_boundary(self) -> None:
        # At small magnitudes the ULP arm is stricter, proving both arms apply.
        result = self.compare([0.0], [1e-5])
        self.assertEqual(result.status, "FAIL")
        self.assertLessEqual(result.maximum_absolute_error, 1e-5)

    def test_absolute_bound_failure_even_when_ulp_policy_widened_is_not_possible(self) -> None:
        result = self.compare([128.0], [step(128.0, 2)])
        self.assertEqual(result.status, "FAIL")
        self.assertGreater(result.maximum_absolute_error, 1e-5)

    def test_exact_float_pass(self) -> None:
        result = self.compare([1.0], [1.0], policy_name="EXACT_FLOAT32_V1", reference_backend="cpu", candidate_backend="cpu", reference_identity=self.exact_identity, candidate_identity=self.exact_identity)
        self.assertEqual(result.status, "PASS")

    def test_exact_float_fail(self) -> None:
        result = self.compare([1.0], [step(1.0, 1)], policy_name="EXACT_FLOAT32_V1", reference_backend="cpu", candidate_backend="cpu", reference_identity=self.exact_identity, candidate_identity=self.exact_identity)
        self.assertEqual(result.status, "FAIL")

    def test_cross_backend_exact_rejected(self) -> None:
        result = self.compare([1.0], [1.0], policy_name="EXACT_FLOAT32_V1", reference_identity=self.exact_identity, candidate_identity=self.exact_identity)
        self.assertTrue(any("CROSS_BACKEND" in finding for finding in result.findings))

    def test_nan_rejected(self) -> None:
        self.assertEqual(self.compare([math.nan], [math.nan]).status, "FAIL")

    def test_infinity_rejected(self) -> None:
        self.assertEqual(self.compare([math.inf], [math.inf]).status, "FAIL")

    def test_signed_zero_geometry_equal(self) -> None:
        self.assertEqual(self.compare([-0.0], [0.0]).status, "PASS")

    def test_signed_zero_exact_differs(self) -> None:
        result = self.compare([-0.0], [0.0], policy_name="EXACT_FLOAT32_V1", reference_backend="cpu", candidate_backend="cpu", reference_identity=self.exact_identity, candidate_identity=self.exact_identity)
        self.assertEqual(result.status, "FAIL")

    def test_exact_identity_required(self) -> None:
        result = self.compare([1.0], [1.0], policy_name="EXACT_FLOAT32_V1", reference_backend="cpu", candidate_backend="cpu")
        self.assertEqual(result.status, "FAIL")
        self.assertIn("EXACT_IDENTITY", result.findings[0])

    def test_unrepresentable_float_is_reported(self) -> None:
        result = self.compare([1e300], [1e300])
        self.assertEqual(result.status, "FAIL")
        self.assertEqual(result.maximum_absolute_error, None)

    def test_dtype_mismatch(self) -> None:
        self.assertEqual(self.compare([1.0], [1.0], dtype="float64").status, "FAIL")

    def test_shape_mismatch(self) -> None:
        self.assertEqual(self.compare([1.0], [1.0], shape=(2,)).status, "FAIL")

    def test_deterministic_first_and_worst(self) -> None:
        result = self.compare([1.0, 2.0], [step(1.0, 3), step(2.0, 4)], shape=(1, 2))
        self.assertEqual(result.first_violation, (0, 0))
        self.assertEqual(result.worst_ulp_location, (0, 1))
        self.assertEqual(result.violation_count, 2)
