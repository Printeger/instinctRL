"""Deterministic exact and combined ULP-plus-absolute floating comparison."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from typing import Iterable, Sequence


@dataclass(frozen=True)
class FloatPolicy:
    name: str
    dtype: str
    max_ulps: int
    max_absolute: float
    exact: bool
    cross_backend: bool
    signed_zero_equal: bool


FLOAT_POLICIES = {
    "EXACT_FLOAT32_V1": FloatPolicy("EXACT_FLOAT32_V1", "float32", 0, 0.0, True, False, False),
    "EXACT_FLOAT64_V1": FloatPolicy("EXACT_FLOAT64_V1", "float64", 0, 0.0, True, False, False),
    "FLOAT32_GEOMETRY_ULP2_ABS1E5_V1": FloatPolicy(
        "FLOAT32_GEOMETRY_ULP2_ABS1E5_V1", "float32", 2, 1e-5, False, True, True
    ),
}


@dataclass(frozen=True)
class FloatComparison:
    status: str
    policy: str
    dtype: str
    shape: tuple[int, ...]
    maximum_absolute_error: float | None
    maximum_ulp_distance: int | None
    first_violation: tuple[int, ...] | None
    worst_absolute_location: tuple[int, ...] | None
    worst_ulp_location: tuple[int, ...] | None
    violation_count: int
    findings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "policy": self.policy,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_ulp_distance": self.maximum_ulp_distance,
            "first_violation": list(self.first_violation) if self.first_violation is not None else None,
            "worst_absolute_location": list(self.worst_absolute_location) if self.worst_absolute_location is not None else None,
            "worst_ulp_location": list(self.worst_ulp_location) if self.worst_ulp_location is not None else None,
            "violation_count": self.violation_count,
            "findings": list(self.findings),
        }


def quantize(value: float, dtype: str) -> float:
    if dtype == "float32":
        return struct.unpack(">f", struct.pack(">f", value))[0]
    if dtype == "float64":
        return struct.unpack(">d", struct.pack(">d", value))[0]
    raise ValueError(f"unsupported dtype: {dtype}")


def float_bits(value: float, dtype: str) -> int:
    if dtype == "float32":
        return struct.unpack(">I", struct.pack(">f", value))[0]
    if dtype == "float64":
        return struct.unpack(">Q", struct.pack(">d", value))[0]
    raise ValueError(f"unsupported dtype: {dtype}")


def _ordered_bits(value: float, dtype: str) -> int:
    bits = float_bits(value, dtype)
    width = 32 if dtype == "float32" else 64
    sign = 1 << (width - 1)
    mask = (1 << width) - 1
    return ((~bits) & mask) if bits & sign else (bits | sign)


def ulp_distance(left: float, right: float, dtype: str, *, signed_zero_equal: bool) -> int:
    left = quantize(left, dtype)
    right = quantize(right, dtype)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("ULP distance requires finite values")
    if signed_zero_equal and left == 0.0 and right == 0.0:
        return 0
    return abs(_ordered_bits(left, dtype) - _ordered_bits(right, dtype))


def _size(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 0:
            raise ValueError("shape dimensions must be nonnegative integers")
        result *= dimension
    return result


def _location(index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    result = [0] * len(shape)
    for position in range(len(shape) - 1, -1, -1):
        dimension = shape[position]
        if dimension == 0:
            return tuple(result)
        result[position] = index % dimension
        index //= dimension
    return tuple(result)


def compare_floats(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    shape: tuple[int, ...],
    dtype: str,
    policy_name: str,
    reference_backend: str,
    candidate_backend: str,
    reference_identity: dict[str, str] | None = None,
    candidate_identity: dict[str, str] | None = None,
) -> FloatComparison:
    policy = FLOAT_POLICIES.get(policy_name)
    if policy is None:
        raise ValueError(f"unknown float policy: {policy_name}")
    findings: list[str] = []
    if dtype != policy.dtype:
        findings.append(f"DTYPE_MISMATCH: policy requires {policy.dtype}, received {dtype}")
    expected_size = _size(shape)
    if len(reference) != expected_size or len(candidate) != expected_size:
        findings.append(
            f"SHAPE_MISMATCH: shape {shape} requires {expected_size} values, received {len(reference)}/{len(candidate)}"
        )
    if policy.exact:
        identity_fields = {"device", "implementation", "algorithm", "dtype", "execution_mode"}
        if not isinstance(reference_identity, dict) or not isinstance(candidate_identity, dict):
            findings.append("EXACT_IDENTITY: exact policy requires both complete execution identities")
        elif set(reference_identity) != identity_fields or set(candidate_identity) != identity_fields:
            findings.append("EXACT_IDENTITY: identity fields must be device/implementation/algorithm/dtype/execution_mode")
        elif reference_identity != candidate_identity or reference_identity["dtype"] != dtype:
            findings.append("EXACT_IDENTITY: exact policy identities must match the comparison dtype and each other")
    if reference_backend != candidate_backend and not policy.cross_backend:
        findings.append("CROSS_BACKEND_POLICY: exact policy cannot compare different backends")
    if findings:
        return FloatComparison("FAIL", policy_name, dtype, shape, None, None, None, None, None, 0, tuple(findings))
    max_absolute = -1.0
    max_ulp = -1
    max_absolute_index = 0
    max_ulp_index = 0
    first_violation: tuple[int, ...] | None = None
    violations = 0
    finite_values = 0
    for index, (raw_reference, raw_candidate) in enumerate(zip(reference, candidate)):
        location = _location(index, shape)
        try:
            reference_value = quantize(float(raw_reference), dtype)
            candidate_value = quantize(float(raw_candidate), dtype)
        except (OverflowError, TypeError, ValueError):
            violations += 1
            if first_violation is None:
                first_violation = location
            findings.append(f"UNREPRESENTABLE at {location}")
            continue
        if not (math.isfinite(reference_value) and math.isfinite(candidate_value)):
            violations += 1
            if first_violation is None:
                first_violation = location
            findings.append(f"NONFINITE at {location}")
            continue
        finite_values += 1
        absolute = abs(candidate_value - reference_value)
        ulps = ulp_distance(reference_value, candidate_value, dtype, signed_zero_equal=policy.signed_zero_equal)
        if absolute > max_absolute:
            max_absolute = absolute
            max_absolute_index = index
        if ulps > max_ulp:
            max_ulp = ulps
            max_ulp_index = index
        if policy.exact:
            violated = float_bits(reference_value, dtype) != float_bits(candidate_value, dtype)
        else:
            violated = ulps > policy.max_ulps or absolute > policy.max_absolute
        if violated:
            violations += 1
            if first_violation is None:
                first_violation = location
    return FloatComparison(
        status="PASS" if violations == 0 else "FAIL",
        policy=policy_name,
        dtype=dtype,
        shape=shape,
        maximum_absolute_error=max_absolute if finite_values else None,
        maximum_ulp_distance=max_ulp if finite_values else None,
        first_violation=first_violation,
        worst_absolute_location=_location(max_absolute_index, shape) if finite_values else None,
        worst_ulp_location=_location(max_ulp_index, shape) if finite_values else None,
        violation_count=violations,
        findings=tuple(findings),
    )
