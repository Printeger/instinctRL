"""Stable collect-all findings and protocol-defined runtime fail-fast errors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


@dataclass(frozen=True, order=True)
class Finding:
    """One deterministic cheap-validation finding."""

    code: str
    location: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


class FindingCollector:
    """Collect independent cheap defects and return them in stable order."""

    def __init__(self) -> None:
        self._findings: list[Finding] = []

    def add(self, code: str, location: str, message: str) -> None:
        self._findings.append(Finding(code=code, location=location, message=message))

    def extend(self, findings: Iterable[Finding]) -> None:
        self._findings.extend(findings)

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(sorted(set(self._findings)))

    @property
    def ok(self) -> bool:
        return not self._findings

    def as_dicts(self) -> list[dict[str, str]]:
        return [finding.as_dict() for finding in self.findings]


class RuntimeFailureClass(str, Enum):
    PROCESS_CRASH = "PROCESS_CRASH"
    TIMEOUT = "TIMEOUT"
    FORBIDDEN_NONFINITE = "FORBIDDEN_NONFINITE"
    ACTOR_OPTIMIZER_LEAKAGE = "ACTOR_OPTIMIZER_LEAKAGE"
    PRODUCT_SEMANTIC_FAILURE = "PRODUCT_SEMANTIC_FAILURE"
    UNIT_FRAME_SHAPE_DTYPE = "UNIT_FRAME_SHAPE_DTYPE"
    FORBIDDEN_WRITE_OR_ACTION = "FORBIDDEN_WRITE_OR_ACTION"
    FROZEN_IDENTITY_LOSS = "FROZEN_IDENTITY_LOSS"


class RuntimeFailure(RuntimeError):
    """Fail-fast runtime failure retaining its primary classification."""

    def __init__(self, failure_class: RuntimeFailureClass, message: str) -> None:
        self.failure_class = failure_class
        super().__init__(f"{failure_class.value}: {message}")


_FAIL_FAST_ORDER = (
    RuntimeFailureClass.FROZEN_IDENTITY_LOSS,
    RuntimeFailureClass.FORBIDDEN_WRITE_OR_ACTION,
    RuntimeFailureClass.ACTOR_OPTIMIZER_LEAKAGE,
    RuntimeFailureClass.FORBIDDEN_NONFINITE,
    RuntimeFailureClass.UNIT_FRAME_SHAPE_DTYPE,
    RuntimeFailureClass.PROCESS_CRASH,
    RuntimeFailureClass.TIMEOUT,
    RuntimeFailureClass.PRODUCT_SEMANTIC_FAILURE,
)


def classify_runtime_fail_fast(flags: dict[RuntimeFailureClass, bool]) -> RuntimeFailureClass | None:
    """Return the first active class in the protocol's stable safety order."""

    unknown = set(flags) - set(_FAIL_FAST_ORDER)
    if unknown:
        raise ValueError(f"unknown runtime failure classes: {sorted(map(str, unknown))}")
    return next((failure_class for failure_class in _FAIL_FAST_ORDER if flags.get(failure_class, False)), None)
