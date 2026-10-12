"""Immutable launcher receipt construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LauncherReceipt:
    raw_exit: int
    semantic_exit: int
    receipt_status: str
    primary_failure: str | None
    cleanup_status: str
    cleanup_exit: int | None
    timed_out: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_exit": self.raw_exit,
            "semantic_exit": self.semantic_exit,
            "receipt_status": self.receipt_status,
            "primary_failure": self.primary_failure,
            "cleanup_status": self.cleanup_status,
            "cleanup_exit": self.cleanup_exit,
            "timed_out": self.timed_out,
        }


def build_launcher_receipt(
    *,
    raw_exit: int,
    semantic_exit: int,
    semantic_evidence_present: bool,
    cleanup_exit: int | None,
    timed_out: bool,
) -> LauncherReceipt:
    primary: str | None = None
    if timed_out:
        primary = "TIMEOUT"
    elif raw_exit != 0:
        primary = f"RAW_EXIT:{raw_exit}"
    elif not semantic_evidence_present:
        primary = "MISSING_PASS_EVIDENCE"
    elif semantic_exit != 0:
        primary = f"SEMANTIC_EXIT:{semantic_exit}"
    cleanup_status = "NOT_REQUESTED" if cleanup_exit is None else ("PASS" if cleanup_exit == 0 else "FAIL")
    if primary is None and cleanup_status == "FAIL":
        primary = f"CLEANUP_EXIT:{cleanup_exit}"
    passed = (
        raw_exit == 0
        and semantic_exit == 0
        and semantic_evidence_present
        and cleanup_status != "FAIL"
        and not timed_out
    )
    return LauncherReceipt(
        raw_exit=raw_exit,
        semantic_exit=semantic_exit,
        receipt_status="PASS" if passed else "FAIL",
        primary_failure=primary,
        cleanup_status=cleanup_status,
        cleanup_exit=cleanup_exit,
        timed_out=timed_out,
    )
