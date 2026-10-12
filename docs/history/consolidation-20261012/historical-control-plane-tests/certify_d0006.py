#!/usr/bin/env python3
"""Persisted D0006 certification harness with per-test immutable evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path


COORD = Path("/home/mint/rl_dev/instinctRL_coordination")
PRODUCT = Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell")
ATTEMPT = COORD / "attempts/development/CP2-VAL-001/CP2-VAL-001-D0006"
ALLOWLIST = (
    "verifiers/m1a/", "qualification/runner/", "qualification/manifests/examples/",
    "tests/control_plane_v2/",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(data: bytes):
    return hashlib.sha256(data).hexdigest()


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def tests():
    suite = unittest.defaultTestLoader.discover(str(COORD / "tests/control_plane_v2"), pattern="test_*.py", top_level_dir=str(COORD))
    return sorted(test.id() for test in flatten(suite))


def requirement(test_id):
    if "GovernanceIdentity" in test_id:
        return 1
    if "FrozenProductIdentity" in test_id:
        return 2
    if "ManifestSchema" in test_id:
        return 3
    if "SemanticValidator" in test_id:
        return 4
    if "ObjectValidator" in test_id:
        return 5
    if "HistoryValidator" in test_id:
        return 6
    if "StaticContract" in test_id:
        return 7
    if any(token in test_id for token in ("frozen", "certification", "cwd_allowlist", "input_allowlist", "inner_git")):
        return 8
    if "CommandPolicy" in test_id or "LedgerValidator" in test_id:
        return 9
    if any(token in test_id for token in ("interface", "provenance", "hash_preflight", "byte_mutation", "manifest_mutation")):
        return 10
    if "end_to_end" in test_id or "timeout" in test_id or "signal" in test_id:
        return 11
    if "reuse" in test_id or "immutability" in test_id:
        return 12
    if "repeatable" in test_id:
        return 13
    if "CandidateCommit" in test_id:
        return 14
    return 10


def safe_id(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def execute(argv, cwd, env=None):
    start = now()
    proc = subprocess.run(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    end = now()
    return proc, start, end


def inventory(head):
    listed = subprocess.run(
        ["git", "-C", str(COORD), "ls-tree", "-r", "--name-only", head],
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    sources = {}
    for path in sorted(path for path in listed if path.startswith(ALLOWLIST)):
        blob = subprocess.run(
            ["git", "-C", str(COORD), "rev-parse", f"{head}:{path}"],
            text=True, capture_output=True, check=True,
        ).stdout.strip()
        data = subprocess.run(
            ["git", "-C", str(COORD), "cat-file", "blob", blob],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        ).stdout
        working = COORD / path
        sources[path] = {
            "git_blob": blob,
            "sha256": digest(data),
            "size": len(data),
            "working_sha256": digest(working.read_bytes()) if working.is_file() else None,
        }
    return sources


def make_immutable(root):
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def run_preflight(test_ids):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COORD)
    failures = []
    for index, test_id in enumerate(test_ids, 1):
        proc, _, _ = execute([sys.executable, "-m", "unittest", test_id], COORD, env)
        print(f"[{index}/{len(test_ids)}] exit={proc.returncode} {test_id}")
        if proc.returncode:
            failures.append((test_id, proc.stdout.decode(errors="replace"), proc.stderr.decode(errors="replace")))
    for test_id, stdout, stderr in failures:
        print(f"FAIL {test_id}\nSTDOUT\n{stdout}\nSTDERR\n{stderr}", file=sys.stderr)
    return 1 if failures else 0


def run_final(test_ids):
    attempt = json.loads((ATTEMPT / "ATTEMPT.json").read_text(encoding="utf-8"))
    if attempt.get("state") != "ATTEMPT_RUNNING" or attempt.get("attempt_id") != "CP2-VAL-001-D0006":
        raise SystemExit("D0006 is not a mutable ATTEMPT_RUNNING development attempt")
    head = subprocess.run(["git", "-C", str(COORD), "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip()
    parent = subprocess.run(["git", "-C", str(COORD), "rev-parse", "HEAD^"], text=True, capture_output=True, check=True).stdout.strip()
    source_inventory = inventory(head)
    mismatches = [path for path, item in source_inventory.items() if item["working_sha256"] != item["sha256"]]
    if mismatches:
        raise SystemExit(f"working source/test bytes differ from candidate: {mismatches}")
    evidence = ATTEMPT / "evidence"
    evidence.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(COORD)
    ledger = []
    failures = []
    results_by_round = []
    for round_number in (1, 2):
        round_results = []
        for index, test_id in enumerate(test_ids, 1):
            command_id = f"r{round_number:02d}_{index:04d}_{safe_id(test_id)}"
            argv = [sys.executable, "-m", "unittest", test_id]
            proc, start, end = execute(argv, COORD, env)
            command_dir = evidence / command_id
            command_dir.mkdir()
            stdout_path = command_dir / "stdout"
            stderr_path = command_dir / "stderr"
            stdout_path.write_bytes(proc.stdout)
            stderr_path.write_bytes(proc.stderr)
            row = {
                "command_id": command_id, "kind": "test", "required_test_item": requirement(test_id),
                "test_id": test_id, "round": round_number, "argv": argv, "cwd": str(COORD),
                "environment": {"PYTHONPATH": str(COORD), "python": sys.version},
                "start": start, "end": end, "exit_code": proc.returncode,
                "stdout": str(stdout_path.relative_to(ATTEMPT)), "stderr": str(stderr_path.relative_to(ATTEMPT)),
                "stdout_sha256": digest(proc.stdout), "stderr_sha256": digest(proc.stderr),
                "result": "PASS" if proc.returncode == 0 else "FAIL",
            }
            (command_dir / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            ledger.append(row)
            round_results.append((test_id, proc.returncode))
            if proc.returncode:
                failures.append(command_id)
        results_by_round.append(round_results)
    audit_commands = [
        ("product_branch", PRODUCT, ["git", "branch", "--show-current"], 2),
        ("product_head", PRODUCT, ["git", "rev-parse", "HEAD"], 2),
        ("product_parent", PRODUCT, ["git", "rev-parse", "HEAD^"], 2),
        ("product_status", PRODUCT, ["git", "status", "--porcelain=v1"], 2),
        ("product_zero_commits", PRODUCT, ["git", "rev-list", "--count", "6eb521f2174a64030e30ea5e8ce8db1f89006619..HEAD"], 2),
        ("candidate_name_status", COORD, ["git", "diff", "--name-status", f"{parent}..{head}"], 14),
        ("candidate_stat", COORD, ["git", "diff", "--stat", f"{parent}..{head}"], 14),
        ("candidate_full_diff", COORD, ["git", "diff", f"{parent}..{head}"], 14),
        ("candidate_show", COORD, ["git", "show", "--format=fuller", "--stat", "--patch", head], 14),
        ("candidate_diff_check", COORD, ["git", "diff", "--check", f"{parent}..{head}"], 14),
    ]
    for index, (name, cwd, argv, item) in enumerate(audit_commands, 1):
        command_id = f"audit_{index:02d}_{name}"
        proc, start, end = execute(argv, cwd, env)
        command_dir = evidence / command_id
        command_dir.mkdir()
        stdout_path = command_dir / "stdout"
        stderr_path = command_dir / "stderr"
        stdout_path.write_bytes(proc.stdout)
        stderr_path.write_bytes(proc.stderr)
        row = {
            "command_id": command_id, "kind": "audit", "required_test_item": item,
            "argv": argv, "cwd": str(cwd), "environment": {}, "start": start, "end": end,
            "exit_code": proc.returncode, "stdout": str(stdout_path.relative_to(ATTEMPT)),
            "stderr": str(stderr_path.relative_to(ATTEMPT)), "stdout_sha256": digest(proc.stdout),
            "stderr_sha256": digest(proc.stderr), "result": "PASS" if proc.returncode == 0 else "FAIL",
        }
        (command_dir / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        ledger.append(row)
        if proc.returncode:
            failures.append(command_id)
    if results_by_round[0] != results_by_round[1]:
        failures.append("certification_repeatability")
    (ATTEMPT / "command_ledger.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger), encoding="utf-8"
    )
    verifier_inventory = {
        "schema_version": "control-plane-v2/development-inventory-1",
        "candidate_commit": head, "candidate_parent": parent,
        "sources_and_tests": source_inventory,
        "inventory_sha256": digest(json.dumps(source_inventory, sort_keys=True).encode()),
    }
    (ATTEMPT / "verifier_inventory.json").write_text(
        json.dumps(verifier_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    index = [
        "# CP2-VAL-001-D0006 gate/test evidence index", "",
        f"- candidate commit: `{head}`", f"- candidate parent: `{parent}`",
        f"- unique named tests: {len(test_ids)}", "- certification rounds: 2",
        f"- ledger rows covered by protocol audit: {len(ledger)}",
        f"- terminal result: {'FAIL' if failures else 'PASS'}", "",
        "| command | required item | round | test/audit | argv | cwd | start | end | exit | stdout/hash | stderr/hash | result |",
        "| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in ledger:
        index.append(
            f"| {row['command_id']} | {row['required_test_item']} | {row.get('round', '-')} | {row.get('test_id', row['kind'])} | `{json.dumps(row['argv'])}` | {row['cwd']} | {row['start']} | {row['end']} | {row['exit_code']} | {row['stdout']} / {row['stdout_sha256']} | {row['stderr']} / {row['stderr_sha256']} | {row['result']} |"
        )
    index.extend(["", "## Known limitations", "", "- None within the CP2-VAL-001 verifier certification scope."])
    (ATTEMPT / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    final_time = now()
    result = {
        "schema_version": "control-plane-v2/development-result-1",
        "attempt_id": "CP2-VAL-001-D0006", "state": "ATTEMPT_FAILED" if failures else "ATTEMPT_PASSED",
        "candidate_commit": head, "candidate_parent": parent, "unique_tests": len(test_ids),
        "test_executions": len(test_ids) * 2, "audit_commands": len(audit_commands),
        "ledger_rows": len(ledger), "failures": failures, "repeatable": not failures,
        "finalized_at": final_time,
    }
    (ATTEMPT / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    attempt.update({
        "state": result["state"], "finalized_at": final_time,
        "candidate_verifier_commit": head, "candidate_parent": parent,
        "test_names": test_ids, "unique_tests": len(test_ids), "test_executions": len(test_ids) * 2,
        "ledger_rows": len(ledger), "source_inventory_sha256": verifier_inventory["inventory_sha256"],
        "result": {"status": "PASS" if not failures else "FAIL", "failures": failures},
    })
    temporary = ATTEMPT / ".ATTEMPT.json.tmp"
    temporary.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, ATTEMPT / "ATTEMPT.json")
    if failures:
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 1
    make_immutable(ATTEMPT)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "finalize"))
    args = parser.parse_args()
    test_ids = tests()
    if not test_ids:
        raise SystemExit("no certification tests discovered")
    return run_preflight(test_ids) if args.mode == "preflight" else run_final(test_ids)


if __name__ == "__main__":
    raise SystemExit(main())
