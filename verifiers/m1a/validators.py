#!/usr/bin/env python3
"""Fail-closed M1A validators.

Subcommands:
  semantic MANIFEST
  objects MANIFEST
  history MANIFEST
  static --root PRODUCT_ROOT
  ledger FILE

All validators return nonzero on parser, missing-file, missing-fact, mismatch,
or malformed-input failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PRODUCT_COMMIT = "6eb521f2174a64030e30ea5e8ce8db1f89006619"


def parse_manifest_sections(text: str) -> List[Tuple[int, Dict[int, str]]]:
    section_re = re.compile(r"^## (\d+)\. ", re.M)
    headings = [int(m.group(1)) for m in section_re.finditer(text)]
    if headings != list(range(1, 37)):
        raise ValueError(f"section headings are not exactly 1..36: {headings[:80]}")
    out: List[Tuple[int, Dict[int, str]]] = []
    for m in section_re.finditer(text):
        num = int(m.group(1))
        start = m.end()
        end_match = section_re.search(text, start)
        end = end_match.start() if end_match else len(text)
        body = text[start:end]
        raw_fields: List[Tuple[int, str]] = []
        current_label: int | None = None
        current_lines: List[str] = []
        for line in body.splitlines():
            field_match = re.match(r"^(\d+)\.\s+(.+)$", line)
            if field_match:
                if current_label is not None:
                    raw_fields.append((current_label, " ".join(current_lines).strip()))
                current_label = int(field_match.group(1))
                current_lines = [field_match.group(2).strip()]
            elif current_label is not None and line.strip():
                current_lines.append(line.strip())
            elif line.strip() and re.match(r"^[^#].*:\s*", line):
                raise ValueError(f"section {num}: malformed field line: {line.strip()}")
        if current_label is not None:
            raw_fields.append((current_label, " ".join(current_lines).strip()))
        raw_labels = [label for label, _ in raw_fields]
        if len(raw_labels) != len(set(raw_labels)):
            raise ValueError(f"section {num}: duplicate field labels")
        if sorted(raw_labels) != list(range(1, 11)):
            raise ValueError(f"section {num}: field labels are not exactly 1..10: {raw_labels}")
        fields: Dict[int, str] = {}
        for label, value in raw_fields:
            if not value:
                raise ValueError(f"section {num}: empty field {label}")
            fields[label] = value
        out.append((num, fields))
    return out


def validate_semantic(manifest: Path) -> int:
    if not manifest.is_file():
        print(f"SEMANTIC_RESULT=FAIL missing input {manifest}", file=sys.stderr)
        return 1
    try:
        text = manifest.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"SEMANTIC_RESULT=FAIL read error {exc}", file=sys.stderr)
        return 1
    try:
        sections = parse_manifest_sections(text)
    except ValueError as exc:
        print(f"SEMANTIC_RESULT=FAIL {exc}", file=sys.stderr)
        return 1
    errors: List[str] = []
    if len(sections) != 36:
        errors.append(f"section_count={len(sections)} expected=36")
    for num, fields in sections:
        missing = [i for i in range(1, 11) if i not in fields]
        if missing:
            errors.append(f"section {num}: missing fields {missing}")
            continue
        authority = fields[1]
        commit = fields[2]
        path_raw = fields[3]
        blob = fields[4]
        sha = fields[5]
        size_type = fields[6]
        history = fields[7]
        license_ = fields[8]
        consumer = fields[9]
        classification = fields[10]
        if not authority or not consumer or not classification or not license_:
            errors.append(f"section {num}: empty semantic field")
        if "any `" in path_raw:
            if "proven absent" not in classification.lower():
                errors.append(f"section {num}: absence authority mismatch")
            if not re.search(r"\bN/A\b", blob) or not re.search(r"\bN/A\b", sha):
                errors.append(f"section {num}: absence N/A mismatch")
            continue
        if not re.search(r"[0-9a-f]{40}", commit):
            errors.append(f"section {num}: exact commit missing")
        if not re.search(r"[0-9a-f]{40}", blob):
            errors.append(f"section {num}: blob missing")
        if not re.search(r"[0-9a-f]{64}", sha):
            errors.append(f"section {num}: sha missing")
        if not re.search(r"\d+`?\s*bytes", size_type):
            errors.append(f"section {num}: size/type missing")
        if "not separately traced" in history or "not individually" in history:
            errors.append(f"section {num}: unresolved history")
        if not re.search(r"[0-9a-f]{7,40}", history):
            errors.append(f"section {num}: history evidence missing")
        if num in range(18, 23) or num == 36:
            if "unknown" not in history.lower() or "shallow" not in history.lower():
                errors.append(f"section {num}: shallow wording missing")
            if "f8f65d3c6ea3e243335f3444d58dfe241a7655b5" not in history:
                errors.append(f"section {num}: shallow boundary pin missing")
        if num in range(23, 29):
            if "unknown" not in history.lower() or "shallow" not in history.lower():
                errors.append(f"section {num}: Isaac shallow wording missing")
            if "37ddf626871758333d6ed89cf64ad702aef127d0" not in history:
                errors.append(f"section {num}: Isaac snapshot pin missing")
        if num in (2, 4, 7, 8, 9, 10):
            if "e970cbf89887c163ef2f9a711377ac267a2cee3d" not in history:
                errors.append(f"section {num}: e970cbf event missing")
    if errors:
        print("SEMANTIC_RESULT=FAIL")
        for e in errors:
            print(e)
        return 1
    print(f"sections_parsed={len(sections)}")
    print("SEMANTIC_RESULT=PASS")
    return 0


def _repo_for_path(path: str) -> str:
    if path.startswith(("isaac-training/", "docs/")):
        return "/home/mint/rl_dev/worktrees/instinctRL-blackwell"
    if path.startswith("omni_drones/"):
        return "/home/mint/rl_blackwell_test/repos/OmniDrones"
    if path.startswith("source/isaaclab/"):
        return "/home/mint/rl_blackwell_test/repos/IsaacLab"
    raise ValueError(f"no repository mapping for {path}")


def validate_objects(manifest: Path) -> int:
    if not manifest.is_file():
        print("OBJECT_RESULT=FAIL missing input", file=sys.stderr)
        return 1
    text = manifest.read_text(encoding="utf-8")
    try:
        sections = parse_manifest_sections(text)
    except ValueError as exc:
        print(f"OBJECT_RESULT=FAIL {exc}", file=sys.stderr)
        return 1
    if len(sections) != 36:
        print("OBJECT_RESULT=FAIL section count")
        return 1
    errors: List[str] = []
    checked = 0
    for num, f in sections:
        if len(f) < 10:
            errors.append(f"section {num}: missing fields")
            continue
        raw_path = f[3]
        classification = f[10].lower()
        if "any `" in raw_path:
            if "proven absent" not in classification:
                errors.append(f"section {num}: absence classification mismatch")
                continue
            cwd = "/home/mint/rl_blackwell_test/repos/OmniDrones"
            tracked = subprocess.run(["git", "ls-files"], cwd=cwd, text=True, capture_output=True)
            grep = subprocess.run(["rg", "-i", "TaslabUAV|taslab_uav|TASLAB", "omni_drones"], cwd=cwd, text=True, capture_output=True)
            if tracked.returncode != 0 or grep.stdout.strip() or "TaslabUAV" in tracked.stdout or "taslab_uav" in tracked.stdout:
                errors.append(f"section {num}: proven absence failed")
            else:
                checked += 1
            continue
        path_match = re.search(r"`([^`]+)`", raw_path)
        if not path_match:
            errors.append(f"section {num}: path parse failed")
            continue
        rel_path = path_match.group(1)
        try:
            cwd = _repo_for_path(rel_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not Path(cwd).is_dir():
            errors.append(f"section {num}: repository missing {cwd}")
            continue
        commit_matches = re.findall(r"[0-9a-f]{40}", f[2])
        blob_matches = re.findall(r"[0-9a-f]{40}", f[4])
        sha_matches = re.findall(r"[0-9a-f]{64}", f[5])
        size_matches = re.findall(r"(\d+)`?\s*bytes", f[6])
        type_match = re.search(r"`([^`]+)`\s*$", f[6])
        if not (commit_matches and blob_matches and sha_matches and size_matches and type_match):
            errors.append(f"section {num}: missing object values")
            continue
        commit = commit_matches[0]
        expected_blob = blob_matches[0]
        expected_sha = sha_matches[0]
        expected_size = int(size_matches[0])
        expected_type = type_match.group(1).strip()
        blob_r = subprocess.run(["git", "rev-parse", f"{commit}:{rel_path}"], cwd=cwd, text=True, capture_output=True)
        if blob_r.returncode != 0:
            errors.append(f"section {num}: blob lookup failed")
            continue
        actual_blob = blob_r.stdout.strip()
        if actual_blob != expected_blob:
            errors.append(f"section {num}: blob mismatch")
            continue
        cat_r = subprocess.run(["git", "cat-file", "blob", actual_blob], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if cat_r.returncode != 0:
            errors.append(f"section {num}: cat-file failed")
            continue
        data = cat_r.stdout
        actual_sha = hashlib.sha256(data).hexdigest()
        actual_size = len(data)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        type_r = subprocess.run(["file", "-b", str(tmp_path)], cwd=cwd, text=True, capture_output=True)
        actual_type = type_r.stdout.strip()
        tmp_path.unlink()
        if actual_sha != expected_sha or actual_size != expected_size or actual_type != expected_type:
            errors.append(f"section {num}: object/type mismatch")
        checked += 1
    print(f"object_rows_checked={checked}")
    if errors:
        print("OBJECT_RESULT=FAIL")
        for e in errors:
            print(e)
        return 1
    print("OBJECT_RESULT=PASS")
    return 0


def validate_history(manifest: Path) -> int:
    if not manifest.is_file():
        print("HISTORY_RESULT=FAIL missing input", file=sys.stderr)
        return 1
    text = manifest.read_text(encoding="utf-8")
    try:
        sections = parse_manifest_sections(text)
    except ValueError as exc:
        print(f"HISTORY_RESULT=FAIL {exc}", file=sys.stderr)
        return 1
    if len(sections) != 36:
        print("HISTORY_RESULT=FAIL section count")
        return 1
    errors: List[str] = []
    checked = 0
    for num, f in sections:
        if len(f) < 10:
            errors.append(f"section {num}: missing fields")
            continue
        raw_path = f[3]
        history = f[7]
        classification = f[10].lower()
        if "any `" in raw_path:
            if "proven absent" not in classification or "N/A" not in history:
                errors.append(f"section {num}: absence history mismatch")
            else:
                checked += 1
            continue
        path_match = re.search(r"`([^`]+)`", raw_path)
        if not path_match:
            errors.append(f"section {num}: path parse failed")
            continue
        rel_path = path_match.group(1)
        try:
            cwd = _repo_for_path(rel_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not Path(cwd).is_dir():
            errors.append(f"section {num}: repository missing {cwd}")
            continue
        if rel_path.startswith(("isaac-training/", "docs/")):
            pin = "6f6dee3da72d0b0202846b330adcaeeec8ee7bc3"
        elif rel_path.startswith("omni_drones/"):
            pin = "5045c71cc73d224144cc08bc38d65cabeedaec8c"
        else:
            pin = "37ddf626871758333d6ed89cf64ad702aef127d0"
        declared_commits = re.findall(r"[0-9a-f]{40}", f[2])
        if declared_commits != [pin]:
            errors.append(f"section {num}: exact history pin mismatch")
            continue
        log_r = subprocess.run(["git", "log", "--follow", "--format=%H", pin, "--", rel_path], cwd=cwd, text=True, capture_output=True)
        if log_r.returncode != 0:
            errors.append(f"section {num}: git log failed")
            continue
        log_hashes = log_r.stdout.strip().splitlines()
        for claimed in re.findall(r"[0-9a-f]{40}", history):
            if claimed not in log_hashes:
                errors.append(f"section {num}: claimed hash not in log: {claimed}")
        if num in range(18, 23) or num == 36:
            if "unknown" not in history.lower() or "shallow" not in history.lower():
                errors.append(f"section {num}: shallow wording missing")
            parent_r = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "f8f65d3c6ea3e243335f3444d58dfe241a7655b5"], cwd=cwd, text=True, capture_output=True)
            if parent_r.returncode != 0 or len(parent_r.stdout.split()) != 1:
                errors.append(f"section {num}: shallow topology unexpected")
        if num in range(23, 29):
            if "unknown" not in history.lower() or "shallow" not in history.lower():
                errors.append(f"section {num}: Isaac shallow wording missing")
            one_r = subprocess.run(["git", "rev-list", "--parents", "-n", "1", "37ddf626871758333d6ed89cf64ad702aef127d0"], cwd=cwd, text=True, capture_output=True)
            if one_r.returncode != 0 or len(one_r.stdout.split()) != 1:
                errors.append(f"section {num}: Isaac one-commit shallow topology unexpected")
        if num in (2, 4, 7, 8, 9, 10):
            if "e970cbf89887c163ef2f9a711377ac267a2cee3d" not in history:
                errors.append(f"section {num}: e970cbf event missing")
        checked += 1
    print(f"history_rows_checked={checked}")
    if errors:
        print("HISTORY_RESULT=FAIL")
        for e in errors:
            print(e)
        return 1
    print("HISTORY_RESULT=PASS")
    return 0


def validate_static(root: Path, modern_root: Path | None = None) -> int:
    doc_path = root / "docs/blackwell_devlog/m1a/TASLAB_ORIGINAL_VS_MODERN_DIFF.md"
    legacy_base = root / "isaac-training/third_party/OmniDrones/omni_drones"
    modern_base = modern_root or Path("/home/mint/rl_blackwell_test/repos/OmniDrones/omni_drones")
    if not doc_path.is_file():
        print("STATIC_RESULT=FAIL missing product doc", file=sys.stderr)
        return 1
    required_source_files = [
        legacy_base / "utils/torchrl/transforms.py",
        legacy_base / "controllers/lee_position_controller.py",
        legacy_base / "actuators/rotor_group.py",
        legacy_base / "robots/drone/multirotor.py",
        modern_base / "utils/torchrl/transforms.py",
        modern_base / "robots/drone/multirotor.py",
    ]
    for source_file in required_source_files:
        if not source_file.is_file():
            print(f"STATIC_RESULT=FAIL missing source file {source_file}", file=sys.stderr)
            return 1
    doc = doc_path.read_text(encoding="utf-8")
    norm_doc = re.sub(r"\s+", " ", doc).replace("`", "")
    legacy_transforms = (legacy_base / "utils/torchrl/transforms.py").read_text(encoding="utf-8")
    legacy_lee = (legacy_base / "controllers/lee_position_controller.py").read_text(encoding="utf-8")
    legacy_rotor = (legacy_base / "actuators/rotor_group.py").read_text(encoding="utf-8")
    legacy_multi = (legacy_base / "robots/drone/multirotor.py").read_text(encoding="utf-8")
    modern_transforms = (modern_base / "utils/torchrl/transforms.py").read_text(encoding="utf-8")
    modern_multi = (modern_base / "robots/drone/multirotor.py").read_text(encoding="utf-8")
    facts = [
        ("doc_affine", "affine-normalized", norm_doc),
        ("doc_not_guaranteed", "not a guaranteed [-1, 1] value", norm_doc),
        ("doc_per_rotor_thrust", "Per-rotor thrust forces are applied", norm_doc),
        ("doc_aggregate_reaction", "aggregate reaction torque and body force are applied", norm_doc),
        ("legacy_action_key", 'action_key: str = ("agents", "action")', legacy_transforms),
        ("legacy_state_key", 'in_keys_inv=[("info", "drone_state")]', legacy_transforms),
        ("legacy_nan", "torch.nan_to_num_(cmds, 0.)", legacy_transforms),
        ("legacy_affine", "cmd = (cmd / self.max_thrusts) * 2 - 1", legacy_lee),
        ("legacy_clamp", "target_throttle = self.f_inv(torch.clamp((cmds + 1) / 2, 0, 1))", legacy_rotor),
        ("legacy_rotor_apply", "self.rotors_view.apply_forces_and_torques_at_pos(", legacy_multi),
        ("legacy_base_apply", "self.base_link.apply_forces_and_torques_at_pos(", legacy_multi),
        ("modern_action_key", 'action_key: str = ("agents", "action")', modern_transforms),
        ("modern_state_key", 'in_keys_inv=[("info", "drone_state")]', modern_transforms),
        ("modern_nan", "torch.nan_to_num_(cmds, 0.)", modern_transforms),
        ("modern_compute", "cmds = self.controller.compute(", modern_transforms),
        ("modern_rotor_apply", "self.rotors_view.apply_forces_and_torques_at_pos(", modern_multi),
        ("modern_base_apply", "self.base_link.apply_forces_and_torques_at_pos(", modern_multi),
    ]
    errors = []
    for name, needle, haystack in facts:
        normalized_haystack = re.sub(r"\s+", " ", haystack).replace("`", "")
        if needle not in normalized_haystack:
            errors.append(f"missing fact {name}")
    actor_facts = [
        ("actor_boundary", "controller/adapter boundary", norm_doc),
        ("actor_no_observation", "not an authorized actor observation field", norm_doc),
        ("actor_contract_scope", "actor-contract compliance", norm_doc),
    ]
    for name, needle, haystack in actor_facts:
        if needle not in haystack:
            errors.append(f"missing fact {name}")
    if re.search(
        r"deployed[_ -]?actor[_ -]?input\s*[:=].*\b(?:pose|odometry|velocity estimate|map|slam|obstacle ground truth|simulator state)\b",
        norm_doc,
        flags=re.I,
    ):
        errors.append("actor-boundary leakage")
    if errors:
        print("STATIC_RESULT=FAIL")
        for e in errors:
            print(e)
        return 1
    print("STATIC_RESULT=PASS")
    print(f"asserted_facts={len(facts)}")
    return 0


FORBIDDEN_LEDGER_PATTERNS = [
    r"\bbash\s+-c\b", r"\bsh\s+-c\b", r"\bpython\s+-c\b", r"\bpython3\s+-c\b",
    r"\bpython\s+-(\s|$)", r"\bpython3\s+-(\s|$)", r"<\([^)]*\)", r"<<<",
    r"\bheredoc\b", r"\bhere-string\b", r"\bprocess-substitution\b",
    r"\bgit\s+add\b", r"\bgit\s+commit\b", r"\bgit\s+push\b", r"\bgit\s+merge\b",
    r"\bgit\s+rebase\b", r"\bgit\s+fetch\b", r"\bgit\s+config\b", r"\bgit\s+remote\b",
    r"\bgit\s+(?:reset|clean|checkout|restore|switch|cherry-pick|revert|rm|mv|apply|am)\b",
    r"\bpip\s+install\b", r"\bconda\s+install\b", r"\bconda\s+remove\b",
    r"\bapt\s+install\b", r"\bapt\s+remove\b", r"\bcurl\b", r"\bwget\b",
    r"\bgit\s+clone\b", r"\b(?:rm|mv|cp|dd|tee|truncate|touch)\b", r"\bruntime\b", r"\btraining\b", r"\bevaluation\b",
    r"\bmid360\b", r"\bomni_drones\b",
]


def validate_ledger(path: Path) -> int:
    if not path.is_file():
        print("LEDGER_RESULT=FAIL missing input", file=sys.stderr)
        return 1
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    errors = []
    for idx, line in enumerate(lines, 1):
        for pat in FORBIDDEN_LEDGER_PATTERNS:
            if re.search(pat, line, flags=re.I):
                errors.append(f"line={idx} pattern={pat}")
                break
    if errors:
        print("LEDGER_RESULT=FAIL")
        for e in errors:
            print(e)
        return 1
    print(f"ledger_rows={len(lines)}")
    print("LEDGER_RESULT=PASS")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_sem = sub.add_parser("semantic")
    p_sem.add_argument("manifest", type=Path)
    p_obj = sub.add_parser("objects")
    p_obj.add_argument("manifest", type=Path)
    p_his = sub.add_parser("history")
    p_his.add_argument("manifest", type=Path)
    p_static = sub.add_parser("static")
    p_static.add_argument("--root", type=Path, default=Path("/home/mint/rl_dev/worktrees/instinctRL-blackwell"))
    p_ledger = sub.add_parser("ledger")
    p_ledger.add_argument("file", type=Path)
    args = parser.parse_args(argv)
    if args.command == "semantic":
        return validate_semantic(args.manifest)
    if args.command == "objects":
        return validate_objects(args.manifest)
    if args.command == "history":
        return validate_history(args.manifest)
    if args.command == "static":
        return validate_static(args.root)
    if args.command == "ledger":
        return validate_ledger(args.file)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
