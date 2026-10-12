"""CPU-only source, pin, checkpoint, and method audits for the candidate."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
from typing import Any

from .config import load_config, sha256_file


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(["git", "-C", str(repository), *arguments], check=True, capture_output=True, text=True).stdout.strip()


def _entry_import_audit(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    forbidden = [name for name in imports if name == "torch" or name.startswith(("isaaclab", "omni", "pxr"))]
    if forbidden or "AppLauncher" in path.read_text(encoding="utf-8").split("def main", 1)[0]:
        raise ValueError(f"entry has a runtime import before authorization: {forbidden}")
    return {"path": str(path), "top_level_imports": imports, "runtime_imports": False}


def checkpoint_readonly_audit(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("update_counter") != 250:
        raise ValueError("input checkpoint counter/schema mismatch")
    summaries = {}
    for name in ("actor_optimizer", "critic_optimizer"):
        state = payload[name]["state"]
        steps = [int(item["step"].item()) for item in state.values()]
        if not steps or set(steps) != {2000}:
            raise ValueError(f"{name} does not have exact step 2000")
        summaries[name] = {"state_count": len(steps), "step_min": min(steps), "step_max": max(steps)}
    schema = payload.get("schema", {})
    if schema.get("actor_keys") != ("lidar_grid", "state_vec") or schema.get("lidar_shape") != (12, 59, 360) or schema.get("state_dimension") != 52:
        raise ValueError("checkpoint actor boundary mismatch")
    return {"path": str(path), "sha256": sha256_file(path), "counter": 250, "optimizers": summaries, "actor_boundary": schema}


def affine_rank_negative_control() -> dict[str, Any]:
    """Prove affine point rank 3 does not satisfy the exact normal gate."""
    import torch
    from instinctlab.evaluation.observability import compute_exact_observability

    points = torch.tensor(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), dtype=torch.float64)
    affine_rank = int(torch.linalg.matrix_rank(points[1:] - points[0]).item())
    normals = torch.tensor([[[[0., 0., 1.], [1., 0., 0.], [0., 0., 1.], [1., 0., 0.]]]], dtype=torch.float64).clone()
    valid = torch.ones((1, 1, 4), dtype=torch.bool)
    exact = compute_exact_observability(normals, valid)
    numerical_rank = int(exact.numerical_rank.item())
    sigma_min = float(exact.sigma_min.item())
    if (affine_rank, numerical_rank, sigma_min) != (3, 2, 0.0):
        raise ValueError("affine-rank negative control did not distinguish the exact normal gate")
    return {"point_cloud_affine_rank": affine_rank, "normal_jacobian_rank": numerical_rank, "sigma_min": sigma_min, "gate": "REJECT"}


def run_cpu_audit(*, require_candidate_clean: bool = False) -> dict[str, Any]:
    config = load_config()
    paths = config["paths"]; pins = config["pins"]
    product = Path(paths["product_repository"])
    product_identity = {
        "commit": _git(product, "rev-parse", "HEAD"), "tree": _git(product, "rev-parse", "HEAD^{tree}"),
        "status": _git(product, "status", "--porcelain"),
    }
    if product_identity != {"commit": pins["product_commit"], "tree": pins["product_tree"], "status": ""}:
        raise ValueError("frozen product identity/cleanliness mismatch")
    python = Path(paths["python"])
    checkpoint = Path(pins["checkpoint_path"])
    if not python.is_file() or python.is_symlink() or sha256_file(python) != pins["python_sha256"]:
        raise ValueError("pinned interpreter identity/type mismatch")
    if sha256_file(checkpoint) != pins["checkpoint_sha256"]:
        raise ValueError("pinned checkpoint identity mismatch")
    candidate = Path(paths["candidate_repository"])
    candidate_identity = {
        "commit": _git(candidate, "rev-parse", "HEAD"), "tree": _git(candidate, "rev-parse", "HEAD^{tree}"),
        "status": _git(candidate, "status", "--porcelain"),
    }
    if require_candidate_clean and candidate_identity["status"]:
        raise ValueError("candidate worktree is not clean")
    entries = [_entry_import_audit(Path(paths[name])) for name in (
        "training_entry", "evaluation_entry", "station_evaluation_entry", "tracking_evaluation_entry",
    )]
    return {
        "schema_version": "scale-003-cpu-audit.v1", "status": "PASS",
        "product": product_identity, "candidate": candidate_identity,
        "python": {"path": str(python), "sha256": sha256_file(python)},
        "checkpoint": checkpoint_readonly_audit(checkpoint), "entries": entries,
        "method_negative_control": affine_rank_negative_control(),
        "runtime_started": False, "cuda_initialized": False, "environment_transitions": 0,
        "optimizer_updates": 0, "checkpoint_writes": 0,
    }
