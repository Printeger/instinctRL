"""Build existing-tooling-compatible future attempt documents without writing an attempt."""

from __future__ import annotations

import hashlib
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Iterable

from .config import canonical_bytes, load_config, sha256_file
from .handoff import validate_training_handoff
from .runtime_support import runtime_environment


def input_row(path: Path, role: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {resolved}")
    return {"bytes": resolved.stat().st_size, "path": str(resolved), "role": role, "sha256": sha256_file(resolved)}


def build_attempt_documents(
    *, mode: str, attempt_id: str, attempt_dir: Path, execution_authorization: Path,
    checkpoint: Path, source_dependencies: Iterable[Path], candidate_commit: str, candidate_tree: str,
    training_handoff: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config()
    if mode not in {"training", "evaluation"}:
        raise ValueError("manifest mode must be training or evaluation")
    if mode == "training":
        if training_handoff is not None or checkpoint.resolve() != Path(config["pins"]["checkpoint_path"]).resolve() or sha256_file(checkpoint) != config["pins"]["checkpoint_sha256"]:
            raise ValueError("training manifest must bind only the frozen update-250 input")
    else:
        if training_handoff is None:
            raise ValueError("evaluation manifest requires a sealed independent training-attempt handoff")
        handoff = validate_training_handoff(training_handoff, evaluation_attempt=attempt_dir)
        if checkpoint.resolve() != handoff.final_checkpoint_path:
            raise ValueError("evaluation manifest checkpoint differs from the sealed training handoff")
    source_root = Path(config["paths"]["candidate_source_root"])
    entries = ([source_root / "formal_scale_training.py"] if mode == "training" else [
        source_root / "formal_station_evaluation.py", source_root / "formal_tracking_evaluation.py",
    ])
    files = [
        input_row(Path(config["paths"]["python"]), "interpreter"),
        input_row(Path(__file__).with_name("candidate_config.json"), "workload_config"),
        input_row(Path(config["proposal"]["path"]), "approved_proposal"),
        input_row(execution_authorization, "live_execution_authorization"),
        input_row(checkpoint, "input_checkpoint"),
    ]
    fixed_dependencies = list(source_root.glob("*.py"))
    fixed_dependencies.extend(Path(config["paths"]["product_repository"]) / "isaaclab-training" / relative for relative in config["pins"]["product_source_sha256"])
    fixed_dependencies.extend((
        Path(config["pins"]["eval_002_scene_path"]),
        Path(config["pins"]["eval_002_gpu_gate_path"]),
        Path(config["pins"]["nvidia_smi_path"]),
        Path(config["pins"]["app_launcher_path"]),
        Path(config["pins"]["simulation_app_path"]),
    ))
    files.extend(input_row(path, "workload_dependency") for path in (*fixed_dependencies, *source_dependencies))
    files.extend(input_row(entry, "command_source") for entry in entries)
    if mode == "evaluation":
        files.extend(input_row(source_root / f"evaluation_{case}.json", "workload_config") for case in ("station", "tracking"))
        files.extend(input_row(path, "sealed_training_handoff") for path in (
            handoff.terminal_manifest_path, handoff.training_semantic_path, handoff.training_status_path,
        ))
    unique = {row["path"]: row for row in files}
    input_hashes = {
        "schema_version": "evaluation-input-hashes.v1",
        "product_commit": config["pins"]["product_commit"],
        "tooling_commit": config["pins"]["tooling_commit"],
        "files": [unique[path] for path in sorted(unique)],
    }
    input_hash = hashlib.sha256(canonical_bytes(input_hashes)).hexdigest()
    common_environment = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "CUDA_VISIBLE_DEVICES": "0",
        "NVIDIA_TF32_OVERRIDE": "0", "OMP_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "20260907", "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": f"{config['paths']['candidate_repository']}:{config['paths']['product_repository']}/isaaclab-training",
        "SCALE003_ATTEMPT_DIR": str(attempt_dir),
        "SCALE003_EXECUTION_AUTHORIZATION": str(execution_authorization),
        **runtime_environment(attempt_dir),
    }
    commands = []
    cases = [None] if mode == "training" else ["station", "tracking"]
    for command_index, case in enumerate(cases):
        entry = entries[command_index]
        environment = dict(common_environment)
        argv = [config["paths"]["python"], "-B", str(entry)]
        if case is not None:
            environment["SCALE003_EVALUATION_CHECKPOINT"] = str(checkpoint)
            environment["SCALE003_TRAINING_ATTEMPT_DIR"] = str(handoff.training_attempt_path)
        semantic = "results/training/semantic.json" if case is None else f"results/{case}/semantic.json"
        commands.append({
            "label": "scale_training" if case is None else f"evaluate_{case}", "argv": argv,
            "cwd": config["paths"]["product_repository"], "environment": environment,
            "expected_exit": 0, "timeout_seconds": 7200 if case is None else 1800,
            "semantic_result_path": semantic, "cleanup_argv": None,
        })
    required = [command["semantic_result_path"] for command in commands]
    manifest = {
        "schema_version": "evaluation-attempt-manifest.v1", "attempt_id": attempt_id,
        "product_commit": config["pins"]["product_commit"], "tooling_commit": config["pins"]["tooling_commit"],
        "tooling_repository": config["paths"]["coordination_repository"], "tooling_tree": config["pins"]["tooling_tree"],
        "product_repository": config["paths"]["product_repository"], "product_tree": config["pins"]["product_tree"],
        "product_clean_required": True, "input_hashes_path": "INPUT_HASHES.json", "input_hashes_sha256": input_hash,
        "allowed_input_roots": sorted({str(Path(row["path"]).parent) for row in input_hashes["files"]}),
        "allowed_working_roots": [config["paths"]["product_repository"]],
        "float_policy": "FLOAT32_GEOMETRY_ULP2_ABS1E5_V1", "commands": commands,
        "external_inventories": [], "required_results": required,
    }
    if candidate_commit == "" or candidate_tree == "":
        raise ValueError("candidate identity is required by the execution authorization even though the existing manifest stores it through pinned source rows")
    return manifest, input_hashes


def build_training_handoff_document(*, training_attempt: Path, evaluation_attempt: Path) -> dict[str, Any]:
    """Build only after the training attempt is sealed; never predicts final bytes."""
    training_attempt = training_attempt.resolve()
    terminal = training_attempt / "TERMINAL_MANIFEST.json"
    semantic = training_attempt / "results/training/semantic.json"
    status = training_attempt / "results/training/status.json"
    checkpoint = training_attempt / "results/training/checkpoints/update_1954_final.pt"
    for path in (terminal, semantic, status, checkpoint):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sealed training handoff input is absent: {path}")
    value = {
        "schema_version": "scale-003-training-handoff.v1",
        "training_attempt_id": training_attempt.name, "training_attempt_path": str(training_attempt),
        "terminal_manifest_path": str(terminal), "terminal_manifest_sha256": sha256_file(terminal),
        "training_semantic_path": str(semantic), "training_semantic_sha256": sha256_file(semantic),
        "training_status_path": str(status), "training_status_sha256": sha256_file(status),
        "final_checkpoint_path": str(checkpoint), "final_checkpoint_sha256": sha256_file(checkpoint),
        "final_checkpoint_bytes": checkpoint.stat().st_size, "final_checkpoint_counter": 1954,
    }
    validate_training_handoff(value, evaluation_attempt=evaluation_attempt)
    return value
