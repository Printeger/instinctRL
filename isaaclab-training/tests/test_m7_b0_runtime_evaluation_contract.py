"""Pure exact contract tests for M7-012 B0 runtime evaluation collection."""

from __future__ import annotations

import ast
from collections import OrderedDict
import copy
from dataclasses import dataclass, FrozenInstanceError, fields, replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from types import ModuleType
import sys

import pytest
import torch

from instinctlab.evaluation.baseline_run import B0RunReceipt
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)
from instinctlab.evaluation.metrics import (
    EVALUATION_INPUT_KEYS,
    EvaluationMetricConfig,
    ObservabilitySource,
    aggregate_metrics,
    compute_step_metrics,
)
from instinctlab.evaluation.runtime import ExactRuntimeObservability


@dataclass(frozen=True, slots=True)
class B0EnvironmentCoverage:
    policy_steps: int
    physics_steps: int
    lifecycle_calls: int
    wrench_applications: int
    environment_samples: int
    num_envs: int
    decimation: int


@dataclass(frozen=True, slots=True)
class B0EnvironmentResult:
    receipt: B0RunReceipt
    coverage: B0EnvironmentCoverage


fake_environment = ModuleType("instinctlab.evaluation.baseline_environment")
fake_environment.B0EnvironmentCoverage = B0EnvironmentCoverage
fake_environment.B0EnvironmentResult = B0EnvironmentResult
previous_environment = sys.modules.get(fake_environment.__name__)
sys.modules[fake_environment.__name__] = fake_environment
try:
    baseline_runtime = importlib.import_module("instinctlab.evaluation.baseline_runtime")
finally:
    if previous_environment is None:
        sys.modules.pop(fake_environment.__name__, None)
    else:
        sys.modules[fake_environment.__name__] = previous_environment

ANCHOR_KEYS = baseline_runtime.ANCHOR_KEYS
B0RuntimeEvaluationConfig = baseline_runtime.B0RuntimeEvaluationConfig
B0RuntimeEvaluationProvenance = baseline_runtime.B0RuntimeEvaluationProvenance
B0_SOURCE_MANIFEST = baseline_runtime.B0_SOURCE_MANIFEST
CONTROL_KEYS = baseline_runtime.CONTROL_KEYS
ICS_KEYS = baseline_runtime.ICS_KEYS
OBSERVABILITY_DETAIL_KEYS = baseline_runtime.OBSERVABILITY_DETAIL_KEYS
PRE_RESET_KEYS = baseline_runtime.PRE_RESET_KEYS
RECORD_KEYS = baseline_runtime.RECORD_KEYS
SOURCE_METADATA_KEYS = baseline_runtime.SOURCE_METADATA_KEYS
TASK_KEYS = baseline_runtime.TASK_KEYS
TERMINATION_KEYS = baseline_runtime.TERMINATION_KEYS
TERMINATION_MASK_KEYS = baseline_runtime.TERMINATION_MASK_KEYS
_row_hashes = baseline_runtime._row_hashes
begin_ready_b0_runtime_evaluation = baseline_runtime.begin_ready_b0_runtime_evaluation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/baseline_runtime.py"
PRODUCT_COMMIT = "69d94b870ef02843e77caed916cc51f4730c4db7"
SEED = 7012
COMMANDS = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((0.25, -0.10, 0.05), (-0.20, 0.15, 0.10)),
    ((3.0, 4.0, 0.0), (-4.0, 0.0, 3.0)),
    ((0.35, 0.10, -0.05), (0.10, -0.25, 0.15)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((-3.0, -4.0, 0.0), (0.0, 4.0, -3.0)),
    ((0.10, 0.20, 0.30), (-0.30, 0.20, 0.10)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
)


def _bindings() -> tuple[CapabilityBinding, ...]:
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in resolve_baseline("B0").capability_requirements
    )


def _manifest(
    status: ExecutionStatus = ExecutionStatus.READY,
    scenario_id: str = "M7-012.runtime-evaluation",
) -> BaselineRunManifest:
    row = resolve_baseline("B0")
    return build_run_manifest(
        product_commit=PRODUCT_COMMIT,
        baseline_id="B0",
        config_switch=row.config_switch,
        capability_bindings=_bindings(),
        scenario_id=scenario_id,
        seed=SEED,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=status,
        skip_reason=None,
    )


def _config() -> B0RuntimeEvaluationConfig:
    return B0RuntimeEvaluationConfig(
        8,
        2,
        21240,
        EvaluationMetricConfig(),
        B0RuntimeEvaluationProvenance(
            PRODUCT_COMMIT, "m7-012-b0-two-env-eight-step-v1", "pytest", SEED,
        ),
    )


def _critic(position_x: tuple[float, float] = (0.0, 10.0)) -> torch.Tensor:
    value = torch.zeros((2, 18), dtype=torch.float32)
    value[:, 0] = torch.tensor(position_x)
    value[:, 2] = 1.5
    value[:, 3] = 1.0
    value[:, 16] = 3.0
    return value


def _exact(step: int) -> ExactRuntimeObservability:
    valid_count = torch.tensor((12000 + step, 10000 + step), dtype=torch.long)
    total_count = torch.full((2,), 21240, dtype=torch.long)
    source_identity = torch.stack(tuple(
        torch.tensor(tuple(hashlib.sha256(f"{step}:{index}".encode()).digest()), dtype=torch.uint8)
        for index in range(2)
    ))
    values = OrderedDict((
        ("numerical_rank", torch.tensor((1, 3), dtype=torch.long)),
        ("singular_values", torch.tensor(((0.0, 0.0, 10.0 + step), (1.0, 2.0, 3.0)), dtype=torch.float32)),
        ("sigma_min", torch.tensor((0.0, 1.0), dtype=torch.float32)),
        ("condition_number", torch.tensor((0.0, 3.0), dtype=torch.float32)),
        ("condition_valid", torch.tensor((False, True))),
        ("weak_direction_w", torch.zeros((2, 3), dtype=torch.float32)),
        ("weak_direction_valid", torch.zeros(2, dtype=torch.bool)),
        ("valid_ray_count", valid_count),
        ("total_ray_count", total_count),
        ("valid_beam_fraction", valid_count.to(torch.float32) / total_count.to(torch.float32)),
        ("hit_max_error_m", torch.zeros(2, dtype=torch.float32)),
        ("distance_max_error_m", torch.zeros(2, dtype=torch.float32)),
        ("source_step", torch.full((2,), step, dtype=torch.long)),
        ("source_reset_mask", torch.zeros(2, dtype=torch.bool)),
        ("source_identity_sha256", source_identity),
    ))
    return ExactRuntimeObservability(**values, result_identity_sha256=_row_hashes(values))


def _reseal(value: ExactRuntimeObservability, **changes: torch.Tensor) -> ExactRuntimeObservability:
    values = OrderedDict(
        (field.name, changes.get(field.name, getattr(value, field.name)).detach().clone())
        for field in fields(value) if field.name != "result_identity_sha256"
    )
    return ExactRuntimeObservability(**values, result_identity_sha256=_row_hashes(values))


def _pre_reset(step: int, command: torch.Tensor) -> tuple[OrderedDict[str, object], torch.Tensor, torch.Tensor]:
    root = torch.zeros((2, 13), dtype=torch.float32)
    root[:, 0] = torch.tensor((float(step), 10.0 + step))
    root[:, 2] = 1.5
    root[:, 3] = 1.0
    actual = torch.tensor(((0.1 * step, 0.0, 0.0), (0.0, -0.1 * step, 0.0)), dtype=torch.float32)
    norm = torch.linalg.vector_norm(command, dim=-1, keepdim=True)
    issued = command * torch.clamp(2.0 / torch.clamp(norm, min=torch.finfo(torch.float32).eps), max=1.0)
    false = torch.zeros((2, 1), dtype=torch.bool)
    zeros = torch.zeros((2, 1), dtype=torch.float32)
    anchor = OrderedDict((
        ("active", zeros.clone()), ("reset_code", torch.zeros((2, 1), dtype=torch.long)),
        ("usable_fraction", zeros.clone()), ("mean_absolute_error_m", zeros.clone()),
        ("huber_loss", zeros.clone()),
    ))
    ics = OrderedDict((
        ("beta", torch.ones((2, 1), dtype=torch.float32)),
        ("active_beam_count", zeros.clone()), ("minimum_clearance_m", zeros.clone()),
        ("worst_margin_m_s", zeros.clone()), ("emergency", zeros.clone()),
    ))
    control = OrderedDict((
        ("alpha", torch.ones((2, 1), dtype=torch.float32)),
        ("v_corr_b", torch.zeros((2, 3), dtype=torch.float32)),
        ("v_gov_b", issued.clone()), ("v_final_b", issued.clone()),
        ("v_final_w", issued.clone()), ("anchor_metrics", anchor), ("ics_metrics", ics),
    ))
    masks = OrderedDict((
        ("collision", false.clone()), ("below_height", false.clone()),
        ("above_height", false.clone()), ("timeout", false.clone()),
    ))
    termination = OrderedDict((
        ("terminated", false.clone()), ("truncated", false.clone()),
        ("done", false.clone()), ("reason_code", torch.zeros((2, 1), dtype=torch.long)),
        ("masks", masks),
    ))
    clearance = torch.tensor(((3.0 + 0.1 * step,), (4.0 + 0.1 * step,)), dtype=torch.float32)
    task = OrderedDict((
        ("root_state_w", root.clone()), ("actual_velocity_b", actual),
        ("exact_clearance_m", clearance), ("collision", false.clone()),
        ("termination", termination),
    ))
    episode = torch.full((2,), step, dtype=torch.long)
    pre = OrderedDict((
        ("reset_mask", torch.zeros(2, dtype=torch.bool)),
        ("episode_length", episode.clone()), ("root_state_w", root.clone()),
        ("joint_pos", torch.zeros((2, 4), dtype=torch.float32)),
        ("joint_vel", torch.zeros((2, 4), dtype=torch.float32)),
        ("control", control),
        ("lee", OrderedDict((("normalized", torch.zeros((2, 4))),))),
        ("wrench", OrderedDict((("forces", torch.zeros((2, 3))),))),
        ("reward", OrderedDict((("total", torch.zeros((2, 1))),))), ("task", task),
    ))
    critic = torch.cat((root, actual, clearance, false.to(torch.float32)), dim=1)
    return pre, critic, episode


def _collector():
    return begin_ready_b0_runtime_evaluation(_manifest(), _critic(), _config())


def _commit(collector, step: int, values=COMMANDS[0]) -> None:
    original = torch.tensor(values, dtype=torch.float32)
    staged = collector.prepare_command(original)
    pre, critic, episode = _pre_reset(step, staged)
    collector.commit_transition(
        staged_command=staged,
        pre_reset=pre,
        exact_observability=_exact(step),
        returned_critic=critic,
        returned_episode_length=episode,
    )


def _environment_result() -> B0EnvironmentResult:
    receipt = B0RunReceipt(
        "m7.b0-run-receipt.v1", _manifest(ExecutionStatus.READY),
        _manifest(ExecutionStatus.EXECUTED), 160, "a" * 64,
    )
    return B0EnvironmentResult(receipt, B0EnvironmentCoverage(8, 160, 160, 160, 320, 2, 20))


def _fingerprint(collector) -> tuple[object, ...]:
    return (
        collector.state_token, collector.row_count, collector.has_pending_command,
        None if collector._pending_command is None else collector._pending_command.clone(),
        None if collector._pending_snapshot is None else collector._pending_snapshot.clone(),
        len(collector._seen_sources),
    )


def test_exact_factory_config_manifest_and_direct_import_isolation() -> None:
    config = _config()
    assert (config.expected_policy_steps, config.num_envs, config.rays_per_environment) == (8, 2, 21240)
    with pytest.raises(FrozenInstanceError):
        config.num_envs = 3
    invalid = (
        (7, 2, 21240, EvaluationMetricConfig(), config.provenance),
        (8, 3, 21240, EvaluationMetricConfig(), config.provenance),
        (8, 2, 1, EvaluationMetricConfig(), config.provenance),
        (8, 2, 21240, object(), config.provenance),
        (8, 2, 21240, EvaluationMetricConfig(), object()),
    )
    for args in invalid:
        with pytest.raises((TypeError, ValueError)):
            B0RuntimeEvaluationConfig(*args)
    collector = _collector()
    assert collector.row_count == 0 and collector.state_token == "ACTIVE"
    with pytest.raises(TypeError):
        type("Forbidden", (type(collector),), {})
    with pytest.raises(TypeError):
        type(collector)(_manifest(), _critic(), _config(), object())
    with pytest.raises(TypeError, match="initial_critic"):
        begin_ready_b0_runtime_evaluation(_manifest(), [[0.0] * 18] * 2, _config())
    assert tuple(inspect.signature(begin_ready_b0_runtime_evaluation).parameters) == (
        "manifest", "initial_critic", "config",
    )
    import instinctlab.evaluation as aggregate
    assert "B0RuntimeEvaluationCollector" not in aggregate.__all__
    assert not hasattr(aggregate, "B0RuntimeEvaluationCollector")


def test_command_state_machine_owns_exact_staged_tensor_and_commits_once() -> None:
    collector = _collector()
    command = torch.tensor(COMMANDS[1], dtype=torch.float32)
    snapshot = command.clone()
    staged = collector.prepare_command(command)
    assert staged is collector._pending_command
    assert staged is not command and torch.equal(staged, command)
    command.add_(10.0)
    assert torch.equal(staged, snapshot)
    with pytest.raises(RuntimeError, match="already pending"):
        collector.prepare_command(snapshot)
    pre, critic, episode = _pre_reset(1, staged)
    collector.commit_transition(
        staged_command=staged, pre_reset=pre, exact_observability=_exact(1),
        returned_critic=critic, returned_episode_length=episode,
    )
    assert collector.row_count == 1 and not collector.has_pending_command
    with pytest.raises(RuntimeError, match="prepared command"):
        collector.commit_transition(
            staged_command=staged, pre_reset=pre, exact_observability=_exact(1),
            returned_critic=critic, returned_episode_length=episode,
        )


@pytest.mark.parametrize("mutation", ("type", "shape", "dtype", "gradient", "nonfinite", "view", "alias"))
def test_malformed_command_rejects_atomically(mutation: str) -> None:
    collector = _collector()
    command: object = torch.tensor(COMMANDS[0], dtype=torch.float32)
    if mutation == "type":
        command = object()
    elif mutation == "shape":
        command = torch.zeros((2, 4), dtype=torch.float32)
    elif mutation == "dtype":
        command = torch.zeros((2, 3), dtype=torch.float64)
    elif mutation == "gradient":
        command = command.requires_grad_()
    elif mutation == "nonfinite":
        command[0, 0] = float("nan")
    elif mutation == "view":
        command = torch.zeros((2, 4), dtype=torch.float32)[:, :3]
    elif mutation == "alias":
        command = collector._station_reference
    before = _fingerprint(collector)
    with pytest.raises((TypeError, ValueError)):
        collector.prepare_command(command)
    assert _fingerprint(collector) == before


@pytest.mark.parametrize("mutation", (
    "wrong_staged", "mutated_staged", "missing", "extra", "reordered", "malformed_control",
    "control_reordered", "task_reordered", "termination_reordered", "reset", "done",
    "episode", "root_mismatch", "critic", "alpha", "correction", "issued", "anchor",
    "beta", "emergency", "clearance", "collision", "alias", "observability_type",
    "observability_stale", "observability_reset", "observability_duplicate",
))
def test_row_negative_matrix_preserves_pending_and_rows(mutation: str) -> None:
    collector = _collector()
    if mutation == "observability_duplicate":
        _commit(collector, 1, COMMANDS[0])
        staged = collector.prepare_command(torch.tensor(COMMANDS[1], dtype=torch.float32))
        pre, critic, episode = _pre_reset(2, staged)
        exact = _reseal(_exact(2), source_identity_sha256=collector._source_identities[0].clone())
        before_pending = collector._pending_snapshot.clone()
        with pytest.raises(ValueError, match="duplicated"):
            collector.commit_transition(
                staged_command=staged, pre_reset=pre, exact_observability=exact,
                returned_critic=critic, returned_episode_length=episode,
            )
        assert collector.row_count == 1 and collector.has_pending_command
        assert torch.equal(collector._pending_snapshot, before_pending)
        return
    staged = collector.prepare_command(torch.tensor(COMMANDS[1], dtype=torch.float32))
    pre, critic, episode = _pre_reset(1, staged)
    exact: object = _exact(1)
    supplied = staged
    if mutation == "wrong_staged":
        supplied = staged.clone()
    elif mutation == "mutated_staged":
        staged[0, 0] += 1.0
    elif mutation == "missing":
        pre.pop("reward")
    elif mutation == "extra":
        pre["unexpected"] = object()
    elif mutation == "reordered":
        pre.move_to_end("reset_mask")
    elif mutation == "malformed_control":
        pre["control"] = torch.zeros(2)
    elif mutation == "control_reordered":
        pre["control"].move_to_end("alpha")
    elif mutation == "task_reordered":
        pre["task"].move_to_end("root_state_w")
    elif mutation == "termination_reordered":
        pre["task"]["termination"].move_to_end("terminated")
    elif mutation == "reset":
        pre["reset_mask"][0] = True
    elif mutation == "done":
        pre["task"]["termination"]["done"][0, 0] = True
    elif mutation == "episode":
        episode[0] = 2
    elif mutation == "root_mismatch":
        pre["task"]["root_state_w"][0, 0] += 1.0
    elif mutation == "critic":
        critic[0, 0] += 1.0
    elif mutation == "alpha":
        pre["control"]["alpha"][0, 0] = 0.5
    elif mutation == "correction":
        pre["control"]["v_corr_b"][0, 0] = 0.1
    elif mutation == "issued":
        pre["control"]["v_final_b"][0, 0] += 0.1
    elif mutation == "anchor":
        pre["control"]["anchor_metrics"]["active"][0, 0] = 1.0
    elif mutation == "beta":
        pre["control"]["ics_metrics"]["beta"][0, 0] = 0.5
    elif mutation == "emergency":
        pre["control"]["ics_metrics"]["emergency"][0, 0] = 1.0
    elif mutation == "clearance":
        pre["task"]["exact_clearance_m"][0, 0] = -1.0
    elif mutation == "collision":
        pre["task"]["collision"][0, 0] = True
    elif mutation == "alias":
        pre["control"]["v_final_b"] = pre["control"]["v_gov_b"]
    elif mutation == "observability_type":
        exact = object()
    elif mutation == "observability_stale":
        exact = _reseal(exact, source_step=torch.zeros(2, dtype=torch.long))
    elif mutation == "observability_reset":
        exact = _reseal(exact, source_reset_mask=torch.tensor((True, False)))
    before_rows = collector.row_count
    before_pending = collector._pending_snapshot.clone()
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        collector.commit_transition(
            staged_command=supplied, pre_reset=pre, exact_observability=exact,
            returned_critic=critic, returned_episode_length=episode,
        )
    assert collector.row_count == before_rows
    assert collector.has_pending_command and torch.equal(collector._pending_snapshot, before_pending)


def test_every_evaluation_input_metric_and_canonical_field_has_exact_oracle() -> None:
    collector = _collector()
    for step, command in enumerate(COMMANDS, 1):
        _commit(collector, step, command)
    result = collector.finalize(_environment_result())
    values = OrderedDict(result.evaluation_inputs)
    details = OrderedDict(result.observability_details)
    assert tuple(values) == EVALUATION_INPUT_KEYS
    assert tuple(details) == OBSERVABILITY_DETAIL_KEYS
    assert values["commanded_velocity_b_m_s"].shape == (8, 2, 3)
    assert torch.equal(values["commanded_velocity_b_m_s"], torch.tensor(COMMANDS, dtype=torch.float32))
    assert values["actual_velocity_b_m_s"][:, 0, 0].tolist() == pytest.approx([0.1 * step for step in range(1, 9)])
    assert torch.equal(values["station_reference_position_w_m"], _critic()[:, :3].expand(8, -1, -1))
    assert not values["anchor_valid"].any() and values["clearance_valid"].all()
    assert torch.equal(values["ics_beta"], torch.ones((8, 2)))
    assert not values["ics_emergency"].any() and not values["terminated"].any()
    assert values["observability_source"].eq(int(ObservabilitySource.EXACT)).all()
    assert details["source_step"][:, 0].tolist() == list(range(1, 9))
    assert tuple(name for name, _ in B0_SOURCE_MANIFEST) == EVALUATION_INPUT_KEYS
    assert all(tuple(name for name, _ in metadata) == SOURCE_METADATA_KEYS for _, metadata in B0_SOURCE_MANIFEST)
    oracle_step = compute_step_metrics(values, EvaluationMetricConfig())
    oracle_aggregate = aggregate_metrics(oracle_step)
    assert result.aggregate_metrics == oracle_aggregate
    record = result.record.to_mapping()
    assert tuple(record) == RECORD_KEYS
    assert record["schema_version"] == "m7.b0-runtime-evaluation.v1"
    assert record["actor_surface"] == {
        "kind": "command-only", "input_fields": ["commanded_body_velocity_m_s"],
        "learned_actor_called": False, "normalized_action_present": False,
    }
    assert record["coverage"] == {
        "num_envs": 2, "policy_steps": 8, "metric_samples": 16,
        "physics_steps": 160, "lifecycle_calls": 160, "wrench_applications": 160,
        "environment_samples": 320, "rays_per_environment": 21240,
        "ray_samples": 339840,
    }
    assert record["observability_source_counts"] == {"UNAVAILABLE": 0, "PROXY": 0, "EXACT": 16}
    assert record["termination_counts"]["done"] == 0
    assert record["termination_counts"]["reason_none"] == 16
    assert json.dumps(record, ensure_ascii=True, allow_nan=False, separators=(",", ":")) == result.record.canonical_json
    with pytest.raises(FrozenInstanceError):
        result.schema_version = "changed"


def test_finalize_exact_once_receipt_coverage_ownership_retention_and_cache_clear() -> None:
    collector = _collector()
    with pytest.raises(RuntimeError, match="coverage"):
        collector.finalize(_environment_result())
    for step, command in enumerate(COMMANDS, 1):
        _commit(collector, step, command)
    original = _environment_result()
    result = collector.finalize(original)
    assert collector.state_token == "FINALIZED"
    assert collector._station_reference is None and not collector._rows and not collector._details
    assert not collector._seen_sources and not collector.has_pending_command
    assert result.ready_manifest_sha256 == original.receipt.ready_manifest.manifest_sha256
    assert result.executed_manifest_sha256 == original.receipt.executed_manifest.manifest_sha256
    assert result.trajectory_sha256 == "a" * 64
    assert result.command_identity_sha256.shape == (8, 2, 32)
    assert result.source_identity_sha256.shape == (8, 2, 32)
    public_fields = {field.name for field in fields(result)}
    assert not public_fields.intersection({
        "raw_rays", "ray_starts_w", "ray_directions_w", "surface_normals_w",
        "raw_surface_normals_w", "mesh", "actor_private_lidar", "history",
    })
    pointers = [value.untyped_storage().data_ptr() for _, value in result.evaluation_inputs]
    pointers += [value.untyped_storage().data_ptr() for _, value in result.observability_details]
    assert len(pointers) == len(set(pointers))
    with pytest.raises(RuntimeError, match="exactly once"):
        collector.finalize(original)
    with pytest.raises(RuntimeError, match="finalized"):
        collector.prepare_command(torch.zeros((2, 3), dtype=torch.float32))


@pytest.mark.parametrize("mutation", ("type", "ready", "coverage", "count", "successor"))
def test_finalize_forgery_matrix_emits_no_record(mutation: str) -> None:
    collector = _collector()
    for step, command in enumerate(COMMANDS, 1):
        _commit(collector, step, command)
    result: object = _environment_result()
    if mutation == "type":
        result = object()
    elif mutation == "ready":
        other = _manifest(ExecutionStatus.READY, "different")
        result = B0EnvironmentResult(
            B0RunReceipt(
                "m7.b0-run-receipt.v1", other,
                _manifest(ExecutionStatus.EXECUTED, "different"), 160, "a" * 64,
            ),
            result.coverage,
        )
    elif mutation == "coverage":
        result = B0EnvironmentResult(result.receipt, B0EnvironmentCoverage(7, 160, 160, 160, 320, 2, 20))
    elif mutation == "count":
        result = B0EnvironmentResult(
            B0RunReceipt("m7.b0-run-receipt.v1", _manifest(), _manifest(ExecutionStatus.EXECUTED), 159, "a" * 64),
            result.coverage,
        )
    elif mutation == "successor":
        other_executed = build_run_manifest(
            product_commit=PRODUCT_COMMIT, baseline_id="B0", config_switch="direct_velocity",
            capability_bindings=_bindings(), scenario_id="different", seed=SEED,
            actor_audit_status=ActorAuditStatus.PASS,
            execution_status=ExecutionStatus.EXECUTED, skip_reason=None,
        )
        with pytest.raises(ValueError):
            B0RunReceipt("m7.b0-run-receipt.v1", _manifest(), other_executed, 160, "a" * 64)
        return
    with pytest.raises((TypeError, ValueError)):
        collector.finalize(result)
    assert collector.state_token == "ACTIVE" and collector.row_count == 8


def test_source_privacy_imports_and_no_learned_or_actuation_path() -> None:
    source = SOURCE.read_text()
    tree = ast.parse(source)
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module, tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    runtime_imports = [names for module, names in imports if module == "runtime"]
    assert runtime_imports == [("ExactRuntimeObservability",)]
    assert not any(token in source for token in (
        "HistoryActor", "build_evaluation_record", "governor_action", "checkpoint", "optimizer",
        "acquire_exact_observability_from_source", "raycast_mesh",
        "ControlCorePipeline", "taslab_velocity_control", "wrench_adapter",
    ))
    assert not any(call.endswith("step") for call in calls)
    requirements = resolve_baseline("B0").capability_requirements
    assert len(requirements) == 8
    assert tuple(item.name for item in requirements[-2:]) == (
        "m7.b0_taslab_environment", "m7.b0_runtime_evaluation",
    )
    assert requirements[-1].accepted_provider == (
        "instinctlab.evaluation.baseline_runtime.B0RuntimeEvaluationCollector"
    )
    assert resolve_baseline("B0").actor_input_fields == ("commanded_body_velocity_m_s",)
    assert resolve_baseline("B1").readiness.value == "NOT_READY"
    assert resolve_baseline("B8").readiness.value == "NOT_READY"


def test_cuda_runner_and_comparator_are_bounded_complete_and_fail_closed() -> None:
    runner = (ROOT / "tests/run_m7_b0_runtime_evaluation_smoke.py").read_text()
    comparator = (ROOT / "tests/compare_m7_b0_runtime_evaluation_smoke.py").read_text()
    assert all(token in runner for token in (
        "NUM_ENVS = 2", "POLICY_STEPS = 8", "PHYSICS_STEPS = 160",
        "METRIC_SAMPLES = 16", "RAY_SAMPLES = 339840", "COMMAND_SCHEDULE",
        "prepare_command", "commit_transition", "compute_step_metrics",
        "aggregate_metrics", "torch.save", "actor_calls", "training_calls",
    ))
    assert "HistoryActor" not in runner and "RuntimeEvaluationCollector" not in runner
    assert all(token in comparator for token in (
        "weights_only=True", "type(left) is not type(right)",
        "list(left) != list(right)", "left.shape != right.shape",
        "left.dtype != right.dtype", "torch.equal(left, right)",
        "missing, extra, or reordered keys",
    ))
    assert "allclose" not in comparator and "isclose" not in comparator
