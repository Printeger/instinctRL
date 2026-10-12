"""Strict independent validator for one EVAL-002 semantic tree and mirror."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Mapping
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from attempt_binding import AttemptBinding, binding_from_environment, bound_path, path_from_environment
from constants import (
    ACTION_MODE, ACTOR_SHA256, CHECKPOINT_SHA256, DECIMATION,
    ENVIRONMENT_SEED, ENVIRONMENT_TRANSITIONS, INFERENCE_SEED, NUM_ENVS,
    PHYSICS_STEPS_PER_ENVIRONMENT, POLICY_STEPS, PRODUCT_COMMIT, PRODUCT_TREE,
    RAYS_PER_ENVIRONMENT_PER_STEP, RAY_SAMPLES, RUN_IDS, SOURCE_SHA256,
    TOP_LEVEL_KEYS,
)
from io_utils import atomic_canonical_replace, canonical_bytes, canonical_value, finite_tree, sha256_file


KIT_SHUTDOWN_MARKERS = (
    b"SimulationApp.close: Closing application",
    b"SimulationApp.close: Replicator shutdown finished",
    b"SimulationApp.close: Stage closed",
    b"Simulation App Shutting Down",
    b"SimulationApp.close: shutting down app and releasing framework",
)

CANONICAL_RECORD_KEYS = (
    "schema_version", "platform", "sensor", "claim", "actor_audit",
    "input_dimensions", "evaluation_input_order", "metric_config",
    "observability_source_counts", "metrics", "termination_counts", "provenance",
)
ACTOR_AUDIT_KEYS = (
    "status", "keys", "lidar_grid_shape", "state_vec_shape", "dtype", "device",
    "lidar_grid_sha256", "state_vec_sha256",
)
METRIC_FIELDS = (
    "reduction", "unit", "available", "value", "numerator", "denominator",
    "valid_count", "total_count",
)
METRIC_SPECS = OrderedDict((
    ("tracking_rmse_m_s", ("root_mean_square", "m/s")),
    ("null_station_drift_mean_m", ("mean", "m")),
    ("null_station_speed_mean_m_s", ("mean", "m/s")),
    ("anchor_error_mean_m", ("mean", "m")),
    ("minimum_clearance_m", ("minimum", "m")),
    ("collision_rate", ("mean", "fraction")),
    ("ics_violation_rate", ("mean", "fraction")),
    ("command_preservation_ratio", ("mean", "ratio")),
    ("intervention_frequency", ("mean", "fraction")),
    ("observability_mean", ("mean", "source-defined")),
))
TERMINATION_COUNT_KEYS = (
    "collision", "below_height", "above_height", "timeout", "terminated",
    "truncated", "done", "simultaneous_terminated_truncated", "reason_none",
    "reason_timeout", "reason_above_height", "reason_below_height",
    "reason_collision",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class EvidenceValidationError(ValueError):
    """A malformed or semantically unequal external evidence mirror."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> OrderedDict[str, Any]:
    value: OrderedDict[str, Any] = OrderedDict()
    for key, child in pairs:
        if key in value:
            raise EvidenceValidationError(f"duplicate JSON object key: {key}")
        value[key] = child
    return value


def _reject_nonstandard_constant(token: str) -> None:
    raise EvidenceValidationError(f"non-standard JSON constant: {token}")


def _canonical_difference_count(expected: Any, actual: Any) -> int:
    """Compare canonical JSON values exactly while ignoring mapping order."""

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        differences = len(expected_keys ^ actual_keys)
        for key in expected_keys & actual_keys:
            differences += _canonical_difference_count(expected[key], actual[key])
        return differences
    if type(expected) is not type(actual):
        return 1
    if isinstance(expected, list):
        differences = abs(len(expected) - len(actual))
        differences += sum(
            _canonical_difference_count(expected_child, actual_child)
            for expected_child, actual_child in zip(expected, actual)
        )
        return differences
    return int(expected != actual)


def validate_semantic_mirror(value: OrderedDict[str, Any], path: Path) -> OrderedDict[str, Any]:
    """Load and exactly compare a JSON mirror without object-order sensitivity."""

    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise EvidenceValidationError(
            "semantic mirror path must be an absolute regular non-symlink file"
        )
    try:
        raw = path.read_bytes()
        mirror = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except EvidenceValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"invalid semantic JSON mirror: {error}") from error
    if not isinstance(mirror, OrderedDict):
        raise EvidenceValidationError("semantic JSON mirror root must be an object")
    expected = canonical_value(value)
    try:
        stored_canonical = canonical_bytes(mirror)
        expected_canonical = canonical_bytes(expected)
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError(f"invalid semantic JSON mirror value: {error}") from error
    differences = _canonical_difference_count(expected, mirror)
    if differences != 0 or stored_canonical != expected_canonical:
        raise EvidenceValidationError(
            "canonical JSON mirror does not match semantic PT "
            f"({differences} value differences)"
        )
    return OrderedDict((
        ("mirror", mirror),
        ("pt_root_key_order", list(expected)),
        ("mirror_root_key_order", list(mirror)),
        ("root_insertion_order_differs", tuple(expected) != tuple(mirror)),
        ("canonical_bytes_equal", True),
        ("semantic_value_difference_count", 0),
    ))


def failure_result(error: BaseException) -> dict[str, Any]:
    """Classify mirror-evidence failures separately from semantic failures."""

    failure_class = (
        "EVIDENCE_VALIDATION_FAILURE"
        if isinstance(error, EvidenceValidationError)
        else "PRODUCT_SEMANTIC_FAILURE"
    )
    return {
        "failure_class": failure_class,
        "message": f"{type(error).__name__}: {error}",
        "semantic_exit": 1,
        "status": "FAIL",
    }


def load_semantic(path: Path) -> OrderedDict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("semantic path must be an absolute regular non-symlink file")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, OrderedDict):
        raise ValueError("semantic root must be OrderedDict")
    return value


def _exact_keys(value: Any, expected: tuple[str, ...], name: str) -> None:
    if not isinstance(value, OrderedDict) or tuple(value) != expected:
        raise ValueError(f"{name} missing, extra, or reordered field")


def _exact_key_set(value: Any, expected: tuple[str, ...], name: str) -> OrderedDict[str, Any]:
    """Require a JSON object schema without assigning meaning to object order."""

    if not isinstance(value, OrderedDict) or set(value) != set(expected):
        raise ValueError(f"{name} missing or schema-unknown field")
    return value


def _decode_canonical_record(canonical_record: Any) -> OrderedDict[str, Any]:
    """Decode the nested record with the same fail-closed JSON rules as evidence files."""

    if type(canonical_record) is not str or not canonical_record:
        raise TypeError("canonical record must be one nonempty string")
    if any(ord(character) > 127 for character in canonical_record):
        raise ValueError("canonical record must contain ASCII only")
    try:
        record = json.loads(
            canonical_record,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except EvidenceValidationError as error:
        raise ValueError(f"invalid canonical record: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid canonical record JSON: {error}") from error
    if not isinstance(record, OrderedDict):
        raise ValueError("canonical record root must be an object")
    encoded = json.dumps(
        record, ensure_ascii=True, allow_nan=False, separators=(",", ":")
    )
    if encoded != canonical_record:
        raise ValueError("canonical record does not use its exact canonical encoding")
    return record


def _expected_termination_counts(inputs: Mapping[str, torch.Tensor]) -> OrderedDict[str, int]:
    from instinctlab.task.termination import (
        TERMINATION_ABOVE_HEIGHT, TERMINATION_BELOW_HEIGHT, TERMINATION_COLLISION,
        TERMINATION_NONE, TERMINATION_TIMEOUT,
    )

    terminated = inputs["terminated"]
    truncated = inputs["truncated"]
    reason = inputs["termination_reason"]
    return OrderedDict((
        ("collision", int(inputs["termination_collision"].sum().item())),
        ("below_height", int(inputs["termination_below_height"].sum().item())),
        ("above_height", int(inputs["termination_above_height"].sum().item())),
        ("timeout", int(inputs["termination_timeout"].sum().item())),
        ("terminated", int(terminated.sum().item())),
        ("truncated", int(truncated.sum().item())),
        ("done", int((terminated | truncated).sum().item())),
        ("simultaneous_terminated_truncated", int((terminated & truncated).sum().item())),
        ("reason_none", int((reason == TERMINATION_NONE).sum().item())),
        ("reason_timeout", int((reason == TERMINATION_TIMEOUT).sum().item())),
        ("reason_above_height", int((reason == TERMINATION_ABOVE_HEIGHT).sum().item())),
        ("reason_below_height", int((reason == TERMINATION_BELOW_HEIGHT).sum().item())),
        ("reason_collision", int((reason == TERMINATION_COLLISION).sum().item())),
    ))


def _validate_canonical_record(
    canonical_record: Any,
    inputs: Mapping[str, torch.Tensor],
    run_id: str | None,
) -> OrderedDict[str, Any]:
    """Validate the fixed record schema and its own aggregation facts.

    GPU numerator/value fields are checked for exact internal arithmetic. Counts and
    availability are replayed from the saved inputs, while CPU/GPU reductions are
    deliberately not compared through an unapproved tolerance.
    """

    from instinctlab.evaluation.metrics import (
        EVALUATION_INPUT_KEYS, EvaluationMetricConfig, ObservabilitySource,
        aggregate_metrics, compute_step_metrics,
    )

    record = _exact_key_set(
        _decode_canonical_record(canonical_record), CANONICAL_RECORD_KEYS,
        "canonical record",
    )
    if (
        record["schema_version"] != "m7.evaluation.v1"
        or record["platform"] != "TASLAB_UAV"
        or record["sensor"] != "Livox MID360"
        or record["claim"]
        != "pure-tensor contract record; no simulator evaluation or policy-performance claim"
    ):
        raise ValueError("canonical record identity or claim mismatch")

    actor = _exact_key_set(record["actor_audit"], ACTOR_AUDIT_KEYS, "actor audit")
    if (
        actor["status"] != "PASS"
        or actor["keys"] != ["lidar_grid", "state_vec"]
        or actor["lidar_grid_shape"] != [NUM_ENVS, 12, 59, 360]
        or actor["state_vec_shape"] != [NUM_ENVS, 52]
        or actor["dtype"] != "torch.float32"
        or type(actor["device"]) is not str
        or not actor["device"]
        or any(
            type(actor[name]) is not str or SHA256_PATTERN.fullmatch(actor[name]) is None
            for name in ("lidar_grid_sha256", "state_vec_sha256")
        )
    ):
        raise ValueError("canonical actor audit mismatch")

    dimensions = _exact_key_set(
        record["input_dimensions"], ("time_steps", "environments", "samples"),
        "input dimensions",
    )
    if dimensions != {"time_steps": POLICY_STEPS, "environments": NUM_ENVS, "samples": ENVIRONMENT_TRANSITIONS}:
        raise ValueError("canonical input dimensions mismatch")
    if record["evaluation_input_order"] != list(EVALUATION_INPUT_KEYS):
        raise ValueError("canonical evaluation input order mismatch")

    config = EvaluationMetricConfig()
    metric_config = _exact_key_set(
        record["metric_config"],
        ("schema_version", "null_command_threshold_m_s", "full_compliance_beta"),
        "metric config",
    )
    if metric_config != {
        "schema_version": config.schema_version,
        "null_command_threshold_m_s": config.null_command_threshold_m_s,
        "full_compliance_beta": config.full_compliance_beta,
    }:
        raise ValueError("canonical metric config mismatch")

    source_counts = _exact_key_set(
        record["observability_source_counts"],
        ("UNAVAILABLE", "PROXY", "EXACT"), "observability source counts",
    )
    source = inputs["observability_source"]
    expected_sources = OrderedDict(
        (label.name, int((source == int(label)).sum().item()))
        for label in (
            ObservabilitySource.UNAVAILABLE, ObservabilitySource.PROXY,
            ObservabilitySource.EXACT,
        )
    )
    if source_counts != expected_sources:
        raise ValueError("canonical observability source counts mismatch")

    aggregate = aggregate_metrics(compute_step_metrics(inputs, config))
    expected_metrics = {metric.name: metric for metric in aggregate.metrics}
    metrics = _exact_key_set(record["metrics"], tuple(METRIC_SPECS), "canonical metrics")
    for name, (reduction, unit) in METRIC_SPECS.items():
        metric = _exact_key_set(metrics[name], METRIC_FIELDS, f"metric {name}")
        expected = expected_metrics[name]
        if metric["reduction"] != reduction or metric["unit"] != unit:
            raise ValueError(f"metric identity mismatch: {name}")
        for count_name in ("denominator", "valid_count", "total_count"):
            if type(metric[count_name]) is not int:
                raise TypeError(f"metric count type mismatch: {name}.{count_name}")
        if (
            type(metric["available"]) is not bool
            or metric["available"] != expected.available
            or metric["denominator"] != expected.denominator
            or metric["valid_count"] != expected.valid_count
            or metric["total_count"] != expected.total_count
        ):
            raise ValueError(f"metric availability/count semantics mismatch: {name}")
        if not metric["available"]:
            if (metric["value"], metric["numerator"], metric["denominator"], metric["valid_count"]) != (None, None, 0, 0):
                raise ValueError(f"unavailable metric representation mismatch: {name}")
            continue
        if (
            type(metric["value"]) is not float
            or type(metric["numerator"]) is not float
            or not math.isfinite(metric["value"])
            or not math.isfinite(metric["numerator"])
            or metric["valid_count"] <= 0
        ):
            raise ValueError(f"available metric must use finite float values: {name}")
        if reduction == "minimum":
            consistent = metric["denominator"] == 1 and metric["value"] == metric["numerator"]
        elif reduction == "root_mean_square":
            consistent = (
                metric["denominator"] == metric["valid_count"]
                and metric["value"]
                == math.sqrt(metric["numerator"] / metric["denominator"])
            )
        else:
            consistent = (
                metric["denominator"] == metric["valid_count"]
                and metric["value"] == metric["numerator"] / metric["denominator"]
            )
        if not consistent:
            raise ValueError(f"metric aggregation facts mismatch: {name}")

    termination = _exact_key_set(
        record["termination_counts"], TERMINATION_COUNT_KEYS, "termination counts"
    )
    if termination != _expected_termination_counts(inputs):
        raise ValueError("canonical termination counts mismatch")

    provenance = _exact_key_set(
        record["provenance"], ("product_commit", "fixture_id", "generator_id", "seed"),
        "canonical provenance",
    )
    if provenance != {
        "product_commit": PRODUCT_COMMIT,
        "fixture_id": f"eval-002-floor-corner-{run_id}",
        "generator_id": "evaluation_workloads/eval_002/diagnostic_evaluator.py",
        "seed": ENVIRONMENT_SEED,
    }:
        raise ValueError("canonical provenance mismatch")
    return record


def validate_semantic(value: OrderedDict[str, Any], *, run_id: str | None = None) -> None:
    if run_id not in RUN_IDS:
        raise ValueError("unknown run id")
    _exact_keys(value, TOP_LEVEL_KEYS, "semantic root")
    if value["schema_version"] != "eval-002-formal-semantic.v2":
        raise ValueError("semantic schema mismatch")
    scope = value["scope"]
    expected_scope = OrderedDict((
        ("environment", "TaslabDirectEnv combined floor and corner"),
        ("case", run_id),
        ("performance_gate", False),
    ))
    if scope != expected_scope or tuple(scope) != tuple(expected_scope):
        raise ValueError("scope or performance claim mismatch")
    identity = value["identity"]
    _exact_keys(identity, ("product_commit", "product_tree", "checkpoint_sha256", "checkpoint_counter", "source_sha256"), "identity")
    if identity != OrderedDict((
        ("product_commit", PRODUCT_COMMIT), ("product_tree", PRODUCT_TREE),
        ("checkpoint_sha256", CHECKPOINT_SHA256), ("checkpoint_counter", 250),
        ("source_sha256", SOURCE_SHA256),
    )):
        raise ValueError("product/checkpoint/source identity mismatch")
    campaign = value["campaign"]
    campaign_keys = (
        "action_mode", "environment_seed", "inference_seed", "num_envs",
        "policy_steps", "decimation", "physics_steps_per_environment",
        "environment_transitions", "rays_per_environment_per_step", "ray_samples",
    )
    _exact_keys(campaign, campaign_keys, "campaign")
    expected_campaign = OrderedDict((
        ("action_mode", ACTION_MODE), ("environment_seed", ENVIRONMENT_SEED),
        ("inference_seed", INFERENCE_SEED), ("num_envs", NUM_ENVS),
        ("policy_steps", POLICY_STEPS), ("decimation", DECIMATION),
        ("physics_steps_per_environment", PHYSICS_STEPS_PER_ENVIRONMENT),
        ("environment_transitions", ENVIRONMENT_TRANSITIONS),
        ("rays_per_environment_per_step", RAYS_PER_ENVIRONMENT_PER_STEP),
        ("ray_samples", RAY_SAMPLES),
    ))
    if campaign != expected_campaign:
        raise ValueError("action/seed/dimension/count mismatch")
    before = value["checkpoint_state_before"]
    after = value["checkpoint_state_after"]
    if before != after or before["actor"]["sha256"] != ACTOR_SHA256:
        raise ValueError("actor hash or checkpoint state mutation")
    for module_name in ("actor", "critic"):
        _exact_keys(before[module_name], ("sha256", "parameter_count", "all_finite"), module_name)
        if before[module_name]["all_finite"] is not True:
            raise ValueError(f"{module_name} state is nonfinite")
    for optimizer_name in ("actor_optimizer", "critic_optimizer"):
        _exact_keys(before[optimizer_name], ("sha256", "state_count", "step_min", "step_max", "all_steps_equal"), optimizer_name)
        optimizer = before[optimizer_name]
        if optimizer["step_min"] != 2000 or optimizer["step_max"] != 2000 or optimizer["all_steps_equal"] is not True:
            raise ValueError("optimizer step/state mismatch")
    gradients = OrderedDict((
        ("actor_grad_tensor_count", 0), ("critic_grad_tensor_count", 0),
        ("global_gradient_enabled", False),
    ))
    if value["gradients"] != gradients or tuple(value["gradients"]) != tuple(gradients):
        raise ValueError("gradient claim mismatch")
    claims = OrderedDict((
        ("training_performed", False), ("optimizer_updates", 0),
        ("checkpoint_writes", 0), ("one_million_training", False),
        ("merge", False), ("push", False),
    ))
    if value["training_claims"] != claims or tuple(value["training_claims"]) != tuple(claims):
        raise ValueError("training/update/write/1M/merge/push claim mismatch")
    boundary = OrderedDict((
        ("actor_keys", ["lidar_grid", "state_vec"]),
        ("privileged_actor_input", False),
        ("action_source", "actor(policy).mean_action"),
        ("critic_influenced_action", False),
        ("evaluation_influenced_action", False),
        ("action_passed_unchanged", True),
    ))
    if value["actor_boundary"] != boundary or tuple(value["actor_boundary"]) != tuple(boundary):
        raise ValueError("actor boundary or action source violation")
    runtime = value["runtime_evaluation"]
    expected_contract = OrderedDict((
        ("num_envs", NUM_ENVS), ("policy_steps", POLICY_STEPS),
        ("samples", ENVIRONMENT_TRANSITIONS),
        ("rays_per_environment", RAYS_PER_ENVIRONMENT_PER_STEP),
        ("ray_samples", RAY_SAMPLES),
    ))
    if runtime["contract"] != expected_contract or tuple(runtime["contract"]) != tuple(expected_contract):
        raise ValueError("runtime coverage mismatch")
    if len(runtime["policy_hashes"]) != POLICY_STEPS or len(runtime["action_hashes"]) != POLICY_STEPS:
        raise ValueError("runtime hash coverage mismatch")
    _validate_canonical_record(runtime["canonical_record"], runtime["evaluation_inputs"], run_id)
    for name in (
        "action_diagnostics", "alpha_diagnostics", "beta_diagnostics",
        "entropy_diagnostics", "log_probability_diagnostics", "reward_diagnostics",
    ):
        diagnostic = value[name]
        expected_keys = ("name", "shape", "dtype", "count", "sum", "square_sum", "minimum", "maximum", "mean", "sha256")
        _exact_keys(diagnostic, expected_keys, name)
    action = value["action_diagnostics"]
    if action["shape"] != [POLICY_STEPS, NUM_ENVS, 4] or action["dtype"] != "torch.float32" or action["count"] != POLICY_STEPS * NUM_ENVS * 4:
        raise ValueError("action diagnostic shape/dtype/count mismatch")
    if action["minimum"] < 0.0 or action["maximum"] > 1.0:
        raise ValueError("action diagnostic range mismatch")
    counts = value["termination_reason_counts"]
    if not isinstance(counts, torch.Tensor) or counts.dtype != torch.long or counts.shape != (5,) or int(counts.sum().item()) != ENVIRONMENT_TRANSITIONS:
        raise ValueError("termination reason count mismatch")
    if not isinstance(value["reset_count"], int) or not 0 <= value["reset_count"] <= ENVIRONMENT_TRANSITIONS:
        raise ValueError("reset count mismatch")
    for phase in ("before", "after"):
        resources = value["cuda_resources"][phase]
        if resources["free_memory_bytes"] < 1024 ** 3:
            raise ValueError("less than 1 GiB CUDA memory remained")
    validate_diagnostic(value, run_id)
    finite_tree(value)


def validate_diagnostic(value, run_id):
    from scene import command_at, MESH_PATH, validate_geometry
    from summary import summarize
    from diagnostic_evaluator import tensor_hash
    import diagnostic_evaluator
    diagnostic_evaluator.torch = torch
    from instinctlab.evaluation.metrics import validate_evaluation_inputs
    from instinctlab.evaluation.runtime import SOURCE_MANIFEST
    geometry = value['geometry']
    if geometry['mesh_path'] != MESH_PATH:
        raise ValueError('wrong selected mesh')
    validate_geometry(geometry['points'],geometry['indices'])
    import hashlib
    original = {k:v for k,v in geometry.items() if k!='sha256'}
    if hashlib.sha256(json.dumps(original,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=geometry['sha256']:
        raise ValueError('geometry digest mismatch')
    initial=value['initial_observability']
    if initial['numerical_rank'] != [3,3] or set(initial['surface_hits']) != {'floor','x_wall','y_wall'}:
        raise ValueError('FIXTURE_NOT_OBSERVABLE')
    if any(type(x) is not int or x<=0 for counts in initial['surface_hits'].values() for x in counts):
        raise ValueError('initial surface visibility mismatch')
    if len(initial['sigma_min']) != 2 or any(not isinstance(x,float) or x<=0 for x in initial['sigma_min']):
        raise ValueError('initial sigma mismatch')
    runtime=value['runtime_evaluation']
    inputs=runtime['evaluation_inputs']
    if validate_evaluation_inputs(inputs)!=(POLICY_STEPS,NUM_ENVS) or runtime['source_manifest']!=SOURCE_MANIFEST:
        raise ValueError('runtime schema/provenance mismatch')
    if len(value['rows'])!=POLICY_STEPS or len(value['resources_per_step'])!=POLICY_STEPS:
        raise ValueError('trace/resource coverage mismatch')
    for index,row in enumerate(value['rows']):
        expected=torch.tensor(command_at(run_id,index),dtype=torch.float32).repeat(NUM_ENVS,1)
        if row['global_index']!=index or not torch.equal(row['command_b'],expected) or not torch.equal(row['state_vec'][:,-7:-4],expected) or not torch.equal(inputs['commanded_velocity_b_m_s'][index],expected):
            raise ValueError('command/action schedule alignment mismatch')
        next_command=torch.tensor(command_at(run_id,index+1),dtype=torch.float32).repeat(NUM_ENVS,1)
        if row['state_vec'].shape!=(2,52) or row['returned_state_vec'].shape!=(2,52) or not torch.equal(row['returned_state_vec'][:,-7:-4],next_command):
            raise ValueError('returned observation block boundary mismatch')
        if index+1<POLICY_STEPS and not torch.equal(row['returned_state_vec'],value['rows'][index+1]['state_vec']):
            raise ValueError('stale returned observation')
        action=row['action']
        if action.shape!=(2,4) or action.dtype!=torch.float32 or bool(((action<0)|(action>1)).any()) or tensor_hash(action)!=runtime['action_hashes'][index]:
            raise ValueError('action trace/hash mismatch')
        pre=row['pre_reset']; task=pre['task']; control=pre['control']
        for name,source in [('actual_velocity_b_m_s',task['actual_velocity_b']),('issued_velocity_b_m_s',control['v_final_b']),('position_w_m',task['root_state_w'][:,:3]),('minimum_clearance_m',task['exact_clearance_m'][:,0]),('collision',task['collision'][:,0]),('termination_reason',task['termination']['reason_code'][:,0])]:
            if not torch.equal(inputs[name][index],source):
                raise ValueError('pre-reset trace mismatch: '+name)
        if not torch.equal(pre['reset_mask'],runtime['reset_masks'][index]) or not torch.equal(row['returned_episode_length'],runtime['returned_episode_lengths'][index]):
            raise ValueError('reset trace mismatch')
        expected_episode=torch.where(pre['reset_mask'],torch.zeros_like(pre['episode_length']),pre['episode_length'])
        if not torch.equal(expected_episode,row['returned_episode_length']):
            raise ValueError('episode ownership mismatch')
    resources=[value['cuda_resources']['before'],value['cuda_resources']['after'],*value['resources_per_step']]
    for resource in resources:
        for key in ('free_memory_bytes','total_memory_bytes','memory_allocated_bytes','max_memory_allocated_bytes'):
            if type(resource[key]) is not int or resource[key]<0:
                raise ValueError('malformed resource metadata')
        if not 1024**3<=resource['free_memory_bytes']<=resource['total_memory_bytes']:
            raise ValueError('unsafe memory')
    replay=summarize(runtime,value['rows'])
    if canonical_bytes(replay)!=canonical_bytes(value['diagnostic_summary']):
        raise ValueError('diagnostic metric replay mismatch')
    counts=torch.bincount(inputs['termination_reason'].flatten(),minlength=5)
    if not torch.equal(counts,value['termination_reason_counts']) or value['reset_count']!=int(runtime['reset_masks'].sum()):
        raise ValueError('termination accounting mismatch')


def _load_canonical_object(path: Path, name: str) -> OrderedDict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} path must be an absolute regular non-symlink file")
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"), object_pairs_hook=OrderedDict)
    if raw != canonical_bytes(value) or not isinstance(value, OrderedDict):
        raise ValueError(f"{name} must be canonical JSON object")
    return value


def validate_closed_kit_log(path: Path) -> OrderedDict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("Kit log path must be an absolute regular non-symlink file")
    raw = path.read_bytes()
    if not raw:
        raise ValueError("Kit log is empty")
    offsets: list[int] = []
    cursor = 0
    for marker in KIT_SHUTDOWN_MARKERS:
        offset = raw.find(marker, cursor)
        if offset < 0:
            raise ValueError(f"Kit log lacks ordered shutdown evidence: {marker.decode('ascii')}")
        offsets.append(offset)
        cursor = offset + len(marker)
    return OrderedDict((
        ("bytes", len(raw)),
        ("sha256", sha256_file(path)),
        ("shutdown_markers", [marker.decode("ascii") for marker in KIT_SHUTDOWN_MARKERS]),
        ("shutdown_offsets", offsets),
    ))


def validate_files(
    binding: AttemptBinding,
    semantic_path: Path,
    mirror_path: Path,
    receipt_path: Path,
    status_path: Path,
    kit_log_path: Path,
    *,
    run_id: str,
) -> OrderedDict[str, Any]:
    semantic_path = bound_path(binding, semantic_path, f"results/{run_id}/semantic.pt", kind="file", must_exist=True)
    mirror_path = bound_path(binding, mirror_path, f"results/{run_id}/semantic.json", kind="file", must_exist=True)
    receipt_path = bound_path(binding, receipt_path, f"results/{run_id}/receipt.json", kind="file", must_exist=True)
    status_path = bound_path(binding, status_path, f"results/{run_id}/status.json", kind="file", must_exist=True)
    kit_log_path = bound_path(binding, kit_log_path, f"results/{run_id}/kit.log", kind="file", must_exist=True)
    value = load_semantic(semantic_path)
    validate_semantic(value, run_id=run_id)
    initial_path = bound_path(binding,semantic_path.parent/'initial_rays.pt',f'results/{run_id}/initial_rays.pt',kind='file',must_exist=True)
    if sha256_file(initial_path)!=value['initial_rays_sha256']:
        raise ValueError('initial ray source hash mismatch')
    validate_semantic_mirror(value, mirror_path)
    receipt = _load_canonical_object(receipt_path, "process receipt")
    receipt_keys = (
        "exit_status", "kit_log", "kit_setting", "pid", "run_id",
        "schema_version", "semantic_completed_at", "semantic_json",
        "semantic_json_sha256", "semantic_pt", "semantic_pt_sha256", "start",
    )
    if tuple(receipt) != receipt_keys:
        raise ValueError("process receipt fields mismatch")
    if receipt["schema_version"] != "eval-002-process-receipt.v3" or receipt["run_id"] != run_id or receipt["exit_status"] != "PASS":
        raise ValueError("process receipt identity/status mismatch")
    if not isinstance(receipt["pid"], int) or receipt["pid"] <= 0:
        raise ValueError("process receipt PID mismatch")
    if receipt["kit_log"] != str(kit_log_path) or receipt["kit_setting"] != str(kit_log_path):
        raise ValueError("Kit log receipt/setting mismatch")
    closed_log = validate_closed_kit_log(kit_log_path)
    if receipt["semantic_pt"] != str(semantic_path) or receipt["semantic_json"] != str(mirror_path):
        raise ValueError("semantic receipt path mismatch")
    if receipt["semantic_pt_sha256"] != sha256_file(semantic_path) or receipt["semantic_json_sha256"] != sha256_file(mirror_path):
        raise ValueError("semantic receipt hash mismatch")
    status = _load_canonical_object(status_path, "formal status")
    status_keys = (
        "pid", "process_receipt_sha256", "run_id", "schema_version",
        "semantic_exit", "semantic_json_sha256", "semantic_pt_sha256", "status",
    )
    if tuple(status) != status_keys:
        raise ValueError("formal status fields mismatch")
    if status.get("schema_version") != "eval-002-formal-status.v2" or status.get("status") != "PASS" or status.get("semantic_exit") != 0:
        raise ValueError("formal status mismatch")
    if status.get("run_id") != run_id or status.get("pid") != receipt["pid"]:
        raise ValueError("formal status identity mismatch")
    if status.get("semantic_pt_sha256") != receipt["semantic_pt_sha256"] or status.get("semantic_json_sha256") != receipt["semantic_json_sha256"] or status.get("process_receipt_sha256") != sha256_file(receipt_path):
        raise ValueError("formal status binding mismatch")
    return OrderedDict((
        ("schema_version", "eval-002-semantic-validation.v2"),
        ("status", "PASS"),
        ("run_id", run_id),
        ("semantic_pt_sha256", sha256_file(semantic_path)),
        ("semantic_json_sha256", sha256_file(mirror_path)),
        ("process_receipt_sha256", sha256_file(receipt_path)),
        ("formal_status_sha256", sha256_file(status_path)),
        ("kit_log_bytes", closed_log["bytes"]),
        ("kit_log_sha256", closed_log["sha256"]),
        ("kit_shutdown_markers", closed_log["shutdown_markers"]),
        ("kit_shutdown_offsets", closed_log["shutdown_offsets"]),
        ("pid", receipt["pid"]),
        ("metric_count", 10),
        ("environment_transitions", ENVIRONMENT_TRANSITIONS),
        ("ray_samples", RAY_SAMPLES),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=RUN_IDS, required=True)
    args = parser.parse_args()
    binding = binding_from_environment()
    output_path = bound_path(
        binding, path_from_environment("EVAL_VALIDATION_OUTPUT"),
        f"results/{args.run_id}/validation.json",
        must_exist=False,
    )
    try:
        result = validate_files(
            binding,
            path_from_environment("EVAL_SEMANTIC_PT"),
            path_from_environment("EVAL_SEMANTIC_JSON"),
            path_from_environment("EVAL_PROCESS_RECEIPT"),
            path_from_environment("EVAL_FORMAL_STATUS"),
            path_from_environment("EVAL_KIT_LOG"),
            run_id=args.run_id,
        )
        atomic_canonical_replace(output_path, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        atomic_canonical_replace(output_path, failure_result(error))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
