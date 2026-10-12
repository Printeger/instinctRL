"""CPU contract and mutation-negative tests for the EVAL-001 workload."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from contextlib import redirect_stdout
import copy
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

WORKLOAD = Path("/home/mint/rl_dev/instinctRL_coordination/evaluation_workloads/eval_001")
if str(WORKLOAD) not in sys.path:
    sys.path.insert(0, str(WORKLOAD))

import torch

from attempt_binding import AttemptBinding, bind_attempt, bound_path
from checkpoint_audit import audit_checkpoint
from constants import (
    ACTION_MODE, ACTOR_SHA256, CHECKPOINT_PATH, CHECKPOINT_SHA256, DECIMATION,
    ENVIRONMENT_SEED, ENVIRONMENT_TRANSITIONS, INFERENCE_SEED, NUM_ENVS,
    PHYSICS_STEPS_PER_ENVIRONMENT, POLICY_STEPS, PRODUCT_COMMIT, PRODUCT_TREE,
    RAYS_PER_ENVIRONMENT_PER_STEP, RAY_SAMPLES, SOURCE_SHA256,
)
from io_utils import atomic_canonical_replace, canonical_bytes, canonical_value
import formal_evaluator
from pair_comparator import V2_LIVE_METADATA_PATHS, compare_exact, compare_v2
from semantic_validator import (
    EvidenceValidationError,
    KIT_SHUTDOWN_MARKERS,
    failure_result,
    validate_closed_kit_log,
    validate_semantic,
    validate_semantic_mirror,
)


def _module(sha256: str) -> OrderedDict:
    return OrderedDict((("sha256", sha256), ("parameter_count", 1), ("all_finite", True)))


def _optimizer() -> OrderedDict:
    return OrderedDict((
        ("sha256", "d" * 64), ("state_count", 1), ("step_min", 2000),
        ("step_max", 2000), ("all_steps_equal", True),
    ))


def _diagnostic(name: str, shape: list[int], count: int, minimum: float = 0.0, maximum: float = 0.0) -> OrderedDict:
    return OrderedDict((
        ("name", name), ("shape", shape), ("dtype", "torch.float32"),
        ("count", count), ("sum", 0.0), ("square_sum", 0.0),
        ("minimum", minimum), ("maximum", maximum), ("mean", 0.0),
        ("sha256", "a" * 64),
    ))


def fixture() -> OrderedDict:
    actor = _module(ACTOR_SHA256)
    critic = _module("c" * 64)
    state = OrderedDict((
        ("actor", actor), ("critic", critic),
        ("actor_optimizer", _optimizer()), ("critic_optimizer", _optimizer()),
    ))
    metric = OrderedDict((
        ("reduction", "mean"), ("unit", "ratio"), ("available", True),
        ("value", 0.0), ("numerator", 0.0), ("denominator", ENVIRONMENT_TRANSITIONS),
        ("valid_count", ENVIRONMENT_TRANSITIONS), ("total_count", ENVIRONMENT_TRANSITIONS),
    ))
    metrics = OrderedDict((f"metric_{index}", copy.deepcopy(metric)) for index in range(10))
    record = OrderedDict((("metrics", metrics),))
    resources = OrderedDict((
        ("device_name", "NVIDIA GeForce RTX 5070"), ("device_capability", [12, 0]),
        ("total_memory_bytes", 12 * 1024 ** 3), ("free_memory_bytes", 2 * 1024 ** 3),
        ("memory_allocated_bytes", 1), ("max_memory_allocated_bytes", 1),
    ))
    return OrderedDict((
        ("schema_version", "eval-001-formal-semantic.v2"),
        ("scope", OrderedDict((
            ("environment", "TaslabDirectEnv static plane"),
            ("commanded_body_velocity_m_s", [0.0, 0.0, 0.0]),
            ("performance_gate", False),
        ))),
        ("identity", OrderedDict((
            ("product_commit", PRODUCT_COMMIT), ("product_tree", PRODUCT_TREE),
            ("checkpoint_sha256", CHECKPOINT_SHA256), ("checkpoint_counter", 250),
            ("source_sha256", SOURCE_SHA256),
        ))),
        ("campaign", OrderedDict((
            ("action_mode", ACTION_MODE), ("environment_seed", ENVIRONMENT_SEED),
            ("inference_seed", INFERENCE_SEED), ("num_envs", NUM_ENVS),
            ("policy_steps", POLICY_STEPS), ("decimation", DECIMATION),
            ("physics_steps_per_environment", PHYSICS_STEPS_PER_ENVIRONMENT),
            ("environment_transitions", ENVIRONMENT_TRANSITIONS),
            ("rays_per_environment_per_step", RAYS_PER_ENVIRONMENT_PER_STEP),
            ("ray_samples", RAY_SAMPLES),
        ))),
        ("checkpoint_state_before", state),
        ("checkpoint_state_after", copy.deepcopy(state)),
        ("gradients", OrderedDict((
            ("actor_grad_tensor_count", 0), ("critic_grad_tensor_count", 0),
            ("global_gradient_enabled", False),
        ))),
        ("training_claims", OrderedDict((
            ("training_performed", False), ("optimizer_updates", 0),
            ("checkpoint_writes", 0), ("one_million_training", False),
            ("merge", False), ("push", False),
        ))),
        ("actor_boundary", OrderedDict((
            ("actor_keys", ["lidar_grid", "state_vec"]),
            ("privileged_actor_input", False),
            ("action_source", "actor(policy).mean_action"),
            ("critic_influenced_action", False),
            ("evaluation_influenced_action", False),
            ("action_passed_unchanged", True),
        ))),
        ("action_diagnostics", _diagnostic("action", [POLICY_STEPS, NUM_ENVS, 4], POLICY_STEPS * NUM_ENVS * 4, 0.0, 1.0)),
        ("alpha_diagnostics", _diagnostic("alpha", [POLICY_STEPS, NUM_ENVS, 4], POLICY_STEPS * NUM_ENVS * 4)),
        ("beta_diagnostics", _diagnostic("beta", [POLICY_STEPS, NUM_ENVS, 4], POLICY_STEPS * NUM_ENVS * 4)),
        ("entropy_diagnostics", _diagnostic("entropy", [POLICY_STEPS, NUM_ENVS], POLICY_STEPS * NUM_ENVS)),
        ("log_probability_diagnostics", _diagnostic("log_probability", [POLICY_STEPS, NUM_ENVS], POLICY_STEPS * NUM_ENVS)),
        ("reward_diagnostics", _diagnostic("reward", [POLICY_STEPS, NUM_ENVS], POLICY_STEPS * NUM_ENVS)),
        ("termination_reason_counts", torch.tensor([ENVIRONMENT_TRANSITIONS, 0, 0, 0, 0], dtype=torch.long)),
        ("reset_count", 0),
        ("cuda_resources", OrderedDict((("before", resources), ("after", copy.deepcopy(resources))))),
        ("raycaster_provenance", OrderedDict()),
        ("runtime_evaluation", OrderedDict((
            ("contract", OrderedDict((
                ("num_envs", NUM_ENVS), ("policy_steps", POLICY_STEPS),
                ("samples", ENVIRONMENT_TRANSITIONS),
                ("rays_per_environment", RAYS_PER_ENVIRONMENT_PER_STEP),
                ("ray_samples", RAY_SAMPLES),
            ))),
            ("policy_hashes", tuple("p" for _ in range(POLICY_STEPS))),
            ("action_hashes", tuple("a" for _ in range(POLICY_STEPS))),
            ("canonical_record", json.dumps(record, separators=(",", ":"))),
        ))),
    ))


def must_reject(name: str, mutation) -> None:
    value = fixture()
    mutation(value)
    try:
        validate_semantic(value)
    except (ValueError, TypeError, KeyError, IndexError):
        return
    raise AssertionError(f"negative mutation accepted: {name}")


class NonReturningClose(BaseException):
    pass


def verify_preclose_persistence() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output_dir = root / "mean_cert_a"
        output_dir.mkdir()
        log_path = output_dir / "kit.log"
        semantic = fixture()

        class FakeApplication:
            def close(self) -> None:
                for name in ("semantic.pt", "semantic.json", "receipt.json", "status.json"):
                    if not (output_dir / name).is_file():
                        raise AssertionError(f"{name} was not durable before close")
                raise NonReturningClose()

        class FakeLauncher:
            def __init__(self, **_kwargs):
                self.app = FakeApplication()

        class FakeSettings:
            def get(self, _name):
                return str(log_path)

        isaaclab_module = types.ModuleType("isaaclab")
        app_module = types.ModuleType("isaaclab.app")
        app_module.AppLauncher = FakeLauncher
        isaaclab_module.app = app_module
        carb_module = types.ModuleType("carb")
        carb_module.settings = types.SimpleNamespace(get_settings=lambda: FakeSettings())
        argv = [
            "formal_evaluator.py", "--run-id", "mean_cert_a",
            "--action-mode", ACTION_MODE, "--environment-seed", str(ENVIRONMENT_SEED),
            "--inference-seed", str(INFERENCE_SEED), "--checkpoint", str(CHECKPOINT_PATH),
        ]
        environment = {"EVAL_OUTPUT_DIR": str(output_dir), "EVAL_KIT_LOG": str(log_path)}
        with (
            mock.patch.object(
                formal_evaluator, "binding_from_environment",
                return_value=AttemptBinding("EVAL-001-D0042", root, root.parent),
            ),
            mock.patch.object(formal_evaluator, "validate_product_sources"),
            mock.patch.object(formal_evaluator, "validate_paths", return_value=(output_dir, log_path)),
            mock.patch.object(formal_evaluator, "evaluate", return_value=semantic),
            mock.patch.dict(sys.modules, {"isaaclab": isaaclab_module, "isaaclab.app": app_module, "carb": carb_module}),
            mock.patch.dict(os.environ, environment, clear=False),
            mock.patch.object(sys, "argv", argv),
        ):
            try:
                with redirect_stdout(io.StringIO()):
                    formal_evaluator.main()
            except NonReturningClose:
                pass
            else:
                raise AssertionError("mock terminal close unexpectedly returned")
        status = json.loads((output_dir / "status.json").read_text(encoding="ascii"))
        if status.get("status") != "PASS" or status.get("semantic_exit") != 0:
            raise AssertionError("pre-close status is not semantic PASS")
        if "kit_log_sha256" in status:
            raise AssertionError("pre-close status claimed a final Kit log hash")
        receipt = json.loads((output_dir / "receipt.json").read_text(encoding="ascii"))
        if receipt.get("schema_version") != "eval-001-process-receipt.v3" or "kit_log_sha256" in receipt:
            raise AssertionError("process receipt incorrectly claimed final log facts")


def verify_closed_log_validation() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        log_path = Path(temporary) / "kit.log"
        log_path.write_bytes(b"prefix\n" + b"\n".join(KIT_SHUTDOWN_MARKERS) + b"\n")
        result = validate_closed_kit_log(log_path.resolve())
        if result["bytes"] != log_path.stat().st_size or len(result["shutdown_offsets"]) != len(KIT_SHUTDOWN_MARKERS):
            raise AssertionError("closed Kit log facts mismatch")
        for malformed in (b"", b"\n".join(KIT_SHUTDOWN_MARKERS[:-1]), b"\n".join(reversed(KIT_SHUTDOWN_MARKERS))):
            log_path.write_bytes(malformed)
            try:
                validate_closed_kit_log(log_path.resolve())
            except ValueError:
                pass
            else:
                raise AssertionError("missing/incomplete/reordered Kit shutdown evidence passed")
        log_path.write_bytes(b"\n".join(KIT_SHUTDOWN_MARKERS))
        link_path = Path(temporary) / "linked.log"
        link_path.symlink_to(log_path)
        try:
            validate_closed_kit_log(link_path.absolute())
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked Kit log passed")


def verify_attempt_binding() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        attempts_root = Path(temporary).resolve() / "attempts/development/EVAL-001"
        attempts_root.mkdir(parents=True)
        current = attempts_root / "EVAL-001-D0019"
        future = attempts_root / "EVAL-001-D0042"
        current.mkdir()
        future.mkdir()
        for attempt_id, directory in (
            ("EVAL-001-D0019", current), ("EVAL-001-D0042", future),
        ):
            binding = bind_attempt(attempt_id, directory, attempts_root=attempts_root)
            candidate = directory / "results/value.json"
            if bound_path(binding, candidate, "results/value.json", must_exist=False) != candidate:
                raise AssertionError("attempt-neutral positive binding changed path")

        binding = bind_attempt("EVAL-001-D0019", current, attempts_root=attempts_root)
        prior_a = attempts_root / "EVAL-001-D0017"
        prior_b = attempts_root / "EVAL-001-D0018"
        prior_a.mkdir()
        prior_b.mkdir()
        rejected = (
            lambda: bind_attempt("OTHER-001-D0019", current, attempts_root=attempts_root),
            lambda: bind_attempt("EVAL-001-D019", current, attempts_root=attempts_root),
            lambda: bind_attempt("EVAL-001-D0019", future, attempts_root=attempts_root),
            lambda: bind_attempt("EVAL-001-D0019", Path("EVAL-001-D0019"), attempts_root=attempts_root),
            lambda: bind_attempt("EVAL-001-D0019", current / ".." / "EVAL-001-D0019", attempts_root=attempts_root),
            lambda: bound_path(binding, prior_a / "results/value.json", "results/value.json", must_exist=False),
            lambda: bound_path(binding, prior_b / "results/value.json", "results/value.json", must_exist=False),
            lambda: bound_path(binding, attempts_root / "escape.json", "results/value.json", must_exist=False),
            lambda: bound_path(binding, Path("results/value.json"), "results/value.json", must_exist=False),
        )
        for case in rejected:
            try:
                case()
            except ValueError:
                pass
            else:
                raise AssertionError("invalid attempt binding/path was accepted")

        real = attempts_root / "EVAL-001-D0043-real"
        real.mkdir()
        linked = attempts_root / "EVAL-001-D0043"
        linked.symlink_to(real, target_is_directory=True)
        try:
            bind_attempt("EVAL-001-D0043", linked, attempts_root=attempts_root)
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked attempt directory was accepted")

        alias_parent = Path(temporary).resolve() / "attempts-alias"
        alias_parent.symlink_to(attempts_root.parent, target_is_directory=True)
        aliased_root = alias_parent / "EVAL-001"
        try:
            bind_attempt("EVAL-001-D0019", aliased_root / "EVAL-001-D0019", attempts_root=aliased_root)
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked parent component was accepted")

        results = current / "results"
        results.mkdir()
        real_file = results / "real.json"
        real_file.write_text("{}", encoding="ascii")
        link_file = results / "value.json"
        link_file.symlink_to(real_file)
        try:
            bound_path(binding, link_file, "results/value.json", kind="file", must_exist=True)
        except ValueError:
            pass
        else:
            raise AssertionError("symlinked attempt file was accepted")


def _reordered_objects(value):
    if isinstance(value, dict):
        return OrderedDict(
            (key, _reordered_objects(child))
            for key, child in reversed(tuple(value.items()))
        )
    if isinstance(value, list):
        return [_reordered_objects(child) for child in value]
    return value


def verify_canonical_mirror_contract() -> None:
    semantic = fixture()
    expected = canonical_value(semantic)
    with tempfile.TemporaryDirectory() as temporary:
        path = (Path(temporary) / "semantic.json").resolve()

        path.write_bytes(canonical_bytes(expected))
        sorted_result = validate_semantic_mirror(semantic, path)
        if sorted_result["semantic_value_difference_count"] != 0:
            raise AssertionError("sorted canonical mirror has value differences")

        shuffled = _reordered_objects(expected)
        path.write_text(
            json.dumps(shuffled, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n",
            encoding="ascii",
        )
        shuffled_result = validate_semantic_mirror(semantic, path)
        if not shuffled_result["root_insertion_order_differs"]:
            raise AssertionError("shuffled mirror did not exercise root object order")
        if not shuffled_result["canonical_bytes_equal"]:
            raise AssertionError("shuffled mirror canonical bytes differ")

        def reject_bytes(name: str, raw: bytes) -> None:
            path.write_bytes(raw)
            try:
                validate_semantic_mirror(semantic, path)
            except EvidenceValidationError:
                return
            raise AssertionError(f"canonical mirror negative accepted: {name}")

        def reject_value(name: str, value) -> None:
            reject_bytes(name, canonical_bytes(value))

        missing = copy.deepcopy(expected)
        missing.pop("reward_diagnostics")
        reject_value("missing_key", missing)
        extra = copy.deepcopy(expected)
        extra["unknown_schema_field"] = False
        reject_value("extra_unknown_key", extra)
        changed_value = copy.deepcopy(expected)
        changed_value["reset_count"] = 1
        reject_value("changed_scalar_value", changed_value)
        changed_type = copy.deepcopy(expected)
        changed_type["reset_count"] = False
        reject_value("changed_scalar_type", changed_type)
        changed_dtype = copy.deepcopy(expected)
        changed_dtype["termination_reason_counts"]["tensor_dtype"] = "torch.int32"
        reject_value("changed_tensor_dtype", changed_dtype)
        changed_shape = copy.deepcopy(expected)
        changed_shape["termination_reason_counts"]["tensor_shape"] = [1, 5]
        reject_value("changed_tensor_shape", changed_shape)
        changed_data = copy.deepcopy(expected)
        changed_data["termination_reason_counts"]["tensor_data"][0] -= 1
        reject_value("changed_tensor_data", changed_data)
        reordered_list = copy.deepcopy(expected)
        reordered_list["actor_boundary"]["actor_keys"].reverse()
        reject_value("reordered_ordered_list", reordered_list)
        reject_bytes("duplicate_key", b'{"duplicate":1,"duplicate":1}\n')
        reject_bytes("malformed_utf8", b"{\"value\":\"\xff\"}\n")
        reject_bytes("malformed_json", b"{\n")
        reject_bytes("trailing_content", canonical_bytes(expected) + b"{}\n")
        reject_bytes("nan", b'{"value":NaN}\n')
        reject_bytes("positive_infinity", b'{"value":Infinity}\n')
        reject_bytes("negative_infinity", b'{"value":-Infinity}\n')
        if failure_result(EvidenceValidationError("mirror"))["failure_class"] != "EVIDENCE_VALIDATION_FAILURE":
            raise AssertionError("mirror failure was not classified as evidence validation")
        if failure_result(ValueError("semantic"))["failure_class"] != "PRODUCT_SEMANTIC_FAILURE":
            raise AssertionError("semantic failure lost product classification")


def verify_v2_pair_contract() -> None:
    def assert_pass(left, right, expected_paths: list[str]) -> None:
        result = compare_v2(left, right)
        if result["status"] != "PASS" or result["all_difference_paths"] != expected_paths:
            raise AssertionError(f"V2 positive failed: {result}")
        if result["deterministic_domain_maximum_error"] != 0.0:
            raise AssertionError("V2 positive has nonzero deterministic error")
        if [row["path"] for row in result["live_metadata"]] != list(V2_LIVE_METADATA_PATHS):
            raise AssertionError("V2 metadata path set/order changed")

    equal = fixture()
    assert_pass(equal, copy.deepcopy(equal), [])
    for path in V2_LIVE_METADATA_PATHS:
        left = fixture()
        right = copy.deepcopy(left)
        phase = path.split(".")[-2]
        right["cuda_resources"][phase]["free_memory_bytes"] += 65536
        assert_pass(left, right, [path])
    left = fixture()
    right = copy.deepcopy(left)
    right["cuda_resources"]["before"]["free_memory_bytes"] += 65536
    right["cuda_resources"]["after"]["free_memory_bytes"] -= 65536
    assert_pass(left, right, list(V2_LIVE_METADATA_PATHS))

    for phase in ("before", "after"):
        for invalid in (True, 1.5, 0, 1024 ** 3 - 1, 13 * 1024 ** 3):
            left = fixture()
            right = copy.deepcopy(left)
            right["cuda_resources"][phase]["free_memory_bytes"] = invalid
            result = compare_v2(left, right)
            if result["status"] != "FAIL" or result["failure_class"] != "INVALID_LIVE_RESOURCE_METADATA":
                raise AssertionError(f"unsafe V2 metadata accepted: {phase}={invalid!r}")
        left = fixture()
        right = copy.deepcopy(left)
        right["cuda_resources"][phase].pop("free_memory_bytes")
        if compare_v2(left, right)["status"] != "FAIL":
            raise AssertionError("missing V2 metadata accepted")

    deterministic_mutations = (
        lambda x: x.__setitem__("reset_count", 1),
        lambda x: x["termination_reason_counts"].__setitem__(0, ENVIRONMENT_TRANSITIONS - 1),
        lambda x: x.pop("reward_diagnostics"),
        lambda x: x.__setitem__("extra", False),
        lambda x: x.move_to_end("scope"),
        lambda x: x["actor_boundary"]["actor_keys"].reverse(),
        lambda x: x["cuda_resources"]["before"].__setitem__("total_memory_bytes", 13 * 1024 ** 3),
        lambda x: x["cuda_resources"]["before"].__setitem__("memory_allocated_bytes", 2),
        lambda x: x["cuda_resources"]["after"].__setitem__("max_memory_allocated_bytes", 2),
        lambda x: x["cuda_resources"]["after"].__setitem__("device_name", "other"),
        lambda x: x["cuda_resources"]["after"]["device_capability"].reverse(),
    )
    for mutation in deterministic_mutations:
        left = fixture()
        right = copy.deepcopy(left)
        mutation(right)
        result = compare_v2(left, right)
        if result["status"] != "FAIL" or result["failure_class"] != "PRODUCT_SEMANTIC_FAILURE":
            raise AssertionError(f"deterministic V2 mismatch accepted: {result}")
    left = fixture()
    right = copy.deepcopy(left)
    right["reset_count"] = 1
    right["training_claims"]["merge"] = True
    result = compare_v2(left, right)
    if len(result["deterministic_differences"]) != 2:
        raise AssertionError("V2 comparator did not collect all deterministic differences")


def run_tests() -> OrderedDict:
    value = fixture()
    validate_semantic(value)
    leaf_count, maximum_error = compare_exact(value, copy.deepcopy(value))
    if leaf_count <= 0 or maximum_error != 0.0:
        raise AssertionError("exact positive comparison failed")
    mutations = OrderedDict((
        ("missing_field", lambda x: x.pop("reward_diagnostics")),
        ("extra_field", lambda x: x.__setitem__("extra", False)),
        ("reordered_field", lambda x: x.move_to_end("scope")),
        ("wrong_product", lambda x: x["identity"].__setitem__("product_commit", "0" * 40)),
        ("wrong_checkpoint", lambda x: x["identity"].__setitem__("checkpoint_sha256", "0" * 64)),
        ("wrong_actor", lambda x: x["checkpoint_state_before"]["actor"].__setitem__("sha256", "0" * 64)),
        ("wrong_action", lambda x: x["actor_boundary"].__setitem__("action_source", "actor.sample(policy)")),
        ("wrong_seed", lambda x: x["campaign"].__setitem__("inference_seed", 0)),
        ("wrong_count", lambda x: x["campaign"].__setitem__("environment_transitions", 8191)),
        ("nonfinite", lambda x: x["reward_diagnostics"].__setitem__("mean", math.nan)),
        ("privileged_actor_input", lambda x: x["actor_boundary"].__setitem__("privileged_actor_input", True)),
        ("training_claim", lambda x: x["training_claims"].__setitem__("training_performed", True)),
        ("optimizer_update_claim", lambda x: x["training_claims"].__setitem__("optimizer_updates", 1)),
        ("checkpoint_write_claim", lambda x: x["training_claims"].__setitem__("checkpoint_writes", 1)),
    ))
    for name, mutation in mutations.items():
        must_reject(name, mutation)
    mismatch = copy.deepcopy(value)
    mismatch["reward_diagnostics"]["sum"] = 1.0
    try:
        compare_exact(value, mismatch)
    except ValueError:
        pass
    else:
        raise AssertionError("pair mismatch was accepted")
    checkpoint_result = audit_checkpoint(CHECKPOINT_PATH)
    if checkpoint_result["status"] != "PASS":
        raise AssertionError("checkpoint audit did not pass")
    if canonical_value(torch.tensor([1.0]))["tensor_dtype"] != "torch.float32":
        raise AssertionError("tensor mirror encoding failed")
    try:
        canonical_bytes({"status": "PASS", "value": math.nan})
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite canonical output was accepted")
    verify_preclose_persistence()
    verify_closed_log_validation()
    verify_attempt_binding()
    verify_canonical_mirror_contract()
    verify_v2_pair_contract()
    return OrderedDict((
        ("schema_version", "eval-001-cpu-tests.v1"), ("status", "PASS"),
        ("fixed_num_envs", NUM_ENVS), ("fixed_policy_steps", POLICY_STEPS),
        ("fixed_transitions_per_process", ENVIRONMENT_TRANSITIONS),
        ("fixed_ray_samples_per_process", RAY_SAMPLES),
        ("negative_mutation_count", len(mutations)),
        ("pair_mismatch_negative", "PASS"),
        ("fail_closed_nonfinite_output", "PASS"),
        ("checkpoint_audit", "PASS"),
        ("preclose_persistence", "PASS"),
        ("closed_log_validation", "PASS"),
        ("attempt_neutral_binding", "PASS"),
        ("canonical_mirror_contract", "PASS"),
        ("v2_pair_contract", "PASS"),
        ("positive_compared_leaf_or_tensor_element_count", leaf_count),
    ))


def main() -> int:
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    result = run_tests()
    if "EVAL_CPU_TEST_OUTPUT" in os.environ:
        from attempt_binding import binding_from_environment, path_from_environment
        binding = binding_from_environment()
        output = bound_path(
            binding, path_from_environment("EVAL_CPU_TEST_OUTPUT"),
            "results/cpu_tests.json", must_exist=False,
        )
        atomic_canonical_replace(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
