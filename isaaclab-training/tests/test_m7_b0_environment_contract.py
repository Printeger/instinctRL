"""Pure exact contract tests for the M7-011 B0 TASLAB environment mode."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from instinctlab.control.baseline_b0 import B0DirectConfig
from instinctlab.evaluation.baseline_run import B0RunConfig, begin_ready_b0_run
from instinctlab.evaluation.baselines import (
    ActorAuditStatus,
    BaselineId,
    BaselineRunManifest,
    CapabilityBinding,
    ExecutionStatus,
    build_run_manifest,
    resolve_baseline,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "instinctlab/evaluation/baseline_environment.py"
SOURCE_SHA256 = "b1ecdaa8b7d01ff0f9dd49891c2266908cea5a2540e8a91c33b9626464ded584"
DECIMATION = 20


class FakeTaslabDirectEnvCfg:
    def __init__(self) -> None:
        self.decimation = DECIMATION
        self.scene = SimpleNamespace(num_envs=2)
        self.sim = SimpleNamespace(device="cuda:0")
        self.action_noise_model = None


class FakeDirectRLEnv:
    simulator_steps = 0

    def step(self, action):
        self._pre_physics_step(action)
        for _ in range(self.cfg.decimation):
            self._apply_action()
            self._fake_sim_steps += 1
            type(self).simulator_steps += 1
        return "observation", "reward", "terminated", "truncated", "extras"

    def _validated_root_state(self):
        return self._robot.data.root_state_w

    def _reset_idx(self, env_ids) -> None:
        self._fake_resets += 1


class FakeTaslabDirectEnv(FakeDirectRLEnv):
    constructor_calls = 0

    def __init__(self, cfg, render_mode) -> None:
        type(self).constructor_calls += 1
        self.cfg = cfg
        self.render_mode = render_mode
        self.num_envs = cfg.scene.num_envs
        self.device = cfg.sim.device


def _load_module():
    fake_parent = ModuleType("instinctlab.envs")
    fake_parent.__path__ = []
    fake_source = ModuleType("instinctlab.envs.taslab_direct_env")
    fake_source.DECIMATION = DECIMATION
    fake_source.TaslabDirectEnv = FakeTaslabDirectEnv
    fake_source.TaslabDirectEnvCfg = FakeTaslabDirectEnvCfg
    names = ("instinctlab.envs", "instinctlab.envs.taslab_direct_env")
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules[names[0]] = fake_parent
    sys.modules[names[1]] = fake_source
    try:
        return importlib.import_module("instinctlab.evaluation.baseline_environment")
    finally:
        for name in names:
            if previous[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous[name]


environment = _load_module()


def _bindings() -> tuple[CapabilityBinding, ...]:
    return tuple(
        CapabilityBinding(item.name, item.accepted_provider, item.accepted_evidence_sha256)
        for item in resolve_baseline(BaselineId.B0).capability_requirements
    )


def _manifest(seed: int = 71) -> BaselineRunManifest:
    row = resolve_baseline(BaselineId.B0)
    return build_run_manifest(
        product_commit="a538ed7f096340d7b525b0d407455ef976b0c9d7",
        baseline_id=BaselineId.B0,
        config_switch=row.config_switch,
        capability_bindings=_bindings(),
        scenario_id="M7-011.environment",
        seed=seed,
        actor_audit_status=ActorAuditStatus.PASS,
        execution_status=ExecutionStatus.READY,
        skip_reason=None,
    )


class FakeWrenchAdapter:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.command_ids = []

    def apply(self, robot, commands):
        self.calls += 1
        self.command_ids.append(id(commands))
        if self.fail:
            raise RuntimeError("injected wrench failure")
        return SimpleNamespace(normalized_commands=commands)


def _root() -> torch.Tensor:
    root = torch.zeros((2, 13), dtype=torch.float64)
    root[:, 2] = 1.5
    for index, yaw in enumerate((0.0, 35.0)):
        root[index, 3] = math.cos(math.radians(yaw) / 2.0)
        root[index, 6] = math.sin(math.radians(yaw) / 2.0)
    return root


def _new_env(expected_policy_steps: int = 2):
    value = object.__new__(environment.TaslabB0EvaluationEnv)
    value.cfg = FakeTaslabDirectEnvCfg()
    value.num_envs = 2
    value.device = "cpu"
    value._command_b = torch.zeros((2, 3), dtype=torch.float64)
    value._previous_issued_v_final_b = torch.zeros((2, 3), dtype=torch.float64)
    value._reward_previous_v_final_b = torch.zeros((2, 3), dtype=torch.float64)
    value._actions = torch.zeros((2, 4), dtype=torch.float64)
    value._robot = SimpleNamespace(data=SimpleNamespace(root_state_w=_root()))
    value._wrench_adapter = FakeWrenchAdapter()
    value._control = None
    value._lee = None
    value._wrench = None
    value._b0_session = begin_ready_b0_run(
        _manifest(),
        B0RunConfig(expected_policy_steps * DECIMATION, B0DirectConfig(2.0)),
    )
    value._b0_expected_policy_steps = expected_policy_steps
    value._b0_expected_physics_steps = expected_policy_steps * DECIMATION
    value._b0_policy_steps = 0
    value._b0_physics_steps = 0
    value._b0_lifecycle_calls = 0
    value._b0_wrench_applications = 0
    value._b0_state = "ACTIVE"
    value._b0_current_command = None
    value._b0_motion_started = False
    value._fake_sim_steps = 0
    value._fake_resets = 0
    return value


def _command(values=((0.0, 0.0, 0.0), (3.0, 4.0, 0.0))) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def _fingerprint(value) -> tuple[object, ...]:
    return (
        value.state_token,
        value.policy_steps,
        value.physics_steps,
        value.lifecycle_calls,
        value.wrench_applications,
        value._b0_session.next_step,
        value._fake_sim_steps,
        value._fake_resets,
        None if value._b0_current_command is None else value._b0_current_command.clone(),
    )


def test_exact_config_api_factory_and_import_isolation() -> None:
    cfg = FakeTaslabDirectEnvCfg()
    config = environment.B0EnvironmentConfig(8, B0DirectConfig(2.0), cfg)
    assert config.expected_policy_steps == 8 and config.direct_config.velocity_limit_m_s == 2.0
    with pytest.raises(FrozenInstanceError):
        config.expected_policy_steps = 9
    invalid = (
        (True, B0DirectConfig(), cfg),
        (0, B0DirectConfig(), cfg),
        (1, object(), cfg),
        (1, B0DirectConfig(), object()),
    )
    for args in invalid:
        with pytest.raises((TypeError, ValueError)):
            environment.B0EnvironmentConfig(*args)
    for name, value in (("decimation", 19), ("num_envs", 1), ("device", "cpu")):
        bad = FakeTaslabDirectEnvCfg()
        if name == "num_envs":
            bad.scene.num_envs = value
        elif name == "device":
            bad.sim.device = value
        else:
            bad.decimation = value
        with pytest.raises(ValueError):
            environment.B0EnvironmentConfig(1, B0DirectConfig(), bad)
    session_env = environment.begin_ready_b0_environment(_manifest(), config, None)
    assert type(session_env) is environment.TaslabB0EvaluationEnv
    assert session_env.expected_policy_steps == 8
    assert session_env.expected_physics_steps == 160
    assert session_env._b0_session.next_step == 0
    with pytest.raises(TypeError):
        environment.TaslabB0EvaluationEnv(_manifest(), config, None, object())
    with pytest.raises(TypeError):
        type("Forbidden", (environment.TaslabB0EvaluationEnv,), {})
    with pytest.raises(ValueError):
        environment.begin_ready_b0_environment(_manifest(), config, "human")
    assert tuple(inspect.signature(environment.begin_ready_b0_environment).parameters) == (
        "manifest", "config", "render_mode"
    )
    import instinctlab.evaluation as aggregate
    assert "TaslabB0EvaluationEnv" not in aggregate.__all__
    assert not hasattr(aggregate, "TaslabB0EvaluationEnv")


def test_environment_config_owns_nested_copy_and_ignores_caller_mutation() -> None:
    original = FakeTaslabDirectEnvCfg()
    config = environment.B0EnvironmentConfig(1, B0DirectConfig(2.0), original)
    assert config.environment_config is not original
    assert config.environment_config.scene is not original.scene
    assert config.environment_config.sim is not original.sim
    original.decimation = 7
    original.scene.num_envs = 9
    original.sim.device = "cpu"
    original.action_noise_model = object()
    assert config.environment_config.decimation == DECIMATION
    assert config.environment_config.scene.num_envs == 2
    assert config.environment_config.sim.device == "cuda:0"
    assert config.environment_config.action_noise_model is None
    value = environment.begin_ready_b0_environment(_manifest(), config, None)
    assert value.cfg is not config.environment_config
    assert value.cfg.scene is not config.environment_config.scene
    assert value.cfg.sim is not config.environment_config.sim


@pytest.mark.parametrize("mutation", ("action_noise", "device", "num_envs", "decimation"))
def test_post_wrapper_nested_drift_rejects_before_constructor_lifecycle_or_simulator(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = environment.B0EnvironmentConfig(1, B0DirectConfig(2.0), FakeTaslabDirectEnvCfg())
    if mutation == "action_noise":
        config.environment_config.action_noise_model = object()
    elif mutation == "device":
        config.environment_config.sim.device = "cpu"
    elif mutation == "num_envs":
        config.environment_config.scene.num_envs = 1
    else:
        config.environment_config.decimation = DECIMATION - 1
    lifecycle_calls = []

    def forbidden_lifecycle(*args, **kwargs):
        lifecycle_calls.append((args, kwargs))
        raise AssertionError("lifecycle construction must not run after nested config drift")

    monkeypatch.setattr(environment, "begin_ready_b0_run", forbidden_lifecycle)
    constructor_calls = FakeTaslabDirectEnv.constructor_calls
    simulator_steps = FakeTaslabDirectEnv.simulator_steps
    with pytest.raises(ValueError):
        environment.begin_ready_b0_environment(_manifest(), config, None)
    assert lifecycle_calls == []
    assert FakeTaslabDirectEnv.constructor_calls == constructor_calls
    assert FakeTaslabDirectEnv.simulator_steps == simulator_steps


def test_two_policy_steps_exact_call_control_wrench_and_finalize_contract() -> None:
    value = _new_env(2)
    first = _command()
    second = _command(((0.25, -0.5, 0.75), (-3.0, 0.0, 0.0)))
    assert value.step(first)[0] == "observation"
    assert (value.policy_steps, value.physics_steps, value.lifecycle_calls) == (1, 20, 20)
    assert value.wrench_applications == value._wrench_adapter.calls == value._fake_sim_steps == 20
    assert value._b0_current_command is None
    control = value._control
    assert torch.equal(control.alpha, torch.ones((2, 1), dtype=torch.float64))
    assert torch.count_nonzero(control.v_corr_b) == 0
    assert torch.equal(control.v_gov_b, control.v_final_b)
    assert torch.equal(control.v_final_b, value._previous_issued_v_final_b)
    assert torch.count_nonzero(control.anchor_metrics["active"]) == 0
    assert torch.count_nonzero(control.anchor_metrics["huber_loss"]) == 0
    assert torch.equal(control.ics_metrics["beta"], torch.ones((2, 1), dtype=torch.float64))
    assert torch.count_nonzero(control.ics_metrics["emergency"]) == 0
    assert id(value._lee.normalized_rotor_commands) == value._wrench_adapter.command_ids[-1]
    value.step(second)
    assert (value.policy_steps, value.physics_steps, value.lifecycle_calls) == (2, 40, 40)
    result = value.finalize()
    assert type(result) is environment.B0EnvironmentResult
    assert type(result.coverage) is environment.B0EnvironmentCoverage
    assert result.receipt.step_count == 40
    assert tuple(getattr(result.coverage, item) for item in (
        "policy_steps", "physics_steps", "lifecycle_calls", "wrench_applications",
        "environment_samples", "num_envs", "decimation",
    )) == (2, 40, 40, 40, 80, 2, 20)
    assert value.finalized and value.state_token == "FINALIZED"
    assert value._b0_current_command is None
    assert value._control is value._lee is value._wrench is None
    assert torch.count_nonzero(value._command_b) == 0
    assert tuple(field.name for field in fields(environment.B0EnvironmentResult)) == ("receipt", "coverage")
    assert all(
        field.type in (int, "int") for field in fields(environment.B0EnvironmentCoverage)
    )


@pytest.mark.parametrize("mutation", ("type", "shape", "dtype", "nonfinite", "gradient", "view", "alias"))
def test_malformed_action_rejects_before_lifecycle_or_simulator(mutation: str) -> None:
    value = _new_env(1)
    action: object = _command()
    if mutation == "type":
        action = object()
    elif mutation == "shape":
        action = torch.zeros((2, 4), dtype=torch.float64)
    elif mutation == "dtype":
        action = _command().float()
    elif mutation == "nonfinite":
        action = _command(); action[0, 0] = float("nan")
    elif mutation == "gradient":
        action = _command().requires_grad_()
    elif mutation == "view":
        action = torch.zeros((2, 4), dtype=torch.float64)[:, :3]
    elif mutation == "alias":
        action = value._command_b
    before = _fingerprint(value)
    with pytest.raises((TypeError, ValueError)):
        value.step(action)
    assert _fingerprint(value) == before


def test_early_overrun_repeated_and_post_finalize_reject_without_calls() -> None:
    value = _new_env(1)
    before = _fingerprint(value)
    with pytest.raises(RuntimeError, match="before the exact horizon"):
        value.finalize()
    assert _fingerprint(value) == before
    value.step(_command())
    after_step = _fingerprint(value)
    with pytest.raises(RuntimeError, match="must be finalized"):
        value.step(_command())
    assert _fingerprint(value) == after_step
    value.finalize()
    after_finalize = _fingerprint(value)
    with pytest.raises(RuntimeError, match="exactly once"):
        value.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        value.step(_command())
    assert _fingerprint(value) == after_finalize


def test_forged_envelope_and_downstream_apply_failure_boundaries() -> None:
    forged = _new_env(1)
    calls = []
    forged._b0_session = SimpleNamespace(step=lambda *args: calls.append(args) or object())
    with pytest.raises(TypeError, match="exact controller envelope"):
        forged.step(_command())
    assert len(calls) == 1
    assert (forged.lifecycle_calls, forged.wrench_applications, forged.physics_steps, forged._fake_sim_steps) == (0, 0, 0, 0)
    assert forged._b0_current_command is None

    downstream = _new_env(1)
    downstream._wrench_adapter = FakeWrenchAdapter(fail=True)
    with pytest.raises(RuntimeError, match="injected wrench failure"):
        downstream.step(_command())
    assert downstream._b0_session.next_step == 1
    assert downstream.lifecycle_calls == 1
    assert (downstream.wrench_applications, downstream.physics_steps, downstream._fake_sim_steps) == (0, 0, 0)
    assert downstream.policy_steps == 0 and downstream._b0_current_command is None


def test_source_api_import_privacy_and_preserved_seventh_binding_audit() -> None:
    source = SOURCE.read_text()
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA256
    tree = ast.parse(source)
    imports = set()
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
        elif isinstance(node, ast.Call):
            calls.append(ast.unparse(node.func))
    assert not any(token in source for token in (
        "HistoryActor", "ControlCorePipeline(", "decode_governor(", "taslab_velocity_control(",
        "MeasurementAnchor(", "attenuate_command(", "collector", "optimizer", "checkpoint",
    ))
    assert calls.count("self._b0_session.step") == 1
    assert calls.count("self._wrench_adapter.apply") == 1
    assert "instinctlab.evaluation" not in imports
    requirements = resolve_baseline(BaselineId.B0).capability_requirements
    assert len(requirements) == 8
    assert tuple(item.name for item in requirements[-3:]) == (
        "m7.b0_run_lifecycle", "m7.b0_taslab_environment", "m7.b0_runtime_evaluation"
    )
    assert requirements[-2].accepted_provider == (
        "instinctlab.evaluation.baseline_environment.TaslabB0EvaluationEnv"
    )
    assert requirements[-2].accepted_evidence_sha256 == SOURCE_SHA256
    assert resolve_baseline(BaselineId.B0).actor_input_fields == ("commanded_body_velocity_m_s",)
    assert tuple(item.readiness.value for item in (
        resolve_baseline("B1"), resolve_baseline("B8")
    )) == ("NOT_READY", "NOT_READY")


def test_resigned_stale_six_rejects_under_current_registry() -> None:
    mapping = _manifest().to_mapping()
    mapping["capability_bindings"] = mapping["capability_bindings"][:6]
    payload = {name: mapping[name] for name in mapping if name != "manifest_sha256"}
    mapping["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    canonical = json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    with pytest.raises(ValueError, match="capability bindings"):
        BaselineRunManifest.from_json(canonical)


def test_resigned_stale_seven_rejects_under_current_registry() -> None:
    mapping = _manifest().to_mapping()
    mapping["capability_bindings"] = mapping["capability_bindings"][:7]
    payload = {name: mapping[name] for name in mapping if name != "manifest_sha256"}
    mapping["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    canonical = json.dumps(mapping, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    with pytest.raises(ValueError, match="capability bindings"):
        BaselineRunManifest.from_json(canonical)


def test_cuda_runner_and_complete_comparator_are_bounded_and_fail_closed() -> None:
    runner = (ROOT / "tests/run_m7_b0_environment_smoke.py").read_text()
    comparator = (ROOT / "tests/compare_m7_b0_environment_smoke.py").read_text()
    assert all(token in runner for token in (
        "NUM_ENVS = 2", "POLICY_STEPS = 8", "PHYSICS_STEPS = 160",
        "ENVIRONMENT_SAMPLES = 320", "COMMAND_SCHEDULE", "torch.save",
        "pipeline_calls", "reset_calls", "clip_vector_norm",
        "rotate_body_to_world", "taslab_velocity_control",
        '"body_clip_value_exact"', '"world_rotation_value_exact"',
        '"lee_direct_chain_exact"', '"oracle_non_actuation"',
        'os._exit(0 if gates["all_gates_pass"] else 2)',
    ))
    assert "HistoryActor" not in runner and "training" not in runner
    assert all(token in comparator for token in (
        'weights_only=True', "type(left) is not type(right)",
        "list(left) != list(right)", "left.shape != right.shape",
        "left.dtype != right.dtype", "torch.equal(left, right)",
        "bundle root keys are missing, extra, or reordered",
    ))
    assert "allclose" not in comparator and "isclose" not in comparator
