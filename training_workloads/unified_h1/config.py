"""Strict method, repository identity and complete execution-source binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

CONFIG_PATH = Path(__file__).with_name("candidate_config.json")
METHOD_PATH = Path(__file__).resolve().parents[2] / "coordination/contracts/R1_UNIFIED_HOVER_TRAINING_PLAN.md"
METHOD_SHA256 = "4d0186ecef8d1feda7af6d27766e50bbf78c9db9f4446ab224c074736f71b612"
VERSIONS = {"curriculum": "curriculum-h1-v1", "reward": "reward-h1-v2", "critic": "critic-h1-v2",
            "checkpoint": "checkpoint-h1-v2", "logging": "learning-log-h1-v1", "metric": "metric-h1-v1"}
NEUTRALITY_METHOD_PATH = Path(__file__).resolve().parents[2] / "coordination/contracts/R5_ZERO_COMMAND_RESIDUAL_NEUTRALITY.md"
NEUTRALITY_METHOD_SHA256 = "ecc75a8e3d581ca1bd43e8c3e34326a5f18dd55cfa1739fe3b26f4e25fe1c14e"
NEUTRALITY_VERSIONS = {**VERSIONS, "reward": "reward-h1-zero-neutrality-v1"}
RESEARCH_METHOD_PATH = Path(__file__).resolve().parents[2] / "coordination/contracts/R5_FORMAL_READINESS_RESEARCH_V1.md"
RESEARCH_METHOD_SHA256 = "679f5ce9b4d43a562332a355c8a64768e0e2d150741acb9cd43a13161f2ee5ec"
RESEARCH_VERSIONS = {**NEUTRALITY_VERSIONS, "curriculum": "curriculum-h1-research-pure-hover-v1"}
PRODUCT_SOURCES = {
    "isaaclab-training/instinctlab/assets/taslab_uav/articulation.py",
    "isaaclab-training/instinctlab/control/anchor.py", "isaaclab-training/instinctlab/control/baseline_b0.py", "isaaclab-training/instinctlab/control/pipeline.py",
    "isaaclab-training/instinctlab/control/ics.py", "isaaclab-training/instinctlab/control/governor.py",
    "isaaclab-training/instinctlab/control/lee_velocity_controller.py", "isaaclab-training/instinctlab/control/runtime_adapter.py",
    "isaaclab-training/instinctlab/envs/__init__.py",
    "isaaclab-training/instinctlab/envs/taslab_direct_env.py", "isaaclab-training/instinctlab/envs/unified_h1_direct_env.py",
    "isaaclab-training/instinctlab/task/critic_h1.py", "isaaclab-training/instinctlab/task/rewards_h1.py",
    "isaaclab-training/instinctlab/task/scene_h1.py", "isaaclab-training/instinctlab/task/termination.py",
    "isaaclab-training/instinctlab/task/__init__.py", "isaaclab-training/instinctlab/observations/actor.py",
    "isaaclab-training/instinctlab/sensors/mid360.py", "isaaclab-training/instinctlab/training/__init__.py",
    "isaaclab-training/instinctlab/evaluation/raycaster_observability.py",
    "isaaclab-training/instinctlab/training/checkpoint.py", "isaaclab-training/instinctlab/training/checkpoint_h1.py",
    "isaaclab-training/instinctlab/training/networks.py", "isaaclab-training/instinctlab/training/networks_h1.py",
    "isaaclab-training/instinctlab/training/ppo.py", "isaaclab-training/instinctlab/training/rollout.py",
    "isaaclab-training/instinctlab/training/rollout_h1.py",
}
WORKLOAD_SOURCES = {
    "training_workloads/unified_h1/__init__.py",
    "training_workloads/unified_h1/asset_closure.py",
    "training_workloads/unified_h1/authorization.py", "training_workloads/unified_h1/backend.py",
    "training_workloads/unified_h1/control_diagnostic.py", "training_workloads/unified_h1/pilot_diagnostic.py",
    "training_workloads/unified_h1/pilot_contract.py", "training_workloads/unified_h1/research_contract.py",
    "training_workloads/unified_h1/mean_prior.py",
    "training_workloads/unified_h1/checkpoint_compatibility.py", "training_workloads/unified_h1/checkpoint_recalculation.py",
    "training_workloads/unified_h1/w_closure.py",
    "training_workloads/unified_h1/w_independent.py",
    "training_workloads/unified_h1/resource_admission.py",
    "training_workloads/unified_h1/transition_diagnostic.py", "training_workloads/unified_h1/formal_segment.py",
    "training_workloads/unified_h1/budget.py", "training_workloads/unified_h1/config.py",
    "training_workloads/unified_h1/driver.py", "training_workloads/unified_h1/evaluation.py",
    "training_workloads/unified_h1/formal_unified_h1.py", "training_workloads/unified_h1/live_adapter.py",
    "training_workloads/unified_h1/logging.py", "training_workloads/unified_h1/r3_sequence.py",
    "training_workloads/unified_h1/schedule.py", "training_workloads/unified_h1/preflight.py",
    "training_workloads/unified_h1/local_dispatch.py",
    "training_workloads/unified_h1/b0_enhanced.py", "training_workloads/unified_h1/b0_observer.py",
    "training_workloads/unified_h1/b0_handoff.py", "training_workloads/unified_h1/b0_causal_analyzer.py",
    "training_workloads/unified_h1/evidence_boundary.py",
    "training_workloads/scale_003/lifecycle.py",
    "training_workloads/scale_003/persistence.py", "training_workloads/scale_003/resources.py",
}


R6_CONFIG_SCHEMA = "unified-h1-r6-config.v1"
R6_PRODUCT_SOURCES = PRODUCT_SOURCES | {
    "isaaclab-training/instinctlab/observations/robustness.py",
    "isaaclab-training/instinctlab/observations/robustness_adapter.py",
    "isaaclab-training/instinctlab/observations/robustness_physics.py",
}
R6_WORKLOAD_SOURCES = WORKLOAD_SOURCES | {
    "training_workloads/unified_h1/" + name + ".py" for name in (
        "robustness", "robustness_scenarios", "robustness_evaluation",
        "robustness_contract", "robustness_prelaunch", "robustness_audit",
        "range_archive", "force_archive", "robustness_analysis",
        "robustness_measurement_replay", "robustness_physics_replay",
        "robustness_training_storage", "robustness_training_rng", "robustness_training_plan",
        "robustness_training_driver", "lossless_evidence",
        "native_volume", "extension_inputs", "write_boundary", "result_volume",
    )
}


# Executable package initializers and transitive local dependencies for R6 only.
R6_PRODUCT_SOURCES |= {
    'isaaclab-training/instinctlab/__init__.py',
    'isaaclab-training/instinctlab/assets/__init__.py',
    'isaaclab-training/instinctlab/assets/taslab_uav/__init__.py',
    'isaaclab-training/instinctlab/assets/taslab_uav/config.py',
    'isaaclab-training/instinctlab/control/__init__.py',
    'isaaclab-training/instinctlab/control/command_adapter.py',
    'isaaclab-training/instinctlab/control/rotor_actuation.py',
    'isaaclab-training/instinctlab/evaluation/__init__.py',
    'isaaclab-training/instinctlab/evaluation/audit.py',
    'isaaclab-training/instinctlab/evaluation/baseline_controller_execution.py',
    'isaaclab-training/instinctlab/evaluation/baseline_execution.py',
    'isaaclab-training/instinctlab/evaluation/baseline_run.py',
    'isaaclab-training/instinctlab/evaluation/baselines.py',
    'isaaclab-training/instinctlab/evaluation/matrix.py',
    'isaaclab-training/instinctlab/evaluation/metrics.py',
    'isaaclab-training/instinctlab/evaluation/observability.py',
    'isaaclab-training/instinctlab/evaluation/runtime.py',
    'isaaclab-training/instinctlab/evaluation/scenarios.py',
    'isaaclab-training/instinctlab/observations/__init__.py',
    'isaaclab-training/instinctlab/observations/audit.py',
    'isaaclab-training/instinctlab/sensors/__init__.py',
    'isaaclab-training/instinctlab/task/critic.py',
    'isaaclab-training/instinctlab/task/rewards.py',
}
R6_WORKLOAD_SOURCES |= {
    'training_workloads/scale_003/__init__.py',
    'training_workloads/scale_003/config.py',
    'training_workloads/scale_003/schedule.py',
}


def required_source_sets(schema: str) -> tuple[set[str], set[str]]:
    if schema == R6_CONFIG_SCHEMA:
        return R6_PRODUCT_SOURCES, R6_WORKLOAD_SOURCES
    if schema == "unified-h1-r2-config.v3":
        return PRODUCT_SOURCES, WORKLOAD_SOURCES
    raise ValueError("unified-h1 config schema mismatch")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def resource_binding_pins() -> dict[str, str]:
    """Bind reused resource tools independently of the old SCALE product.

    These are the same frozen sources and thresholds; H1 source identity is
    already checked by its own complete product/workload manifests.
    """
    pins = {
        'app_launcher_path':'/home/mint/rl_blackwell_test/repos/IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py',
        'app_launcher_sha256':'7ab2c742cfa64e2c4e2ea4601d9441e5e8c9bb8f556a121b0d10c337f9b6093e',
        'simulation_app_path':'/home/mint/miniconda3/envs/navrl_blackwell_test/lib/python3.11/site-packages/isaacsim/exts/isaacsim.simulation_app/isaacsim/simulation_app/simulation_app.py',
        'simulation_app_sha256':'7cbaa6f00e935a6f14bf1c28ec0db089fd924e931f3b0deee07a822f9b7d0090',
        'nvidia_smi_path':'/usr/bin/nvidia-smi',
        'nvidia_smi_sha256':'22964713c1701fb62b4dd10b26b0dd25d174e100af5bda20c65e0b0fcc32b3be',
        'eval_002_gpu_gate_path':str(Path(__file__).resolve().parents[2]/'evaluation_workloads/eval_002/gpu_gate.py'),
        'eval_002_gpu_gate_sha256':'29fd12bec882ac0b2b0ac8a81142f0102927b63da6aa351328d02fbfe4ed581a'}
    for name in ('app_launcher','simulation_app','nvidia_smi','eval_002_gpu_gate'):
        path = Path(pins[name+'_path'])
        if path.is_symlink() or not path.is_file() or sha256_file(path) != pins[name+'_sha256']:
            raise ValueError('H1 frozen resource source identity mismatch: '+name)
    return pins


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def _exact_mapping(value: object, expected: Mapping[str, Any], label: str) -> None:
    if value != expected: raise ValueError(f"{label} mismatch")


def _validate_repository(spec: object, *, label: str, required_sources: set[str], self_identity: bool = False) -> dict[str, str]:
    required = {"path", "base_commit", "candidate_commit", "candidate_tree", "branch", "source_sha256"}
    if not isinstance(spec, dict) or set(spec) != required: raise ValueError(f"{label} repository schema mismatch")
    root = Path(spec["path"])
    if not root.is_absolute() or root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError(f"{label} repository path is not an exact checkout")
    try: repository_root=Path(_git(root,"rev-parse","--show-toplevel")).resolve()
    except (OSError,subprocess.CalledProcessError) as error: raise ValueError(f"{label} repository is unavailable") from error
    if repository_root not in (root,*root.parents): raise ValueError(f"{label} repository root mismatch")
    actual = {"commit": _git(root, "rev-parse", "HEAD"), "tree": _git(root, "rev-parse", "HEAD^{tree}"),
              "branch": _git(root, "branch", "--show-current")}
    if _git(root, "merge-base", "--is-ancestor", spec["base_commit"], actual["commit"]) != "":
        raise AssertionError("git merge-base emitted unexpected output")
    expected_commit = actual["commit"] if self_identity and spec["candidate_commit"] == "SELF" else spec["candidate_commit"]
    expected_tree = actual["tree"] if self_identity and spec["candidate_tree"] == "SELF" else spec["candidate_tree"]
    if actual != {"commit": expected_commit, "tree": expected_tree, "branch": spec["branch"]}:
        raise ValueError(f"{label} candidate identity mismatch")
    sources = spec["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != required_sources: raise ValueError(f"{label} source manifest set mismatch")
    for relative, expected_hash in sources.items():
        if not isinstance(relative, str) or relative.startswith("/") or ".." in Path(relative).parts:
            raise ValueError(f"{label} source path is invalid")
        source = (root / relative).resolve()
        if root not in source.parents or source.is_symlink() or not source.is_file() or sha256_file(source) != expected_hash:
            raise ValueError(f"{label} source changed: {relative}")
    return actual


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    if not path.is_absolute(): path = path.resolve()
    if path.is_symlink() or not path.is_file(): raise ValueError("config must be a regular resolved file")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema_version", "status", "method", "product", "workload", "runtime", "execution", "actor", "training", "versions", "future_entry"}
    if not isinstance(value, dict) or set(value) != expected or value["schema_version"] not in ("unified-h1-r2-config.v3", R6_CONFIG_SCHEMA):
        raise ValueError("unified-h1 config schema mismatch")
    if value["status"] != "R2_CPU_CANDIDATE_NOT_AUTHORIZED_FOR_LIVE_EXECUTION": raise ValueError("configuration overclaims live authorization")
    method = value["method"]
    expected_method = {"version": "unified-h1-v1", "proposal_path": str(METHOD_PATH), "proposal_sha256": METHOD_SHA256,
                       "hashes": {"method": METHOD_SHA256, **VERSIONS}}
    method_path, method_sha, versions = METHOD_PATH, METHOD_SHA256, VERSIONS
    if isinstance(method, dict) and method.get("version") == "unified-h1-zero-neutrality-v1":
        method_path, method_sha, versions = NEUTRALITY_METHOD_PATH, NEUTRALITY_METHOD_SHA256, NEUTRALITY_VERSIONS
        expected_method = {"version": "unified-h1-zero-neutrality-v1", "proposal_path": str(method_path),
                           "proposal_sha256": method_sha, "hashes": {"method": method_sha, **versions}}
    if isinstance(method, dict) and method.get("version") == "unified-h1-formal-readiness-v1":
        method_path, method_sha, versions = RESEARCH_METHOD_PATH, RESEARCH_METHOD_SHA256, RESEARCH_VERSIONS
        expected_method = {"version": "unified-h1-formal-readiness-v1", "proposal_path": str(method_path), "proposal_sha256": method_sha, "hashes": {"method": method_sha, **versions}}
    _exact_mapping(method, expected_method, "method identity")
    if sha256_file(method_path) != method_sha: raise ValueError("approved method bytes changed")
    _exact_mapping(value["versions"], versions, "method component versions")
    _exact_mapping(value["runtime"], {"live_process_starts": 0, "simulator_initializations": 0, "gpu_probes": 0,
        "real_environment_transitions": 0, "real_training_updates": 0, "real_checkpoints": 0}, "R2 runtime budget")
    execution = value["execution"]
    expected_execution = {"python_path":"/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11",
        "python_sha256":"efa5468dc4ff3850e950cb68c0e430b3a1621e2c6bd9e0ef48f784ba90a9667a",
        "frozen_runtime_root":"/home/mint/rl_blackwell_test",
        "outer_timeout_required":True, "cuda_visible_devices_required":"0",
        "isaacsim_asset_root_env":"ISAACSIM_ASSET_ROOT",
        "isaacsim_asset_root":"/home/mint/rl_dev/instinctRL-dev/runtime_assets/Assets/Isaac/5.1",
        "isaacsim_asset_manifest_sha256":"f64a0561477efcc612761f6b3034f9612f2d3cb42832138e86ff7a4c18e1ef52"}
    _exact_mapping(execution, expected_execution, "future execution environment")
    interpreter = Path(sys.executable).resolve()
    if interpreter != Path(execution["python_path"]).resolve() or sha256_file(interpreter) != execution["python_sha256"]:
        raise ValueError("running interpreter identity differs from the future entry contract")
    runtime_root = Path(execution["frozen_runtime_root"])
    if runtime_root.is_symlink() or not runtime_root.is_dir(): raise ValueError("frozen runtime root is unavailable")
    from .asset_closure import validate_asset_closure
    validate_asset_closure(Path(execution["isaacsim_asset_root"]), execution["isaacsim_asset_manifest_sha256"])
    _exact_mapping(value["actor"], {"keys": ["lidar_grid", "state_vec"], "lidar_shape": [12,59,360], "state_dimension": 52}, "actor boundary")
    _exact_mapping(value["training"], {"num_envs":32, "horizon":16, "samples_per_update":512, "minibatch":64,
        "optimizer_steps_per_model_per_update":8, "seed":20260908, "evaluation_seed":20260909}, "training arithmetic")
    product_sources, workload_sources = required_source_sets(value["schema_version"])
    product = _validate_repository(value["product"], label="product", required_sources=product_sources, self_identity=True)
    workload = _validate_repository(value["workload"], label="workload", required_sources=workload_sources, self_identity=True)
    entry = value["future_entry"]
    if not isinstance(entry, dict) or set(entry) != {"status", "module", "modes", "required_argv", "resource_checks", "budget_fields", "sequences"}:
        raise ValueError("future entry schema mismatch")
    if entry["status"] != "PREPARED_NOT_AUTHORIZED" or entry["module"] != "training_workloads.unified_h1.formal_unified_h1":
        raise ValueError("future entry identity mismatch")
    expected_modes = ["main", "controlled-failure", "r4-training", "evaluation", "r5-resume", "control-diagnostic", "pilot-training", "pilot-evaluation", "enhanced-b0-diagnostic"]
    if method["version"] == "unified-h1-formal-readiness-v1": expected_modes += ["research-training", "research-evaluation", "research-transition", "research-formal-training", "research-formal-evaluation"]
    if value["schema_version"] == R6_CONFIG_SCHEMA:
        if method["version"] != "unified-h1-formal-readiness-v1":
            raise ValueError("R6 config requires frozen research method")
        expected_modes += ["research-r6-evaluation"]
    if entry["modes"] != expected_modes:
        raise ValueError("future mode set mismatch")
    result = dict(value)
    result["resolved_identity"] = {"product_commit": product["commit"], "product_tree": product["tree"],
        "workload_commit": workload["commit"], "workload_tree": workload["tree"], "config_sha256": sha256_file(path)}
    validate_loaded_module_origins(result)
    resource_binding_pins()
    return result


def validate_loaded_module_origins(config: Mapping[str, Any], *, require_product: bool = False) -> None:
    """Reject imports that resolve outside the checkouts validated above."""
    product = Path(config["product"]["path"]).resolve(); workload = Path(config["workload"]["path"]).resolve()
    seen_product = False
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if not filename: continue
        resolved = Path(filename).resolve()
        if name.startswith("training_workloads.unified_h1"):
            if workload not in resolved.parents: raise ValueError(f"loaded workload module escaped validated root: {name}")
        elif name.startswith("instinctlab"):
            seen_product = True
            if product not in resolved.parents: raise ValueError(f"loaded product module escaped validated root: {name}")
    if require_product and not seen_product: raise ValueError("no validated product module is loaded")


def validate_prelaunch_environment(config: Mapping[str, Any]) -> None:
    """Reject statically knowable execution/source mismatches before a probe or AppLauncher."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["execution"]["cuda_visible_devices_required"]:
        raise PermissionError("live CUDA visibility does not match the frozen entry environment")
    from .asset_closure import validate_asset_environment
    validate_asset_environment(config["execution"])
    expected = Path(config["product"]["path"]).resolve() / "isaaclab-training/instinctlab/envs/unified_h1_direct_env.py"
    resolved = None
    for entry in sys.path:
        root = Path(entry or ".").resolve()
        candidate = root / "instinctlab/envs/unified_h1_direct_env.py"
        if candidate.is_file():
            resolved = candidate.resolve(); break
    if resolved != expected:
        raise ValueError("prelaunch product module source does not resolve from the validated checkout")
    validate_loaded_module_origins(config)


def validate_live_environment(config: Mapping[str, Any]) -> None:
    validate_prelaunch_environment(config)
    validate_loaded_module_origins(config, require_product=True)


__all__ = ["CONFIG_PATH", "METHOD_PATH", "METHOD_SHA256", "PRODUCT_SOURCES", "VERSIONS", "WORKLOAD_SOURCES", "load_config", "sha256_file", "validate_live_environment", "validate_loaded_module_origins", "validate_prelaunch_environment"]
