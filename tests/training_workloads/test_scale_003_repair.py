from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from training_workloads.scale_003.config import load_config
from training_workloads.scale_003.lifecycle import persist_failure_semantic, run_lifecycle
from training_workloads.scale_003.live_scene import run_initial_gate
from training_workloads.scale_003.live_training import LiveTrainingBackend
from training_workloads.scale_003.manifest import build_attempt_documents
from training_workloads.scale_003.numeric_diagnostics import analyze_ray_distance_consistency
from training_workloads.scale_003.runtime_support import (
    configure_simulation_logging,
    prepare_runtime_support,
    runtime_environment,
    runtime_support_paths,
)


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11")


def test_failure_lifecycle_subprocess_returns_nonzero_and_retains_cleanup_errors(tmp_path):
    output = tmp_path / "results/training"
    output.mkdir(parents=True)
    marker = tmp_path / "cleanup.txt"
    code = """
from pathlib import Path
import sys
from training_workloads.scale_003.lifecycle import persist_failure_semantic, run_lifecycle
output = Path(sys.argv[1]); marker = Path(sys.argv[2])
def operation():
    raise RuntimeError('original preflight failure')
def cleanup_one():
    marker.write_text('one\\n')
    raise OSError('cleanup one failed')
def cleanup_two():
    with marker.open('a') as stream: stream.write('two\\n')
code = run_lifecycle(operation, [('one', cleanup_one), ('two', cleanup_two)], lambda value: None,
                     lambda outcome: persist_failure_semantic(output, 'training', outcome))
raise SystemExit(code)
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", code, str(output), str(marker)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 1
    assert marker.read_text().splitlines() == ["one", "two"]
    semantic = json.loads((output / "semantic.json").read_text())
    assert semantic["status"] == "FAIL" and semantic["semantic_exit"] == 1
    assert semantic["failure_class"] == "PRODUCT_SEMANTIC_FAILURE"
    assert semantic["primary_error"]["message"] == "original preflight failure"
    assert semantic["cleanup_errors"][0]["message"] == "cleanup one failed"


def test_default_launcher_disables_immediate_shutdown_in_cpu_subprocess(tmp_path):
    marker = tmp_path / "graceful-close.txt"
    code = """
import os, sys, types
from pathlib import Path
marker = Path(sys.argv[1])
class FakeApp:
    def __init__(self, config): self.config = config
    def close(self):
        if self.config.get('fast_shutdown', True): os._exit(0)
        marker.write_text('close returned\\n')
class FakeLauncher:
    def __init__(self, **kwargs): self.app = FakeApp(kwargs)
package = types.ModuleType('isaaclab'); package.__path__ = []
module = types.ModuleType('isaaclab.app'); module.AppLauncher = FakeLauncher
sys.modules['isaaclab'] = package; sys.modules['isaaclab.app'] = module
from training_workloads.scale_003.resources import start_after_coexistence_gate, validate_graceful_shutdown_configuration
gate, launcher, app = start_after_coexistence_gate(
    marker.parent, mode='training', attempt_id='INERT', authorization_sha256='a' * 64,
    gate_runner=lambda *args, **kwargs: {'status': 'PASS'},
)
assert validate_graceful_shutdown_configuration(app)['fast_shutdown'] is False
app.close()
raise SystemExit(0 if marker.is_file() else 2)
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", code, str(marker)], cwd=ROOT, env=env,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0
    assert marker.read_text() == "close returned\n"


def test_existing_runner_reads_failure_semantic_instead_of_reporting_it_missing(tmp_path):
    sys.path.insert(0, "/home/mint/rl_dev/instinctRL_coordination")
    import importlib.util
    from evaluation_tooling.canonical import canonical_load
    from evaluation_tooling.runner import EvaluationRunner
    helper_spec = importlib.util.spec_from_file_location(
        "scale004_runner_helpers",
        "/home/mint/rl_dev/instinctRL_coordination/tests/evaluation_tooling/helpers.py",
    )
    assert helper_spec is not None and helper_spec.loader is not None
    helpers = importlib.util.module_from_spec(helper_spec)
    helper_spec.loader.exec_module(helpers)

    semantic = tmp_path / "attempt/results/result.json"
    source = (
        "from pathlib import Path\n"
        f"path=Path({str(semantic)!r})\n"
        "path.parent.mkdir(parents=True,exist_ok=True)\n"
        "path.write_text('{\\\"failure_class\\\":\\\"PRODUCT_SEMANTIC_FAILURE\\\",\\\"semantic_exit\\\":1,\\\"status\\\":\\\"FAIL\\\"}\\n')\n"
        "raise SystemExit(1)\n"
    )
    command = {
        "label": "failure", "argv": helpers.python_source_argv(tmp_path, "failure", source),
        "cwd": str(tmp_path), "environment": {}, "expected_exit": 0, "timeout_seconds": 5,
        "semantic_result_path": "results/result.json", "cleanup_argv": None,
    }
    attempt = helpers.make_attempt(tmp_path, commands=[command], required_results=["results/result.json"])
    runner = EvaluationRunner(attempt, tooling_commit=helpers.TOOLING_COMMIT, product_commit=helpers.PRODUCT_COMMIT)
    assert runner.prepare()["status"] == "PASS"
    receipt, exit_code = runner.run_command(0)
    persisted = canonical_load(attempt / "launcher_receipts/0001.json")
    assert exit_code == 1 and receipt.semantic_exit == 1
    assert persisted["primary_failure"] != "MISSING_PASS_EVIDENCE"


def _authorization(attempt: Path, document: Path) -> SimpleNamespace:
    return SimpleNamespace(mode="training", attempt_path=attempt.resolve(), attempt_id=attempt.name, document_path=document)


def test_training_live_initialization_failure_is_runner_readable(monkeypatch, tmp_path):
    from training_workloads.scale_003 import live_training

    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    expected = runtime_environment(attempt)
    for name, value in expected.items(): monkeypatch.setenv(name, value)
    monkeypatch.setattr(live_training, "start_after_coexistence_gate", lambda *a, **k: (_ for _ in ()).throw(ValueError("inert initialization failure")))
    assert live_training.run_live(_authorization(attempt, authorization), attempt) == 1
    semantic = json.loads((attempt / "results/training/semantic.json").read_text())
    failure = json.loads((attempt / "results/training/failure.json").read_text())
    assert semantic["status"] == "FAIL" and semantic["semantic_exit"] == 1
    assert semantic["primary_error"]["message"] == "inert initialization failure"
    assert failure["message"] == "inert initialization failure"


def test_training_live_collects_all_cleanup_failures_without_overwriting_primary(monkeypatch, tmp_path):
    from training_workloads.scale_003 import live_training

    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    expected = runtime_environment(attempt)
    for name, value in expected.items(): monkeypatch.setenv(name, value)
    calls: list[str] = []
    class App:
        config = {"fast_shutdown": False}
        def close(self):
            calls.append("app")
            assert json.loads((attempt / "results/training/semantic.json").read_text())["status"] == "FAIL"
            raise OSError("app cleanup failed")
    monkeypatch.setattr(live_training, "start_after_coexistence_gate", lambda *a, **k: ({"status": "PASS"}, object(), App()))
    monkeypatch.setattr(live_training, "run_training", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("original training failure")))
    def close_backend(self):
        calls.append("backend")
        raise OSError("backend cleanup failed")
    monkeypatch.setattr(live_training.LiveTrainingBackend, "close", close_backend)
    assert live_training.run_live(_authorization(attempt, authorization), attempt) == 1
    assert calls == ["backend", "app"]
    semantic = json.loads((attempt / "results/training/semantic.json").read_text())
    assert semantic["primary_error"]["message"] == "original training failure"
    assert [row["message"] for row in semantic["cleanup_errors"]] == ["backend cleanup failed", "app cleanup failed"]


def test_training_live_success_persists_pending_result_then_pass_after_app_close(monkeypatch, tmp_path):
    from training_workloads.scale_003 import live_training

    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    for name, value in runtime_environment(attempt).items(): monkeypatch.setenv(name, value)
    output = attempt / "results/training"
    class App:
        config = {"fast_shutdown": False}
        def close(self):
            assert json.loads((output / "semantic_pending_shutdown.json").read_text())["status"] == "PASS"
            assert json.loads((output / "shutdown_boundary.json").read_text())["status_before_app_close"] == "PENDING_SUCCESS"
            assert not (output / "semantic.json").exists()
    monkeypatch.setattr(live_training, "start_after_coexistence_gate", lambda *a, **k: ({"status": "PASS"}, object(), App()))
    monkeypatch.setattr(live_training, "run_training", lambda *a, **k: {"status": "PASS", "mode": "training"})
    monkeypatch.setattr(live_training, "validate_training_semantic", lambda *a, **k: None)
    assert live_training.run_live(_authorization(attempt, authorization), attempt) == 0
    assert json.loads((output / "semantic.json").read_text())["status"] == "PASS"
    assert json.loads((output / "status.json").read_text())["status"] == "PASS"
    shutdown = json.loads((output / "launcher_shutdown_configuration.json").read_text())
    assert shutdown["fast_shutdown"] is False
    assert set(shutdown["runtime_sources"]) == {"app_launcher", "simulation_app"}


def test_training_live_app_cleanup_error_cannot_leave_false_pass(monkeypatch, tmp_path):
    from training_workloads.scale_003 import live_training

    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    for name, value in runtime_environment(attempt).items(): monkeypatch.setenv(name, value)
    output = attempt / "results/training"
    class App:
        config = {"fast_shutdown": False}
        def close(self): raise OSError("terminal cleanup failed")
    monkeypatch.setattr(live_training, "start_after_coexistence_gate", lambda *a, **k: ({"status": "PASS"}, object(), App()))
    monkeypatch.setattr(live_training, "run_training", lambda *a, **k: {"status": "PASS", "mode": "training"})
    monkeypatch.setattr(live_training, "validate_training_semantic", lambda *a, **k: None)
    assert live_training.run_live(_authorization(attempt, authorization), attempt) == 1
    assert json.loads((output / "semantic_pending_shutdown.json").read_text())["status"] == "PASS"
    semantic = json.loads((output / "semantic.json").read_text())
    assert semantic["status"] == "FAIL" and semantic["failure_class"] == "PROCESS_CRASH"
    assert semantic["cleanup_errors"][0]["message"] == "terminal cleanup failed"


def _evaluation_authorization(attempt: Path, document: Path, checkpoint: Path) -> SimpleNamespace:
    return SimpleNamespace(
        mode="evaluation", attempt_path=attempt.resolve(), attempt_id=attempt.name,
        document_path=document,
        input_handoff=SimpleNamespace(final_checkpoint_path=checkpoint.resolve()),
    )


@pytest.mark.parametrize("scenario, expected_exit", [
    ("success", 0), ("operation_failure", 1), ("initialization_failure", 1), ("cleanup_failure", 1),
])
def test_evaluation_live_shutdown_wiring(monkeypatch, tmp_path, scenario, expected_exit):
    from training_workloads.scale_003 import live_evaluation

    attempt = tmp_path / scenario; (attempt / "results").mkdir(parents=True)
    authorization = tmp_path / f"{scenario}-authorization.json"; authorization.write_text("{}\n")
    checkpoint = tmp_path / f"{scenario}-final.pt"; checkpoint.write_bytes(b"inert")
    for name, value in runtime_environment(attempt).items(): monkeypatch.setenv(name, value)
    output = attempt / "results/station"

    class App:
        config = {"fast_shutdown": False}
        def close(self):
            if scenario == "cleanup_failure":
                raise OSError("evaluation app cleanup failed")
            if scenario == "operation_failure":
                assert json.loads((output / "semantic.json").read_text())["status"] == "FAIL"
            if scenario == "success":
                assert json.loads((output / "semantic_pending_shutdown.json").read_text())["status"] == "PASS"

    if scenario == "initialization_failure":
        start = lambda *a, **k: (_ for _ in ()).throw(ValueError("evaluation initialization failed"))
    else:
        start = lambda *a, **k: ({"status": "PASS"}, object(), App())
    monkeypatch.setattr(live_evaluation, "start_after_coexistence_gate", start)
    if scenario == "operation_failure":
        monkeypatch.setattr(live_evaluation, "run_evaluation", lambda *a, **k: (_ for _ in ()).throw(ValueError("evaluation operation failed")))
    else:
        monkeypatch.setattr(live_evaluation, "run_evaluation", lambda *a, **k: {"status": "PASS", "mode": "evaluation", "case": "station"})

    returned = live_evaluation.run_live(
        _evaluation_authorization(attempt, authorization, checkpoint), attempt, "station", checkpoint,
    )
    assert returned == expected_exit
    semantic = json.loads((output / "semantic.json").read_text())
    assert semantic["status"] == ("PASS" if scenario == "success" else "FAIL")
    if scenario == "cleanup_failure":
        assert semantic["cleanup_errors"][0]["message"] == "evaluation app cleanup failed"


@pytest.mark.parametrize("mode", ["training", "evaluation"])
@pytest.mark.parametrize("scenario, expected_exit", [
    ("success", 0), ("operation_failure", 1), ("initialization_failure", 1), ("cleanup_failure", 1),
])
def test_live_entry_shutdown_matrix_in_cpu_subprocess(mode, scenario, expected_exit, tmp_path):
    case_root = tmp_path / f"{mode}-{scenario}"
    code = """
import json, os, sys
from pathlib import Path
from types import SimpleNamespace
from training_workloads.scale_003.runtime_support import runtime_environment
mode, scenario, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
attempt = root / 'attempt'; (attempt / 'results').mkdir(parents=True)
document = root / 'authorization.json'; document.write_text('{}\\n')
for name, value in runtime_environment(attempt).items(): os.environ[name] = value
output = attempt / 'results' / ('training' if mode == 'training' else 'station')
marker = root / 'app-close-marker'
class App:
    config = {'fast_shutdown': False}
    def close(self):
        if self.config.get('fast_shutdown', True): os._exit(0)
        marker.write_text('entered\\n')
        if scenario == 'operation_failure':
            assert json.loads((output / 'semantic.json').read_text())['status'] == 'FAIL'
        else:
            assert json.loads((output / 'semantic_pending_shutdown.json').read_text())['status'] == 'PASS'
        if scenario == 'cleanup_failure': raise OSError(f'{mode} cleanup failed')
def start(*args, **kwargs):
    if scenario == 'initialization_failure': raise ValueError(f'{mode} initialization failed')
    return {'status': 'PASS'}, object(), App()
if mode == 'training':
    from training_workloads.scale_003 import live_training as live
    live.start_after_coexistence_gate = start
    live.run_training = (lambda *a, **k: (_ for _ in ()).throw(ValueError('training operation failed'))) if scenario == 'operation_failure' else (lambda *a, **k: {'status': 'PASS', 'mode': 'training'})
    live.validate_training_semantic = lambda *a, **k: None
    auth = SimpleNamespace(mode='training', attempt_path=attempt.resolve(), attempt_id=attempt.name, document_path=document)
    result = live.run_live(auth, attempt)
else:
    from training_workloads.scale_003 import live_evaluation as live
    live.start_after_coexistence_gate = start
    live.run_evaluation = (lambda *a, **k: (_ for _ in ()).throw(ValueError('evaluation operation failed'))) if scenario == 'operation_failure' else (lambda *a, **k: {'status': 'PASS', 'mode': 'evaluation', 'case': 'station'})
    checkpoint = root / 'final.pt'; checkpoint.write_bytes(b'inert')
    auth = SimpleNamespace(mode='evaluation', attempt_path=attempt.resolve(), attempt_id=attempt.name, document_path=document, input_handoff=SimpleNamespace(final_checkpoint_path=checkpoint.resolve()))
    result = live.run_live(auth, attempt, 'station', checkpoint)
semantic = json.loads((output / 'semantic.json').read_text())
assert semantic['status'] == ('PASS' if scenario == 'success' else 'FAIL')
assert marker.exists() == (scenario != 'initialization_failure')
if scenario == 'cleanup_failure': assert semantic['cleanup_errors'][0]['message'] == f'{mode} cleanup failed'
print('VERIFIED', mode, scenario, result, flush=True)
raise SystemExit(result)
"""
    env = dict(
        os.environ,
        PYTHONPATH=f"{ROOT}:/home/mint/rl_dev/instinctRL-dev/isaaclab-training",
        CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
    )
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", code, mode, scenario, str(case_root)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == expected_exit, completed.stderr
    assert completed.stdout.strip() == f"VERIFIED {mode} {scenario} {expected_exit}"


def _cpu_ray_source():
    import torch

    return OrderedDict((
        ("ray_starts_w", torch.tensor([[[500.0, 500.0, 1.0]]], dtype=torch.float32)),
        ("ray_directions_w", torch.tensor([[[0.4942087233066559, 0.8693432807922363, 0.0]]], dtype=torch.float32)),
        ("accepted_ray_hits_w", torch.tensor([[[519.2741088867188, 533.9043579101562, 1.0]]], dtype=torch.float32)),
        ("accepted_beam_valid", torch.tensor([[True]], dtype=torch.bool)),
        ("mesh_path", "/World/MID360StaticScene/Scale003CombinedMesh"),
        ("maximum_distance_m", 40.0),
        ("source_step", torch.tensor([1], dtype=torch.long)),
        ("reset_mask", torch.tensor([False], dtype=torch.bool)),
    ))


def test_late_preflight_failure_preserves_geometry_and_raw_source(monkeypatch, tmp_path):
    from instinctlab.evaluation import raycaster_observability
    from training_workloads.scale_003 import live_scene, live_training

    source = _cpu_ray_source()
    monkeypatch.setattr(live_scene, "initial_source", lambda env: source)
    monkeypatch.setattr(
        raycaster_observability,
        "acquire_exact_observability_from_source",
        lambda value: (_ for _ in ()).throw(RuntimeError("injected acquisition failure")),
    )
    geometry = {"mesh_prim_path": source["mesh_path"], "env_origins": [[500.0, 500.0, 0.0]]}
    monkeypatch.setattr(live_training, "live_scene_readback", lambda env: geometry)
    class Env:
        def close(self): pass
    backend = LiveTrainingBackend(tmp_path, SimpleNamespace(), SimpleNamespace(), {})
    backend.env = Env()
    with pytest.raises(RuntimeError, match="injected acquisition failure"):
        backend.run_initial_preflight()
    paths = (
        tmp_path / "initial_preflight/geometry.json",
        tmp_path / "initial_preflight/raw_ray_source.pt",
        tmp_path / "initial_preflight/raw_ray_source.json",
        tmp_path / "initial_preflight/observability_failure.json",
    )
    assert all(path.is_file() for path in paths)
    before = [path.read_bytes() for path in paths]
    backend.close()
    assert [path.read_bytes() for path in paths] == before
    assert not (tmp_path / "initial_preflight/complete.json").exists()
    failure = json.loads(paths[-1].read_text())
    assert failure["completed_stage"] == "raw_ray_source"
    assert "validated_observability_gate" in failure["unfinished"]


def test_gate_validation_failure_preserves_completed_acquisition(monkeypatch, tmp_path):
    import torch
    from instinctlab.evaluation import raycaster_observability
    from training_workloads.scale_003 import live_scene, live_training

    source = _cpu_ray_source()
    acquisition = SimpleNamespace(
        ray_hits_w=source["accepted_ray_hits_w"].clone(),
        ray_distances_m=torch.tensor([[39.0]], dtype=torch.float32),
        raw_surface_normals_w=torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32),
        beam_valid=torch.tensor([[[True]]], dtype=torch.bool),
        hit_max_error_m=torch.tensor(0.0, dtype=torch.float32),
        distance_max_error_m=torch.tensor(0.0, dtype=torch.float32),
        reconstruction_max_error_m=torch.tensor(0.0, dtype=torch.float32),
        observability=SimpleNamespace(
            numerical_rank=torch.tensor([[3]], dtype=torch.int64),
            sigma_min=torch.tensor([[1.0]], dtype=torch.float32),
            singular_values=torch.tensor([[[1.0, 1.0, 1.0]]], dtype=torch.float32),
        ),
    )
    monkeypatch.setattr(live_scene, "initial_source", lambda env: source)
    monkeypatch.setattr(raycaster_observability, "acquire_exact_observability_from_source", lambda value: acquisition)
    monkeypatch.setattr(live_scene, "visibility", lambda *args: {name: [1] * 32 for name in ("floor", "x_wall", "y_wall")})
    monkeypatch.setattr(live_scene, "validate_exact_observability_gate", lambda **kwargs: (_ for _ in ()).throw(ValueError("injected final gate failure")))
    monkeypatch.setattr(live_training, "live_scene_readback", lambda env: {"mesh_prim_path": source["mesh_path"]})
    backend = LiveTrainingBackend(tmp_path, SimpleNamespace(), SimpleNamespace(), {})
    backend.env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros((1, 3), dtype=torch.float32)), close=lambda: None)
    with pytest.raises(ValueError, match="injected final gate failure"):
        backend.run_initial_preflight()
    assert (tmp_path / "initial_preflight/raw_acquisition.pt").is_file()
    assert (tmp_path / "initial_preflight/raw_acquisition.json").is_file()
    acquisition_metadata = json.loads(
        (tmp_path / "initial_preflight/raw_acquisition.json").read_text()
    )
    assert acquisition_metadata["schema_version"] == "scale-004-raw-acquisition.v2"
    assert acquisition_metadata["geometry_comparison_standard"] == "forward-reconstruction-v1"
    assert "reconstruction_max_error_m" in acquisition_metadata["tensors"]
    assert (tmp_path / "initial_preflight/observability_diagnostics.json").is_file()
    failure = json.loads((tmp_path / "initial_preflight/observability_failure.json").read_text())
    assert failure["completed_stage"] == "observability_diagnostics"
    assert not (tmp_path / "initial_preflight/complete.json").exists()


def test_runtime_support_paths_config_and_manifest_are_attempt_confined(monkeypatch, tmp_path):
    attempt = tmp_path / "SCALE-004-INERT"; (attempt / "results").mkdir(parents=True)
    expected = runtime_environment(attempt)
    for name, value in expected.items(): monkeypatch.setenv(name, value)
    receipt = prepare_runtime_support(attempt, mode="training")
    paths = runtime_support_paths(attempt)
    assert all(path.is_dir() for path in paths.values())
    assert receipt["external_inventories_required"] is False
    cfg = SimpleNamespace(sim=SimpleNamespace(log_dir=None))
    configured = configure_simulation_logging(cfg, attempt)
    assert configured == str(paths["logs"].resolve()) == cfg.sim.log_dir

    authorization = tmp_path / "authorization.json"; authorization.write_text("{}\n")
    config = load_config()
    manifest, _ = build_attempt_documents(
        mode="training", attempt_id=attempt.name, attempt_dir=attempt,
        execution_authorization=authorization, checkpoint=Path(config["pins"]["checkpoint_path"]),
        source_dependencies=[], candidate_commit="a" * 40, candidate_tree="b" * 40,
    )
    environment = manifest["commands"][0]["environment"]
    assert all(environment[name] == value for name, value in expected.items())
    assert manifest["external_inventories"] == []
    assert str(ROOT / "training_workloads/scale_003/runtime_support.py") in {row["path"] for row in manifest["commands"] and _["files"]}


def test_evaluation_temp_default_resolves_under_precreated_attempt_root(tmp_path):
    attempt = tmp_path / "attempt"; (attempt / "results").mkdir(parents=True)
    expected = runtime_environment(attempt)
    Path(expected["TMPDIR"]).mkdir(parents=True)
    env = dict(os.environ, **expected, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    completed = subprocess.run(
        [str(PYTHON), "-B", "-c", "import tempfile,sys; print(tempfile.gettempdir()); raise SystemExit(0 if tempfile.gettempdir()==sys.argv[1] else 1)", expected["TMPDIR"]],
        env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0 and completed.stdout.strip() == expected["TMPDIR"]


def test_large_world_float32_diagnostic_has_negative_controls():
    import torch

    start = torch.tensor([[[500.0, 500.0, 1.0]]], dtype=torch.float32)
    direction = torch.tensor([[[0.4942087233066559, 0.8693432807922363, 0.0]]], dtype=torch.float32)
    parameter = torch.tensor([[39.0]], dtype=torch.float32)
    returned = start + parameter.unsqueeze(-1) * direction
    accepted = returned.clone()
    report = analyze_ray_distance_consistency(start, direction, returned, accepted, parameter)
    assert report["reverse_world_subtraction_max_error_m"] == pytest.approx(4.1961669921875e-05, abs=0.0)
    assert report["forward_reconstruction_max_error_m"] == 0.0
    assert report["same_state_hit_max_error_m"] == 0.0
    assert report["decomposition_against_float64_evaluation_of_float32_inputs"]["information_recovery_claimed"] is False
    assert not torch.cuda.is_initialized()

    altered_parameter = torch.tensor([[39.001]], dtype=torch.float32)
    negative = analyze_ray_distance_consistency(start, direction, returned, accepted, altered_parameter)
    assert negative["forward_reconstruction_max_error_m"] > 1.0e-5
    altered_hit = returned.clone(); altered_hit[0, 0, 0] += 1.0e-3
    identity_negative = analyze_ray_distance_consistency(start, direction, returned, altered_hit, parameter)
    assert identity_negative["same_state_hit_max_error_m"] > 1.0e-5
    with pytest.raises((TypeError, ValueError)):
        analyze_ray_distance_consistency(start.to(torch.float64), direction, returned, accepted, parameter)
