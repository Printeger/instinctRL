"""One direct current-product deterministic-mean EVAL-001 process."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any

from attempt_binding import AttemptBinding, binding_from_environment, bound_path
from constants import (
    ACTION_MODE, ACTOR_SHA256, CHECKPOINT_PATH, CHECKPOINT_SHA256,
    DECIMATION, ENVIRONMENT_SEED, ENVIRONMENT_TRANSITIONS, INFERENCE_SEED,
    NUM_ENVS, PHYSICS_STEPS_PER_ENVIRONMENT, POLICY_STEPS, PRODUCT_COMMIT,
    PRODUCT_ROOT, PRODUCT_TREE, RAYS_PER_ENVIRONMENT_PER_STEP, RAY_SAMPLES,
    RUN_IDS, SOURCE_SHA256,
)
from io_utils import (
    atomic_canonical_replace,
    atomic_torch_replace,
    canonical_value,
    finite_tree,
    sha256_file,
)


torch: Any = None


def tensor_hash(value: Any) -> str:
    cpu = value.detach().cpu().contiguous()
    header = f"{cpu.dtype}|{tuple(cpu.shape)}|".encode("ascii")
    return hashlib.sha256(header + cpu.numpy().tobytes(order="C")).hexdigest()


def hash_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        digest.update(type(item).__name__.encode("ascii"))
        digest.update(b"|")
        if isinstance(item, torch.Tensor):
            digest.update(tensor_hash(item).encode("ascii"))
        elif isinstance(item, dict):
            for key, child in item.items():
                visit(key)
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode("utf-8"))

    visit(value)
    return digest.hexdigest()


def module_summary(module: Any) -> OrderedDict[str, Any]:
    state = module.state_dict()
    digest = hashlib.sha256()
    for name, value in state.items():
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(str(tuple(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes(order="C"))
    return OrderedDict((
        ("sha256", digest.hexdigest()),
        ("parameter_count", sum(value.numel() for value in state.values())),
        ("all_finite", all(
            bool(torch.isfinite(value).all().item())
            for value in state.values() if value.is_floating_point()
        )),
    ))


def optimizer_summary(optimizer: Any) -> OrderedDict[str, Any]:
    state = optimizer.state_dict()
    steps = [int(item["step"].item()) for item in state["state"].values()]
    if not steps:
        raise ValueError("optimizer state is empty")
    return OrderedDict((
        ("sha256", hash_tree(state)), ("state_count", len(state["state"])),
        ("step_min", min(steps)), ("step_max", max(steps)),
        ("all_steps_equal", len(set(steps)) == 1),
    ))


def diagnostic_summary(tensors: list[Any], name: str) -> OrderedDict[str, Any]:
    value = torch.stack([item.detach() for item in tensors])
    finite_tree(value, name)
    double = value.double()
    return OrderedDict((
        ("name", name), ("shape", list(value.shape)), ("dtype", str(value.dtype)),
        ("count", value.numel()), ("sum", float(double.sum().item())),
        ("square_sum", float(double.square().sum().item())),
        ("minimum", float(double.min().item())),
        ("maximum", float(double.max().item())),
        ("mean", float(double.mean().item())), ("sha256", tensor_hash(value)),
    ))


def resource_observation() -> OrderedDict[str, Any]:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    if free < 1024 ** 3:
        raise RuntimeError("formal process has less than 1 GiB free CUDA memory")
    properties = torch.cuda.get_device_properties(0)
    return OrderedDict((
        ("device_name", properties.name),
        ("device_capability", list(torch.cuda.get_device_capability(0))),
        ("total_memory_bytes", int(total)), ("free_memory_bytes", int(free)),
        ("memory_allocated_bytes", int(torch.cuda.memory_allocated())),
        ("max_memory_allocated_bytes", int(torch.cuda.max_memory_allocated())),
    ))


def validate_paths(binding: AttemptBinding, run_id: str, checkpoint: Path, output_dir: Path, log_path: Path) -> tuple[Path, Path]:
    expected_output = binding.directory / "results" / run_id
    expected_log = expected_output / "kit.log"
    if checkpoint != CHECKPOINT_PATH or not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("checkpoint path/type mismatch")
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("checkpoint hash mismatch")
    output_dir = bound_path(binding, output_dir, f"results/{run_id}", must_exist=False)
    log_path = bound_path(binding, log_path, f"results/{run_id}/kit.log", must_exist=False)
    if output_dir.exists() or log_path.exists():
        raise FileExistsError("formal output directory or Kit log already exists")
    if output_dir.parent != binding.directory / "results" or not output_dir.parent.is_dir():
        raise ValueError("certified results root is unavailable")
    output_dir.mkdir(mode=0o755)
    return output_dir, log_path


def validate_product_sources() -> None:
    actual = OrderedDict((path, sha256_file(PRODUCT_ROOT / path)) for path in SOURCE_SHA256)
    if actual != SOURCE_SHA256:
        raise ValueError("product source identity mismatch")


def evaluate(run_id: str, checkpoint: Path) -> OrderedDict[str, Any]:
    from instinctlab.envs import TaslabDirectEnv, TaslabDirectEnvCfg
    from instinctlab.evaluation import RuntimeEvaluationCollector, RuntimeEvaluationConfig
    from instinctlab.evaluation.raycaster_observability import (
        acquire_exact_observability_from_source,
        frozen_raycaster_provenance,
    )
    from instinctlab.evaluation.runtime import distill_exact_runtime_observability
    from instinctlab.training import (
        HistoryActor, NetworkConfig, PPOConfig, PPOTrainer, PrivilegedCritic,
        load_checkpoint,
    )

    torch.manual_seed(ENVIRONMENT_SEED)
    torch.cuda.manual_seed_all(ENVIRONMENT_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = TaslabDirectEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    if cfg.decimation != DECIMATION:
        raise ValueError("environment decimation mismatch")
    env = TaslabDirectEnv(cfg)
    try:
        observation, _ = env.reset(seed=ENVIRONMENT_SEED)
        network_config = NetworkConfig()
        ppo_config = PPOConfig(minibatch_size=64, epoch_count=1)
        actor = HistoryActor(network_config).to(env.device)
        critic = PrivilegedCritic(network_config).to(env.device)
        trainer = PPOTrainer(actor, critic, ppo_config)
        load_checkpoint(checkpoint, trainer)
        if trainer.update_counter != 250:
            raise ValueError("checkpoint counter mismatch")
        actor.eval()
        critic.eval()
        before = OrderedDict((
            ("actor", module_summary(actor)), ("critic", module_summary(critic)),
            ("actor_optimizer", optimizer_summary(trainer.actor_optimizer)),
            ("critic_optimizer", optimizer_summary(trainer.critic_optimizer)),
        ))
        if before["actor"]["sha256"] != ACTOR_SHA256:
            raise ValueError("checkpoint actor hash mismatch")
        for name in ("actor_optimizer", "critic_optimizer"):
            if before[name]["step_min"] != 2000 or before[name]["step_max"] != 2000:
                raise ValueError(f"{name} step mismatch")
        torch.manual_seed(INFERENCE_SEED)
        torch.cuda.manual_seed_all(INFERENCE_SEED)
        torch.set_grad_enabled(False)
        if torch.is_grad_enabled():
            raise RuntimeError("global gradients remain enabled")
        collector = RuntimeEvaluationCollector(
            observation["critic"].detach().clone(),
            RuntimeEvaluationConfig(num_envs=NUM_ENVS, expected_policy_steps=POLICY_STEPS),
        )
        resources_before = resource_observation()
        actions: list[Any] = []
        alphas: list[Any] = []
        betas: list[Any] = []
        entropies: list[Any] = []
        log_probabilities: list[Any] = []
        rewards: list[Any] = []
        termination_reason_counts = torch.zeros(5, dtype=torch.long)
        reset_count = 0
        for _step in range(1, POLICY_STEPS + 1):
            audited = collector.prepare_policy(observation["policy"])
            if tuple(audited) != ("lidar_grid", "state_vec"):
                raise ValueError("actor policy keys/order mismatch")
            actor_output = actor(audited)
            action = actor_output.mean_action
            log_probability = actor.log_probability_from_output(actor_output, action)
            entropy = actor.entropy_from_output(actor_output)
            if action.shape != (NUM_ENVS, 4) or action.dtype != torch.float32:
                raise ValueError("action shape/dtype mismatch")
            if not bool(torch.isfinite(action).all().item()) or bool(((action < 0.0) | (action > 1.0)).any().item()):
                raise ValueError("action range/finite mismatch")
            action = action.detach()
            unchanged = action.clone()
            observation, reward, terminated, truncated, _ = env.step(action)
            if not torch.equal(action, unchanged):
                raise RuntimeError("environment step mutated action")
            pre_reset = env.extras["pre_reset"]
            ray_source = env.extras["evaluation_ray_source"]
            acquisition = acquire_exact_observability_from_source(ray_source)
            exact_observability = distill_exact_runtime_observability(ray_source, acquisition)
            collector.record_transition(
                pre_action_policy=audited, action=action, pre_reset=pre_reset,
                exact_observability=exact_observability,
                returned_policy=observation["policy"],
                returned_critic=observation["critic"].detach().clone(),
                returned_episode_length=env.episode_length_buf.detach().clone(),
            )
            reasons = pre_reset["task"]["termination"]["reason_code"][:, 0].detach().cpu()
            termination_reason_counts += torch.bincount(reasons, minlength=5)
            reset_count += int((terminated | truncated).sum().item())
            actions.append(action.cpu().clone())
            alphas.append(actor_output.alpha.cpu().clone())
            betas.append(actor_output.beta.cpu().clone())
            entropies.append(entropy.cpu().clone())
            log_probabilities.append(log_probability.cpu().clone())
            rewards.append(reward.detach().cpu().clone())
        provenance = OrderedDict((
            ("product_commit", PRODUCT_COMMIT),
            ("fixture_id", "eval-001-static-plane-zero-command-deterministic-mean-v2"),
            ("generator_id", "evaluation_workloads/eval_001/formal_evaluator.py"),
            ("seed", INFERENCE_SEED),
        ))
        evaluation = collector.finalize(provenance)
        resources_after = resource_observation()
        after = OrderedDict((
            ("actor", module_summary(actor)), ("critic", module_summary(critic)),
            ("actor_optimizer", optimizer_summary(trainer.actor_optimizer)),
            ("critic_optimizer", optimizer_summary(trainer.critic_optimizer)),
        ))
        if before != after:
            raise RuntimeError("actor/critic/optimizer state mutated")
        gradients = OrderedDict((
            ("actor_grad_tensor_count", sum(parameter.grad is not None for parameter in actor.parameters())),
            ("critic_grad_tensor_count", sum(parameter.grad is not None for parameter in critic.parameters())),
            ("global_gradient_enabled", torch.is_grad_enabled()),
        ))
        expected_gradients = OrderedDict((
            ("actor_grad_tensor_count", 0), ("critic_grad_tensor_count", 0),
            ("global_gradient_enabled", False),
        ))
        if gradients != expected_gradients:
            raise RuntimeError("formal inference created gradients")
        semantic = OrderedDict((
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
            ("checkpoint_state_before", before), ("checkpoint_state_after", after),
            ("gradients", gradients),
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
            ("action_diagnostics", diagnostic_summary(actions, "action")),
            ("alpha_diagnostics", diagnostic_summary(alphas, "alpha")),
            ("beta_diagnostics", diagnostic_summary(betas, "beta")),
            ("entropy_diagnostics", diagnostic_summary(entropies, "entropy")),
            ("log_probability_diagnostics", diagnostic_summary(log_probabilities, "log_probability")),
            ("reward_diagnostics", diagnostic_summary(rewards, "reward")),
            ("termination_reason_counts", termination_reason_counts),
            ("reset_count", reset_count),
            ("cuda_resources", OrderedDict((
                ("before", resources_before), ("after", resources_after),
            ))),
            ("raycaster_provenance", frozen_raycaster_provenance()),
            ("runtime_evaluation", evaluation),
        ))
        finite_tree(semantic)
        return semantic
    finally:
        env.close()


def persist_success_evidence(
    output_dir: Path,
    log_path: Path,
    semantic: OrderedDict[str, Any],
    *,
    run_id: str,
    pid: int,
    start: str,
    kit_setting: str,
) -> OrderedDict[str, Any]:
    """Durably persist semantic evidence before the terminal runtime close."""

    if kit_setting != str(log_path):
        raise RuntimeError("Kit /log/file setting changed before persistence")
    semantic_path = output_dir / "semantic.pt"
    mirror_path = output_dir / "semantic.json"
    atomic_torch_replace(semantic_path, semantic)
    atomic_canonical_replace(mirror_path, canonical_value(semantic))
    receipt = OrderedDict((
        ("schema_version", "eval-001-process-receipt.v3"),
        ("run_id", run_id), ("pid", pid), ("start", start),
        ("semantic_completed_at", datetime.now().astimezone().isoformat(timespec="microseconds")),
        ("exit_status", "PASS"), ("kit_setting", kit_setting),
        ("kit_log", str(log_path)), ("semantic_pt", str(semantic_path)),
        ("semantic_pt_sha256", sha256_file(semantic_path)),
        ("semantic_json", str(mirror_path)),
        ("semantic_json_sha256", sha256_file(mirror_path)),
    ))
    receipt_path = output_dir / "receipt.json"
    atomic_canonical_replace(receipt_path, receipt)
    status = OrderedDict((
        ("schema_version", "eval-001-formal-status.v2"), ("status", "PASS"),
        ("run_id", run_id), ("semantic_exit", 0), ("pid", pid),
        ("semantic_pt_sha256", receipt["semantic_pt_sha256"]),
        ("semantic_json_sha256", receipt["semantic_json_sha256"]),
        ("process_receipt_sha256", sha256_file(receipt_path)),
    ))
    atomic_canonical_replace(output_dir / "status.json", status)
    return status


def persist_failure_status(output_dir: Path, error: BaseException) -> None:
    atomic_canonical_replace(output_dir / "status.json", {
        "failure_class": "PRODUCT_SEMANTIC_FAILURE",
        "message": f"{type(error).__name__}: {error}",
        "semantic_exit": 1,
        "status": "FAIL",
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", choices=RUN_IDS, required=True)
    parser.add_argument("--action-mode", choices=(ACTION_MODE,), required=True)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    start = datetime.now().astimezone().isoformat(timespec="microseconds")
    pid = os.getpid()
    output_dir: Path | None = None
    log_path: Path | None = None
    simulation_app: Any = None
    kit_setting: Any = None
    exit_code = 1
    try:
        binding = binding_from_environment()
        if (args.action_mode, args.environment_seed, args.inference_seed) != (ACTION_MODE, ENVIRONMENT_SEED, INFERENCE_SEED):
            raise ValueError("formal arguments violate fixed deterministic-mean campaign")
        validate_product_sources()
        try:
            declared_output = Path(os.environ["EVAL_OUTPUT_DIR"])
            declared_log = Path(os.environ["EVAL_KIT_LOG"])
        except KeyError as error:
            raise ValueError(f"missing exact formal path environment: {error.args[0]}") from error
        output_dir, log_path = validate_paths(
            binding, args.run_id, args.checkpoint, declared_output, declared_log,
        )
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(
            headless=True, enable_cameras=False,
            kit_args=f"--/log/file={log_path}",
        )
        simulation_app = launcher.app
        import carb

        kit_setting = carb.settings.get_settings().get("/log/file")
        if kit_setting != str(log_path):
            raise RuntimeError(f"Kit /log/file mismatch: {kit_setting!r}")
        global torch
        import torch as torch_module

        torch = torch_module
        semantic = evaluate(args.run_id, args.checkpoint)
        status = persist_success_evidence(
            output_dir,
            log_path,
            semantic,
            run_id=args.run_id,
            pid=pid,
            start=start,
            kit_setting=kit_setting,
        )
        print(json.dumps(status, sort_keys=True), flush=True)
        exit_code = 0
    except BaseException as error:
        if output_dir is not None:
            try:
                persist_failure_status(output_dir, error)
            except BaseException:
                pass
        traceback.print_exception(error, file=sys.stderr)
    if simulation_app is None:
        return exit_code
    # This is deliberately the final call: some Kit runtimes terminate here.
    simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
