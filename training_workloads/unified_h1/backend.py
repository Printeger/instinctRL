"""Lazy live backend for R3 and the later R4/R5 training/evaluation calls."""

from __future__ import annotations

import hashlib
from functools import partial
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping

import numpy as np
import torch

from training_workloads.scale_003.lifecycle import LifecycleOutcome, exception_record
from training_workloads.scale_003.persistence import atomic_json_replace
from training_workloads.scale_003.resources import RuntimeResourceGuard, run_prelaunch_gate, start_after_coexistence_gate, validate_graceful_shutdown_configuration

from .budget import ExecutionCounters
from .config import load_config, sha256_file, validate_live_environment
from .control_diagnostic import CASES as DIAGNOSTIC_CASES, DiagnosticChunkWriter, TIMING_SEMANTICS, fixed_b0_action, select_actor_input
from .driver import UnifiedH1Driver, UnifiedH1TrainingLoop
from .evaluation import CASES, EpisodeTrace, case_commands, evaluate_checkpoint, evaluate_episode
from .logging import LearningLog
from .pilot_contract import pilot_actor_learning_rate, pilot_residual_weight, pilot_method_hashes
from .preflight import initial_source, live_geometry_readback, persist_initial_source, validate_initial_observability
from .schedule import ScheduleClock, command_at


def _state_fingerprint(value: Any) -> str:
    """Canonical digest for nested tensor state used only as reload evidence."""
    digest = hashlib.sha256()
    def update(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0" + repr(tuple(tensor.shape)).encode() + b"\0")
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()); return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda candidate: repr(candidate)):
                update(key); update(item[key])
            return
        if isinstance(item, (tuple, list)):
            digest.update(("tuple" if isinstance(item, tuple) else "list").encode() + b"\0")
            for child in item: update(child)
            return
        digest.update(type(item).__name__.encode() + b"\0" + repr(item).encode() + b"\0")
    update(value)
    return digest.hexdigest()


class UnifiedH1LiveBackend:
    """The authorized Isaac session behind the backend-independent sequence."""

    def __init__(self, *, authorization: Mapping[str, Any], attempt: Path, app: Any,
                 environment: Any, resource_guard: RuntimeResourceGuard,
                 counters: ExecutionCounters | None = None,
                 process_started_monotonic: float | None = None) -> None:
        self.research = authorization["parameters"] if authorization["mode"].startswith("research-") else None
        self.authorization = authorization; self.attempt = attempt; self.app = app
        self.environment = environment; self.resource_guard = resource_guard
        self.output = attempt / "results" / str(authorization["mode"])
        if not self.output.is_dir(): raise ValueError("lifecycle must prepare the mode output before resource construction")
        self.counters = counters or ExecutionCounters(); self.observation = None; self.trainer = None
        self.driver = None; self.loop = None; self.curriculum_generator = torch.Generator(device="cpu").manual_seed(self.research["training_seed"] if self.research else 20260908)
        self.journal: LearningLog | None = None
        self._evaluation_rows: dict[str, list[EpisodeTrace]] = {}
        now = time.monotonic()
        self._started = now if process_started_monotonic is None else float(process_started_monotonic)
        if not 0 < self._started <= now: raise ValueError("process wall-clock boundary is invalid")
        self._evaluation_update: int | None = None
        self.initialization_measurement: dict[str, Any] = {
            "status":"NOT_REACHED", "observed_physics_ticks":None,
            "measurement_boundary":"after_environment_construction_before_explicit_reset_or_policy_step",
            "reserved_simulator_initializations":1,
        }

    def resource_check(self, stage: str, ordinal: int) -> None:
        if time.monotonic() - self._started > self.authorization["budgets"]["wall_seconds"]:
            raise TimeoutError("authorized wall-time budget exhausted")
        self.resource_guard.check(stage=stage, ordinal=ordinal)

    def initial_raw_rays(self) -> None:
        source = initial_source(self.environment)
        persist_initial_source(self.output, source)
        atomic_json_replace(self.output, "initial_raw_rays.json", {
            "schema_version": "unified-h1-initial-rays.v2", "envs": 32,
            "mesh_path": source["mesh_path"],
            "valid_counts": source["accepted_beam_valid"].sum(dim=1).detach().cpu().tolist(),
        })

    def initial_preflight(self) -> None:
        geometry = live_geometry_readback(self.environment)
        atomic_json_replace(self.output, "initial_preflight/geometry.json", geometry)
        source = initial_source(self.environment); persist_initial_source(self.output, source)
        observability = validate_initial_observability(source, self.output)
        atomic_json_replace(self.output, "initial_preflight.json", {
            "schema_version": "unified-h1-preflight.v1", "status": "PASS",
            "scene": "combined-finite-floor-x-wall-y-wall", "num_envs": 32,
            "geometry": geometry, "observability": observability,
        })

    def evaluate_b0(self, case: str) -> None:
        self._evaluate_case(case, direct_b0=True, evaluation_label="b0")

    def _preserving_reset(self, *, seed: int) -> tuple[Any, Mapping[str, Any]]:
        cpu = torch.get_rng_state().clone(); py = random.getstate(); numpy = np.random.get_state()
        cuda = torch.cuda.get_rng_state_all() if torch.device(self.environment.device).type == "cuda" else []
        self.counters.reserve_reset(automatic=False, maximum=self.authorization["budgets"]["explicit_resets"])
        try:
            return self.environment.reset(seed=seed)
        finally:
            random.setstate(py); np.random.set_state(numpy); torch.set_rng_state(cpu)
            if cuda: torch.cuda.set_rng_state_all(cuda)

    def _evaluate_case(self, case: str, *, direct_b0: bool, evaluation_label: str) -> None:
        evaluation_seed = self.research["evaluation_seed"] if getattr(self, "research", None) else 20260909
        if case not in CASES: raise ValueError("unknown B0 case")
        self.environment.set_h1_evaluation_case(case)
        commands = case_commands(case).to(self.environment.device)
        self.environment.set_h1_initial_command(commands[0], 0)
        self.resource_check("before_evaluation_reset", self.counters.explicit_resets + 1)
        observation, _ = self._preserving_reset(seed=evaluation_seed)
        self.environment._refresh_task_snapshot()
        initial_source_snapshot = self.environment._evaluation_ray_snapshot
        initial_sensor = initial_source_snapshot.ray_starts_w[:, 0].detach().cpu()
        initial_root_state = self.environment._robot.data.root_state_w.detach().cpu().clone()
        raw_log = LearningLog(self.output / "evaluation_raw")
        raw_relative = f"{evaluation_label}/{case}.jsonl"
        raw_log.append(raw_relative, {"schema_version":"metric-h1-v1.raw-trace.v1", "kind":"T0",
            "case":case, "seed":evaluation_seed, "initial_sensor_position_w_m":initial_sensor,
            "initial_root_state_w":initial_root_state, "env_origins_w_m":self.environment.scene.env_origins.detach().cpu()})
        episode_ids = torch.zeros(32, dtype=torch.int64)
        lane_rows = [[] for _ in range(32)]
        for step in range(CASES[case].steps):
            self.resource_check("before_evaluation_step", step)
            self.counters.require_step_capacity(kind="nontraining", lanes=32,
                maximum=self.authorization["budgets"]["nontraining_transitions"])
            self.counters.require_reset_capacity(automatic=True, lanes=32,
                maximum=self.authorization["budgets"]["automatic_resets"])
            current = commands[step]; following = commands[min(step + 1, len(commands) - 1)]
            if direct_b0:
                self.environment.prepare_b0_step(command_b=current, next_command_b=following)
                action = torch.zeros(32, 4, device=self.environment.device)
            else:
                self.environment.prepare_h1_step(global_policy_step=step, command_b=current, next_command_b=following)
                with torch.no_grad(): action = self.trainer.actor(observation["policy"]).mean_action.detach()
            self.counters.attempt_step()
            atomic_json_replace(self.output, "evaluation_progress.json", {"case":case,"step":step,
                "attempted_policy_steps":self.counters.attempted_policy_steps,
                "completed_policy_steps":self.counters.completed_policy_steps,
                "completed_transitions":self.counters.real_environment_transitions})
            observation, _, terminated, truncated, extras = self.environment.step(action)
            self.counters.complete_step(32, kind="nontraining")
            pre = extras["pre_reset"]; source = extras["evaluation_ray_source"]
            q = extras["reward_diagnostics"]["q"]
            ics_beta = pre["control"]["ics_metrics"]["beta"]
            if (not isinstance(q, torch.Tensor) or q.shape != (32, 1)
                    or not isinstance(ics_beta, torch.Tensor) or ics_beta.shape != (32, 1)):
                raise ValueError("product q and ICS beta must have the approved [32,1] shape")
            reset_mask = (terminated | truncated).detach().cpu()
            self.counters.reserve_reset(automatic=True, lanes=int(reset_mask.sum()),
                maximum=self.authorization["budgets"]["automatic_resets"])
            raw_log.append(raw_relative, {"schema_version":"metric-h1-v1.raw-trace.v1", "kind":"STEP",
                "case":case, "step":step, "command_b":current, "actual_velocity_b":pre["task"]["actual_velocity_b"],
                "sensor_positions_w_m":source["ray_starts_w"][:,0], "episode_ids":episode_ids,
                "collision":pre["task"]["termination"]["masks"]["collision"][:,0],
                "below_height":pre["task"]["termination"]["masks"]["below_height"][:,0],
                "above_height":pre["task"]["termination"]["masks"]["above_height"][:,0],
                "reset_mask":reset_mask, "ray_clearance_m":pre["task"]["exact_clearance_m"][:,0],
                "q":q[:, 0], "ics_beta":ics_beta[:, 0],
                **({"issued_action":action,"v_corr_b":pre["control"]["v_corr_b"],
                    "v_final_b":pre["control"]["v_final_b"],"v_final_w":pre["control"]["v_final_w"],
                    "root_state_w":pre["root_state_w"],"anchor_metrics":pre["control"]["anchor_metrics"],
                    "diagnostic_timing":"action_t staged before integration; pre_reset root/control and same-state sensor returned at t+1; no additional sampling"}
                   if getattr(self,"research",None) else {})})
            for lane in range(32):
                lane_rows[lane].append({"command": current[lane].detach().cpu(),
                    "velocity": pre["task"]["actual_velocity_b"][lane].detach().cpu(),
                    "sensor": source["ray_starts_w"][lane, 0].detach().cpu(),
                    "episode": int(episode_ids[lane]), "collision": pre["task"]["termination"]["masks"]["collision"][lane, 0].detach().cpu(),
                    "below": pre["task"]["termination"]["masks"]["below_height"][lane, 0].detach().cpu(),
                    "above": pre["task"]["termination"]["masks"]["above_height"][lane, 0].detach().cpu(),
                    "ray": pre["task"]["exact_clearance_m"][lane, 0].detach().cpu(),
                    "q": q[lane, 0].detach().cpu(), "ics": ics_beta[lane, 0].detach().cpu()})
            episode_ids += (terminated | truncated).detach().cpu().to(torch.int64)
        origins = self.environment.scene.env_origins.detach().cpu()
        traces = []
        for lane, rows in enumerate(lane_rows):
            traces.append(EpisodeTrace(case, lane, torch.stack([r["command"] for r in rows]),
                torch.stack([r["velocity"] for r in rows]), torch.stack([r["sensor"] for r in rows]), origins[lane],
                tuple(r["episode"] for r in rows), torch.stack([r["collision"] for r in rows]),
                torch.stack([r["below"] for r in rows]), torch.stack([r["above"] for r in rows]),
                torch.zeros(len(rows), dtype=torch.bool), torch.stack([r["ray"] for r in rows])))
            traces[-1] = EpisodeTrace(**{**vars(traces[-1]),
                "initial_sensor_position_w_m": initial_sensor[lane],
                "initial_root_position_w_m": initial_root_state[lane, :3],
                "initial_velocity_b": initial_root_state[lane, 7:10],
                "initial_root_quaternion_wxyz": initial_root_state[lane, 3:7],
                "sample_policy_steps": tuple(range(len(rows))), "seed": evaluation_seed,
                "q": torch.stack([r["q"] for r in rows]), "ics_beta": torch.stack([r["ics"] for r in rows])})
        self._evaluation_rows[case] = traces
        prefix = "b0" if direct_b0 else "policy"
        summary_name = f"{evaluation_label}_{case}.json" if direct_b0 else f"{evaluation_label}_{prefix}_{case}.json"
        atomic_json_replace(self.output, summary_name, {"schema_version": f"metric-h1-v1.{prefix}.v1",
                            "case": case, "episodes": [evaluate_episode(trace) for trace in traces]})

    def cold_start(self, *, pilot_group: str | None = None) -> None:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
        from .pilot_diagnostic import FirstUpdateDiagnostic
        from instinctlab.training.networks_h1 import PrivilegedCriticH1

        research = getattr(self, "research", None)
        training_seed = research["training_seed"] if research else 20260908
        torch.manual_seed(training_seed)
        network = NetworkConfig()
        actor_lr = research["actor_lr"] if research else (3.0e-4 if pilot_group is None else pilot_actor_learning_rate(pilot_group))
        if research: self._apply_research_profile()
        if pilot_group in {"C0", "C1"}:
            self.environment.set_h1_residual_neutrality_weight(pilot_residual_weight(pilot_group))
            if self.environment._h1_residual_neutrality_weight != self.authorization["parameters"]["residual_neutrality_weight"]:
                raise RuntimeError("actual reward coefficient differs from authorization")
            atomic_json_replace(self.output, "pilot_reward_binding.json", {
                "group": pilot_group, "actual_residual_neutrality_weight": self.environment._h1_residual_neutrality_weight,
                "authorization_sha256": self.authorization["authorization_sha256"],
                "method_hashes": pilot_method_hashes(load_config(), pilot_group)})
        ppo = PPOConfig(minibatch_size=64, epoch_count=1, actor_learning_rate=actor_lr, critic_learning_rate=3.0e-4)
        self.trainer = PPOTrainer(HistoryActor(network).to(self.environment.device), PrivilegedCriticH1(network).to(self.environment.device), ppo)
        from .mean_prior import configure_trainer
        configure_trainer(self.trainer, research)
        if research and 'mean_prior_weight' in research:
            atomic_json_replace(self.output, 'actor_mean_prior_binding.json', {
                'mean_prior_contract': research['mean_prior_contract'],
                'actual_weight': research['mean_prior_weight'],
                'hook_installed': self.trainer.actor_auxiliary is not None,
                'actor_only': True, 'runtime_action_replacement': False,
                'authorization_sha256': self.authorization['authorization_sha256']})
        self.environment.set_h1_evaluation_case(None)
        initial = command_at(0, device=self.environment.device)
        self.environment.set_h1_initial_command(initial, 0)
        self.resource_check("before_training_reset", self.counters.explicit_resets + 1)
        self.observation, _ = self._preserving_reset(seed=training_seed)
        journal = LearningLog(self.output / "learning_log"); self.journal = journal
        self.driver = UnifiedH1Driver(self.environment, self.observation, clock=ScheduleClock(0, pure_hover=bool(research)), counters=self.counters, journal=journal,
            before_step=lambda ordinal: self.resource_check("before_training_step", ordinal),
            transition_limit=self.authorization["budgets"]["training_transitions"],
            automatic_reset_limit=self.authorization["budgets"]["automatic_resets"])
        diagnostic = FirstUpdateDiagnostic(self.output / "pilot_diagnostic", enabled=pilot_group is not None or bool(research))
        self.loop = UnifiedH1TrainingLoop(self.driver, self.trainer, journal, self._write_checkpoint,
            before_rollout=lambda ordinal: self.resource_check("before_training_rollout", ordinal),
            before_optimizer=lambda ordinal: self.resource_check("before_optimizer_update", ordinal),
            update_limit=self.authorization["budgets"]["optimizer_updates"], diagnostic=diagnostic)
        if pilot_group is not None or research:
            actual = {"actor":[float(group["lr"]) for group in self.trainer.actor_optimizer.param_groups],
                      "critic":[float(group["lr"]) for group in self.trainer.critic_optimizer.param_groups]}
            if actual != {"actor":[actor_lr],"critic":[3.0e-4]}: raise RuntimeError("pilot optimizer learning rates differ from the bound group")
            atomic_json_replace(self.output,"pilot_optimizer_lrs.json",{"schema_version":"instinctrl-pilot-optimizer-lrs/1","group":pilot_group,"actual":actual})

    def _write_checkpoint(self, label: str, trainer: Any, driver: UnifiedH1Driver) -> None:
        from instinctlab.training.checkpoint_h1 import save_checkpoint_h1
        self.resource_check("before_checkpoint_write", self.counters.real_checkpoint_writes + 1)
        self.counters.reserve_checkpoint(self.authorization["budgets"]["checkpoint_writes"])
        config = load_config(); target = (self.output / "checkpoints" / f"{label}.pt").resolve(); target.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint_h1(target, trainer, method_hashes=self._checkpoint_method_hashes(config),
            write_guard=__import__("training_workloads.scale_003.persistence", fromlist=["guard_stream"]).guard_stream,
            source_identity=config["resolved_identity"], transition_counter=trainer.update_counter * 512,
            global_policy_step=trainer.update_counter * 16,
            curriculum={"version": ("curriculum-h1-research-pure-hover-v1" if getattr(self,"research",None) and not getattr(self,"_formal_segment",False) else "curriculum-h1-v1"), "completed_policy_steps": trainer.update_counter * 16},
            training_seed=(self.research["training_seed"] if getattr(self,"research",None) else 20260908), reset_number=getattr(self,"_checkpoint_reset_offset",0) + self.counters.explicit_resets + self.counters.automatic_resets,
            curriculum_generator=self.curriculum_generator)
        self.counters.real_checkpoint_writes += 1
        atomic_json_replace(self.output, f"checkpoints/{label}.json", {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size})

    def save_checkpoint(self, label: str) -> None:
        if self.trainer is None or self.driver is None: raise RuntimeError("trainer is not initialized")
        self._write_checkpoint(label, self.trainer, self.driver)

    def smoke_update(self, profile: str) -> None:
        if self.loop is None: raise RuntimeError("training loop is not initialized")
        if profile not in {"zero", "mixed", "role-swap-a", "role-swap-b"}: raise ValueError("unknown smoke profile")
        starts = {"zero": 0, "mixed": 1024, "role-swap-a": 1152, "role-swap-b": 1280}
        start = starts[profile]
        # R3 is a diagnostic four-update profile, not the contiguous production
        # curriculum.  Replace only the command slots in the already-returned
        # observation: no extra sensor read or actor-history advance occurs.
        self.driver.clock = ScheduleClock(start)
        command = command_at(start, device=self.environment.device,
                             dtype=self.driver.observation["policy"]["state_vec"].dtype)
        self.environment.set_h1_initial_command(command, start)
        policy = {key: value.clone() for key, value in self.driver.observation["policy"].items()}
        policy["state_vec"][:, 45:48] = command
        critic = self.driver.observation["critic"].clone(); critic[:, -3:] = command
        internal = getattr(self.environment, "_policy", {}).get("state_vec")
        if (not isinstance(internal, torch.Tensor) or not torch.equal(internal[:, 45:48], command)
                or not torch.equal(getattr(self.environment, "_command_b", None), command)):
            raise RuntimeError("environment did not synchronize the diagnostic command")
        self.driver.observation = {**self.driver.observation, "policy": policy, "critic": critic}
        before = self.counters.training_transitions
        self.loop.run_update()
        atomic_json_replace(self.output, f"smoke_profiles/{profile}.json", {
            "schema_version":"unified-h1-r3-diagnostic-profile.v1", "profile":profile,
            "production_curriculum":False, "diagnostic_clock_start":start,
            "diagnostic_clock_end":self.driver.clock.completed_policy_steps,
            "training_transitions":self.counters.training_transitions - before})

    def reload_final(self) -> None:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
        from instinctlab.training.checkpoint_h1 import load_checkpoint_h1
        from instinctlab.training.networks_h1 import PrivilegedCriticH1
        if self.trainer is None: raise RuntimeError("trainer is not initialized")
        original = self.trainer
        before = {"actor":_state_fingerprint(original.actor.state_dict()),
                  "critic":_state_fingerprint(original.critic.state_dict()),
                  "actor_optimizer":_state_fingerprint(original.actor_optimizer.state_dict()),
                  "critic_optimizer":_state_fingerprint(original.critic_optimizer.state_dict())}
        config = load_config(); network = NetworkConfig(); loaded = PPOTrainer(HistoryActor(network).to(self.environment.device), PrivilegedCriticH1(network).to(self.environment.device), PPOConfig(minibatch_size=64, epoch_count=1))
        checkpoint = (self.output / "checkpoints/smoke_final.pt").resolve()
        load_checkpoint_h1(checkpoint, loaded,
            expected_method_hashes=config["method"]["hashes"], expected_source_identity=config["resolved_identity"],
            curriculum_generator=torch.Generator(device="cpu"))
        after = {"actor":_state_fingerprint(loaded.actor.state_dict()),
                 "critic":_state_fingerprint(loaded.critic.state_dict()),
                 "actor_optimizer":_state_fingerprint(loaded.actor_optimizer.state_dict()),
                 "critic_optimizer":_state_fingerprint(loaded.critic_optimizer.state_dict())}
        exact = {name: before[name] == after[name] for name in before}
        if not all(exact.values()) or loaded.update_counter != original.update_counter:
            raise RuntimeError("final checkpoint reload differs from the saved trainer state")
        atomic_json_replace(self.output, "reload_consistency.json", {
            "schema_version":"unified-h1-reload-consistency.v1", "checkpoint":str(checkpoint),
            "checkpoint_sha256":sha256_file(checkpoint), "original_state_sha256":before,
            "loaded_state_sha256":after, "exact":exact,
            "original_update":original.update_counter, "loaded_update":loaded.update_counter})
        self.trainer = loaded

    def train_to_update(self, target_update: int, *, checkpoint_interval: int = 64) -> None:
        """R4/R5 training path with fixed 64-update nodes and no hidden retry."""
        if self.loop is None or self.trainer is None: raise RuntimeError("training loop is not initialized")
        if target_update < self.trainer.update_counter or target_update > 2048 or checkpoint_interval != 64:
            raise ValueError("training target exceeds the frozen R4/R5 nodes")
        while self.trainer.update_counter < target_update:
            ordinal = self.trainer.update_counter + 1; self.resource_check("before_training_update", ordinal)
            self.loop.run_update(checkpoint_label=f"u{ordinal:04d}" if ordinal % checkpoint_interval == 0 else None)


    def _checkpoint_method_hashes(self, config):
        if self.authorization["mode"] == "research-r6-evaluation":
            from .robustness_contract import baseline
            return baseline(self.research)["method_hashes"]
        if self.authorization['mode'] in {'research-formal-training', 'research-formal-evaluation'}:
            from .formal_segment import method_hashes
            return method_hashes(config, self.research)
        if getattr(self, "research", None):
            from .research_contract import method_hashes
            return method_hashes(config, self.research)
        return pilot_method_hashes(config, self.authorization["parameters"].get("group"))

    def _apply_research_profile(self):
        p = self.research
        self.environment.set_h1_residual_neutrality_weight(p["residual_neutrality_weight"])
        self.environment.set_h1_research_profile(pure_hover=True, anchor_weight=p["anchor_weight"])
        if self.environment._h1_residual_neutrality_weight != p["residual_neutrality_weight"] or self.environment._h1_anchor_weight != p["anchor_weight"]:
            raise RuntimeError("actual research reward profile differs")
        atomic_json_replace(self.output,"research_profile_binding.json", {
            "parameters":p,"actual_residual_neutrality_weight":self.environment._h1_residual_neutrality_weight,
            "actual_anchor_weight":self.environment._h1_anchor_weight,
            "pure_hover":p["command_profile"] == "pure-hover",
            "center_reset_only":self.environment._h1_pure_hover_training,
            "initialization":("COLD_START" if self.authorization["mode"] == "research-training" else
                "FULL_TRAINER_RESTORE_FRESH_SIM_RESET" if self.authorization["mode"] == "research-formal-training" else "EVALUATION_FRESH_RESET"),
            "authorization_sha256":self.authorization["authorization_sha256"],
            "method_hashes":self._checkpoint_method_hashes(load_config())})

    def train_research(self):
        p = self.research
        self.cold_start(); self.save_checkpoint("init0000")
        nodes=set(p["checkpoint_updates"])
        while self.trainer.update_counter < p["max_updates"]:
            ordinal=self.trainer.update_counter+1; self.resource_check("before_research_update",ordinal)
            self.loop.run_update(checkpoint_label=f"u{ordinal}" if ordinal in nodes else None)

    def evaluate_research(self):
        p = self.research; self._apply_research_profile()
        for row in p["checkpoints"]:
            path=Path(row["path"]); update=row["update"]
            if sha256_file(path)!=row["sha256"]: raise ValueError("research checkpoint drift")
            meta=self._load_evaluation_checkpoint(path,actor_learning_rate=p["actor_lr"])
            if self.trainer.update_counter != update or meta["training_seed"] != p["training_seed"] or meta["curriculum"]["version"] != "curriculum-h1-research-pure-hover-v1":
                raise ValueError("research checkpoint seed/stage/node mismatch")
            self.trainer.actor.eval(); self.trainer.critic.eval(); self._evaluation_rows.clear()
            label=f"research_{p['group']}_u{update}"
            self._evaluate_case("Z",direct_b0=False,evaluation_label=label)
            atomic_json_replace(self.output,f"{label}_Z.json",{
                "schema_version":"instinctrl-research-z-evaluation/1","group":p["group"],"update":update,
                "training_seed":p["training_seed"],"evaluation_seed":p["evaluation_seed"],
                "episodes":[evaluate_episode(trace) for trace in self._evaluation_rows["Z"]]})

    def run_research_transition(self):
        """Read-only actual-checkpoint recalculation and bounded command check."""
        from .checkpoint_recalculation import recalculate_actor_probe
        from .transition_diagnostic import execute_finite_transition
        from .pilot_diagnostic import (FLOAT32_ATOL, FLOAT32_RTOL, _digest, _state,
                                      _rng_snapshot, _rng_equal, _restore_rng)
        if self.authorization['mode'] != 'research-transition':
            raise PermissionError('transition backend requires exact mode')
        p = self.research
        self._apply_research_profile()
        rec = p['recalculation']

        def load(row):
            path = Path(row['path'])
            self.resource_check('before_transition_checkpoint_read', row['update'])
            if sha256_file(path) != row['sha256']:
                raise ValueError('transition checkpoint drift')
            meta = self._load_evaluation_checkpoint(path, actor_learning_rate=p['actor_lr'])
            if (self.trainer.update_counter != row['update'] or meta['training_seed'] != p['training_seed']
                    or meta['curriculum']['version'] != 'curriculum-h1-research-pure-hover-v1'):
                raise ValueError('transition checkpoint stage/seed/node mismatch')
            self.trainer.actor.eval()
            self.trainer.critic.eval()
            return meta

        load(rec['checkpoint'])
        for key in ['input', 'probe']:
            if sha256_file(Path(rec[key]['path'])) != rec[key]['sha256']:
                raise ValueError('transition retained input/probe drift')
        data = torch.load(rec['input']['path'], map_location=self.environment.device, weights_only=True)
        probe = torch.load(rec['probe']['path'], map_location=self.environment.device, weights_only=True)
        if (data['update'] != 1 or probe['schema_version'] != 'instinctrl-pilot-first-update-probe/2'
                or probe['sample_count'] != 512 or len(probe['minibatches']) != 8):
            raise ValueError('transition actual u1 probe boundary invalid')
        reference = probe['minibatches'][-1]['after']
        if (reference['boundary'] != 'after_minibatch_8'
                or not torch.equal(reference['indices'], torch.arange(512, device=reference['indices'].device))):
            raise ValueError('transition actual u1 probe indices invalid')
        result = recalculate_actor_probe(self.trainer, data['policy'], data['actions'], reference)
        atomic_json_replace(self.output, 'checkpoint_actor_recalculation.json', {
            **result, 'actual_inputs': rec, 'original_source_identity': p['checkpoint_source_identity'],
            'scope': 'Actual saved u1 actor boundary; not a missing historical final-checkpoint output'})
        del data, probe, reference

        final = p['checkpoints'][0]
        original_meta = load(final)

        def reload_check(observation):
            # Fresh reset has preserved the restored checkpoint RNG. The same
            # returned observation is reused without a sensor/history advance.
            frozen = {k: v.detach().clone() for k, v in observation['policy'].items()}
            before = _digest(_state(self.trainer))
            rng = _rng_snapshot()
            try:
                with torch.no_grad():
                    first = self.trainer.actor(frozen)
                    action = first.mean_action.detach().clone()
                    expected = {'alpha': first.alpha.detach().clone(), 'beta': first.beta.detach().clone(),
                                'mean': action, 'logp': self.trainer.actor.log_probability_from_output(first, action).detach().clone()}
                if _digest(_state(self.trainer)) != before:
                    raise RuntimeError('transition original load forward changed state')
                meta = load(final)
                if meta != original_meta or _digest(_state(self.trainer)) != before:
                    raise RuntimeError('transition independent final load differs')
                with torch.no_grad():
                    second = self.trainer.actor(frozen)
                    actual = {'alpha': second.alpha, 'beta': second.beta, 'mean': second.mean_action,
                              'logp': self.trainer.actor.log_probability_from_output(second, action)}
                errors = {}
                for key, value in actual.items():
                    if (value.shape != expected[key].shape or not torch.isfinite(value).all()
                            or not torch.allclose(value, expected[key], atol=FLOAT32_ATOL, rtol=FLOAT32_RTOL)):
                        raise RuntimeError('transition final reload numeric mismatch: ' + key)
                    errors[key] = float((value - expected[key]).abs().max().item())
                if _digest(_state(self.trainer)) != before:
                    raise RuntimeError('transition reloaded forward changed state')
                return {'status': 'PASS', 'checkpoint': final, 'metadata': meta,
                        'state_sha256': before, 'samples': 32, 'device': str(self.environment.device),
                        'atol': FLOAT32_ATOL, 'rtol': FLOAT32_RTOL, 'maximum_absolute_errors': errors,
                        'scope': 'Same actual fresh-reset input across two strict loads; no historical final output invented',
                        'optimizer_calls': 0, 'policy_rng_consumed': False}
            finally:
                if not _rng_equal(rng, _rng_snapshot()):
                    _restore_rng(rng)
                    raise RuntimeError('transition final reload/forward changed RNG')

        return execute_finite_transition(self, checkpoint_reload_check=reload_check)

    def train_pilot(self, group: str) -> None:
        """True cold-start 64-update pilot with the exact seven checkpoint names."""
        self.cold_start(pilot_group=group); self.save_checkpoint("init0000")
        checkpoints={1:"u1",2:"u2",8:"u8",16:"u16",32:"u32",64:"u64"}
        while self.trainer.update_counter < 64:
            ordinal=self.trainer.update_counter+1; self.resource_check("before_pilot_update",ordinal)
            self.loop.run_update(checkpoint_label=checkpoints.get(ordinal))

    def evaluate_pilot_z(self, group: str, checkpoints: list[Mapping[str, Any]]) -> None:
        """Development-only Z evaluation at u16/u64; no other case is exposed."""
        if group not in {"P0","PLR","C0","C1"} or [row.get("update") for row in checkpoints] != [16,64]: raise ValueError("pilot evaluation is exactly group Z at u16/u64")
        if group in {"C0", "C1"}:
            self.environment.set_h1_residual_neutrality_weight(pilot_residual_weight(group))
            atomic_json_replace(self.output, "pilot_reward_binding.json", {
                "group": group, "actual_residual_neutrality_weight": self.environment._h1_residual_neutrality_weight,
                "authorization_sha256": self.authorization["authorization_sha256"],
                "method_hashes": pilot_method_hashes(load_config(), group)})
        for row in checkpoints:
            path=Path(str(row["path"])); update=int(row["update"])
            if sha256_file(path) != row["sha256"]: raise ValueError("pilot evaluation checkpoint hash changed")
            self._load_evaluation_checkpoint(path, actor_learning_rate=pilot_actor_learning_rate(group))
            if self.trainer.update_counter != update: raise ValueError("pilot evaluation update mismatch")
            self.trainer.actor.eval(); self.trainer.critic.eval(); self._evaluation_rows.clear()
            self._evaluate_case("Z",direct_b0=False,evaluation_label=f"pilot_{group}_u{update}")
            atomic_json_replace(self.output,f"pilot_{group}_u{update}_Z.json",{"schema_version":"instinctrl-pilot-z-evaluation/1","group":group,"update":update,"case":"Z","development_only":True,"episodes":[evaluate_episode(trace) for trace in self._evaluation_rows["Z"]]})

    def resume_from_checkpoint(self, checkpoint: Path) -> Mapping[str, Any]:
        """Restore all H1 state, then perform a fresh reset without consuming training RNG."""
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
        from instinctlab.training.checkpoint_h1 import load_checkpoint_h1
        from instinctlab.training.networks_h1 import PrivilegedCriticH1
        config = load_config(); network = NetworkConfig()
        trainer = PPOTrainer(HistoryActor(network).to(self.environment.device), PrivilegedCriticH1(network).to(self.environment.device), PPOConfig(minibatch_size=64, epoch_count=1))
        metadata = load_checkpoint_h1(checkpoint.resolve(), trainer, expected_method_hashes=config["method"]["hashes"],
            expected_source_identity=config["resolved_identity"], curriculum_generator=self.curriculum_generator)
        command = command_at(metadata["counters"]["global_policy_step"], device=self.environment.device)
        self.environment.set_h1_evaluation_case(None); self.environment.set_h1_initial_command(command, metadata["counters"]["global_policy_step"])
        self.resource_check("before_resume_reset", self.counters.explicit_resets + 1)
        observation, _ = self._preserving_reset(seed=20260908 + metadata["reset_number"] + 1)
        journal = LearningLog(self.output / "learning_log"); self.journal = journal
        self.trainer = trainer; self.observation = observation
        self.driver = UnifiedH1Driver(self.environment, observation,
            clock=ScheduleClock(metadata["counters"]["global_policy_step"]), counters=self.counters, journal=journal,
            before_step=lambda ordinal: self.resource_check("before_training_step", ordinal),
            transition_limit=self.authorization["budgets"]["training_transitions"],
            automatic_reset_limit=self.authorization["budgets"]["automatic_resets"])
        self.loop = UnifiedH1TrainingLoop(self.driver, trainer, journal, self._write_checkpoint,
            before_rollout=lambda ordinal: self.resource_check("before_training_rollout", ordinal),
            before_optimizer=lambda ordinal: self.resource_check("before_optimizer_update", ordinal),
            update_limit=self.authorization["budgets"]["optimizer_updates"])
        return metadata

    def evaluate_policy_checkpoint(self, checkpoint: Path, *, update: int) -> Mapping[str, Any]:
        """Independent-process deterministic evaluation from all raw episode rows."""
        self._load_evaluation_checkpoint(checkpoint)
        if self.trainer.update_counter != update: raise ValueError("evaluation checkpoint update mismatch")
        self.trainer.actor.eval(); self.trainer.critic.eval(); self._evaluation_rows.clear()
        for ordinal, case in enumerate(CASES, 1):
            self.resource_check("before_policy_case", ordinal); self._evaluate_case(case, direct_b0=False, evaluation_label=f"u{update:04d}")
        result = evaluate_checkpoint([trace for case in CASES for trace in self._evaluation_rows[case]], update=update)
        atomic_json_replace(self.output, f"evaluation_u{update:04d}.json", result)
        return result

    def evaluate_r6_blocks(self) -> list[Mapping[str, Any]]:
        """Separate exact-authorized R6 entry; never inherits an R5 receipt."""
        import json
        from .robustness_evaluation import evaluate_block
        if self.authorization['mode'] != 'research-r6-evaluation':
            raise PermissionError('R6 blocks require a new exact evaluation authorization')
        parameters=self.authorization['parameters']['r6']
        protocol_path=Path(parameters['protocol']['path'])
        if not protocol_path.is_absolute() or protocol_path.is_symlink() or sha256_file(protocol_path)!=parameters['protocol']['sha256']:
            raise PermissionError('R6 protocol pin drift')
        protocol=json.loads(protocol_path.read_text())
        if protocol['frozen'] is not True:raise PermissionError('R6 protocol is not independently frozen')
        blocks={block['id']:block for block in protocol['blocks']}
        selected=parameters['block_ids']
        if not selected or len(set(selected))!=len(selected) or any(key not in blocks for key in selected):
            raise ValueError('R6 exact block selection')
        required=32*sum(blocks[key]['steps'] for key in selected)
        if self.counters.nontraining_transitions+required>self.authorization['budgets']['nontraining_transitions']:
            raise ValueError('R6 complete block batch exceeds exact transition budget')
        results=[]
        for key in selected:
            quota=parameters['output_quotas'][key]
            results.append(evaluate_block(self,blocks[key],run_id=parameters['run_id'],seed=parameters['seed'],
                split=parameters['split'],expected_mass_kg=parameters['expected_mass_kg'],
                range_quota_bytes=quota['range_bytes'],audit_quota_bytes=quota['audit_bytes']))
        atomic_json_replace(self.output,'r6_block_results.json',results)
        return results

    def _load_evaluation_checkpoint(self, checkpoint: Path, *, actor_learning_rate: float = 3.0e-4,
                                    expected_method_hashes=None, source_identity=None,
                                    restore_curriculum_generator: bool = False) -> Mapping[str, Any]:
        from instinctlab.training import HistoryActor, NetworkConfig, PPOConfig, PPOTrainer
        from instinctlab.training.checkpoint_h1 import load_checkpoint_h1
        from instinctlab.training.networks_h1 import PrivilegedCriticH1
        config = load_config(); network = NetworkConfig()
        trainer = PPOTrainer(HistoryActor(network).to(self.environment.device), PrivilegedCriticH1(network).to(self.environment.device), PPOConfig(minibatch_size=64, epoch_count=1, actor_learning_rate=actor_learning_rate, critic_learning_rate=3.0e-4))
        metadata = load_checkpoint_h1(checkpoint.resolve(), trainer,
            expected_method_hashes=self._checkpoint_method_hashes(config) if expected_method_hashes is None else expected_method_hashes,
            expected_source_identity=(self.authorization["parameters"].get("checkpoint_source_identity", config["resolved_identity"])
                                      if source_identity is None else source_identity),
            curriculum_generator=self.curriculum_generator if restore_curriculum_generator else torch.Generator(device="cpu"))
        self.trainer = trainer
        return metadata

    def evaluate_policy_checkpoints(self, checkpoints: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        results = []
        for row in checkpoints:
            path = Path(str(row["path"])); update = int(row["update"])
            if sha256_file(path) != row["sha256"]: raise ValueError("evaluation checkpoint hash changed after authorization")
            results.append(self.evaluate_policy_checkpoint(path, update=update))
        return results

    def run_enhanced_b0_diagnostic(self, parameters: Mapping[str, Any]) -> None:
        """Attach the read-only observer before the exact environment-owned Z/T/S run."""
        import json
        from .b0_enhanced import TraceValidator
        from .b0_observer import EnhancedB0ExecutionAccounting, FormalEnhancedB0EnvironmentAdapter, ReadOnlyObserver, attach_supported_post_physics
        validator = TraceValidator()
        target = self.output / "enhanced_b0_physics.jsonl"
        if target.exists(): raise RuntimeError("enhanced B0 raw output already exists")
        stream = target.open("x", encoding="utf-8")
        written_bytes = written_rows = attempted_rows = validated_rows = 0
        coverage = observation = receipt = measurement_failure = None
        observer = None
        last_attempted_coordinate = last_written_coordinate = None
        phase = "PREPARED"
        def preserve_raw_manifest(status, error=None):
            digest = hashlib.sha256(); file_rows = file_bytes = 0; last_byte = b""
            with target.open("rb") as raw:
                for chunk in iter(lambda:raw.read(1024*1024),b""):
                    digest.update(chunk); file_rows += chunk.count(b"\n"); file_bytes += len(chunk); last_byte = chunk[-1:]
            atomic_json_replace(self.output, "enhanced_b0_manifest.json", {
                "schema_version":"instinctrl-enhanced-b0-raw-manifest/2",
                "experiment_id":"UNIFIED-H1-R5-READINESS-B0-PLANT-R2-20261012",
                "authorization_sha256":self.authorization["authorization_sha256"],
                "source_identity":{key:self.authorization[key] for key in
                    ("product_commit","product_tree","workload_commit","workload_tree","config_sha256")},
                "status":status,"phase":phase,"hook_receipt":receipt,
                "coverage":coverage,"observation":observation,
                "raw":{"path":target.name,"attempted_rows":attempted_rows,
                    "validated_rows":validated_rows,"written_rows":file_rows,"stream_accepted_rows":written_rows,
                    "encoded_bytes":written_bytes,"file_bytes":file_bytes,
                    "sha256":digest.hexdigest(),"trailing_partial_row":bool(last_byte and last_byte!=b"\n"),
                    "last_attempted_coordinate":last_attempted_coordinate,
                    "last_written_coordinate":last_written_coordinate if file_rows==written_rows else None,
                    "stream_closed":stream.closed},
                "observer_progress":None if observer is None else {
                    "completed_callback_calls":observer.calls,"completed_callback_rows":observer.rows,
                    "coverage_finalized":observation is not None},
                "primary_error":None if error is None else exception_record(error),
                "measurement_failure":measurement_failure,
                "condition":"ENHANCED_B0_ZERO_TARGET",
                "measurement_sources":{"mass":"Articulation.root_physx_view.get_masses",
                    "inertia":"Articulation.root_physx_view.get_inertias",
                    "body":"data.body_link_state_w","root_response":"data.root_link_state_w",
                    "controller_feedback":"data.root_state_w"},
                "automatic_retries":0,"training_transitions":0,"optimizer_updates":0,
                "checkpoint_writes":0,"policy_evaluations":0,
                "execution_counters":self.counters.snapshot()})
        def write(row):
            nonlocal written_bytes, written_rows, attempted_rows, validated_rows, last_attempted_coordinate, last_written_coordinate
            attempted_rows += 1
            last_attempted_coordinate={key:row.get(key) for key in ("case","policy_step","physics_step","lane","episode_id")}
            validator.append(row)
            validated_rows += 1
            encoded=json.dumps(row, sort_keys=False, separators=(",", ":"), allow_nan=False) + "\n"
            size=len(encoded.encode("utf-8"))
            if written_bytes + size > 16 * 1024**3: raise RuntimeError("enhanced B0 16 GiB diagnostic ceiling exhausted")
            stream.write(encoded); written_bytes += size; written_rows += 1
            last_written_coordinate=dict(last_attempted_coordinate)
        try:
            preserve_raw_manifest("PREPARED")
            formal_environment = FormalEnhancedB0EnvironmentAdapter(self.environment)
            observer = ReadOnlyObserver(write, formal_environment.read_only_diagnostic_state_fingerprints)
            accounting = EnhancedB0ExecutionAccounting(self.counters, self.authorization["budgets"], self.resource_check)
            formal_environment.configure_execution_accounting(accounting)
            receipt = attach_supported_post_physics(formal_environment, observer)
            phase = "COLLECTION"
            formal_environment.run_frozen_enhanced_b0_zts(seed=20260909, cases=("Z", "T", "S"), target_velocity_b_m_s=(0.0, 0.0, 0.0), retries=0)
            phase = "FINALIZATION"
            coverage = validator.finalize(); observation = observer.finalize()
            stream.flush(); os.fsync(stream.fileno())
            stream.close()
            phase = "ANALYSIS"
            # A watchdog may kill the process during the expensive CPU analysis.
            # Persist the successfully returned collection and observed counters
            # first; this is neither an analysis verdict nor a close receipt.
            preserve_raw_manifest("COLLECTION_COMPLETE_ANALYSIS_PENDING")
            from .b0_causal_analyzer import analyze_jsonl
            analysis = analyze_jsonl(target)
            atomic_json_replace(self.output, "enhanced_b0_causal_analysis.json", analysis)
            phase = "COMPLETE"
            preserve_raw_manifest("COMPLETE")
        except BaseException as primary:
            if getattr(primary,"measurement_failure",None) is not None:
                try:
                    from .b0_enhanced import safe_failure_primitives
                    context=safe_failure_primitives(primary.measurement_failure)
                    atomic_json_replace(self.output,"enhanced_b0_measurement_failure.json",context)
                    failure_path=self.output/"enhanced_b0_measurement_failure.json"
                    measurement_failure={"path":failure_path.name,"status":"UNVALIDATED",
                        "sha256":sha256_file(failure_path),"file_bytes":failure_path.stat().st_size}
                except BaseException as save_error:
                    primary.add_note("measurement failure artifact save failed: " + repr(save_error))
            try:
                stream.close()
            except BaseException as close_error:
                primary.add_note("raw stream close failed: " + repr(close_error))
            try:
                preserve_raw_manifest("PARTIAL_FAILURE", primary)
            except BaseException as evidence_error:
                primary.add_note("partial raw manifest failed: " + repr(evidence_error))
            raise

    def run_control_diagnostic(self, parameters: Mapping[str, Any]) -> None:
        """Run one independently authorized Z/T/S control-chain condition.

        B0 uses the fixed normalized action through the unchanged H1 governor,
        ICS, Lee and rotor path.  It deliberately does not use prepare_b0_step,
        which is the legacy direct-controller comparison and a different test.
        """
        condition = str(parameters["condition"])
        from .control_diagnostic import is_command_passthrough, ACTOR_NOT_EVALUATED
        passthrough=is_command_passthrough(condition,parameters['experiment_id'])
        checkpoint=None; metadata=None
        if not passthrough:
            checkpoint = Path(str(parameters["checkpoint"]))
            if sha256_file(checkpoint) != parameters["checkpoint_sha256"]:
                raise ValueError("diagnostic checkpoint hash changed after authorization")
            metadata = self._load_evaluation_checkpoint(checkpoint)
            if self.trainer.update_counter != 256:
                raise ValueError("control diagnostic is bound to the R4 u0256 checkpoint")
            self.trainer.actor.eval(); self.trainer.critic.eval()
        writer = DiagnosticChunkWriter(self.output / "control_trace", condition=condition,
                                       experiment_id=str(parameters["experiment_id"]),
                                       aggregate_ledger=(self.output / ".command_passthrough_disk_ledger.json" if passthrough else self.attempt.parent / ".d7_control_diagnostic_disk_ledger.json"))
        metric_log = LearningLog(self.output / "evaluation_raw") if passthrough else None
        for case in DIAGNOSTIC_CASES:
            self.environment.set_h1_evaluation_case(case)
            commands = case_commands(case).to(self.environment.device)
            self.environment.set_h1_initial_command(commands[0], 0)
            self.resource_check("before_control_diagnostic_reset", self.counters.explicit_resets + 1)
            observation, _ = self._preserving_reset(seed=20260909)
            if passthrough:
                self.environment._refresh_task_snapshot()
                initial_sensor=self.environment._evaluation_ray_snapshot.ray_starts_w[:,0].detach().cpu()
                metric_relative=f"command_passthrough/{case}.jsonl"
                metric_log.append(metric_relative,{"schema_version":"metric-h1-v1.raw-trace.v1","kind":"T0",
                    "case":case,"seed":20260909,"initial_sensor_position_w_m":initial_sensor,
                    "initial_root_state_w":self.environment._robot.data.root_state_w.detach().cpu().clone(),
                    "env_origins_w_m":self.environment.scene.env_origins.detach().cpu()})
                episode_ids=torch.zeros(32,dtype=torch.int64)
            for step in range(CASES[case].steps):
                self.resource_check("before_control_diagnostic_step", step)
                self.counters.require_step_capacity(kind="nontraining", lanes=32,
                    maximum=self.authorization["budgets"]["nontraining_transitions"])
                self.counters.require_reset_capacity(automatic=True, lanes=32,
                    maximum=self.authorization["budgets"]["automatic_resets"])
                current = commands[step]; following = commands[min(step + 1, len(commands) - 1)]
                self.environment.prepare_h1_step(global_policy_step=step, command_b=current,
                                                 next_command_b=following)
                actor_input = select_actor_input(observation["policy"])
                actor_output=None
                if not passthrough:
                    with torch.no_grad(): actor_output = self.trainer.actor(observation["policy"])
                action = (actor_output.mean_action.detach() if condition == "D7-R" else
                          fixed_b0_action(32, device=self.environment.device,
                                          dtype=observation["policy"]["state_vec"].dtype))
                pre_step = {"actor_input": actor_input, "actor_alpha": dict(ACTOR_NOT_EVALUATED) if passthrough else actor_output.alpha,
                            "actor_beta": dict(ACTOR_NOT_EVALUATED) if passthrough else actor_output.beta, "actor_mean": dict(ACTOR_NOT_EVALUATED) if passthrough else actor_output.mean_action,
                            "issued_action": action, "command_b": current,
                            "root_state_w": self.environment._robot.data.root_state_w.detach().clone()}
                self.counters.attempt_step()
                observation, reward, terminated, truncated, extras = self.environment.step(action)
                self.counters.complete_step(32, kind="nontraining")
                reset_mask = terminated | truncated
                self.counters.reserve_reset(automatic=True, lanes=int(reset_mask.sum()),
                    maximum=self.authorization["budgets"]["automatic_resets"])
                if passthrough:
                    pre=extras['pre_reset'];source=extras['evaluation_ray_source']
                    metric_log.append(metric_relative,{"schema_version":"metric-h1-v1.raw-trace.v1","kind":"STEP",
                        "case":case,"step":step,"command_b":current,"actual_velocity_b":pre['task']['actual_velocity_b'],
                        "sensor_positions_w_m":source['ray_starts_w'][:,0],"episode_ids":episode_ids,
                        "collision":pre['task']['termination']['masks']['collision'][:,0],
                        "below_height":pre['task']['termination']['masks']['below_height'][:,0],
                        "above_height":pre['task']['termination']['masks']['above_height'][:,0],
                        "reset_mask":reset_mask,"ray_clearance_m":pre['task']['exact_clearance_m'][:,0],
                        "q":extras['reward_diagnostics']['q'][:,0],"ics_beta":pre['control']['ics_metrics']['beta'][:,0]})
                    episode_ids+=reset_mask.detach().cpu().to(torch.int64)
                # UnifiedH1's durable pre_reset owns the action->ICS/task state.
                # Lee/wrench caches are observable after step only for lanes not
                # reset by DirectRLEnv; validity makes that limitation explicit.
                ordinal = sum(CASES[name].steps for name in DIAGNOSTIC_CASES[:DIAGNOSTIC_CASES.index(case)]) + step
                writer.append(case, step, {"schema_version":"unified-h1-control-diagnostic.step.v3" if passthrough else "unified-h1-control-diagnostic.step.v2",
                    "condition":condition, "case":case, "step":step, "ordinal":ordinal,
                    "timing_semantics":TIMING_SEMANTICS, "pre_step":pre_step,
                    "post_step":{"reward":reward, "terminated":terminated, "truncated":truncated,
                                 "pre_reset":extras["pre_reset"], "lee":self.environment._lee,
                                 "wrench":self.environment._wrench,
                                 "downstream_cache_valid":~reset_mask}})
        manifest = writer.finalize()
        if manifest["transitions"] != parameters["transitions"]:
            raise RuntimeError("control diagnostic transition settlement mismatch")
        atomic_json_replace(self.output, "control_diagnostic.json", {
            "schema_version":"unified-h1-control-diagnostic.summary.v1", "condition":condition,
            "experiment_id":parameters["experiment_id"], "checkpoint":None if checkpoint is None else str(checkpoint),
            "condition_semantics":"B0_COMMAND_PASSTHROUGH" if passthrough else ("D7_R_FROZEN_POLICY" if condition=="D7-R" else "HISTORICAL_B0_COMMAND_PASSTHROUGH"),
            "checkpoint_sha256":parameters["checkpoint_sha256"], "checkpoint_metadata":metadata,
            "transitions":manifest["transitions"], "automatic_retries":0,
            "downstream_reset_lane_limitation":"Lee/wrench post-step caches are invalid for lanes reset in that step"})

    def mixed_reload_rollout(self, steps: int) -> None:
        from instinctlab.training.rollout import RolloutConfig
        from instinctlab.training.rollout_h1 import collect_h1_rollout
        if steps != 16 or self.trainer is None: raise ValueError("reload audit is exactly 16 steps")
        mixed = command_at(1024, device=self.environment.device)
        self.environment.set_h1_initial_command(mixed, 1024)
        self.resource_check("before_reload_reset", self.counters.explicit_resets + 1)
        observation, _ = self._preserving_reset(seed=20260908 + 10000)
        journal = self.journal or LearningLog(self.output / "learning_log")
        driver = UnifiedH1Driver(self.environment, observation, clock=ScheduleClock(1024), counters=self.counters,
            journal=journal, before_step=lambda ordinal: self.resource_check("before_reload_step", ordinal),
            transition_limit=self.authorization["budgets"]["nontraining_transitions"],
            automatic_reset_limit=self.authorization["budgets"]["automatic_resets"], step_kind="nontraining")
        collect_h1_rollout(driver, self.trainer, observation, RolloutConfig(16, 32))

    def preserve(self, status: str, outcome: LifecycleOutcome | None = None) -> None:
        if status in {"PASS_PENDING_CLOSE", "FAIL_PENDING_CLOSE", "EXPECTED_FAILURE_PENDING_CLOSE"} and self.journal is not None:
            self.journal.finalize_right_censored()
        atomic_json_replace(self.output, "status.json", {"schema_version": "unified-h1-status.v3", "status": status,
            "primary_error": None if outcome is None or outcome.primary_error is None else exception_record(outcome.primary_error),
            "cleanup_errors": [] if outcome is None else [{"cleanup": row.name, **exception_record(row.error)} for row in outcome.cleanup_failures],
            "initialization_measurement":dict(self.initialization_measurement),
            "counters": vars(self.counters)})

    def close_bounded(self) -> None:
        errors = []
        for name, resource in (("environment", self.environment), ("app", self.app)):
            try:
                if resource is not None: resource.close()
            except BaseException as error:
                errors.append((name, error))
        if errors:
            raise RuntimeError("; ".join(f"{name}: {type(error).__name__}: {error}" for name, error in errors))


def create_live_backend(*, authorization: Mapping[str, Any], attempt: Path,
                        resources: dict[str, Any] | None = None,
                        counters: ExecutionCounters | None = None) -> UnifiedH1LiveBackend:
    """Start live resources only after formal authorization/config checks have succeeded."""
    from .config import validate_prelaunch_environment, resource_binding_pins
    config = load_config(); validate_prelaunch_environment(config); attempt = attempt.resolve()
    results = attempt / "results"
    if not attempt.is_dir() or not results.is_dir(): raise ValueError("authorized attempt/results directory is absent")
    # Training's resource trace is immutable; evaluation owns its own gate.
    gate_output = (results / "evaluation" / "prelaunch" if authorization["mode"] == "evaluation"
                   else results / "prelaunch")
    gate_output.mkdir(exist_ok=False)
    resource_pins = resource_binding_pins()
    from .resource_admission import select_gate
    selected_gate = select_gate(authorization, resource_pins, partial(run_prelaunch_gate, resource_pins=resource_pins))
    gate, launcher, app = start_after_coexistence_gate(gate_output, mode=str(authorization["mode"]),
        attempt_id=attempt.name, authorization_sha256=str(authorization["authorization_sha256"]),
        gate_runner=selected_gate, resource_pins=resource_pins,
        asset_root=Path(config["execution"]["isaacsim_asset_root"]),
        native_volume_pin=authorization.get('parameters',{}).get('native_volume'))
    if resources is not None: resources.update(launcher=launcher, app=app)
    if authorization['mode']=='research-r6-evaluation':
        # Register the application first so a readback failure uses the existing
        # bounded lifecycle cleanup rather than losing the constructed resource.
        from training_workloads.scale_003.resources import verify_r6_native_paths
        verify_r6_native_paths(gate_output,authorization.get('parameters',{}).get('native_volume'))
    validate_graceful_shutdown_configuration(app, resource_pins=resource_pins)
    from instinctlab.envs.unified_h1_direct_env import UnifiedH1DirectEnv, UnifiedH1DirectEnvCfg
    validate_live_environment(config)
    environment = UnifiedH1DirectEnv(UnifiedH1DirectEnvCfg())
    if resources is not None: resources["environment"] = environment
    return UnifiedH1LiveBackend(authorization=authorization, attempt=attempt, app=app,
        environment=environment, resource_guard=RuntimeResourceGuard(gate_output), counters=counters,
        process_started_monotonic=None if resources is None else resources.get("process_started_monotonic"))


__all__ = ["UnifiedH1LiveBackend", "create_live_backend"]
