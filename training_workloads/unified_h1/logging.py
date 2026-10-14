"""Durable, computation-owning ``learning-log-h1-v1`` writer."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from training_workloads.scale_003.persistence import reserve_output_bytes


def _statistics(values: torch.Tensor) -> dict[str, float | int | str]:
    if not isinstance(values, torch.Tensor):
        raise TypeError("statistics input must be a tensor")
    total = values.numel(); finite_mask = torch.isfinite(values); finite_count = int(finite_mask.sum())
    if finite_count != total:
        return {"status": "UNAVAILABLE", "reason": "NONFINITE_SAMPLES", "count": total,
                "finite_count": finite_count, "nonfinite_count": total - finite_count}
    if total == 0:
        return {"status": "UNAVAILABLE", "reason": "NO_SAMPLES", "count": 0,
                "finite_count": 0, "nonfinite_count": 0}
    flat = values.to(torch.float64).flatten(); cost = -flat
    return {"status": "AVAILABLE", "count": total, "finite_count": total, "nonfinite_count": 0,
            "signed_mean": float(flat.mean()), "sum": float(flat.sum()),
            "cost_p95": float(torch.quantile(cost, .95))}


def _distribution(values: torch.Tensor) -> dict[str, float | int | str]:
    status = _statistics(values)
    if status["status"] != "AVAILABLE": return status
    flat = values.to(torch.float64).flatten()
    return {**status, "std": float(flat.std(unbiased=False)), "min": float(flat.min()), "max": float(flat.max())}


def _jsonable(value: Any, *, field: str | None = None) -> Any:
    if isinstance(value, torch.Tensor): return _jsonable(value.detach().cpu().tolist(), field=field)
    if isinstance(value, Mapping): return {str(key): _jsonable(item, field=str(key)) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_jsonable(item, field=field) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if field == "accepted_ray_hits_w" and value == float("inf"):
            return {"status":"RAY_MISS", "encoding":"POSITIVE_INFINITY"}
        kind = "NAN" if math.isnan(value) else ("POSITIVE_INFINITY" if value > 0 else "NEGATIVE_INFINITY")
        return {"status":"UNAVAILABLE", "reason":"NONFINITE_VALUE", "kind":kind}
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return repr(value)


def _compact_ray_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Retain per-step ray validity evidence without a 32-lane dense JSON dump."""
    result = {name: value for name, value in source.items()
              if name not in {"ray_starts_w", "ray_directions_w", "accepted_ray_hits_w", "accepted_beam_valid"}}
    hits = source.get("accepted_ray_hits_w"); valid = source.get("accepted_beam_valid")
    if (not isinstance(hits, torch.Tensor) or hits.ndim != 3 or hits.shape[-1] != 3
            or not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or valid.shape != hits.shape[:-1]):
        result["dense_contract"] = {"status":"UNAVAILABLE", "reason":"RAY_SOURCE_SHAPE_OR_DTYPE_MISMATCH"}
        return result
    finite = torch.isfinite(hits).all(dim=-1)
    legal_miss = (~valid) & torch.isposinf(hits).all(dim=-1)
    invalid_nonfinite = valid & ~finite
    malformed_miss = (~valid) & ~legal_miss
    result["dense_contract"] = {
        "status":"INVALID_VALID_BEAM" if bool(invalid_nonfinite.any()) else "RECORDED",
        "shape":list(hits.shape),
        "valid_counts":valid.sum(dim=1),
        "legal_positive_infinity_miss_counts":legal_miss.sum(dim=1),
        "valid_nonfinite_counts":invalid_nonfinite.sum(dim=1),
        "noncanonical_invalid_counts":malformed_miss.sum(dim=1),
        "dense_values":"SELECTED_MEASUREMENT_WINDOWS_ONLY",
    }
    return result


def _compact_step_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(evidence)
    source = compact.get("evaluation_ray_source")
    if isinstance(source, Mapping): compact["evaluation_ray_source"] = _compact_ray_source(source)
    return compact


class LearningLog:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self._last_command: torch.Tensor | None = None
        self._block = 0
        self._episode_return: torch.Tensor | None = None
        self._episode_length: torch.Tensor | None = None
        self._episode_id: torch.Tensor | None = None
        self._reset_id: torch.Tensor | None = None
        self._measurement_steps = 0
        self._failure_window_written = False
        self._diagnostic_ring: list[Mapping[str, Any]] = []
        self._failure_post_rows: list[Mapping[str, Any]] = []
        self._failure_post_remaining = 0
        self._durable_step_rows = 0
        self._episode_components: dict[str, torch.Tensor] = {}
        self._episode_start_position: torch.Tensor | None = None
        self._episode_max_drift: torch.Tensor | None = None
        self._episode_max_height: torch.Tensor | None = None
        self._episode_min_clearance: torch.Tensor | None = None
        self._episode_start_types: list[str] | None = None
        self._episode_start_pending: torch.Tensor | None = None
        self._block_states: list[dict[str, Any] | None] = []
        self._next_block_id = 0
        self._blocks_finalized = False

    @property
    def durable_step_rows(self) -> int:
        return self._durable_step_rows

    def append(self, relative: str, record: Mapping[str, Any]) -> None:
        target = (self.root / relative).resolve()
        if self.root not in target.parents or ".." in Path(relative).parts: raise ValueError("log path escapes root")
        target.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(_jsonable(dict(record)), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        reserve_output_bytes(target, len(data.encode("utf-8")))
        with target.open("a", encoding="utf-8") as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())

    def record_update_attempt(self, *, attempted_updates: int, completed_updates: int,
                              attempted_policy_steps: int, completed_policy_steps: int) -> None:
        self.append("accounting.jsonl", {"version": "learning-log-h1-v1", "event": "UPDATE_ATTEMPT",
                    "attempted_updates": attempted_updates, "completed_updates": completed_updates,
                    "attempted_policy_steps": attempted_policy_steps, "completed_policy_steps": completed_policy_steps})

    def record_step_accounting(self, *, event: str, attempted_policy_steps: int,
                               completed_policy_steps: int, transitions: int) -> None:
        if event not in {"ENV_STEP_ATTEMPT", "ENV_STEP_COMPLETE"}: raise ValueError("invalid step accounting event")
        self.append("accounting.jsonl", {"version": "learning-log-h1-v1", "event": event,
                    "attempted_policy_steps": attempted_policy_steps,
                    "completed_policy_steps": completed_policy_steps,
                    "completed_transitions": transitions})

    def record_step_evidence(self, evidence: Mapping[str, Any], *, completed_policy_steps: int) -> None:
        """Fsync each returned step so a later rollout failure cannot erase it."""
        self.append("partial_rollout_steps.jsonl", {"version": "learning-log-h1-v1",
                    "completed_policy_steps": completed_policy_steps, "evidence": _compact_step_evidence(evidence)})
        self._durable_step_rows += 1

    def record_partial_rollout_boundary(self, *, attempted_policy_steps: int,
                                        completed_policy_steps: int,
                                        durable_step_rows: int, buffered_step_rows: int,
                                        error: BaseException) -> None:
        self.append("partial_rollout_boundaries.jsonl", {"version": "learning-log-h1-v1",
                    "attempted_policy_steps": attempted_policy_steps,
                    "completed_policy_steps": completed_policy_steps,
                    "durable_step_rows": durable_step_rows,
                    "buffered_step_rows": buffered_step_rows,
                    "unproven_attempts": attempted_policy_steps - completed_policy_steps,
                    "error_type": type(error).__name__, "message": str(error)})

    def record_partial_episodes(self, evidence: Sequence[Mapping[str, Any]]) -> None:
        """Retain right-censored lane state when rollout assembly never returns."""
        if not evidence: return
        for lane in range(32):
            rewards=[]; positions=[]; clearances=[]; reset_count=0; components: dict[str,float]={}
            start_type: object={"status":"UNAVAILABLE", "reason":"ROOT_ORIGIN_EVIDENCE_MISSING"}
            for item in evidence:
                driver=item.get("_h1_driver", {}); pre=item.get("pre_reset", {}); task=pre.get("task", {}) if isinstance(pre,Mapping) else {}
                reward=driver.get("reward") if isinstance(driver,Mapping) else None
                root=pre.get("root_state_w") if isinstance(pre,Mapping) else None
                origins=driver.get("env_origins_w_m") if isinstance(driver,Mapping) else None
                if isinstance(reward,torch.Tensor): rewards.append(float(reward.reshape(-1)[lane]))
                if isinstance(root,torch.Tensor): positions.append(root.detach().cpu()[lane,:3])
                if len(positions)==1 and isinstance(origins,torch.Tensor):
                    local=positions[0]-origins.detach().cpu()[lane]
                    start_type=("near_x_wall" if abs(float(local[0])-4.6)<=.1 else
                                "near_y_wall" if abs(float(local[1])-4.6)<=.1 else "center")
                clearance=task.get("exact_clearance_m") if isinstance(task,Mapping) else None
                if isinstance(clearance,torch.Tensor): clearances.append(float(clearance.reshape(-1)[lane]))
                terminated=driver.get("terminated") if isinstance(driver,Mapping) else None
                truncated=driver.get("truncated") if isinstance(driver,Mapping) else None
                if isinstance(terminated,torch.Tensor) and isinstance(truncated,torch.Tensor):
                    reset_count += int(bool((terminated|truncated).reshape(-1)[lane]))
                values=item.get("reward_components", {})
                if isinstance(values,Mapping):
                    for name,value in values.items():
                        if isinstance(value,torch.Tensor): components[name]=components.get(name,0.)+float(value.reshape(-1)[lane])
            drift=(max(float(torch.linalg.vector_norm(value[:2]-positions[0][:2])) for value in positions) if positions else None)
            height=(max(abs(float(value[2]-positions[0][2])) for value in positions) if positions else None)
            self.append("partial_episodes.jsonl", {"version":"learning-log-h1-v1", "env":lane,
                "completed_steps":len(evidence), "return":sum(rewards) if len(rewards)==len(evidence) else None,
                "average_reward":sum(rewards)/len(rewards) if rewards else None,
                "reward_component_sums":components, "start_type":start_type, "reset_count":reset_count,
                "behavior":{"max_horizontal_drift_m":drift,"max_height_deviation_m":height,
                    "minimum_ray_clearance_m":min(clearances) if clearances else None},
                "reason":"PARTIAL_ROLLOUT_BOUNDARY", "right_censored":True})

    def record_rollout(self, segment: Any, evidence: Sequence[Mapping[str, Any]], *, update: int) -> dict[str, Any]:
        if len(segment.transitions) != len(evidence): raise ValueError("every transition requires its actual environment evidence")
        commands = segment.policy["state_vec"][..., 45:48]
        zero = torch.linalg.vector_norm(commands, dim=-1) <= .001
        statistics: dict[str, Any] = {
            "returns": _distribution(segment.returns), "advantages": _distribution(segment.advantages),
            "values": _distribution(segment.values), "zero_count": int(zero.sum()),
            "nonzero_count": int((~zero).sum()), "terminal_count": int(segment.terminated.sum()),
            "timeout_count": int((segment.truncated & ~segment.terminated).sum()),
        }
        target = segment.returns.flatten(); prediction = segment.values.flatten(); variance = target.var(unbiased=False)
        statistics["explained_variance"] = ({"status": "UNAVAILABLE", "reason": "ZERO_TARGET_VARIANCE"}
            if float(variance) == 0. else {"status": "AVAILABLE", "value": float(1 - (target - prediction).var(unbiased=False) / variance)})
        residual = (segment.values - segment.returns).abs().squeeze(-1)
        statistics["value_residual"] = {"zero": _distribution(residual[zero]), "nonzero": _distribution(residual[~zero])}
        q_rows = []
        for item in evidence:
            diagnostics = item.get("reward_diagnostics", {})
            if isinstance(diagnostics, Mapping) and isinstance(diagnostics.get("q"), torch.Tensor): q_rows.append(diagnostics["q"].reshape(-1).to(torch.bool))
        statistics["value_residual"]["q"] = _distribution(residual.flatten()[torch.cat(q_rows)]) if len(q_rows) == len(evidence) else {"status": "UNAVAILABLE", "reason": "Q_EVIDENCE_MISSING"}
        for step, (transition, item) in enumerate(zip(segment.transitions, evidence, strict=True)):
            reward_components = item.get("reward_components", {})
            if not isinstance(reward_components, Mapping): raise ValueError("actual reward components are missing")
            compact = {"version": "learning-log-h1-v1", "update": update, "step": step,
                       "command_b": commands[step], "action_mean": transition.policy.get("_actor_output_mean"),
                       "action_sample": transition.action,
                       "value": transition.value, "reward_total": transition.reward,
                       "terminated": transition.terminated, "truncated": transition.truncated,
                       "reward_components": reward_components,
                       "pre_reset": item.get("pre_reset"), "right_censored": False}
            self.append("compact_rollouts.jsonl", compact)
            self._record_blocks_and_episodes(transition, item, commands[step], update=update, step=step)
            self._record_diagnostic_window(item, transition, update=update, step=step)
        self.append("rollouts.jsonl", {"version": "learning-log-h1-v1", "update": update,
                    "rollout_steps": len(evidence), "statistics": statistics,
                    "groups": grouped_reward_summary_from_evidence(evidence, zero)})
        return statistics

    def _record_blocks_and_episodes(self, transition: Any, evidence: Mapping[str, Any], command: torch.Tensor,
                                    *, update: int, step: int) -> None:
        batch = command.shape[0]
        if self._episode_return is None:
            self._episode_return = torch.zeros(batch); self._episode_length = torch.zeros(batch, dtype=torch.long)
            self._episode_id = torch.zeros(batch, dtype=torch.long); self._reset_id = torch.zeros(batch, dtype=torch.long)
            self._episode_start_types = ["center"] * batch
            self._episode_max_drift = torch.zeros(batch); self._episode_max_height = torch.zeros(batch)
            self._episode_min_clearance = torch.full((batch,), float("inf"))
            self._episode_start_pending = torch.zeros(batch, dtype=torch.bool)
        self._episode_return += transition.reward.detach().cpu().reshape(-1)
        self._episode_length += 1
        current = command.detach().cpu()
        done = transition.reset_boundary.detach().cpu().reshape(-1)
        pre = evidence.get("pre_reset", {})
        task = pre.get("task", {}) if isinstance(pre, Mapping) else {}
        termination = task.get("termination", {}) if isinstance(task, Mapping) else {}
        masks = termination.get("masks", {}) if isinstance(termination, Mapping) else {}
        reason = termination.get("reason_code") if isinstance(termination, Mapping) else None
        root = pre.get("root_state_w") if isinstance(pre, Mapping) else None
        self._update_command_blocks(evidence, current, transition.reset_boundary.detach().cpu().reshape(-1))
        if isinstance(root, torch.Tensor) and root.shape[0] == batch:
            position = root.detach().cpu()[:, :3]
            if self._episode_start_position is None: self._episode_start_position = position.clone()
            if self._episode_start_pending is not None and bool(self._episode_start_pending.any()):
                self._episode_start_position[self._episode_start_pending] = position[self._episode_start_pending]
                self._episode_start_pending[self._episode_start_pending] = False
            drift = torch.linalg.vector_norm(position[:, :2] - self._episode_start_position[:, :2], dim=-1)
            height = (position[:, 2] - self._episode_start_position[:, 2]).abs()
            self._episode_max_drift = torch.maximum(self._episode_max_drift, drift)
            self._episode_max_height = torch.maximum(self._episode_max_height, height)
        clearance = task.get("exact_clearance_m") if isinstance(task, Mapping) else None
        if isinstance(clearance, torch.Tensor):
            self._episode_min_clearance = torch.minimum(self._episode_min_clearance, clearance.detach().cpu().reshape(-1))
        components = evidence.get("reward_components")
        if isinstance(components, Mapping):
            for name, value in components.items():
                if isinstance(value, torch.Tensor):
                    row = value.detach().cpu().reshape(-1)
                    self._episode_components[name] = self._episode_components.get(name, torch.zeros_like(row)) + row
        for lane in torch.nonzero(done).flatten().tolist():
            mask_values = {name: bool(value[lane].reshape(-1)[0]) for name, value in masks.items()} if isinstance(masks, Mapping) else {}
            self.record_episode({"env": lane, "episode": int(self._episode_id[lane]), "reset": int(self._reset_id[lane]),
                "start_type": self._episode_start_types[lane],
                "length": int(self._episode_length[lane]), "return": float(self._episode_return[lane]),
                "average_reward": float(self._episode_return[lane] / self._episode_length[lane]),
                "reward_component_sums": {name: float(value[lane]) for name, value in self._episode_components.items()},
                "behavior": {"max_horizontal_drift_m": float(self._episode_max_drift[lane]),
                    "max_height_deviation_m": float(self._episode_max_height[lane]),
                    "minimum_ray_clearance_m": float(self._episode_min_clearance[lane])},
                "termination_masks": mask_values, "reason": int(reason[lane].reshape(-1)[0]) if isinstance(reason, torch.Tensor) else "UNAVAILABLE",
                "right_censored": False})
            self._episode_return[lane] = 0; self._episode_length[lane] = 0
            self._episode_id[lane] += 1; self._reset_id[lane] += 1
            for value in self._episode_components.values(): value[lane] = 0
            if self._episode_start_pending is not None: self._episode_start_pending[lane] = True
            self._episode_max_drift[lane] = 0; self._episode_max_height[lane] = 0; self._episode_min_clearance[lane] = float("inf")
            driver = evidence.get("_h1_driver", {})
            next_step = int(driver.get("global_policy_step", -1)) + 1 if isinstance(driver, Mapping) else -1
            mixed_reset = not (isinstance(driver, Mapping) and (driver.get("pure_hover") is True or driver.get("center_reset_only") is True))
            self._episode_start_types[lane] = ("near_x_wall" if mixed_reset and next_step >= 1024 and lane % 8 == 6
                                                else "near_y_wall" if mixed_reset and next_step >= 1024 and lane % 8 == 7
                                                else "center")

    def _finish_command_block(self, state: Mapping[str, Any], *, right_censored: bool) -> None:
        samples = int(state["samples"])
        confirmation = state["stop_confirmation"]
        self.record_command_block({"block":int(state["block"]), "env":int(state["env"]),
            "previous_command":state["incoming_command"], "command":state["command"],
            "effective_step":int(state["start_step"]), "effective_end_step":int(state["last_step"]),
            "duration_steps":samples, "fixed_reference":state["reference_position_w_m"],
            "reference_source":state["reference_source"], "reset_interrupted":bool(state["reset_count"]),
            "reset_count":int(state["reset_count"]),
            "speed_rms_m_s":float((state["speed_square_sum"] / max(samples, 1)) ** .5),
            "max_horizontal_drift_m":float(state["max_drift_m"]),
            "stop_confirmation":confirmation,
            "braking_displacement_m":state["braking_displacement_m"],
            "post_stop_reference_w_m":state["post_stop_reference_w_m"],
            "post_stop_max_drift_m":state["post_stop_max_drift_m"],
            "right_censored":right_censored})

    def _update_command_blocks(self, evidence: Mapping[str, Any], command: torch.Tensor,
                               reset: torch.Tensor) -> None:
        pre = evidence.get("pre_reset", {})
        task = pre.get("task", {}) if isinstance(pre, Mapping) else {}
        root = pre.get("root_state_w") if isinstance(pre, Mapping) else None
        velocity = task.get("actual_velocity_b") if isinstance(task, Mapping) else None
        driver = evidence.get("_h1_driver", {})
        policy_step = driver.get("global_policy_step") if isinstance(driver, Mapping) else None
        if (not isinstance(root, torch.Tensor) or root.shape[0] != command.shape[0]
                or not isinstance(velocity, torch.Tensor) or velocity.shape != command.shape
                or type(policy_step) is not int):
            raise ValueError("command-block analysis requires actual root/velocity/clock evidence")
        positions = root.detach().cpu()[:, :3]; speeds = torch.linalg.vector_norm(velocity.detach().cpu(), dim=-1)
        if not self._block_states: self._block_states = [None] * command.shape[0]
        for lane in range(command.shape[0]):
            state = self._block_states[lane]
            if state is None or not torch.equal(state["command"], command[lane]):
                prior_command = torch.zeros(3) if state is None else state["command"].clone()
                reference = positions[lane].clone() if state is None else state["last_position_w_m"].clone()
                source = "FIRST_RETURNED_POST_STEP" if state is None else "PREVIOUS_COMPLETED_STEP"
                if state is not None: self._finish_command_block(state, right_censored=False)
                state = {"block":self._next_block_id, "env":lane, "incoming_command":prior_command,
                    "command":command[lane].clone(), "start_step":policy_step, "last_step":policy_step,
                    "reference_position_w_m":reference, "reference_source":source, "last_position_w_m":positions[lane].clone(),
                    "samples":0, "reset_count":0, "speed_square_sum":0., "max_drift_m":0.,
                    "qualifying_run":0, "stop_confirmation":{"status":"NOT_APPLICABLE"},
                    "braking_displacement_m":None, "post_stop_reference_w_m":None,
                    "post_stop_max_drift_m":None}
                if torch.linalg.vector_norm(prior_command) > .001 and torch.linalg.vector_norm(command[lane]) <= .001:
                    state["stop_confirmation"] = {"status":"UNAVAILABLE", "reason":"NO_11_SAMPLE_CONFIRMATION"}
                self._next_block_id += 1; self._block_states[lane] = state
            state["last_step"] = policy_step; state["last_position_w_m"] = positions[lane].clone()
            state["samples"] += 1; state["reset_count"] += int(reset[lane])
            state["speed_square_sum"] += float(speeds[lane].square())
            drift = float(torch.linalg.vector_norm(positions[lane, :2] - state["reference_position_w_m"][:2]))
            state["max_drift_m"] = max(float(state["max_drift_m"]), drift)
            if state["stop_confirmation"]["status"] != "NOT_APPLICABLE":
                if state["stop_confirmation"]["status"] == "UNAVAILABLE":
                    state["qualifying_run"] = state["qualifying_run"] + 1 if float(speeds[lane]) <= .08 else 0
                    if state["qualifying_run"] == 11:
                        state["stop_confirmation"] = {"status":"CONFIRMED",
                            "confirmation_step":policy_step,
                            "confirmation_time_s":(policy_step + 1 - int(state["start_step"])) * .1,
                            "effective_boundary":"PRE_STEP"}
                        state["post_stop_reference_w_m"] = positions[lane].clone()
                        state["braking_displacement_m"] = drift
                        state["post_stop_max_drift_m"] = 0.
                else:
                    post = float(torch.linalg.vector_norm(positions[lane, :2] - state["post_stop_reference_w_m"][:2]))
                    state["post_stop_max_drift_m"] = max(float(state["post_stop_max_drift_m"]), post)
            if bool(reset[lane]):
                if state["stop_confirmation"]["status"] == "UNAVAILABLE":
                    state["stop_confirmation"] = {"status":"UNAVAILABLE", "reason":"RESET_INTERRUPTED_NO_SPLICE"}
                self._finish_command_block(state, right_censored=False)
                self._block_states[lane] = None

    def _record_diagnostic_window(self, evidence: Mapping[str, Any], transition: Any, *, update: int, step: int) -> None:
        lanes = (0, 15, 16, 31)
        source = evidence.get("evaluation_ray_source")
        selected: Mapping[str, Any]
        if isinstance(source, Mapping):
            selected = {name: (value[list(lanes)] if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 32 else value)
                        for name, value in source.items()}
        else: selected = {"status":"UNAVAILABLE","reason":"RAW_RAY_SOURCE_MISSING"}
        row = {"update":update,"step":step,"lanes":lanes,"source":selected}
        self._diagnostic_ring.append(row); self._diagnostic_ring = self._diagnostic_ring[-16:]
        if self._measurement_steps < 16:
            self.record_measurement_window({"kind":"PROCESS_INITIAL_16","rows":[row]}); self._measurement_steps += 1
        if self._failure_post_remaining:
            self._failure_post_rows.append(row); self._failure_post_remaining -= 1
            if self._failure_post_remaining == 0:
                self.record_measurement_window({"kind":"FIRST_TASK_FAILURE_POST_WINDOW","rows":self._failure_post_rows})
                self._failure_window_written = True
        if bool(transition.terminated.any()) and not self._failure_window_written and not self._failure_post_remaining:
            self.record_measurement_window({"kind":"FIRST_TASK_FAILURE_PRE_WINDOW","rows":self._diagnostic_ring})
            self._failure_post_remaining = 16; self._failure_post_rows = []

    def finalize_right_censored(self) -> None:
        if self._episode_return is None: return
        if not self._blocks_finalized:
            for state in self._block_states:
                if state is not None: self._finish_command_block(state, right_censored=True)
            self._blocks_finalized = True
        for lane in range(len(self._episode_return)):
            if int(self._episode_length[lane]) > 0:
                self.record_episode({"env":lane,"episode":int(self._episode_id[lane]),"reset":int(self._reset_id[lane]),
                    "start_type":self._episode_start_types[lane],
                    "length":int(self._episode_length[lane]),"return":float(self._episode_return[lane]),
                    "average_reward":float(self._episode_return[lane] / self._episode_length[lane]),
                    "reward_component_sums":{name:float(value[lane]) for name,value in self._episode_components.items()},
                    "behavior":{"max_horizontal_drift_m":float(self._episode_max_drift[lane]),
                        "max_height_deviation_m":float(self._episode_max_height[lane]),
                        "minimum_ray_clearance_m":float(self._episode_min_clearance[lane])},
                    "termination_masks":{},"reason":"RUN_END","right_censored":True})
        if self._failure_post_remaining:
            self.record_measurement_window({"kind":"FIRST_TASK_FAILURE_POST_WINDOW_INCOMPLETE",
                "rows":self._failure_post_rows, "missing_rows":self._failure_post_remaining})

    def record_update(self, result: Mapping[str, Any], *, attempted_updates: int, actor_optimizer_steps: int,
                      critic_optimizer_steps: int, statistics: Mapping[str, Any]) -> None:
        records = result.get("records")
        if not isinstance(records, list) or not records: raise ValueError("every PPO minibatch record must be retained")
        for ordinal, record in enumerate(records):
            self.append("minibatches.jsonl", {"version": "learning-log-h1-v1", "update": result["update_counter"],
                        "attempted_update": attempted_updates, "minibatch_ordinal": ordinal, **record})
        coverage = result.get("coverage")
        self.append("updates.jsonl", {"version": "learning-log-h1-v1", "attempted_updates": attempted_updates,
                    "completed_updates": result["update_counter"], "actor_optimizer_steps": actor_optimizer_steps,
                    "critic_optimizer_steps": critic_optimizer_steps,
                    "coverage": coverage, "statistics": dict(statistics)})

    def record_episode(self, record: Mapping[str, Any]) -> None:
        required = {"env", "episode", "reset", "start_type", "length", "return", "termination_masks", "reason", "right_censored"}
        if not required.issubset(record): raise ValueError("episode record is incomplete")
        self.append("episodes.jsonl", {"version": "learning-log-h1-v1", **record})

    def record_command_block(self, record: Mapping[str, Any]) -> None:
        required = {"block", "env", "previous_command", "command", "effective_step", "effective_end_step",
                    "duration_steps", "fixed_reference", "reset_interrupted", "reset_count", "stop_confirmation",
                    "braking_displacement_m", "post_stop_reference_w_m", "post_stop_max_drift_m", "right_censored"}
        if not required.issubset(record): raise ValueError("command-block record is incomplete")
        self.append("command_blocks.jsonl", {"version": "learning-log-h1-v1", **record})

    def record_measurement_window(self, record: Mapping[str, Any]) -> None:
        self.append("measurement_windows.jsonl", {"version": "learning-log-h1-v1", **record})

    def record_partial_failure(self, *, attempted_updates: int, completed_updates: int,
                               actor_optimizer_steps: int, critic_optimizer_steps: int, error: BaseException) -> None:
        self.append("failures.jsonl", {"version": "learning-log-h1-v1", "attempted_updates": attempted_updates,
                    "completed_updates": completed_updates, "actor_optimizer_steps": actor_optimizer_steps,
                    "critic_optimizer_steps": critic_optimizer_steps, "error_type": type(error).__name__, "message": str(error)})


def grouped_reward_summary(components: Mapping[str, torch.Tensor], masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group, mask in masks.items():
        if mask.dtype != torch.bool: raise TypeError("group masks must be boolean")
        output[group] = {name: _statistics(value[mask]) for name, value in components.items()}
    return output


def grouped_reward_summary_from_evidence(evidence: Sequence[Mapping[str, Any]], zero: torch.Tensor) -> dict[str, Any]:
    names = None; stacked: dict[str, list[torch.Tensor]] = {}
    q = []
    for item in evidence:
        components = item.get("reward_components")
        diagnostics = item.get("reward_diagnostics")
        if not isinstance(components, Mapping) or not components: raise ValueError("reward component evidence is missing")
        current = set(components); names = current if names is None else names
        if current != names: raise ValueError("reward component schema changed within rollout")
        for name, value in components.items(): stacked.setdefault(name, []).append(value.reshape(-1))
        if not isinstance(diagnostics, Mapping) or not isinstance(diagnostics.get("q"), torch.Tensor): raise ValueError("q evidence is missing")
        q.append(diagnostics["q"].reshape(-1).to(torch.bool))
    components = {name: torch.cat(values) for name, values in stacked.items()}; flat_zero = zero.flatten(); flat_q = torch.cat(q)
    return grouped_reward_summary(components, {"zero": flat_zero, "nonzero": ~flat_zero, "q": flat_q})


__all__ = ["LearningLog", "grouped_reward_summary", "grouped_reward_summary_from_evidence"]
