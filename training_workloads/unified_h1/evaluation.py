"""Raw-episode implementation of the frozen ``metric-h1-v1`` contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from statistics import NormalDist
from typing import Iterable, Mapping, Sequence

import torch

from .budget import EVALUATION_CASE_STEPS
from .schedule import DIRECTIONS, NUM_ENVS

CASE_ORDER = ("Z", "T", "S", "X", "W")
ERROR_THRESHOLDS = {
    "z_speed": .08, "z_drift": .30, "z_height": .20, "t_rmse": .15,
    "s_stop_time": 3.0, "s_drift": .30, "x_rmse": .15, "w_min_distance": .8,
}
EVIDENCE_FAILURE_REASONS = frozenset({
    "MISSING_REQUIRED_SAMPLE", "CASE_COMMAND_MISMATCH", "CASE_SEED_MISMATCH",
    "CASE_TIMING_MISMATCH", "INITIAL_STATE_MISSING", "INITIAL_STATE_MISMATCH",
    "RESET_INTERRUPTED_NO_SPLICE",
})


@dataclass(frozen=True)
class CaseConfig:
    name: str
    steps: int
    transitions: int
    seed: int = 20260909
    num_envs: int = NUM_ENVS


@dataclass(frozen=True)
class EpisodeTrace:
    case: str
    env_index: int
    command_b: torch.Tensor
    velocity_b: torch.Tensor
    sensor_positions_w_m: torch.Tensor
    env_origin_w_m: torch.Tensor
    episode_ids: tuple[int, ...]
    collision: torch.Tensor
    below_height: torch.Tensor
    above_height: torch.Tensor
    missing: torch.Tensor
    ray_clearance_m: torch.Tensor | None = None
    initial_sensor_position_w_m: torch.Tensor | None = None
    initial_root_position_w_m: torch.Tensor | None = None
    initial_velocity_b: torch.Tensor | None = None
    initial_root_quaternion_wxyz: torch.Tensor | None = None
    sample_policy_steps: tuple[int, ...] = ()
    seed: int = 20260909
    q: torch.Tensor | None = None
    ics_beta: torch.Tensor | None = None


CASES = {name: CaseConfig(name, steps, NUM_ENVS * steps) for name, steps in EVALUATION_CASE_STEPS.items()}


def evaluation_manifest() -> dict[str, object]:
    return {
        "version": "metric-h1-v1", "case_order": CASE_ORDER, "seed": 20260909,
        "action_mode": "deterministic_mean", "training_rng_isolated": True,
        "transitions_per_checkpoint": sum(case.transitions for case in CASES.values()),
        "cases": {name: {"steps": case.steps, "num_envs": case.num_envs,
                         "transitions": case.transitions} for name, case in CASES.items()},
        "success_denominator_per_case": 32,
        "early_failure_policy": "FAIL_AND_RETAIN_IN_DENOMINATOR",
        "geometry": {"local_floor_half_extent_m": 42.0, "walls_m": 6.0,
                     "wall_height_m": 12.0, "surface_ids": ["floor", "x_wall", "y_wall"]},
    }


def case_commands(name: str, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if name not in CASES:
        raise ValueError("unknown metric-h1 case")
    result = torch.zeros(CASES[name].steps, NUM_ENVS, 3, dtype=dtype)
    directions = torch.tensor(DIRECTIONS, dtype=dtype)[torch.arange(NUM_ENVS) % 8]
    if name == "T": result[:] = directions
    elif name == "S": result[:100] = directions
    elif name == "X": result[50:125] = directions; result[125:] = -directions
    elif name == "W": result[:, :16, 0] = .2; result[:, 16:, 1] = .2
    return result


def case_initial_positions(name: str) -> torch.Tensor:
    if name not in CASES:
        raise ValueError("unknown metric-h1 case")
    result = torch.zeros(NUM_ENVS, 3); result[:, 2] = 2.0
    if name == "W":
        tangent = (torch.arange(NUM_ENVS) % 4 - 1.5) * .05
        result[:16, 0] = 4.6; result[:16, 1] = tangent[:16]
        result[16:, 1] = 4.6; result[16:, 0] = tangent[16:]
    else:
        offsets = torch.tensor([[-.1, -.1], [-.1, .1], [.1, -.1], [.1, .1]])
        result[:, :2] = offsets[torch.arange(NUM_ENVS) // 8]
    return result


def stopping_confirmation(speed_norm_m_s: Sequence[float], *, zero_effective_index: int,
                          dt_s: float = 0.1, effective_boundary: str = "PRE_STEP") -> dict[str, object]:
    if zero_effective_index < 0 or zero_effective_index >= len(speed_norm_m_s):
        raise ValueError("zero_effective_index is out of range")
    if effective_boundary not in {"PRE_STEP", "OBSERVATION"}:
        raise ValueError("effective_boundary must be PRE_STEP or OBSERVATION")
    run = 0
    for index in range(zero_effective_index, len(speed_norm_m_s)):
        value = float(speed_norm_m_s[index])
        if not math.isfinite(value) or value < 0:
            raise ValueError("speed values must be finite and nonnegative")
        run = run + 1 if value <= .08 else 0
        if run == 11:
            # Row i is the observation returned after policy step i.  A command
            # staged for step k is effective at k*dt, while an observed wall
            # trigger at row k occurs at (k+1)*dt.
            elapsed_steps = index + 1 - zero_effective_index
            if effective_boundary == "OBSERVATION": elapsed_steps -= 1
            return {"status": "CONFIRMED", "first_qualifying_index": index - 10,
                    "confirmation_index": index,
                    "confirmation_time_s": elapsed_steps * dt_s,
                    "effective_boundary": effective_boundary}
    return {"status": "UNAVAILABLE", "reason": "NO_11_SAMPLE_CONFIRMATION"}


def _rms(value: torch.Tensor) -> float:
    return float(torch.sqrt(value.square().mean()).item())


def evaluate_stop_episode(*, speed_norm_m_s: Sequence[float], positions_w_m: torch.Tensor,
                          zero_effective_index: int, episode_ids: Sequence[int],
                          incoming_command_speed_m_s: float = .2) -> dict[str, object]:
    """Evaluate S from the complete trace, including the incoming T segment."""
    if len(speed_norm_m_s) != positions_w_m.shape[0] or len(episode_ids) != len(speed_norm_m_s):
        raise ValueError("stop trace lengths differ")
    if positions_w_m.ndim != 2 or positions_w_m.shape[1] != 3 or not bool(torch.isfinite(positions_w_m).all().item()):
        raise ValueError("positions must be finite [T,3]")
    if len(set(episode_ids)) != 1:
        return {"success": False, "reason": "RESET_INTERRUPTED_NO_SPLICE"}
    if zero_effective_index > 0:
        incoming = torch.as_tensor(speed_norm_m_s[:zero_effective_index], dtype=torch.float64)
        if incoming.numel() != 100:
            return {"success": False, "reason": "INCOMING_TRACKING_WINDOW_MISSING"}
        incoming_error = _rms(incoming - incoming_command_speed_m_s)
        stable_error = _rms(incoming[20:] - incoming_command_speed_m_s)
        if incoming_error > .15 or stable_error > .10:
            return {"success": False, "reason": "INCOMING_TRACKING_FAILED",
                    "incoming_tracking_rmse_m_s": incoming_error,
                    "incoming_stable_rmse_m_s": stable_error}
    confirmation = stopping_confirmation(speed_norm_m_s, zero_effective_index=zero_effective_index)
    if confirmation["status"] != "CONFIRMED" or confirmation["confirmation_time_s"] > 3.0:
        return {"success": False, "reason": "STOP_NOT_CONFIRMED", "confirmation": confirmation}
    index = int(confirmation["confirmation_index"])
    if positions_w_m.shape[0] < index + 201:
        return {"success": False, "reason": "POST_CONFIRMATION_200_STEPS_MISSING", "confirmation": confirmation}
    reference = positions_w_m[index].clone()
    post = positions_w_m[index:index + 201]
    post_speed = torch.as_tensor(speed_norm_m_s[index:index + 201], dtype=torch.float64)
    speed_rms = _rms(post_speed)
    drift = float(torch.linalg.vector_norm(post[:, :2] - reference[:2], dim=-1).max().item())
    height = float((post[:, 2] - reference[2]).abs().max().item())
    success = speed_rms <= .08 and drift <= .30 and height <= .20
    return {"success": success, "reason": "PASS" if success else "POST_STOP_Z_GATE",
            "reference_index": index, "post_stop_speed_rms_m_s": speed_rms,
            "post_stop_max_horizontal_drift_m": drift, "post_stop_max_height_deviation_m": height,
            "confirmation": confirmation}


def wall_motion_prerequisite(projected_speed_m_s: Sequence[float]) -> bool:
    run = 0
    for value in projected_speed_m_s:
        run = run + 1 if math.isfinite(float(value)) and float(value) >= .10 else 0
        if run >= 11:
            return True
    return False


def _rectangle_distance(local: torch.Tensor, *, fixed_axis: int, fixed_value: float,
                        first_axis: int, first_bounds: tuple[float, float],
                        second_axis: int, second_bounds: tuple[float, float]) -> torch.Tensor:
    fixed = (local[:, fixed_axis] - fixed_value).abs()
    zero = torch.zeros_like(fixed)
    first = torch.maximum(torch.maximum(first_bounds[0] - local[:, first_axis], local[:, first_axis] - first_bounds[1]), zero)
    second = torch.maximum(torch.maximum(second_bounds[0] - local[:, second_axis], local[:, second_axis] - second_bounds[1]), zero)
    return torch.sqrt(fixed.square() + first.square() + second.square())


def sensor_origin_surface_distance(position_w_m: torch.Tensor, env_origins_w_m: torch.Tensor | None = None
                                   ) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Distance from mounted sensor origins to finite translated floor/+x/+y rectangles."""
    if not isinstance(position_w_m, torch.Tensor) or position_w_m.ndim != 2 or position_w_m.shape[1] != 3 or not bool(torch.isfinite(position_w_m).all().item()):
        raise ValueError("sensor position must be finite [N,3]")
    if env_origins_w_m is None:
        env_origins_w_m = torch.zeros_like(position_w_m)
    if not isinstance(env_origins_w_m, torch.Tensor) or env_origins_w_m.shape != position_w_m.shape or not bool(torch.isfinite(env_origins_w_m).all().item()):
        raise ValueError("env origins must be finite and match sensor positions")
    local = position_w_m.to(torch.float64) - env_origins_w_m.to(torch.float64)
    distances = torch.stack((
        _rectangle_distance(local, fixed_axis=2, fixed_value=0., first_axis=0, first_bounds=(-42., 42.), second_axis=1, second_bounds=(-42., 42.)),
        _rectangle_distance(local, fixed_axis=0, fixed_value=6., first_axis=1, first_bounds=(-6., 6.), second_axis=2, second_bounds=(0., 12.)),
        _rectangle_distance(local, fixed_axis=1, fixed_value=6., first_axis=0, first_bounds=(-6., 6.), second_axis=2, second_bounds=(0., 12.)),
    ), dim=-1)
    distance, index = distances.min(dim=-1)
    names = ("floor", "x_wall", "y_wall")
    return distance.to(position_w_m.dtype), tuple(names[item] for item in index.tolist())


def _validate_trace(trace: EpisodeTrace) -> int:
    if not isinstance(trace, EpisodeTrace) or trace.case not in CASES or not 0 <= trace.env_index < NUM_ENVS:
        raise ValueError("invalid metric-h1 episode identity")
    steps = CASES[trace.case].steps
    if trace.command_b.shape != (steps, 3) or trace.velocity_b.shape != (steps, 3) or trace.sensor_positions_w_m.shape != (steps, 3):
        raise ValueError("episode tensors do not match the fixed case length")
    for value in (trace.command_b, trace.velocity_b, trace.sensor_positions_w_m):
        if value.dtype not in (torch.float32, torch.float64) or not bool(torch.isfinite(value).all().item()):
            raise ValueError("episode float tensors must be finite")
    for value in (trace.collision, trace.below_height, trace.above_height, trace.missing):
        if value.shape != (steps,) or value.dtype != torch.bool:
            raise ValueError("episode masks must be boolean [T]")
    if (trace.env_origin_w_m.shape != (3,) or not bool(torch.isfinite(trace.env_origin_w_m).all().item())
            or len(trace.episode_ids) != steps):
        raise ValueError("episode origin/id length mismatch")
    if trace.ray_clearance_m is not None and (trace.ray_clearance_m.shape != (steps,)
            or not bool(torch.isfinite(trace.ray_clearance_m).all().item())):
        raise ValueError("ray clearance must be finite [T]")
    if trace.q is not None and (trace.q.shape != (steps,) or trace.q.dtype not in (torch.bool, torch.float32, torch.float64)
            or not bool(torch.isfinite(trace.q).all().item())):
        raise ValueError("q must be finite boolean/float [T]")
    if trace.ics_beta is not None and (trace.ics_beta.shape != (steps,)
            or not bool(torch.isfinite(trace.ics_beta).all().item())):
        raise ValueError("ICS beta must be finite [T]")
    return steps


def _base_failure(trace: EpisodeTrace) -> str | None:
    if bool(trace.missing.any().item()): return "MISSING_REQUIRED_SAMPLE"
    expected_command = case_commands(trace.case, dtype=trace.command_b.dtype)[:, trace.env_index]
    if not torch.equal(trace.command_b.cpu(), expected_command): return "CASE_COMMAND_MISMATCH"
    if trace.seed != CASES[trace.case].seed: return "CASE_SEED_MISMATCH"
    if trace.sample_policy_steps != tuple(range(CASES[trace.case].steps)): return "CASE_TIMING_MISMATCH"
    required = (trace.initial_sensor_position_w_m, trace.initial_root_position_w_m, trace.initial_velocity_b)
    if (any(not isinstance(value, torch.Tensor) or value.shape != (3,) or not bool(torch.isfinite(value).all()) for value in required)
            or not isinstance(trace.initial_root_quaternion_wxyz, torch.Tensor)
            or trace.initial_root_quaternion_wxyz.shape != (4,)
            or not bool(torch.isfinite(trace.initial_root_quaternion_wxyz).all())):
        return "INITIAL_STATE_MISSING"
    expected_root = case_initial_positions(trace.case)[trace.env_index].to(trace.initial_root_position_w_m)
    actual_root = trace.initial_root_position_w_m - trace.env_origin_w_m.to(trace.initial_root_position_w_m)
    expected_sensor_offset = torch.tensor([0., 0., .05], dtype=trace.initial_sensor_position_w_m.dtype,
                                          device=trace.initial_sensor_position_w_m.device)
    identity = torch.tensor([1., 0., 0., 0.], dtype=trace.initial_root_quaternion_wxyz.dtype,
                            device=trace.initial_root_quaternion_wxyz.device)
    if (not torch.allclose(actual_root, expected_root, atol=5e-5, rtol=0)
            or not torch.equal(trace.initial_velocity_b, torch.zeros_like(trace.initial_velocity_b))
            or not torch.allclose(trace.initial_root_quaternion_wxyz, identity, atol=5e-6, rtol=0)
            or not torch.allclose(trace.initial_sensor_position_w_m - trace.initial_root_position_w_m,
                                  expected_sensor_offset, atol=5e-5, rtol=0)):
        return "INITIAL_STATE_MISMATCH"
    if len(set(trace.episode_ids)) != 1: return "RESET_INTERRUPTED_NO_SPLICE"
    if bool(trace.collision.any().item()): return "COLLISION"
    if bool((trace.below_height | trace.above_height).any().item()): return "HEIGHT_FAILURE"
    return None


def _engineering_evidence(trace: EpisodeTrace, failure: str | None) -> dict[str, bool]:
    missing = bool(trace.missing.any().item())
    reset_interrupted = len(set(trace.episode_ids)) != 1
    return {
        "collision": bool(trace.collision.any().item()),
        "height_failure": bool((trace.below_height | trace.above_height).any().item()),
        "missing": missing,
        "reset_interrupted": reset_interrupted,
        "evidence_valid": not missing and not reset_interrupted and failure not in EVIDENCE_FAILURE_REASONS,
    }


def _z_metrics(velocity: torch.Tensor, positions: torch.Tensor,
               initial_position: torch.Tensor | None = None) -> dict[str, float]:
    reference = positions[0] if initial_position is None else initial_position
    return {"speed": _rms(torch.linalg.vector_norm(velocity, dim=-1)),
            "drift": float(torch.linalg.vector_norm(positions[:, :2] - reference[:2], dim=-1).max().item()),
            "height": float((positions[:, 2] - reference[2]).abs().max().item())}


def _tracking_metrics(command: torch.Tensor, velocity: torch.Tensor, positions: torch.Tensor,
                      initial_position: torch.Tensor, stable_start: int = 19) -> dict[str, float]:
    error = torch.linalg.vector_norm(velocity - command, dim=-1)
    return {"rmse": _rms(error), "stable_rmse": _rms(error[stable_start:]),
            "vz_rms": _rms(velocity[stable_start:, 2]),
            "height": float((positions[:, 2] - initial_position[2]).abs().max().item())}


def evaluate_episode(trace: EpisodeTrace) -> dict[str, object]:
    steps = _validate_trace(trace)
    failure = _base_failure(trace)
    metrics: dict[str, float] = {}
    success = failure is None
    if trace.case == "Z":
        metrics = _z_metrics(trace.velocity_b, trace.sensor_positions_w_m, trace.initial_sensor_position_w_m)
        success = success and metrics["speed"] <= .08 and metrics["drift"] <= .30 and metrics["height"] <= .20
    elif trace.case == "T":
        metrics = _tracking_metrics(trace.command_b, trace.velocity_b, trace.sensor_positions_w_m,
                                    trace.initial_sensor_position_w_m)
        success = success and metrics["rmse"] <= .15 and metrics["stable_rmse"] <= .10 and metrics["vz_rms"] <= .08 and metrics["height"] <= .20
    elif trace.case == "S":
        tracking = _tracking_metrics(trace.command_b[:100], trace.velocity_b[:100], trace.sensor_positions_w_m[:100],
                                     trace.initial_sensor_position_w_m)
        stop = evaluate_stop_episode(speed_norm_m_s=torch.linalg.vector_norm(trace.velocity_b, dim=-1).tolist(), positions_w_m=trace.sensor_positions_w_m,
                                     zero_effective_index=100, episode_ids=trace.episode_ids,
                                     incoming_command_speed_m_s=float(torch.linalg.vector_norm(trace.command_b[0]).item()))
        metrics = {"incoming_rmse": tracking["rmse"], "incoming_stable_rmse": tracking["stable_rmse"],
                   "stop_time": float(stop.get("confirmation", {}).get("confirmation_time_s", 15.0)),
                   "drift": float(stop.get("post_stop_max_horizontal_drift_m", 1.5)),
                   "speed": float(stop.get("post_stop_speed_rms_m_s", .4)),
                   "height": float(stop.get("post_stop_max_height_deviation_m", 1.0)),
                   "braking_displacement": float(torch.linalg.vector_norm(
                       trace.sensor_positions_w_m[int(stop.get("reference_index", 100)), :2]
                       - trace.sensor_positions_w_m[99, :2]).item()),
                   "zero_block_drift": float(torch.linalg.vector_norm(
                       trace.sensor_positions_w_m[100:, :2] - trace.sensor_positions_w_m[99, :2], dim=-1).max().item())}
        tracking_pass = tracking["rmse"] <= .15 and tracking["stable_rmse"] <= .10 and tracking["vz_rms"] <= .08 and tracking["height"] <= .20
        success = success and tracking_pass and bool(stop["success"])
        if not tracking_pass: failure = failure or "INCOMING_TRACKING_FAILED"
        if not stop["success"]: failure = failure or str(stop["reason"])
    elif trace.case == "X":
        zero = _z_metrics(trace.velocity_b[:50], trace.sensor_positions_w_m[:50], trace.initial_sensor_position_w_m)
        error = torch.linalg.vector_norm(trace.velocity_b - trace.command_b, dim=-1)
        reaches = []; reach_times = []
        for start in (50, 125):
            run = 0; reached = False; confirmation_time = None
            for index in range(start, min(start + 31, steps)):
                run = run + 1 if float(error[index]) <= .10 else 0
                if run >= 11:
                    confirmation_time = (index + 1 - start) * .1
                    reached = confirmation_time <= 3.0
                    break
            reaches.append(reached)
            reach_times.append(confirmation_time)
        nonzero = torch.cat((error[50:125], error[125:])); stable = torch.cat((error[69:125], error[144:]))
        command_norm = torch.linalg.vector_norm(trace.command_b, dim=-1)
        projection = (trace.velocity_b * trace.command_b).sum(-1) / command_norm.clamp_min(1e-12)
        metrics = {"rmse": _rms(nonzero), "stable_rmse": _rms(stable), "zero_speed": zero["speed"],
                   "zero_drift": zero["drift"], "height": float((trace.sensor_positions_w_m[:, 2] - trace.initial_sensor_position_w_m[2]).abs().max()),
                   "peak_projection": float(projection[50:].max()), "reach_confirmation_time_s": reach_times}
        success = success and all(reaches) and metrics["rmse"] <= .15 and metrics["stable_rmse"] <= .10 and metrics["zero_speed"] <= .08 and metrics["zero_drift"] <= .30 and metrics["height"] <= .20 and metrics["peak_projection"] <= 1.5 * float(command_norm[50])
        if not all(reaches): failure = failure or "NONZERO_REACH_CONFIRMATION_FAILED"
    else:
        geometry, surface_ids = sensor_origin_surface_distance(trace.sensor_positions_w_m, trace.env_origin_w_m.expand(steps, -1))
        projected = (trace.velocity_b * trace.command_b).sum(-1) / torch.linalg.vector_norm(trace.command_b, dim=-1).clamp_min(1e-12)
        reached = torch.nonzero(geometry <= 1.1).flatten()
        proactive = not bool(reached.numel())
        trigger = int(reached[0]) if reached.numel() else 0
        confirmation = stopping_confirmation(torch.linalg.vector_norm(trace.velocity_b, dim=-1).tolist(),
                                             zero_effective_index=trigger,
                                             effective_boundary="OBSERVATION" if reached.numel() else "PRE_STEP")
        confirmed = confirmation["status"] == "CONFIRMED" and (
            confirmation["confirmation_time_s"] <= 3. if reached.numel() else confirmation["confirmation_time_s"] <= 20.)
        hold = False
        if confirmed:
            index = int(confirmation["confirmation_index"]); hold = _rms(torch.linalg.vector_norm(trace.velocity_b[index:], dim=-1)) <= .08
        metrics = {"min_distance": float(geometry.min()), "height": float((trace.sensor_positions_w_m[:, 2] - trace.initial_sensor_position_w_m[2]).abs().max()),
                   "stop_time": float(confirmation.get("confirmation_time_s", 20.0)),
                   "proactive_stop_outside_trigger": proactive and confirmed,
                   "q_true_steps": None if trace.q is None else int((trace.q > .5).sum().item()),
                   "ics_beta_min": None if trace.ics_beta is None else float(trace.ics_beta.min().item())}
        success = success and metrics["min_distance"] >= .8 and metrics["height"] <= .20 and confirmed and hold and (bool(reached.numel()) or wall_motion_prerequisite(projected.tolist()))
        if not success: failure = failure or "WALL_GATE_FAILED"
        return {"case": trace.case, "env_index": trace.env_index, "success": success, "reason": "PASS" if success else failure,
                "metrics": metrics, "surface_ids": surface_ids, "geometry_distance_m": geometry.tolist(),
                "ray_clearance_m": None if trace.ray_clearance_m is None else trace.ray_clearance_m.tolist(),
                "engineering": _engineering_evidence(trace, failure)}
    if not success and failure is None: failure = f"{trace.case}_THRESHOLD_FAILED"
    return {"case": trace.case, "env_index": trace.env_index, "success": success,
            "reason": "PASS" if success else failure, "metrics": metrics,
            "engineering": _engineering_evidence(trace, failure)}


def wilson_interval(successes: int, total: int, confidence: float = .95) -> tuple[float, float]:
    if isinstance(successes, bool) or isinstance(total, bool) or not 0 <= successes <= total or total <= 0:
        raise ValueError("Wilson counts are invalid")
    z = NormalDist().inv_cdf(.5 + confidence / 2); p = successes / total
    denominator = 1 + z * z / total; center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0., center - radius), min(1., center + radius)


def case_outcome(success: Sequence[bool], subgroup: Sequence[str], *, expected_subgroups: Iterable[str]) -> dict[str, object]:
    if len(success) != 32 or len(subgroup) != 32: raise ValueError("all 32 intended samples must remain in the denominator")
    groups = {}
    for name in expected_subgroups:
        rows = [bool(value) for value, group in zip(success, subgroup, strict=True) if group == name]
        if not rows: raise ValueError("required subgroup is empty")
        groups[name] = {"successes": sum(rows), "total": len(rows), "pass": sum(rows) / len(rows) >= .95}
    passed = sum(bool(item) for item in success)
    return {"successes": passed, "total": 32, "pass": passed >= 31 and all(row["pass"] for row in groups.values()),
            "wilson_95": wilson_interval(passed, 32), "subgroups": groups}


def _error_components(results: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, float]:
    selectors = {"z_speed": ("Z", "speed"), "z_drift": ("Z", "drift"), "z_height": ("Z", "height"),
                 "t_rmse": ("T", "rmse"), "s_stop_time": ("S", "stop_time"), "s_drift": ("S", "drift"),
                 "x_rmse": ("X", "rmse"), "w_min_distance": ("W", "min_distance")}
    output = {}
    for key, (case, metric) in selectors.items():
        scores = []
        for row in results[case]:
            observed = float(row["metrics"][metric]); score = max(0., .8 - observed) / .8 if key == "w_min_distance" else observed / ERROR_THRESHOLDS[key]
            if not row["success"]: score = max(score, 5.)
            scores.append(score)
        output[key] = sum(scores) / len(scores)
    return output


def evaluate_checkpoint(traces: Sequence[EpisodeTrace], *, update: int) -> dict[str, object]:
    if type(update) is not int or update < 0:
        raise ValueError("checkpoint update must be a nonnegative integer")
    if len(traces) != 32 * len(CASE_ORDER): raise ValueError("checkpoint evaluation requires the fixed 160 episodes")
    evaluated = [evaluate_episode(trace) for trace in traces]
    by_case = {name: sorted((row for row in evaluated if row["case"] == name), key=lambda row: int(row["env_index"])) for name in CASE_ORDER}
    if any([int(row["env_index"]) for row in rows] != list(range(32)) for rows in by_case.values()):
        raise ValueError("each case requires exactly one episode for every env")
    cases = {}
    for name, rows in by_case.items():
        subgroup = [f"d{i % 8}" for i in range(32)] if name != "W" else ["x_wall" if i < 16 else "y_wall" for i in range(32)]
        expected = [f"d{i}" for i in range(8)] if name != "W" else ["x_wall", "y_wall"]
        cases[name] = case_outcome([bool(row["success"]) for row in rows], subgroup, expected_subgroups=expected)
    components = _error_components(by_case)
    collisions = sum(int(trace.collision.any()) for trace in traces)
    heights = sum(int((trace.below_height | trace.above_height).any()) for trace in traces)
    invalid_evidence = sum(not bool(row["engineering"]["evidence_valid"]) for row in evaluated)
    engineering_pass = (collisions == 0 and heights == 0 and invalid_evidence == 0
                        and all(int(case["total"]) == 32 for case in cases.values()))
    return {"schema_version": "metric-h1-v1.checkpoint.v1", "update": update, "episodes": evaluated,
            "cases": cases, "error_components": components, "loss": sum(components.values()) / 8,
            "failures": {name: 32 - int(cases[name]["successes"]) for name in CASE_ORDER},
            "collisions": collisions, "height_failures": heights,
            "invalid_evidence_episodes": invalid_evidence,
            "engineering_gates_pass": engineering_pass,
            "all_gates_pass": engineering_pass and all(case["pass"] for case in cases.values())}


def normalized_loss(metrics: Mapping[str, float], failed: Mapping[str, bool]) -> float:
    if set(metrics) != set(ERROR_THRESHOLDS): raise ValueError("all eight metric-h1 errors are required")
    values = []
    for key, threshold in ERROR_THRESHOLDS.items():
        observed = float(metrics[key]); score = max(0., .8 - observed) / .8 if key == "w_min_distance" else observed / threshold
        values.append(max(score, 5.) if failed.get(key, False) else score)
    if any(not math.isfinite(value) or value < 0 for value in values): raise ValueError("selection metrics must be finite and nonnegative")
    return sum(values) / 8


def _complete_checkpoint_evidence(row: Mapping[str, object], expected_update: int) -> bool:
    """Verify that every decision field agrees with its episode evidence."""
    required = {"schema_version", "update", "episodes", "cases", "error_components", "loss", "failures",
                "collisions", "height_failures", "invalid_evidence_episodes",
                "engineering_gates_pass", "all_gates_pass"}
    try:
        if (type(expected_update) is not int or expected_update < 0 or not required.issubset(row)
                or row.get("schema_version") != "metric-h1-v1.checkpoint.v1"
                or type(row.get("update")) is not int or row.get("update") != expected_update):
            return False
        episodes = row["episodes"]
        if not isinstance(episodes, Sequence) or isinstance(episodes, (str, bytes)) or len(episodes) != 32 * len(CASE_ORDER):
            return False
        by_case: dict[str, list[Mapping[str, object]]] = {name: [] for name in CASE_ORDER}
        for episode in episodes:
            if (not isinstance(episode, Mapping) or episode.get("case") not in by_case
                    or type(episode.get("env_index")) is not int or type(episode.get("success")) is not bool
                    or not isinstance(episode.get("metrics"), Mapping)
                    or not isinstance(episode.get("reason"), str)
                    or bool(episode["success"]) != (episode["reason"] == "PASS")):
                return False
            engineering = episode.get("engineering")
            if (not isinstance(engineering, Mapping)
                    or set(engineering) != {"collision", "height_failure", "missing", "reset_interrupted", "evidence_valid"}
                    or any(type(value) is not bool for value in engineering.values())):
                return False
            expected_valid = (not engineering["missing"] and not engineering["reset_interrupted"]
                              and episode["reason"] not in EVIDENCE_FAILURE_REASONS)
            if (engineering["evidence_valid"] is not expected_valid
                    or (episode["success"] and any((engineering["collision"], engineering["height_failure"],
                                                    engineering["missing"], engineering["reset_interrupted"])))):
                return False
            by_case[str(episode["case"])].append(episode)
        derived_cases = {}
        for name, episodes_for_case in by_case.items():
            episodes_for_case.sort(key=lambda item: int(item["env_index"]))
            if [int(item["env_index"]) for item in episodes_for_case] != list(range(32)):
                return False
            subgroup = [f"d{i % 8}" for i in range(32)] if name != "W" else ["x_wall" if i < 16 else "y_wall" for i in range(32)]
            expected_groups = [f"d{i}" for i in range(8)] if name != "W" else ["x_wall", "y_wall"]
            derived_cases[name] = case_outcome([bool(item["success"]) for item in episodes_for_case], subgroup,
                                               expected_subgroups=expected_groups)
        cases = row["cases"]
        if not isinstance(cases, Mapping) or set(cases) != set(CASE_ORDER): return False
        for name, expected in derived_cases.items():
            actual = cases[name]
            if not isinstance(actual, Mapping) or set(actual) != set(expected): return False
            if any(actual[key] != expected[key] for key in ("successes", "total", "pass", "subgroups")): return False
            interval = actual["wilson_95"]
            if (not isinstance(interval, Sequence) or len(interval) != 2
                    or any(not math.isclose(float(interval[i]), expected["wilson_95"][i], rel_tol=1e-12, abs_tol=1e-12) for i in range(2))):
                return False
        components = _error_components(by_case)
        actual_components = row["error_components"]
        if not isinstance(actual_components, Mapping) or set(actual_components) != set(ERROR_THRESHOLDS): return False
        if any(not math.isclose(float(actual_components[key]), components[key], rel_tol=1e-9, abs_tol=1e-12) for key in ERROR_THRESHOLDS):
            return False
        if not math.isclose(float(row["loss"]), sum(components.values()) / 8, rel_tol=1e-9, abs_tol=1e-12): return False
        failures = {name: 32 - int(derived_cases[name]["successes"]) for name in CASE_ORDER}
        collision_count = sum(int(bool(item["engineering"]["collision"])) for item in episodes)
        height_count = sum(int(bool(item["engineering"]["height_failure"])) for item in episodes)
        invalid_evidence = sum(not bool(item["engineering"]["evidence_valid"]) for item in episodes)
        engineering_pass = (collision_count == 0 and height_count == 0 and invalid_evidence == 0
                            and all(int(case["total"]) == 32 for case in derived_cases.values()))
        all_pass = engineering_pass and all(bool(case["pass"]) for case in derived_cases.values())
        return (row["failures"] == failures and row["collisions"] == collision_count
                and row["height_failures"] == height_count
                and type(row["invalid_evidence_episodes"]) is int
                and row["invalid_evidence_episodes"] == invalid_evidence
                and row["engineering_gates_pass"] is engineering_pass and row["all_gates_pass"] is all_pass)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def continuation_decision(*, baseline: Mapping[str, object], current: Mapping[str, object],
                          previous: Mapping[str, object] | None = None,
                          u64: Mapping[str, object] | None = None, stage: str = "R4", **legacy: object) -> dict[str, object]:
    """Decide only from complete parser output; incomplete metric maps fail closed."""
    complete = _complete_checkpoint_evidence
    if stage == "R4":
        expected = (0, 256, 192, 64)
    elif stage == "R5":
        target = current.get("update")
        nodes = {512: (256, 448), 1024: (512, 960), 2048: (1024, 1984)}
        if target not in nodes: return {"extend": False, "reason": "R5_NODE_MISMATCH"}
        expected = (nodes[int(target)][0], int(target), nodes[int(target)][1], -1)
    else:
        raise ValueError("stage must be R4 or R5")
    if not complete(baseline, expected[0]) or not complete(current, expected[1]):
        return {"extend": False, "reason": "INCOMPLETE_RAW_EPISODE_EVIDENCE"}
    old = baseline["error_components"]; new = current["error_components"]
    no_regression = all(float(new[key]) <= float(old[key]) + max(float(old[key]) * .10, .10) for key in ERROR_THRESHOLDS)
    failure_safe = all(int(current["failures"][key]) <= int(baseline["failures"][key]) for key in CASE_ORDER)
    if stage == "R4":
        if previous is None or u64 is None or not complete(previous, 192) or not complete(u64, 64): return {"extend": False, "reason": "R4_REQUIRES_U192_U256_AND_U64_EVIDENCE"}
        evidence_valid = all(int(row["invalid_evidence_episodes"]) == 0
                             for row in (baseline, u64, previous, current))
        safe_nodes = all(int(row["collisions"]) == 0 and int(row["height_failures"]) == 0
                         for row in (previous, current))
        forgetting_safe = all(float(current["error_components"][key]) <= float(u64["error_components"][key]) + max(float(u64["error_components"][key]) * .20, .10)
                              for key in ("z_speed", "z_drift", "z_height"))
        behavior = max((float(old[k]) - float(new[k])) / max(float(old[k]), 1e-12) for k in ("z_speed", "z_drift", "s_stop_time", "s_drift")) >= .10
        tracking = max((float(old[k]) - float(new[k])) / max(float(old[k]), 1e-12) for k in ("t_rmse", "x_rmse")) >= .10
        extend = evidence_valid and safe_nodes and forgetting_safe and failure_safe and no_regression and behavior and tracking and float(current["loss"]) <= .9 * float(baseline["loss"])
    elif stage == "R5":
        if previous is None or not complete(previous, expected[2]): return {"extend": False, "reason": "R5_REQUIRES_TWO_LATEST_CHECKPOINTS"}
        evidence_valid = all(int(row["invalid_evidence_episodes"]) == 0
                             for row in (baseline, previous, current))
        safe_nodes = all(int(row["collisions"]) == 0 and int(row["height_failures"]) == 0
                         for row in (previous, current))
        unmet = [key for key in ERROR_THRESHOLDS if float(old[key]) > 1.0]
        improved = any((float(old[key]) - float(new[key])) / max(float(old[key]), 1e-12) >= .10 for key in unmet)
        held = all(not bool(baseline["cases"][case]["pass"]) or bool(current["cases"][case]["pass"]) for case in CASE_ORDER)
        extend = evidence_valid and safe_nodes and failure_safe and no_regression and improved and held and float(current["loss"]) <= .9 * float(baseline["loss"])
    return {"extend": extend, "reason": "SUFFICIENT_CONTINUATION_EVIDENCE" if extend else "NO_SUFFICIENT_CONTINUATION_EVIDENCE", "baseline_loss": float(baseline["loss"]), "current_loss": float(current["loss"])}


def select_checkpoint(rows: Sequence[Mapping[str, object]]) -> Mapping[str, object] | None:
    passing = [row for row in rows if type(row.get("update")) is int
               and _complete_checkpoint_evidence(row, int(row["update"]))
               and row.get("all_gates_pass") is True]
    return min(passing, key=lambda row: (float(row["loss"]), int(row["update"]))) if passing else None


def replay_raw_case(path: str, *, case: str) -> list[EpisodeTrace]:
    """Reconstruct the fixed 32 traces from the durable append-only raw file."""
    from pathlib import Path
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(rows) != CASES[case].steps + 1 or rows[0].get("kind") != "T0" or rows[0].get("case") != case:
        raise ValueError("raw case is incomplete or has the wrong identity")
    initial = rows[0]; steps = rows[1:]
    if [row.get("step") for row in steps] != list(range(CASES[case].steps)) or any(row.get("case") != case for row in steps):
        raise ValueError("raw case timing is incomplete")
    tensors = lambda key, dtype=None: torch.as_tensor([row[key] for row in steps], dtype=dtype)
    command = tensors("command_b"); velocity = tensors("actual_velocity_b")
    positions = tensors("sensor_positions_w_m"); episodes = tensors("episode_ids", torch.int64)
    collision = tensors("collision", torch.bool); below = tensors("below_height", torch.bool)
    above = tensors("above_height", torch.bool); ray = tensors("ray_clearance_m")
    origins = torch.as_tensor(initial["env_origins_w_m"]); initial_sensor = torch.as_tensor(initial["initial_sensor_position_w_m"])
    root = torch.as_tensor(initial["initial_root_state_w"])
    q = tensors("q", torch.bool); ics = tensors("ics_beta")
    return [EpisodeTrace(case, lane, command[:,lane], velocity[:,lane], positions[:,lane], origins[lane],
        tuple(int(value) for value in episodes[:,lane]), collision[:,lane], below[:,lane], above[:,lane],
        torch.zeros(CASES[case].steps,dtype=torch.bool), ray[:,lane], initial_sensor[lane], root[lane,:3],
        root[lane,7:10], root[lane,3:7], tuple(range(CASES[case].steps)), int(initial["seed"]), q[:,lane], ics[:,lane])
        for lane in range(NUM_ENVS)]


__all__ = ["CASES", "CASE_ORDER", "CaseConfig", "EpisodeTrace", "case_commands", "case_initial_positions", "case_outcome", "continuation_decision", "evaluate_checkpoint", "evaluate_episode", "evaluate_stop_episode", "evaluation_manifest", "normalized_loss", "replay_raw_case", "select_checkpoint", "sensor_origin_surface_distance", "stopping_confirmation", "wall_motion_prerequisite", "wilson_interval"]
