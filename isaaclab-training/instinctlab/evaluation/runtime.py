"""Strict environment-independent runtime adapter for accepted M7 records."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from instinctlab.observations.audit import audit_actor_observation
from .audit import EvaluationRecord, build_evaluation_record
from .metrics import EVALUATION_INPUT_KEYS, EvaluationMetricConfig, ObservabilitySource

PRE_RESET_KEYS = (
    "reset_mask", "episode_length", "root_state_w", "joint_pos", "joint_vel",
    "control", "lee", "wrench", "reward", "task",
)
CONTROL_KEYS = (
    "alpha", "v_corr_b", "v_gov_b", "v_final_b", "v_final_w",
    "anchor_metrics", "ics_metrics",
)
ANCHOR_KEYS = (
    "active", "reset_code", "usable_fraction", "mean_absolute_error_m", "huber_loss",
)
ICS_KEYS = ("beta", "active_beam_count", "minimum_clearance_m", "worst_margin_m_s", "emergency")
TASK_KEYS = ("root_state_w", "actual_velocity_b", "exact_clearance_m", "collision", "termination")
TERMINATION_KEYS = ("terminated", "truncated", "done", "reason_code", "masks")
TERMINATION_MASK_KEYS = ("collision", "below_height", "above_height", "timeout")
RUNTIME_TREE_KEYS = (
    "schema_version", "contract", "source_manifest", "policy_hashes", "action_hashes",
    "station_reference_before_w_m", "station_reference_after_w_m", "reset_masks",
    "returned_episode_lengths", "evaluation_inputs", "canonical_record", "gates",
)

SOURCE_MANIFEST = OrderedDict((
    ("commanded_velocity_b_m_s", "pre_action.policy.state_vec latest M4 command slice [-7:-4]; m/s body"),
    ("actual_velocity_b_m_s", "pre_reset.task.actual_velocity_b; m/s body; privileged evaluation"),
    ("issued_velocity_b_m_s", "pre_reset.control.v_final_b; m/s body"),
    ("position_w_m", "pre_reset.task.root_state_w[:,0:3]; m world; privileged evaluation"),
    ("station_reference_position_w_m", "collector-owned detached reset reference; m world"),
    ("anchor_error_m", "pre_reset.control.anchor_metrics.mean_absolute_error_m; m"),
    ("anchor_valid", "anchor active and usable_fraction >= configured minimum"),
    ("minimum_clearance_m", "pre_reset.task.exact_clearance_m; m; privileged evaluation"),
    ("clearance_valid", "true for accepted exact task producer"),
    ("collision", "pre_reset.task.collision; bool"),
    ("ics_beta", "pre_reset.control.ics_metrics.beta; ratio"),
    ("ics_emergency", "pre_reset.control.ics_metrics.emergency exact 0/1 converted to bool"),
    ("ics_worst_margin_m_s", "pre_reset.control.ics_metrics.worst_margin_m_s; m/s"),
    ("terminated", "pre_reset.task.termination.terminated; bool"),
    ("truncated", "pre_reset.task.termination.truncated; bool"),
    ("termination_reason", "pre_reset.task.termination.reason_code; accepted M6 precedence"),
    ("termination_collision", "pre_reset.task.termination.masks.collision; bool"),
    ("termination_below_height", "pre_reset.task.termination.masks.below_height; bool"),
    ("termination_above_height", "pre_reset.task.termination.masks.above_height; bool"),
    ("termination_timeout", "pre_reset.task.termination.masks.timeout; bool"),
    ("observability_value", "finite placeholder excluded by invalid UNAVAILABLE source"),
    ("observability_valid", "false; no authorized algorithm"),
    ("observability_source", "UNAVAILABLE; no authorized algorithm"),
))


@dataclass(frozen=True)
class RuntimeEvaluationConfig:
    num_envs: int = 2
    expected_policy_steps: int = 8
    anchor_min_usable_fraction: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.num_envs, bool) or not isinstance(self.num_envs, int) or self.num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if isinstance(self.expected_policy_steps, bool) or not isinstance(self.expected_policy_steps, int) or self.expected_policy_steps <= 0:
            raise ValueError("expected_policy_steps must be a positive integer")
        value = self.anchor_min_usable_fraction
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= value <= 1.0:
            raise ValueError("anchor_min_usable_fraction must be finite in [0,1]")
        object.__setattr__(self, "anchor_min_usable_fraction", float(value))


def _ordered(name: str, value: object, keys: tuple[str, ...]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or tuple(value.keys()) != keys:
        raise ValueError(f"{name} keys must be exactly ordered as {keys}")
    return value


def _tensor_hash(value: torch.Tensor) -> str:
    header = f"{value.dtype}|{tuple(value.shape)}|".encode()
    payload = value.detach().contiguous().cpu().numpy().tobytes(order="C")
    return hashlib.sha256(header + payload).hexdigest()


def _policy_hash(policy: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in ("lidar_grid", "state_vec"):
        digest.update(name.encode()); digest.update(_tensor_hash(policy[name]).encode())
    return digest.hexdigest()


def _float(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise TypeError(f"{name} must share policy dtype and device")
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be detached and finite")
    return value


def _critic(name: str, value: object, num_envs: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (num_envs, 18):
        raise ValueError(f"{name} must have shape {(num_envs, 18)}")
    if value.dtype != torch.float32 or value.device != reference.device:
        raise TypeError(f"{name} must be float32 on the policy device")
    if value.requires_grad or not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must be detached and finite")
    return value


def _bool(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.bool:
        raise TypeError(f"{name} must be bool with shape {shape}")
    if value.device != reference.device or value.requires_grad:
        raise ValueError(f"{name} must be detached on the policy device")
    return value


def _long(name: str, value: object, shape: tuple[int, ...], reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.long:
        raise TypeError(f"{name} must be torch.long with shape {shape}")
    if value.device != reference.device or value.requires_grad:
        raise ValueError(f"{name} must be detached on the policy device")
    return value


def _reject_source_aliases(policy: Mapping[str, torch.Tensor], sources: Mapping[str, torch.Tensor]) -> None:
    owners: dict[tuple[torch.device, int], str] = {}
    for name, value in tuple(policy.items()) + tuple(sources.items()):
        key = (value.device, value.untyped_storage().data_ptr())
        if key in owners:
            raise ValueError(f"runtime sources {owners[key]} and {name} alias storage")
        owners[key] = name


class RuntimeEvaluationCollector:
    """Collect exact transition rows and own reset-scoped station references."""

    def __init__(
        self,
        initial_critic: torch.Tensor,
        config: RuntimeEvaluationConfig = RuntimeEvaluationConfig(),
    ) -> None:
        if not isinstance(config, RuntimeEvaluationConfig):
            raise TypeError("config must be RuntimeEvaluationConfig")
        if not isinstance(initial_critic, torch.Tensor):
            raise TypeError("initial_critic must be a tensor")
        critic = _critic("initial_critic", initial_critic, config.num_envs, initial_critic)
        self.config = config
        self._station_reference = critic[:, :3].detach().clone()
        self._rows: list[OrderedDict[str, torch.Tensor]] = []
        self._policy_hashes: list[str] = []
        self._action_hashes: list[str] = []
        self._station_before: list[torch.Tensor] = []
        self._station_after: list[torch.Tensor] = []
        self._reset_masks: list[torch.Tensor] = []
        self._returned_episode_lengths: list[torch.Tensor] = []
        self._pending_policy_hash: str | None = None
        self._last_policy: OrderedDict[str, torch.Tensor] | None = None

    def prepare_policy(self, policy: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """Audit exactly once immediately before actor inference; return unchanged mapping."""
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("runtime policy count exceeds configured coverage")
        if self._pending_policy_hash is not None:
            raise RuntimeError("previous audited policy has not been consumed")
        audit_actor_observation(policy)
        if policy["lidar_grid"].dtype != torch.float32:
            raise TypeError("runtime HistoryActor policy must use float32")
        value = _policy_hash(policy)
        self._pending_policy_hash = value
        return policy

    def record_transition(
        self,
        *,
        pre_action_policy: Mapping[str, torch.Tensor],
        action: torch.Tensor,
        pre_reset: Mapping[str, object],
        returned_critic: torch.Tensor,
        returned_episode_length: torch.Tensor,
    ) -> None:
        if len(self._rows) >= self.config.expected_policy_steps:
            raise RuntimeError("runtime transition count exceeds configured coverage")
        if self._pending_policy_hash is None or _policy_hash(pre_action_policy) != self._pending_policy_hash:
            raise ValueError("transition policy does not match the sole pending audited policy")
        policy_ref = pre_action_policy["state_vec"]
        batch = self.config.num_envs
        if not isinstance(action, torch.Tensor) or action.shape != (batch, 4) or action.dtype != torch.float32 or action.device != policy_ref.device:
            raise TypeError("action must be float32 [N,4] on the policy device")
        if action.requires_grad or not bool(torch.isfinite(action).all().item()) or bool(((action < 0.0) | (action > 1.0)).any().item()):
            raise ValueError("action must be detached, finite, and in [0,1]")
        pre = _ordered("pre_reset", pre_reset, PRE_RESET_KEYS)
        control = _ordered("pre_reset.control", pre["control"], CONTROL_KEYS)
        anchor = _ordered("pre_reset.control.anchor_metrics", control["anchor_metrics"], ANCHOR_KEYS)
        ics = _ordered("pre_reset.control.ics_metrics", control["ics_metrics"], ICS_KEYS)
        task = _ordered("pre_reset.task", pre["task"], TASK_KEYS)
        termination = _ordered("pre_reset.task.termination", task["termination"], TERMINATION_KEYS)
        masks = _ordered("pre_reset.task.termination.masks", termination["masks"], TERMINATION_MASK_KEYS)
        root = _float("pre_reset.task.root_state_w", task["root_state_w"], (batch, 13), policy_ref)
        if not torch.equal(root, pre["root_state_w"]):
            raise ValueError("task root_state_w must equal top-level same-state pre_reset root_state_w")
        actual = _float("actual_velocity_b", task["actual_velocity_b"], (batch, 3), policy_ref)
        issued = _float("control.v_final_b", control["v_final_b"], (batch, 3), policy_ref)
        anchor_error = _float("anchor.mean_absolute_error_m", anchor["mean_absolute_error_m"], (batch, 1), policy_ref)
        anchor_active = _float("anchor.active", anchor["active"], (batch, 1), policy_ref)
        anchor_usable = _float("anchor.usable_fraction", anchor["usable_fraction"], (batch, 1), policy_ref)
        clearance = _float("task.exact_clearance_m", task["exact_clearance_m"], (batch, 1), policy_ref)
        collision = _bool("task.collision", task["collision"], (batch, 1), policy_ref)
        beta = _float("ics.beta", ics["beta"], (batch, 1), policy_ref)
        emergency_float = _float("ics.emergency", ics["emergency"], (batch, 1), policy_ref)
        worst = _float("ics.worst_margin_m_s", ics["worst_margin_m_s"], (batch, 1), policy_ref)
        terminated = _bool("termination.terminated", termination["terminated"], (batch, 1), policy_ref)
        truncated = _bool("termination.truncated", termination["truncated"], (batch, 1), policy_ref)
        done = _bool("termination.done", termination["done"], (batch, 1), policy_ref)
        reason = _long("termination.reason_code", termination["reason_code"], (batch, 1), policy_ref)
        mask_values = {name: _bool(f"termination.masks.{name}", masks[name], (batch, 1), policy_ref) for name in TERMINATION_MASK_KEYS}
        reset_mask = _bool("pre_reset.reset_mask", pre["reset_mask"], (batch,), policy_ref)
        episode = _long("pre_reset.episode_length", pre["episode_length"], (batch,), policy_ref)
        returned = _critic("returned_critic", returned_critic, batch, policy_ref)
        returned_episode = _long("returned_episode_length", returned_episode_length, (batch,), policy_ref)
        if not torch.equal(done[:, 0], reset_mask) or not torch.equal(done, terminated | truncated):
            raise ValueError("pre_reset reset/done masks are inconsistent")
        expected_episode = torch.where(reset_mask, torch.zeros_like(episode), episode)
        if not torch.equal(returned_episode, expected_episode):
            raise ValueError("returned episode length violates pre-reset/post-reset ownership")
        live_mask = ~reset_mask
        if live_mask.any() and not torch.equal(root[live_mask], returned[live_mask, :13]):
            raise ValueError("non-reset pre_reset root_state_w is stale or from a different state")
        if reset_mask.any() and torch.equal(root[reset_mask], returned[reset_mask, :13]):
            raise ValueError("terminal pre_reset root_state_w was substituted with post-reset state")
        if bool(((anchor_active != 0.0) & (anchor_active != 1.0)).any().item()) or bool(((emergency_float != 0.0) & (emergency_float != 1.0)).any().item()):
            raise ValueError("accepted anchor/ICS boolean metrics must contain exact 0/1")
        if bool(((anchor_usable < 0.0) | (anchor_usable > 1.0)).any().item()):
            raise ValueError("anchor usable fraction must be in [0,1]")
        sources = OrderedDict((
            ("actual", actual), ("issued", issued), ("root", root), ("anchor_error", anchor_error),
            ("anchor_active", anchor_active), ("anchor_usable", anchor_usable), ("clearance", clearance),
            ("collision", collision), ("beta", beta), ("emergency", emergency_float), ("worst", worst),
            ("terminated", terminated), ("truncated", truncated), ("reason", reason),
            *((f"mask_{name}", mask_values[name]) for name in TERMINATION_MASK_KEYS),
            ("returned_critic", returned), ("returned_episode_length", returned_episode),
        ))
        _reject_source_aliases(pre_action_policy, sources)
        station_before = self._station_reference.detach().clone()
        command = pre_action_policy["state_vec"][:, -7:-4]
        row = OrderedDict((
            ("commanded_velocity_b_m_s", command.detach().clone()),
            ("actual_velocity_b_m_s", actual.detach().clone()),
            ("issued_velocity_b_m_s", issued.detach().clone()),
            ("position_w_m", root[:, :3].detach().clone()),
            ("station_reference_position_w_m", station_before.clone()),
            ("anchor_error_m", anchor_error[:, 0].detach().clone()),
            ("anchor_valid", ((anchor_active[:, 0] == 1.0) & (anchor_usable[:, 0] >= self.config.anchor_min_usable_fraction)).detach().clone()),
            ("minimum_clearance_m", clearance[:, 0].detach().clone()),
            ("clearance_valid", torch.ones(batch, dtype=torch.bool, device=policy_ref.device)),
            ("collision", collision[:, 0].detach().clone()),
            ("ics_beta", beta[:, 0].detach().clone()),
            ("ics_emergency", (emergency_float[:, 0] == 1.0).detach().clone()),
            ("ics_worst_margin_m_s", worst[:, 0].detach().clone()),
            ("terminated", terminated[:, 0].detach().clone()),
            ("truncated", truncated[:, 0].detach().clone()),
            ("termination_reason", reason[:, 0].detach().clone()),
            ("termination_collision", mask_values["collision"][:, 0].detach().clone()),
            ("termination_below_height", mask_values["below_height"][:, 0].detach().clone()),
            ("termination_above_height", mask_values["above_height"][:, 0].detach().clone()),
            ("termination_timeout", mask_values["timeout"][:, 0].detach().clone()),
            ("observability_value", torch.zeros(batch, dtype=policy_ref.dtype, device=policy_ref.device)),
            ("observability_valid", torch.zeros(batch, dtype=torch.bool, device=policy_ref.device)),
            ("observability_source", torch.full((batch,), int(ObservabilitySource.UNAVAILABLE), dtype=torch.long, device=policy_ref.device)),
        ))
        if tuple(row) != EVALUATION_INPUT_KEYS:
            raise RuntimeError("internal runtime evaluation order violation")
        next_reference = self._station_reference.detach().clone()
        next_reference[reset_mask] = returned[reset_mask, :3].detach().clone()
        if not torch.equal(next_reference[~reset_mask], self._station_reference[~reset_mask]):
            raise RuntimeError("non-reset station references changed")
        self._station_reference = next_reference
        self._rows.append(row); self._policy_hashes.append(self._pending_policy_hash); self._action_hashes.append(_tensor_hash(action))
        self._station_before.append(station_before); self._station_after.append(next_reference.detach().clone())
        self._reset_masks.append(reset_mask.detach().clone()); self._returned_episode_lengths.append(returned_episode.detach().clone())
        self._last_policy = OrderedDict((name, value.detach().clone()) for name, value in pre_action_policy.items())
        self._pending_policy_hash = None

    def finalize(
        self,
        provenance: Mapping[str, object],
        metric_config: EvaluationMetricConfig = EvaluationMetricConfig(),
    ) -> OrderedDict[str, object]:
        if self._pending_policy_hash is not None or len(self._rows) != self.config.expected_policy_steps or self._last_policy is None:
            raise RuntimeError("runtime coverage is incomplete or has an unconsumed policy")
        evaluation = OrderedDict((name, torch.stack([row[name] for row in self._rows], dim=0)) for name in EVALUATION_INPUT_KEYS)
        record: EvaluationRecord = build_evaluation_record(OrderedDict((("policy", self._last_policy), ("evaluation", evaluation))), provenance, metric_config)
        reset_masks = torch.stack(self._reset_masks)
        tree = OrderedDict((
            ("schema_version", "m7.runtime-evaluation.v1"),
            ("contract", OrderedDict((("num_envs", self.config.num_envs), ("policy_steps", self.config.expected_policy_steps), ("samples", self.config.num_envs * self.config.expected_policy_steps)))),
            ("source_manifest", SOURCE_MANIFEST.copy()),
            ("policy_hashes", tuple(self._policy_hashes)),
            ("action_hashes", tuple(self._action_hashes)),
            ("station_reference_before_w_m", torch.stack(self._station_before).detach().cpu()),
            ("station_reference_after_w_m", torch.stack(self._station_after).detach().cpu()),
            ("reset_masks", reset_masks.detach().cpu()),
            ("returned_episode_lengths", torch.stack(self._returned_episode_lengths).detach().cpu()),
            ("evaluation_inputs", OrderedDict((name, value.detach().cpu()) for name, value in evaluation.items())),
            ("canonical_record", record.canonical_json),
            ("gates", OrderedDict((("coverage_exact", True), ("all_finite", True), ("observability_unavailable", True), ("reset_count", int(reset_masks.sum().item()))))),
        ))
        if tuple(tree) != RUNTIME_TREE_KEYS or not all(torch.isfinite(value).all() for value in tree["evaluation_inputs"].values() if value.is_floating_point()):
            raise RuntimeError("runtime output tree contract violation")
        json.loads(record.canonical_json)
        return tree


__all__ = [
    "ANCHOR_KEYS", "CONTROL_KEYS", "ICS_KEYS", "PRE_RESET_KEYS", "RUNTIME_TREE_KEYS",
    "RuntimeEvaluationCollector", "RuntimeEvaluationConfig", "SOURCE_MANIFEST", "TASK_KEYS",
    "TERMINATION_KEYS", "TERMINATION_MASK_KEYS",
]
