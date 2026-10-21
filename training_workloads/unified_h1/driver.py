"""Actual schedule-to-environment step adapter and attempt-first accounting."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from contextlib import nullcontext
from typing import Any

import torch

from .budget import ExecutionCounters
from .schedule import ScheduleClock


class UnifiedH1Driver:
    def __init__(self, environment: Any, observation: Mapping[str, Any], *, clock: ScheduleClock | None = None,
                 counters: ExecutionCounters | None = None, journal: Any | None = None,
                 before_step: Any | None = None, transition_limit: int | None = None,
                 automatic_reset_limit: int | None = None, step_kind: str = "training") -> None:
        if not callable(getattr(environment, "prepare_h1_step", None)) or not callable(getattr(environment, "step", None)):
            raise TypeError("environment must expose prepare_h1_step and step")
        self.environment = environment; self.observation = observation
        self.clock = clock or ScheduleClock(); self.counters = counters or ExecutionCounters()
        self.journal = journal
        self.before_step = before_step
        self.transition_limit = transition_limit
        self.automatic_reset_limit = automatic_reset_limit
        if step_kind not in {"training", "nontraining"}: raise ValueError("invalid step kind")
        self.step_kind = step_kind
        self.evidence_buffer: list[Mapping[str, Any]] = []

    def step(self, action: torch.Tensor):
        policy = self.observation["policy"]
        latest_command = policy["state_vec"][:, 45:48]
        plan = self.clock.plan(device=latest_command.device, dtype=latest_command.dtype)
        if not torch.equal(latest_command, plan.command_b):
            raise ValueError("obs_t command does not equal schedule c_t")
        if self.before_step is not None:
            self.before_step(plan.global_policy_step)
        self.counters.require_step_capacity(kind=self.step_kind, lanes=plan.command_b.shape[0], maximum=self.transition_limit)
        self.counters.require_reset_capacity(automatic=True, lanes=plan.command_b.shape[0],
            maximum=self.automatic_reset_limit)
        self.environment.prepare_h1_step(global_policy_step=plan.global_policy_step,
            command_b=plan.command_b, next_command_b=plan.next_command_b)
        self.counters.attempt_step()
        if self.journal is not None:
            self.journal.record_step_accounting(event="ENV_STEP_ATTEMPT",
                attempted_policy_steps=self.counters.attempted_policy_steps,
                completed_policy_steps=self.counters.completed_policy_steps,
                transitions=self.counters.real_environment_transitions)
        returned = self.environment.step(action)
        # A returned env.step consumed physical transitions even when its payload
        # later proves malformed.  Account before interpreting the payload.
        self.counters.complete_step(plan.command_b.shape[0], kind=self.step_kind)
        self.clock.complete_step()
        if self.journal is not None:
            self.journal.record_step_accounting(event="ENV_STEP_COMPLETE",
                attempted_policy_steps=self.counters.attempted_policy_steps,
                completed_policy_steps=self.counters.completed_policy_steps,
                transitions=self.counters.real_environment_transitions)
        if not isinstance(returned, tuple) or len(returned) != 5:
            raise ValueError("environment must return the Gymnasium five-tuple")
        next_observation, reward, terminated, truncated, extras = returned
        if not isinstance(extras, Mapping):
            raise TypeError("environment extras must be a mapping")
        raw = deepcopy(extras)
        scene = getattr(self.environment, "scene", None)
        origins = getattr(scene, "env_origins", None)
        raw["_h1_driver"] = {"global_policy_step": plan.global_policy_step,
            "pure_hover": self.clock.pure_hover,
            "center_reset_only": getattr(self.environment, "_h1_pure_hover_training", False),
            "command_b": plan.command_b.detach().cpu(), "next_command_b": plan.next_command_b.detach().cpu(),
            "action": action.detach().cpu(), "reward": reward.detach().cpu(),
            "terminated": terminated.detach().cpu(), "truncated": truncated.detach().cpu(),
            "env_origins_w_m": (origins.detach().cpu() if isinstance(origins, torch.Tensor)
                else {"status":"UNAVAILABLE", "reason":"ENV_ORIGINS_NOT_EXPOSED"})}
        if self.journal is not None:
            self.journal.record_step_evidence(raw, completed_policy_steps=self.counters.completed_policy_steps)
        reset_lanes = int((terminated | truncated).to(torch.int64).sum().item())
        self.counters.reserve_reset(automatic=True, lanes=reset_lanes, maximum=self.automatic_reset_limit)
        if not torch.equal(next_observation["policy"]["state_vec"][:, 45:48], plan.next_command_b):
            raise ValueError("obs_(t+1) command does not equal schedule c_(t+1)")
        self.evidence_buffer.append(raw)
        self.observation = next_observation
        return returned

    def take_evidence(self) -> tuple[Mapping[str, Any], ...]:
        result = tuple(self.evidence_buffer); self.evidence_buffer.clear(); return result


def optimizer_steps(optimizer: torch.optim.Optimizer) -> int:
    states = optimizer.state_dict()["state"]
    if not states: return 0
    steps = []
    for state in states.values():
        value = state.get("step")
        if not isinstance(value, torch.Tensor) or value.numel() != 1: raise ValueError("partial Adam state")
        scalar = float(value.item())
        if not scalar.is_integer(): raise ValueError("nonintegral Adam step")
        steps.append(int(scalar))
    if len(set(steps)) != 1: raise ValueError("Adam parameter steps disagree")
    return steps[0]


class UnifiedH1TrainingLoop:
    """Actual driver -> rollout -> PPO -> durable log/checkpoint call path."""

    def __init__(self, driver: UnifiedH1Driver, trainer: Any, journal: Any,
                 checkpoint_writer: Any | None = None, before_update: Any | None = None,
                 update_limit: int | None = None, before_rollout: Any | None = None,
                 before_optimizer: Any | None = None, diagnostic: Any | None = None,
                 optimizer_step_origin: tuple[int, int] = (0, 0)) -> None:
        self.driver = driver; self.trainer = trainer; self.journal = journal
        self.checkpoint_writer = checkpoint_writer
        if before_update is not None and (before_rollout is not None or before_optimizer is not None):
            raise ValueError("before_update compatibility callback cannot be combined with explicit stage callbacks")
        self.before_rollout = before_update if before_rollout is None else before_rollout
        self.before_optimizer = before_update if before_optimizer is None else before_optimizer
        self.update_limit = update_limit
        self.diagnostic = diagnostic
        if (not isinstance(optimizer_step_origin, tuple) or len(optimizer_step_origin) != 2
                or any(type(v) is not int or v < 0 for v in optimizer_step_origin)):
            raise ValueError("optimizer step origins must be two nonnegative integers")
        self.optimizer_step_origin = optimizer_step_origin
        self.driver.journal = journal

    def _optimizer_deltas(self, actor, critic):
        result = tuple(value - origin for value, origin in zip((actor, critic), self.optimizer_step_origin))
        if any(value < 0 for value in result):
            raise RuntimeError("optimizer counters fell below restored origin")
        return result

    def run_update(self, *, checkpoint_label: str | None = None) -> Mapping[str, Any]:
        from instinctlab.training.rollout import RolloutConfig
        from instinctlab.training.rollout_h1 import collect_h1_rollout

        counters = self.driver.counters
        durable_at_start = self.journal.durable_step_rows
        if self.before_rollout is not None: self.before_rollout(self.trainer.update_counter + 1)
        counters.attempt_update(self.update_limit)
        self.journal.record_update_attempt(attempted_updates=counters.attempted_updates,
            completed_updates=counters.completed_updates,
            attempted_policy_steps=counters.attempted_policy_steps,
            completed_policy_steps=counters.completed_policy_steps)
        try:
            capture = (self.diagnostic.capture_rollout_distribution(self.trainer)
                       if self.diagnostic is not None else nullcontext())
            with capture:
                rollout = collect_h1_rollout(self.driver, self.trainer, self.driver.observation, RolloutConfig(16, 32))
            evidence = self.driver.take_evidence()
            statistics = self.journal.record_rollout(rollout.segment, evidence, update=self.trainer.update_counter + 1)
            if self.before_optimizer is not None: self.before_optimizer(self.trainer.update_counter + 1)
            update_ordinal = self.trainer.update_counter + 1
            if self.diagnostic is not None:
                self.diagnostic.bind_rollout_distribution(rollout.segment.batch)
                self.diagnostic.retain_rollout(update_ordinal, rollout.segment.batch, self.trainer)
                self.diagnostic.before_update(self.trainer, rollout.segment.batch)
            result = self.trainer.update(rollout.segment.batch)
            if self.diagnostic is not None:
                self.diagnostic.after_update(self.trainer, rollout.segment.batch, result)
            actor = optimizer_steps(self.trainer.actor_optimizer); critic = optimizer_steps(self.trainer.critic_optimizer)
            actor_delta, critic_delta = self._optimizer_deltas(actor, critic)
            counters.complete_update(actor_delta, critic_delta)
            self.journal.record_update(result, attempted_updates=counters.attempted_updates,
                actor_optimizer_steps=actor, critic_optimizer_steps=critic, statistics=statistics)
            if checkpoint_label is not None:
                if not callable(self.checkpoint_writer): raise RuntimeError("checkpoint writer is not configured")
                self.checkpoint_writer(checkpoint_label, self.trainer, self.driver)
            return result
        except BaseException as error:
            if self.diagnostic is not None:
                try: self.diagnostic.abort(error)
                except BaseException as diagnostic_error:
                    error.add_note(f"diagnostic failure preservation failed: {diagnostic_error}")
            # A full output volume can fail every recovery write. Keep trying
            # independent records, but never replace the original rollout/PPO
            # error with an evidence-persistence error.
            def preserve_failure(name, operation):
                try:operation()
                except BaseException as persistence_error:
                    error.add_note(f"{name} preservation failed: {persistence_error!r}")
            preserve_failure('partial episodes', lambda: self.journal.record_partial_episodes(self.driver.evidence_buffer))
            preserve_failure('partial rollout boundary', lambda: self.journal.record_partial_rollout_boundary(
                attempted_policy_steps=counters.attempted_policy_steps,
                completed_policy_steps=counters.completed_policy_steps,
                durable_step_rows=self.journal.durable_step_rows - durable_at_start,
                buffered_step_rows=len(self.driver.evidence_buffer), error=error))
            try: actor = optimizer_steps(self.trainer.actor_optimizer)
            except BaseException: actor = -1
            try: critic = optimizer_steps(self.trainer.critic_optimizer)
            except BaseException: critic = -1
            counters.actor_optimizer_steps = actor - self.optimizer_step_origin[0] if actor >= 0 else -1
            counters.critic_optimizer_steps = critic - self.optimizer_step_origin[1] if critic >= 0 else -1
            preserve_failure('partial failure', lambda: self.journal.record_partial_failure(attempted_updates=counters.attempted_updates,
                completed_updates=counters.completed_updates, actor_optimizer_steps=actor,
                critic_optimizer_steps=critic, error=error))
            raise


__all__ = ["UnifiedH1Driver", "UnifiedH1TrainingLoop", "optimizer_steps"]
