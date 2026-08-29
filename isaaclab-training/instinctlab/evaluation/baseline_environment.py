"""Exact evaluation-only command-driven B0 TASLAB environment mode."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from instinctlab.control.pipeline import ControlCoreResult
from instinctlab.envs.taslab_direct_env import DECIMATION, TaslabDirectEnv, TaslabDirectEnvCfg

from .baseline_controller_execution import B0ControllerExecutionEnvelope
from .baseline_run import (
    B0DirectConfig,
    B0RunConfig,
    B0RunReceipt,
    B0RunSession,
    begin_ready_b0_run,
)
from .baselines import BaselineRunManifest


_ACTIVE = "ACTIVE"
_FINALIZED = "FINALIZED"
_ENVIRONMENT_TOKEN = object()


def _owned_direct_config(config: B0DirectConfig) -> B0DirectConfig:
    if type(config) is not B0DirectConfig:
        raise TypeError("direct_config must be an exact B0DirectConfig")
    return B0DirectConfig(config.velocity_limit_m_s)


def _validate_environment_config(config: TaslabDirectEnvCfg) -> None:
    if type(config) is not TaslabDirectEnvCfg:
        raise TypeError("environment_config must be an exact TaslabDirectEnvCfg")
    if config.decimation != DECIMATION:
        raise ValueError(f"B0 evaluation decimation must remain exactly {DECIMATION}")
    if config.scene.num_envs != 2:
        raise ValueError("B0 evaluation environment must contain exactly two environments")
    if torch.device(config.sim.device) != torch.device("cuda:0"):
        raise ValueError("B0 evaluation environment must use exact cuda:0")
    if config.action_noise_model is not None:
        raise ValueError("B0 evaluation action noise must be disabled")


def _owned_environment_config(config: TaslabDirectEnvCfg) -> TaslabDirectEnvCfg:
    _validate_environment_config(config)
    owned = deepcopy(config)
    if owned is config:
        raise RuntimeError("environment_config copy must be independently owned")
    _validate_environment_config(owned)
    return owned


@dataclass(frozen=True, slots=True)
class B0EnvironmentConfig:
    """Exact bounded policy horizon, direct config, and accepted environment config."""

    expected_policy_steps: int
    direct_config: B0DirectConfig
    environment_config: TaslabDirectEnvCfg

    def __post_init__(self) -> None:
        if type(self.expected_policy_steps) is not int:
            raise TypeError("expected_policy_steps must be an exact integer")
        if self.expected_policy_steps <= 0:
            raise ValueError("expected_policy_steps must be positive")
        object.__setattr__(self, "direct_config", _owned_direct_config(self.direct_config))
        object.__setattr__(self, "environment_config", _owned_environment_config(self.environment_config))


@dataclass(frozen=True, slots=True)
class B0EnvironmentCoverage:
    """Immutable scalar coverage for one finalized environment execution."""

    policy_steps: int
    physics_steps: int
    lifecycle_calls: int
    wrench_applications: int
    environment_samples: int
    num_envs: int
    decimation: int


@dataclass(frozen=True, slots=True)
class B0EnvironmentResult:
    """Privacy-preserving lifecycle receipt and scalar environment coverage."""

    receipt: B0RunReceipt
    coverage: B0EnvironmentCoverage

    def __post_init__(self) -> None:
        if type(self.receipt) is not B0RunReceipt:
            raise TypeError("receipt must be an exact B0RunReceipt")
        if type(self.coverage) is not B0EnvironmentCoverage:
            raise TypeError("coverage must be an exact B0EnvironmentCoverage")


def _disabled_control(envelope: B0ControllerExecutionEnvelope) -> ControlCoreResult:
    body = envelope.b0_execution.result.issued_body_velocity_m_s
    world = envelope.b0_execution.result.issued_world_velocity_m_s
    batch = body.shape[0]
    scalar = (batch, 1)
    zeros = torch.zeros(scalar, device=body.device, dtype=body.dtype)
    anchor_metrics = {
        "active": zeros.clone(),
        "reset_code": torch.zeros(scalar, device=body.device, dtype=torch.long),
        "usable_fraction": zeros.clone(),
        "mean_absolute_error_m": zeros.clone(),
        "huber_loss": zeros.clone(),
    }
    ics_metrics = {
        "beta": torch.ones(scalar, device=body.device, dtype=body.dtype),
        "active_beam_count": zeros.clone(),
        "minimum_clearance_m": zeros.clone(),
        "worst_margin_m_s": zeros.clone(),
        "emergency": zeros.clone(),
    }
    return ControlCoreResult(
        torch.ones(scalar, device=body.device, dtype=body.dtype),
        torch.zeros_like(body),
        body,
        body,
        world,
        anchor_metrics,
        ics_metrics,
    )


class TaslabB0EvaluationEnv(TaslabDirectEnv):
    """Factory-owned command-only B0 execution over the accepted TASLAB environment."""

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("TaslabB0EvaluationEnv must not be subclassed")

    def __init__(
        self,
        manifest: BaselineRunManifest,
        config: B0EnvironmentConfig,
        render_mode: str | None,
        token: object,
    ) -> None:
        if token is not _ENVIRONMENT_TOKEN:
            raise TypeError("TaslabB0EvaluationEnv must be created by begin_ready_b0_environment")
        if type(config) is not B0EnvironmentConfig:
            raise TypeError("config must be an exact B0EnvironmentConfig")
        environment_config = _owned_environment_config(config.environment_config)
        expected_physics_steps = config.expected_policy_steps * DECIMATION
        if expected_physics_steps >= 2**64:
            raise ValueError("environment lifecycle horizon exceeds unsigned 64-bit framing")
        session = begin_ready_b0_run(
            manifest,
            B0RunConfig(expected_physics_steps, config.direct_config),
        )
        self._b0_motion_started = False
        super().__init__(environment_config, render_mode)
        if self.num_envs != 2 or self.cfg.decimation != DECIMATION:
            raise RuntimeError("constructed environment violates exact B0 coverage dimensions")
        if torch.device(self.device) != torch.device("cuda:0"):
            raise RuntimeError("constructed B0 environment must remain on cuda:0")
        self._b0_session = session
        self._b0_expected_policy_steps = config.expected_policy_steps
        self._b0_expected_physics_steps = expected_physics_steps
        self._b0_policy_steps = 0
        self._b0_physics_steps = 0
        self._b0_lifecycle_calls = 0
        self._b0_wrench_applications = 0
        self._b0_state = _ACTIVE
        self._b0_current_command = None

    @property
    def state_token(self) -> str:
        return self._b0_state

    @property
    def policy_steps(self) -> int:
        return self._b0_policy_steps

    @property
    def physics_steps(self) -> int:
        return self._b0_physics_steps

    @property
    def lifecycle_calls(self) -> int:
        return self._b0_lifecycle_calls

    @property
    def wrench_applications(self) -> int:
        return self._b0_wrench_applications

    @property
    def expected_policy_steps(self) -> int:
        return self._b0_expected_policy_steps

    @property
    def expected_physics_steps(self) -> int:
        return self._b0_expected_physics_steps

    @property
    def finalized(self) -> bool:
        return self._b0_state == _FINALIZED

    def _validate_b0_action(self, action: torch.Tensor) -> None:
        if type(action) is not torch.Tensor:
            raise TypeError("B0 environment action must be an exact torch.Tensor")
        if action.shape != (self.num_envs, 3):
            raise ValueError(f"B0 environment action must have shape ({self.num_envs}, 3)")
        if action.device != self._command_b.device:
            raise ValueError("B0 environment action must already share the environment device")
        if action.dtype != self._command_b.dtype:
            raise TypeError("B0 environment action must share the environment floating dtype")
        if action.requires_grad or action.grad_fn is not None:
            raise ValueError("B0 environment action must be detached and gradient-free")
        if not action.is_contiguous() or action.storage_offset() != 0 or (
            action.untyped_storage().nbytes() != action.numel() * action.element_size()
        ):
            raise ValueError("B0 environment action must own one exact contiguous tensor allocation")
        if not bool(torch.isfinite(action).all().item()):
            raise ValueError("B0 environment action must be finite")
        internal_tensors = (
            self._command_b,
            self._previous_issued_v_final_b,
            self._reward_previous_v_final_b,
            self._actions,
            self._robot.data.root_state_w,
        )
        pointer = action.untyped_storage().data_ptr()
        if any(pointer == value.untyped_storage().data_ptr() for value in internal_tensors):
            raise ValueError("B0 environment action must not alias environment state")

    def step(self, action: torch.Tensor):
        """Own one command, run exactly one accepted decimated policy step, then release it."""
        if type(self) is not TaslabB0EvaluationEnv:
            raise TypeError("step requires an exact TaslabB0EvaluationEnv")
        if self._b0_state != _ACTIVE:
            raise RuntimeError("finalized B0 environment cannot step")
        if self._b0_policy_steps >= self._b0_expected_policy_steps:
            raise RuntimeError("B0 environment reached its policy horizon and must be finalized")
        if self._b0_current_command is not None:
            raise RuntimeError("B0 environment command staging is already active")
        self._validate_b0_action(action)
        self._b0_current_command = action.detach().clone()
        self._b0_motion_started = True
        try:
            result = super(TaslabDirectEnv, self).step(self._b0_current_command)
        except Exception:
            self._b0_current_command = None
            raise
        self._b0_policy_steps += 1
        self._b0_current_command = None
        return result

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._validate_b0_action(actions)
        if self._b0_current_command is None or not torch.equal(actions, self._b0_current_command):
            raise RuntimeError("DirectRLEnv changed the owned B0 command before physics")
        self._reward_previous_v_final_b.copy_(self._previous_issued_v_final_b)
        self._command_b.copy_(self._b0_current_command)

    def _apply_action(self) -> None:
        if self._b0_current_command is None:
            raise RuntimeError("B0 command is unavailable during physics application")
        if self._b0_physics_steps >= self._b0_expected_physics_steps:
            raise RuntimeError("B0 environment physics horizon overrun")
        root_snapshot = self._validated_root_state().detach().clone()
        envelope = self._b0_session.step(self._b0_current_command, root_snapshot)
        if type(envelope) is not B0ControllerExecutionEnvelope:
            raise TypeError("B0 lifecycle must return an exact controller envelope")
        self._b0_lifecycle_calls += 1
        control = _disabled_control(envelope)
        wrench = self._wrench_adapter.apply(
            self._robot,
            envelope.lee_result.normalized_rotor_commands,
        )
        self._b0_wrench_applications += 1
        self._control = control
        self._lee = envelope.lee_result
        self._wrench = wrench
        self._previous_issued_v_final_b.copy_(control.v_final_b)
        self._b0_physics_steps += 1

    def _reset_idx(self, env_ids) -> None:
        if getattr(self, "_b0_motion_started", False):
            raise RuntimeError("mid-run reset is not qualified for B0 environment execution")
        super()._reset_idx(env_ids)

    def finalize(self) -> B0EnvironmentResult:
        """Finalize once at the exact policy/physics horizon and clear B0 tensor caches."""
        if type(self) is not TaslabB0EvaluationEnv:
            raise TypeError("finalize requires an exact TaslabB0EvaluationEnv")
        if self._b0_state != _ACTIVE:
            raise RuntimeError("B0 environment may be finalized exactly once")
        if self._b0_current_command is not None:
            raise RuntimeError("B0 environment cannot finalize during an active policy step")
        if self._b0_policy_steps != self._b0_expected_policy_steps or (
            self._b0_physics_steps != self._b0_expected_physics_steps
        ):
            raise RuntimeError("B0 environment cannot finalize before the exact horizon")
        if self._b0_lifecycle_calls != self._b0_expected_physics_steps or (
            self._b0_wrench_applications != self._b0_expected_physics_steps
        ):
            raise RuntimeError("B0 environment coverage counters are inconsistent")
        receipt = self._b0_session.finalize()
        coverage = B0EnvironmentCoverage(
            policy_steps=self._b0_policy_steps,
            physics_steps=self._b0_physics_steps,
            lifecycle_calls=self._b0_lifecycle_calls,
            wrench_applications=self._b0_wrench_applications,
            environment_samples=self._b0_physics_steps * self.num_envs,
            num_envs=self.num_envs,
            decimation=DECIMATION,
        )
        result = B0EnvironmentResult(receipt, coverage)
        self._b0_current_command = None
        self._command_b.zero_()
        self._previous_issued_v_final_b.zero_()
        self._reward_previous_v_final_b.zero_()
        self._control = None
        self._lee = None
        self._wrench = None
        self._b0_state = _FINALIZED
        return result


def begin_ready_b0_environment(
    manifest: BaselineRunManifest,
    config: B0EnvironmentConfig,
    render_mode: str | None,
) -> TaslabB0EvaluationEnv:
    """Create one exact factory-owned B0 TASLAB environment execution."""

    if type(config) is not B0EnvironmentConfig:
        raise TypeError("config must be an exact B0EnvironmentConfig")
    if render_mode is not None:
        raise ValueError("bounded B0 environment execution requires headless render_mode None")
    return TaslabB0EvaluationEnv(manifest, config, render_mode, _ENVIRONMENT_TOKEN)


__all__ = [
    "B0EnvironmentConfig",
    "B0EnvironmentCoverage",
    "B0EnvironmentResult",
    "TaslabB0EvaluationEnv",
    "begin_ready_b0_environment",
]
