"""Formal adapter for an environment-owned supported post-physics hook."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from .b0_enhanced import PHYSICS_ROWS, TraceValidator, canonical_sha256

PROTECTED_STATE=("control","dynamics","rng","reset","actor_history","simulator")

class EnhancedB0ExecutionAccounting:
    """Gate and settle every reset and policy step before the next live effect."""
    def __init__(self, counters: Any, budgets: Mapping[str, int], resource_check: Callable[[str, int], None]):
        self.counters=counters; self.budgets=budgets; self.resource_check=resource_check
    def before_reset(self, case: str) -> None:
        self.resource_check("before_enhanced_b0_reset", self.counters.explicit_resets + 1)
        self.counters.require_reset_capacity(automatic=False, maximum=self.budgets["explicit_resets"])
    def after_reset(self, case: str) -> None:
        self.counters.reserve_reset(automatic=False, maximum=self.budgets["explicit_resets"])
    def before_step(self, case: str, policy_step: int) -> None:
        self.resource_check("before_enhanced_b0_step", self.counters.attempted_policy_steps + 1)
        self.counters.require_step_capacity(kind="nontraining", lanes=32, maximum=self.budgets["nontraining_transitions"])
        self.counters.require_reset_capacity(automatic=True, lanes=32, maximum=self.budgets["automatic_resets"])
        self.counters.attempt_step()
    def after_automatic_reset(self, case: str, policy_step: int, reset_lanes: int) -> None:
        self.counters.reserve_reset(automatic=True, lanes=reset_lanes, maximum=self.budgets["automatic_resets"])
    def after_step(self, case: str, policy_step: int, reset_lanes: int) -> None:
        self.counters.complete_step(32, kind="nontraining")
        if reset_lanes:
            raise RuntimeError("automatic resets must settle at the reset boundary")
    def binding(self) -> dict[str, Callable[..., None]]:
        return {"before_reset":self.before_reset,"after_reset":self.after_reset,"before_step":self.before_step,"after_automatic_reset":self.after_automatic_reset,"after_step":self.after_step}

class ReadOnlyObserver:
    """Observe one lane batch after each physics step without retaining live objects."""
    def __init__(self, writer: Callable[[Mapping[str,Any]],None], state_probe: Callable[[],Mapping[str,Any]]):
        self.writer=writer; self.state_probe=state_probe; self.calls=0; self.rows=0
    def __call__(self, batch: Any) -> None:
        before=copy.deepcopy(dict(self.state_probe()))
        if tuple(before)!=PROTECTED_STATE: raise RuntimeError("read-only probe does not cover the exact protected state")
        before_hash=canonical_sha256(before)
        if not isinstance(batch,(list,tuple)) or len(batch)!=32: raise ValueError("post-physics callback must provide exactly 32 lane rows")
        detached=copy.deepcopy(batch)
        for row in detached: self.writer(row)
        after=dict(self.state_probe())
        if canonical_sha256(after)!=before_hash: raise RuntimeError("observer changed protected live state")
        self.calls+=1; self.rows+=len(detached)
    def finalize(self):
        if self.calls*32!=PHYSICS_ROWS or self.rows!=PHYSICS_ROWS: raise RuntimeError("post-physics callback coverage is not exact")
        return {"callback_calls":self.calls,"physics_rows":self.rows,"write_set":[]}

def attach_supported_post_physics(environment: Any, observer: ReadOnlyObserver) -> Any:
    """Bind only to the explicit supported API; absent/ambiguous hooks fail closed."""
    register=getattr(environment,"register_read_only_post_physics_observer",None)
    if not callable(register): raise RuntimeError("formal backend lacks supported read-only post-physics observer boundary")
    receipt=register(observer, ordering="after_physics_before_reset", exactly_once=True, lanes=32)
    expected={"boundary":"after_physics_before_reset","ordering_unchanged":True,"read_only":True,"exactly_once":True,"lanes":32}
    if not isinstance(receipt,dict) or receipt!=expected: raise RuntimeError("post-physics observer receipt is not exact")
    return receipt


class FormalEnhancedB0EnvironmentAdapter:
    """Production binding to the environment-owned supported diagnostic protocol.

    The protocol is obtained once and must expose exact register/probe/run callables;
    there is deliberately no step monkeypatch or fallback around the reset boundary.
    """
    def __init__(self, environment: Any):
        factory=getattr(environment,"enhanced_b0_read_only_protocol",None)
        if not callable(factory): raise RuntimeError("formal environment does not expose enhanced B0 protocol")
        protocol=factory()
        if not isinstance(protocol,dict) or set(protocol)!={"register","state_probe","accounting","run","identity"} or protocol["identity"]!="post-physics-before-reset/v1":
            raise RuntimeError("enhanced B0 protocol identity/schema mismatch")
        if any(not callable(protocol[name]) for name in ("register","state_probe","accounting","run")):
            raise RuntimeError("enhanced B0 protocol operations are not callable")
        self._environment=environment; self._protocol=protocol; self._registered=False; self._accounted=False

    def register_read_only_post_physics_observer(self, callback: Callable, **binding: Any) -> dict[str,Any]:
        if self._registered: raise RuntimeError("enhanced B0 observer may be registered exactly once")
        expected={"ordering":"after_physics_before_reset","exactly_once":True,"lanes":32}
        if binding!=expected: raise RuntimeError("enhanced B0 observer binding changed")
        receipt=self._protocol["register"](callback,**binding)
        self._registered=True
        return receipt

    def configure_execution_accounting(self, accounting: EnhancedB0ExecutionAccounting) -> None:
        if not isinstance(accounting, EnhancedB0ExecutionAccounting):
            raise TypeError("exact enhanced B0 execution accounting is required")
        if self._accounted: raise RuntimeError("enhanced B0 execution accounting may be bound once")
        self._protocol["accounting"](accounting.binding()); self._accounted=True

    def read_only_diagnostic_state_fingerprints(self) -> Mapping[str,Any]:
        value=self._protocol["state_probe"]()
        if not isinstance(value,Mapping) or tuple(value)!=PROTECTED_STATE:
            raise RuntimeError("formal protocol protected-state fingerprint schema mismatch")
        return value

    def run_frozen_enhanced_b0_zts(self, **parameters: Any) -> None:
        expected={"seed":20260909,"cases":("Z","T","S"),"target_velocity_b_m_s":(0.0,0.0,0.0),"retries":0}
        if parameters!=expected or not self._registered or not self._accounted: raise RuntimeError("frozen enhanced B0 invocation changed or observer/accounting absent")
        report=self._protocol["run"](**parameters)
        if report!={"physics_callback_batches":12600,"physics_rows":PHYSICS_ROWS,"control_physics_reset_observation_order":"UNCHANGED"}:
            raise RuntimeError("formal enhanced B0 execution report is incomplete")
