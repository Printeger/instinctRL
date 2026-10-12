"""Fail-closed first-update diagnostics for the bounded P0/PLR pilot."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import hashlib
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from training_workloads.scale_003.persistence import atomic_json_replace
from .pilot_contract import probability_distribution_gate

SAMPLES = 512
MINIBATCHES = 8
FLOAT32_ATOL = 1.0e-6
FLOAT32_RTOL = 1.0e-5
RETAIN_FULL_INPUT_UPDATES = (1, 2)


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().clone()
    if isinstance(value, Mapping): return {key:_clone(item) for key,item in value.items()}
    if isinstance(value, tuple): return tuple(_clone(item) for item in value)
    if isinstance(value, list): return [_clone(item) for item in value]
    return deepcopy(value)


def _digest(value: Any) -> str:
    digest=hashlib.sha256()
    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor=item.detach().cpu().contiguous(); digest.update(str(tensor.dtype).encode()); digest.update(repr(tuple(tensor.shape)).encode()); digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes()); return
        if isinstance(item, Mapping):
            for key in sorted(item): digest.update(str(key).encode()); add(item[key])
            return
        if isinstance(item,(list,tuple)):
            for child in item: add(child)
            return
        digest.update(repr(item).encode())
    add(value); return digest.hexdigest()


def _rng_snapshot() -> tuple[Any,...]:
    return (torch.get_rng_state().clone(), random.getstate(), np.random.get_state(),
            [state.clone() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])


def _rng_equal(left: tuple[Any,...], right: tuple[Any,...]) -> bool:
    numpy_equal=(left[2][0]==right[2][0] and np.array_equal(left[2][1],right[2][1]) and left[2][2:]==right[2][2:])
    return torch.equal(left[0],right[0]) and left[1]==right[1] and numpy_equal and len(left[3])==len(right[3]) and all(torch.equal(a,b) for a,b in zip(left[3],right[3]))


def _restore_rng(state: tuple[Any,...]) -> None:
    torch.set_rng_state(state[0]); random.setstate(state[1]); np.random.set_state(state[2])
    if state[3]: torch.cuda.set_rng_state_all(state[3])


def _state(trainer: Any) -> dict[str,Any]:
    return {"actor":_clone(trainer.actor.state_dict()),"critic":_clone(trainer.critic.state_dict()),
            "actor_optimizer":_clone(trainer.actor_optimizer.state_dict()),"critic_optimizer":_clone(trainer.critic_optimizer.state_dict())}


def _finite(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all().item()): raise ValueError(f"{name} must be a finite tensor")
    return value


def _batch(batch: Any) -> int:
    size=int(batch.actions.shape[0])
    if size != SAMPLES: raise ValueError("pilot diagnostic requires exactly 512 frozen samples")
    if batch.actions.ndim != 2 or batch.actions.shape[1] != 4: raise ValueError("actions must be [512,4]; broadcasting is forbidden")
    if batch.old_log_probabilities.shape != (SAMPLES,1): raise ValueError("old log probabilities must be [512,1]; broadcasting is forbidden")
    if getattr(batch, "old_alpha", None) is None or getattr(batch, "old_beta", None) is None:
        raise ValueError("rollout-time Beta concentrations are required")
    if batch.old_alpha.shape != (SAMPLES,4) or batch.old_beta.shape != (SAMPLES,4):
        raise ValueError("old Beta concentrations must be [512,4]; broadcasting is forbidden")
    if batch.advantages.shape != (SAMPLES,1) or batch.returns.shape != (SAMPLES,1): raise ValueError("advantage/return must be [512,1]")
    if any(value.shape[0] != SAMPLES for value in batch.policy.values()) or batch.privileged.shape[0] != SAMPLES: raise ValueError("actor/critic inputs must identify the same 512 samples")
    return size


def beta_kl(old_alpha: torch.Tensor, old_beta: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    for name,value in (("old_alpha",old_alpha),("old_beta",old_beta),("alpha",alpha),("beta",beta)): _finite(name,value)
    if old_alpha.shape != old_beta.shape or old_alpha.shape != alpha.shape or old_alpha.shape != beta.shape: raise ValueError("Beta tensors must have identical shapes")
    if bool((torch.stack((old_alpha,old_beta,alpha,beta)) <= 0).any().item()): raise ValueError("Beta concentrations must be positive")
    old_sum=old_alpha+old_beta; new_sum=alpha+beta
    return (torch.lgamma(old_sum)-torch.lgamma(old_alpha)-torch.lgamma(old_beta)
            -torch.lgamma(new_sum)+torch.lgamma(alpha)+torch.lgamma(beta)
            +(old_alpha-alpha)*torch.digamma(old_alpha)+(old_beta-beta)*torch.digamma(old_beta)
            +(alpha+beta-old_sum)*torch.digamma(old_sum)).sum(-1,keepdim=True)


class FirstUpdateDiagnostic:
    """Owns the fixed 512-sample gate/probe without sampling policy actions."""
    def __init__(self, output: Path, *, enabled: bool) -> None:
        self.output=output; self.enabled=bool(enabled); self._probe=None; self._rows=[]; self._handles=[]; self._gradient_handles=[]; self._unclipped_gradients={}; self._before_parameters=None; self._step_count=0

    def _evaluate(self, trainer: Any, batch: Any, boundary: str) -> dict[str,Any]:
        rng=_rng_snapshot()
        try:
            with torch.no_grad():
                output=trainer.actor(self._probe["policy"]); alpha=_finite("alpha",output.alpha); beta=_finite("beta",output.beta)
                logp=_finite("logp",trainer.actor.log_probability_from_output(output,self._probe["actions"]))
                value=_finite("value",trainer.critic(self._probe["privileged"]))
                mean=alpha/(alpha+beta); std=torch.sqrt(alpha*beta/((alpha+beta).square()*(alpha+beta+1)))
                if logp.shape != (SAMPLES,1) or mean.shape != (SAMPLES,4) or value.shape != (SAMPLES,1): raise ValueError("probe output shape permits broadcasting")
                ratio=torch.exp(logp-self._probe["old_log_probabilities"])
                # Use the product decoder and its bounds, including the vertical head.
                from instinctlab.control.governor import decode_governor
                issued_correction=decode_governor(self._probe["actions"],self._probe["policy"]["state_vec"]).v_corr_b
                mean_correction=decode_governor(mean,self._probe["policy"]["state_vec"]).v_corr_b
                row={"boundary":boundary,"indices":self._probe["indices"],"analytic_beta_kl":beta_kl(self._probe["old_alpha"],self._probe["old_beta"],alpha,beta),
                     "legacy_ppo_approximate_kl":self._probe["old_log_probabilities"]-logp,"log_probability":logp,"ratio":ratio,
                     "action":self._probe["actions"],"action_mean":mean,"action_std":std,
                     "issued_v_corr_b":issued_correction,"mean_v_corr_b":mean_correction,
                     "sample_minus_mean_action":self._probe["actions"]-mean,
                     "advantage":self._probe["advantages"],"return":self._probe["returns"],"value":value}
                return _clone(row)
        finally:
            if not _rng_equal(rng,_rng_snapshot()): _restore_rng(rng); raise RuntimeError("diagnostic evaluation consumed policy RNG")

    @contextmanager
    def capture_rollout_distribution(self, trainer: Any):
        """Capture concentrations at each real action-sampling call."""
        if not self.enabled:
            yield
            return
        actor=trainer.actor; original=actor.distribution; captured=[]; rng=_rng_snapshot()
        def distribution(policy):
            result=original(policy)
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError("actor distribution contract changed")
            output=result[1]
            captured.append((_clone(_finite("rollout old_alpha",output.alpha)),
                             _clone(_finite("rollout old_beta",output.beta))))
            return result
        actor.distribution=distribution
        try:
            yield
        finally:
            actor.distribution=original
            self._rollout_distributions=captured
            # Sampling itself advances policy RNG; only the observer must add no draws.
            # Therefore no equality assertion is made against the pre-rollout state.

    def bind_rollout_distribution(self, batch: Any) -> None:
        """Bind sampling-time concentrations to the assembled PPO batch."""
        if not self.enabled: return
        captured=getattr(self,"_rollout_distributions",None)
        if not captured: raise RuntimeError("no rollout-time Beta distributions captured")
        alpha=torch.cat([item[0] for item in captured],dim=0)
        beta=torch.cat([item[1] for item in captured],dim=0)
        if alpha.shape != (SAMPLES,4) or beta.shape != (SAMPLES,4):
            raise RuntimeError("rollout-time Beta distribution coverage is not [512,4]")
        object.__setattr__(batch,"old_alpha",alpha)
        object.__setattr__(batch,"old_beta",beta)
        self._rollout_distributions=None

    def before_update(self, trainer: Any, batch: Any) -> None:
        if not self.enabled or trainer.update_counter != 0: return
        _batch(batch); before=_state(trainer); rng=_rng_snapshot()
        try:
            with torch.no_grad():
                old_output=trainer.actor(batch.policy); recomputed=trainer.actor.log_probability_from_output(old_output,batch.actions)
            if recomputed.shape != (SAMPLES,1) or old_output.alpha.shape != (SAMPLES,4) or old_output.beta.shape != (SAMPLES,4):
                raise RuntimeError("pre-update probability/distribution gate failed")
            try:
                probability_distribution_gate(batch.old_log_probabilities.detach().cpu().reshape(-1).tolist(),recomputed.detach().cpu().reshape(-1).tolist(),
                    batch.old_alpha.detach().cpu().reshape(-1).tolist(),old_output.alpha.detach().cpu().reshape(-1).tolist(),
                    batch.old_beta.detach().cpu().reshape(-1).tolist(),old_output.beta.detach().cpu().reshape(-1).tolist(),atol=FLOAT32_ATOL,rtol=FLOAT32_RTOL)
            except (ValueError,RuntimeError) as error:
                raise RuntimeError("pre-update probability/distribution gate failed") from error
            self._probe={"policy":_clone(batch.policy),"privileged":_clone(batch.privileged),"actions":_clone(batch.actions),
                         "old_log_probabilities":_clone(batch.old_log_probabilities),"old_alpha":_clone(batch.old_alpha),"old_beta":_clone(batch.old_beta),
                         "advantages":_clone(batch.advantages),"returns":_clone(batch.returns),"indices":torch.arange(SAMPLES,dtype=torch.int64)}
            self._before_parameters={name:value.detach().clone() for name,value in trainer.actor.named_parameters()}
            self._rows=[self._evaluate(trainer,batch,"before_minibatch_1")]
            def capture(name):
                def hook(gradient): self._unclipped_gradients[name]=gradient.detach().clone()
                return hook
            self._gradient_handles=[parameter.register_hook(capture(name)) for name,parameter in trainer.actor.named_parameters()]
            def pre_hook(optimizer,args,kwargs):
                self._step_count += 1; ordinal=self._step_count
                if ordinal>MINIBATCHES: raise RuntimeError("actor optimizer exceeded eight pilot minibatches")
                row=self._evaluate(trainer,batch,f"before_minibatch_{ordinal}") if ordinal>1 else self._rows[0]
                row["unclipped_gradients"]=_clone(self._unclipped_gradients)
                row["clipped_gradients"]={name:parameter.grad.detach().clone() for name,parameter in trainer.actor.named_parameters() if parameter.grad is not None}
                row["clipped_gradient_norm"]=float(torch.sqrt(sum((p.grad.detach().square().sum() for p in trainer.actor.parameters() if p.grad is not None),torch.tensor(0.,device=batch.actions.device))).item())
                row["parameter_delta_before"]=float(torch.sqrt(sum(((p.detach()-self._before_parameters[n]).square().sum() for n,p in trainer.actor.named_parameters()),torch.tensor(0.,device=batch.actions.device))).item())
                if ordinal>1: self._rows.append(row)
            def post_hook(optimizer,args,kwargs):
                ordinal=self._step_count; self._rows[-1]["after"]=self._evaluate(trainer,batch,f"after_minibatch_{ordinal}")
                self._rows[-1]["parameter_deltas"]={name:(parameter.detach()-self._before_parameters[name]).clone() for name,parameter in trainer.actor.named_parameters()}
                self._rows[-1]["parameter_delta_after"]=float(torch.sqrt(sum(((p.detach()-self._before_parameters[n]).square().sum() for n,p in trainer.actor.named_parameters()),torch.tensor(0.,device=batch.actions.device))).item())
                self._unclipped_gradients={}
            self._handles=[trainer.actor_optimizer.register_step_pre_hook(pre_hook),trainer.actor_optimizer.register_step_post_hook(post_hook)]
        except BaseException:
            self.detach_hooks()
            if _digest(before) != _digest(_state(trainer)): raise RuntimeError("gate failure changed model or optimizer state")
            if not _rng_equal(rng,_rng_snapshot()): _restore_rng(rng); raise RuntimeError("gate failure consumed policy RNG")
            raise
        if not _rng_equal(rng,_rng_snapshot()): _restore_rng(rng); raise RuntimeError("pre-update gate consumed policy RNG")

    def after_update(self, trainer: Any, batch: Any, result: Mapping[str,Any]) -> None:
        if not self.enabled or self._probe is None: return
        self.detach_hooks()
        if len(self._rows) != MINIBATCHES: raise RuntimeError("pilot first update did not execute exactly eight actor minibatches")
        records=result.get("records",[])
        if len(records) != MINIBATCHES: raise RuntimeError("pilot result did not report eight minibatches")
        for row,record in zip(self._rows,records,strict=True): row["unclipped_gradient_norm"]=record["actor_pre_clip_gradient_norm"]
        self.output.mkdir(parents=True,exist_ok=True)
        torch.save({"schema_version":"instinctrl-pilot-first-update-probe/2","sample_count":SAMPLES,
                    "correction_heads":[1,2,3],"correction_frame":"body","correction_units":"m/s",
                    "sample_minus_mean_action_units":"dimensionless","minibatches":self._rows},self.output/"first_update_probe.pt")
        atomic_json_replace(self.output,"first_update_gate.json",{"schema_version":"instinctrl-pilot-first-update-gate/1","status":"PASS","sample_count":SAMPLES,
            "float32_atol":FLOAT32_ATOL,"float32_rtol":FLOAT32_RTOL,"policy_rng_consumed":False,"minibatches":MINIBATCHES})
        self._probe=None

    def detach_hooks(self) -> None:
        for handle in self._handles+self._gradient_handles: handle.remove()
        self._handles=[]; self._gradient_handles=[]

    def abort(self, error: BaseException) -> None:
        """Release observers and preserve partial diagnostic evidence on loop failure."""
        if not self.enabled: return
        self.detach_hooks()
        self.output.mkdir(parents=True,exist_ok=True)
        torch.save({"schema_version":"instinctrl-pilot-first-update-probe/2","status":"PARTIAL_FAILURE",
                    "probe":self._probe,"minibatches":self._rows},self.output/"first_update_partial_probe.pt")
        atomic_json_replace(self.output,"first_update_failure.json",{
            "schema_version":"instinctrl-pilot-first-update-failure/1","status":"FAILED",
            "error_type":type(error).__name__,"error":str(error),"observed_actor_steps":self._step_count})

    def retain_rollout(self, update: int, batch: Any, trainer: Any) -> None:
        if not self.enabled or update not in RETAIN_FULL_INPUT_UPDATES: return
        _batch(batch); self.output.mkdir(parents=True,exist_ok=True)
        torch.save({"schema_version":"instinctrl-pilot-retained-input/1","update":update,"policy":_clone(batch.policy),"privileged":_clone(batch.privileged),
                    "actions":_clone(batch.actions),"old_log_probabilities":_clone(batch.old_log_probabilities),"old_alpha":_clone(batch.old_alpha),"old_beta":_clone(batch.old_beta),
                    "advantages":_clone(batch.advantages),"returns":_clone(batch.returns)},
                   self.output/f"update_{update}_full_input.pt")


__all__=["FirstUpdateDiagnostic","beta_kl","SAMPLES","MINIBATCHES"]
