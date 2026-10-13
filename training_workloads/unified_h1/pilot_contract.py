"""Standard-library validation for the paired-pilot immutable contract."""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Mapping, Sequence

SAMPLES=512
ACTION_DIM=4
CHECKPOINTS=("init0000","u1","u2","u8","u16","u32","u64")

def validate_shapes(*,policy_rows:Mapping[str,int],privileged_rows:int,action_shape:tuple[int,...],old_logp_shape:tuple[int,...],advantage_shape:tuple[int,...],return_shape:tuple[int,...])->None:
    if action_shape!=(SAMPLES,ACTION_DIM): raise ValueError("actions must be [512,4]; broadcasting is forbidden")
    if old_logp_shape!=(SAMPLES,1): raise ValueError("old log probabilities must be [512,1]; broadcasting is forbidden")
    if advantage_shape!=(SAMPLES,1) or return_shape!=(SAMPLES,1): raise ValueError("learning signals must be [512,1]")
    if privileged_rows!=SAMPLES or any(rows!=SAMPLES for rows in policy_rows.values()): raise ValueError("all inputs must identify the same 512 samples")

def probability_gate(old:Sequence[float],new:Sequence[float],*,atol:float=1e-6,rtol:float=1e-5)->None:
    if len(old)!=SAMPLES or len(new)!=SAMPLES: raise ValueError("probability gate requires 512 matched indices")
    for index,(left,right) in enumerate(zip(old,new,strict=True)):
        if not math.isfinite(left) or not math.isfinite(right) or abs(left-right)>atol+rtol*abs(right): raise RuntimeError(f"probability gate mismatch at frozen index {index}")

def probability_distribution_gate(old_logp:Sequence[float],new_logp:Sequence[float],old_alpha:Sequence[float],new_alpha:Sequence[float],old_beta:Sequence[float],new_beta:Sequence[float],*,atol:float=1e-6,rtol:float=1e-5)->None:
    probability_gate(old_logp,new_logp,atol=atol,rtol=rtol)
    expected=SAMPLES*ACTION_DIM
    for name,old,new in (("alpha",old_alpha,new_alpha),("beta",old_beta,new_beta)):
        if len(old)!=expected or len(new)!=expected: raise ValueError(f"{name} gate requires 512x4 matched concentrations")
        for index,(left,right) in enumerate(zip(old,new,strict=True)):
            if not math.isfinite(left) or not math.isfinite(right) or abs(left-right)>atol+rtol*abs(right):
                raise RuntimeError(f"distribution gate {name} mismatch at frozen index {index}")

def pilot_actor_learning_rate(group:str)->float:
    if group=="P0": return 3.0e-4
    if group in {"PLR", "C0", "C1"}: return 1.0e-4
    raise ValueError("pilot group must be P0, PLR, C0 or C1")

RESIDUAL_WEIGHTS = {"C0": 0.0, "C1": 0.05}
PILOT_OUTPUT_CAP_BYTES = 16 * 1024**3
PILOT_WORKLOAD_WRITE_CAP_BYTES = 13 * 1024**3

def pilot_residual_weight(group: str) -> float:
    pilot_actor_learning_rate(group)
    return RESIDUAL_WEIGHTS.get(group, 0.0)

def pilot_method_hashes(config, group: str | None):
    result = dict(config["method"]["hashes"])
    if group in RESIDUAL_WEIGHTS:
        import hashlib
        # Preserve checkpoint-h1-v2 metadata: the exact coefficient/group
        # receive an explicit SHA instead of relaxing its hash validation.
        binding = f"{group}:residual_neutrality_weight={pilot_residual_weight(group).hex()}"
        result["pilot_reward_binding"] = hashlib.sha256(binding.encode()).hexdigest()
    return result

def frozen_copy(value): return deepcopy(value)

def _digamma(x:float)->float:
    result=0.0
    while x<8.0: result-=1.0/x; x+=1.0
    inv=1.0/x; inv2=inv*inv
    return result+math.log(x)-0.5*inv-inv2*(1/12-inv2*(1/120-inv2/252))

def beta_kl_scalar(a0:float,b0:float,a1:float,b1:float)->float:
    if min(a0,b0,a1,b1)<=0 or not all(math.isfinite(x) for x in (a0,b0,a1,b1)): raise ValueError("Beta concentrations must be positive finite values")
    s0=a0+b0; s1=a1+b1
    return math.lgamma(s0)-math.lgamma(a0)-math.lgamma(b0)-math.lgamma(s1)+math.lgamma(a1)+math.lgamma(b1)+(a0-a1)*_digamma(a0)+(b0-b1)*_digamma(b0)+(s1-s0)*_digamma(s0)
