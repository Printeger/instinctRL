"""Independent raw-evidence causal discriminator for enhanced B0."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping
from .b0_enhanced import (EvidenceError, TraceValidator, WRENCH_CLASSES,
    _add, _norm, _scale, _sub, _vec, close_wrench, infer_momentum_wrench, reconstruct_com, rotor_wrench)

HYPOTHESES=("com_application_point","fixed_force","fixed_torque","gravity_mass","rotor_mixer_frame","drag_aero","contact_joint_constraint","reset_state","axis_relationship")
OUTCOMES=("ROOT_CAUSE_ESTABLISHED_FIX_PREPARED","ROOT_CAUSE_ESTABLISHED_NO_SAFE_FIX","DIAGNOSIS_INCONCLUSIVE")

def _finite_vector(value: Any, name: str) -> tuple[float, float, float]:
    return _vec(value, name)

def derive_mechanics(previous: Mapping[str, Any] | None, row: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute mechanics from primitive row evidence; never consume claimed analysis."""
    system = reconstruct_com(row["bodies"])
    rotor = rotor_wrench(row["rotors"], system["combined_com_world_m"])
    response = row["response"]
    result = {
        "total_mass_kg": system["total_mass_kg"],
        "combined_com_world_m": system["combined_com_world_m"],
        "linear_momentum_kg_m_s": system["linear_momentum_kg_m_s"],
        "angular_momentum_at_com_kg_m2_s": system["angular_momentum_at_com_kg_m2_s"],
        "rotor_force_n": rotor["force_n"],
        "rotor_torque_at_com_nm": rotor["torque_at_com_nm"],
        "rotor_arms_world_m": [tuple(a-b for a,b in zip(_finite_vector(r["application_point_world_m"], "rotor point"), system["combined_com_world_m"])) for r in row["rotors"]],
        "position_world_m": _finite_vector(response["position_world_m"], "position response"),
        "velocity_world_m_s": _finite_vector(response["velocity_world_m_s"], "velocity response"),
        "acceleration_world_m_s2": _finite_vector(response["acceleration_world_m_s2"], "acceleration response"),
        "reset": bool(row["reset"]["mask"]),
    }
    if previous is None:
        result.update({"momentum_wrench": None, "known_residual_wrench": None, "force_acceleration_error_n": None})
        return result
    previous_system = reconstruct_com(previous["bodies"])
    momentum = infer_momentum_wrench(previous_system, system, row["momentum"]["dt_s"])
    closure = close_wrench(momentum, row["wrenches"])
    supplied = row["momentum"]["inferred_wrench"]
    if _norm(_sub(_finite_vector(supplied["force_n"], "supplied momentum force"), momentum["force_n"])) > 1e-8 or _norm(_sub(_finite_vector(supplied["torque_at_com_nm"], "supplied momentum torque"), momentum["torque_at_com_nm"])) > 1e-8:
        raise EvidenceError("independent momentum derivation disagrees with raw row")
    measured_force = _scale(result["acceleration_world_m_s2"], system["total_mass_kg"])
    result.update({
        "momentum_wrench": momentum,
        "known_residual_wrench": closure,
        "force_acceleration_error_n": _norm(_sub(momentum["force_n"], measured_force)),
    })
    return result

def _mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(x) for x in values): raise EvidenceError("analysis metric is absent/non-finite")
    return sum(values) / len(values)

def _stable_score(vectors: list[tuple[float, float, float]]) -> float | None:
    if len(vectors) < 3: return None
    mean=tuple(sum(v[i] for v in vectors)/len(vectors) for i in range(3))
    magnitude=_norm(mean)
    if magnitude <= 1e-9: return 0.0
    scatter=_mean([_norm(_sub(v,mean)) for v in vectors])
    return max(0.0, min(1.0, 1.0-scatter/magnitude))

def discriminate_signatures(*, residual_forces, residual_torques, rotor_torques,
                            class_magnitudes, unavailable, reset_rows):
    """Score mutually testable signatures derived from raw mechanics, not labels."""
    force_stability=_stable_score(residual_forces); torque_stability=_stable_score(residual_torques)
    mean_force=tuple(_mean([v[i] for v in residual_forces]) for i in range(3))
    mean_torque=tuple(_mean([v[i] for v in residual_torques]) for i in range(3))
    force_norm=_norm(mean_force); torque_norm=_norm(mean_torque)
    horizontal=math.hypot(mean_force[0],mean_force[1]); vertical=abs(mean_force[2])
    def activity(name):
        if name in unavailable or not class_magnitudes[name]: return None
        return 1.0 if _mean(class_magnitudes[name])>1e-8 else 0.0
    rotor_activity=activity("rotor"); gravity_activity=activity("gravity")
    drag_activity=activity("drag_aero"); contact_activity=activity("contact"); joint_activity=activity("joint_constraint")
    com_score=None if rotor_activity is None else (_stable_score(rotor_torques) if torque_norm>1e-8 else 0.0)
    return {
        "com_application_point":com_score,
        "fixed_force":0.0 if force_norm<=1e-8 else force_stability,
        "fixed_torque":0.0 if torque_norm<=1e-8 else torque_stability,
        "gravity_mass":None if gravity_activity is None else (force_stability if vertical>5*horizontal and vertical>1e-8 else 0.0),
        "rotor_mixer_frame":None if rotor_activity is None else (force_stability if horizontal>1e-8 and rotor_activity else 0.0),
        "drag_aero":drag_activity,
        "contact_joint_constraint":None if contact_activity is None or joint_activity is None else max(contact_activity,joint_activity),
        "reset_state":1.0 if reset_rows else 0.0,
        "axis_relationship":force_stability if horizontal>1e-8 and vertical>1e-8 else 0.0,
    }

def analyze_raw(rows: Iterable[Mapping[str,Any]]) -> dict[str,Any]:
    validator=TraceValidator(); samples=[]; previous={}
    unavailable=set(); class_magnitudes={name:[] for name in WRENCH_CLASSES}
    for row in rows:
        validator.append(row)
        key=(row["case"],row["lane"]); sample=derive_mechanics(previous.get(key),row); previous[key]=row; samples.append(sample)
        for name,value in row["wrenches"].items():
            if value.get("availability") != "AVAILABLE": unavailable.add(name)
            else: class_magnitudes[name].append(_norm(_finite_vector(value["force_n"], name+" force"))+_norm(_finite_vector(value["torque_at_com_nm"], name+" torque")))
    coverage=validator.finalize()
    adjacent=[s for s in samples if s["momentum_wrench"] is not None]
    if not adjacent: raise EvidenceError("no adjacent momentum samples")
    force_errors=[s["force_acceleration_error_n"] for s in adjacent]
    residual_forces=[s["known_residual_wrench"]["residual_force_n"] for s in adjacent]
    residual_torques=[s["known_residual_wrench"]["residual_torque_at_com_nm"] for s in adjacent]
    mechanics={
        "total_mass_kg":samples[0]["total_mass_kg"],
        "mean_force_acceleration_error_n":_mean(force_errors),
        "max_force_acceleration_error_n":max(force_errors),
        "mean_residual_force_n":tuple(_mean([v[i] for v in residual_forces]) for i in range(3)),
        "mean_residual_torque_nm":tuple(_mean([v[i] for v in residual_torques]) for i in range(3)),
        "position_start_m":samples[0]["position_world_m"],"position_end_m":samples[-1]["position_world_m"],
        "velocity_start_m_s":samples[0]["velocity_world_m_s"],"velocity_end_m_s":samples[-1]["velocity_world_m_s"],
        "reset_rows":sum(s["reset"] for s in samples),
    }
    hypotheses=discriminate_signatures(residual_forces=residual_forces,residual_torques=residual_torques,
        rotor_torques=[s["rotor_torque_at_com_nm"] for s in adjacent],class_magnitudes=class_magnitudes,
        unavailable=unavailable,reset_rows=mechanics["reset_rows"])
    ranked=sorted(((v,k) for k,v in hypotheses.items() if v is not None),reverse=True)
    complete=not unavailable and all(v is not None for v in hypotheses.values())
    closed=mechanics["max_force_acceleration_error_n"] <= 1e-5
    discriminated=complete and len(ranked)>=2 and ranked[0][0]>=0.95 and ranked[0][0]-ranked[1][0]>=0.20
    outcome="ROOT_CAUSE_ESTABLISHED_NO_SAFE_FIX" if closed and discriminated else "DIAGNOSIS_INCONCLUSIVE"
    return {"schema_version":"instinctrl-enhanced-b0-causal-analysis/1","outcome":outcome,"coverage":coverage,"derived_mechanics":mechanics,"mechanically_closed":closed,"hypothesis_scores":hypotheses,"unavailable_wrench_classes":sorted(unavailable),"winner":ranked[0][1] if outcome!="DIAGNOSIS_INCONCLUSIVE" else None,"axis_relationship":{"x_y_share_z_mass_cause":None},"repair":None}

def analyze_jsonl(path:Path):
    path=Path(path)
    if not path.is_file() or path.is_symlink(): raise EvidenceError("raw evidence must be an exact regular file")
    def rows():
        with path.open(encoding="utf-8") as stream:
            for number,line in enumerate(stream,1):
                try: yield json.loads(line)
                except Exception as error: raise EvidenceError(f"invalid JSON row {number}") from error
    return analyze_raw(rows())
