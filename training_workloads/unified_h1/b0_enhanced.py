"""Fail-closed, read-only enhanced-B0 mechanics evidence.

This module is deliberately CPU/stdlib-only.  The live adapter supplies plain
Python snapshots from the supported post-physics hook; this code never owns a
simulator, tensor, controller, reset, or random-number object.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EXPERIMENT_ID = "UNIFIED-H1-R5-READINESS-B0-BASELINE-20261012"
CASES = ("Z", "T", "S")
CASE_POLICY_STEPS = {"Z": 200, "T": 100, "S": 330}
LANES = 32
DECIMATION = 20
PHYSICS_ROWS = sum(CASE_POLICY_STEPS.values()) * LANES * DECIMATION
UNAVAILABLE = "UNAVAILABLE"
WRENCH_CLASSES = ("gravity", "rotor", "drag_aero", "contact", "joint_constraint", "other")
SCHEMA = "instinctrl-enhanced-b0-physics-row/2"

class EvidenceError(ValueError):
    """Raw evidence is absent, contradictory, or not mechanically closed."""

def _finite_number(x: Any, name: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        raise EvidenceError(f"{name} must be finite numeric evidence")
    return float(x)

def _vec(x: Any, name: str, n: int = 3) -> tuple[float, ...]:
    if not isinstance(x, (list, tuple)) or len(x) != n:
        raise EvidenceError(f"{name} must have shape [{n}]")
    return tuple(_finite_number(v, f"{name}[{i}]") for i, v in enumerate(x))

def _add(a, b): return tuple(x+y for x,y in zip(a,b))
def _sub(a, b): return tuple(x-y for x,y in zip(a,b))
def _scale(a, s): return tuple(x*s for x in a)
def _cross(a,b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
def _norm(a): return math.sqrt(sum(x*x for x in a))
def _mat_vec(matrix, vector):
    return tuple(sum(matrix[i][j]*vector[j] for j in range(3)) for i in range(3))
def _transpose(matrix): return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))
def _mat_mul(a,b): return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def _matrix3(value,name):
    if not isinstance(value,(list,tuple)) or len(value)!=3: raise EvidenceError(f"{name} must have shape [3,3]")
    return tuple(_vec(row,f"{name}[{i}]") for i,row in enumerate(value))
def quaternion_matrix(q: Any):
    q=_vec(q,"quaternion_wxyz",4); n=math.sqrt(sum(x*x for x in q))
    if not math.isclose(n,1.0,rel_tol=0.0,abs_tol=1e-5): raise EvidenceError("quaternion is not normalized")
    w,x,y,z=q
    return ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))

def quaternion_rotate(q: Any, v: Any) -> tuple[float,float,float]:
    q = _vec(q, "quaternion_wxyz", 4); v = _vec(v, "vector")
    n = math.sqrt(sum(x*x for x in q))
    if not math.isclose(n, 1.0, rel_tol=0.0, abs_tol=1e-5): raise EvidenceError("quaternion is not normalized")
    w,x,y,z=q; qv=(x,y,z)
    return _add(_add(_scale(v, w*w-sum(t*t for t in qv)), _scale(qv, 2*sum(a*b for a,b in zip(qv,v)))), _scale(_cross(qv,v),2*w))

def reconstruct_com(bodies: Any) -> dict[str, Any]:
    if not isinstance(bodies, list) or not bodies: raise EvidenceError("bodies must be a non-empty exact list")
    names=set(); total=0.0; weighted=(0.0,0.0,0.0); momentum=(0.0,0.0,0.0); parts=[]
    for i,b in enumerate(bodies):
        required={"name","mass_kg","local_com_m","local_com_quaternion_wxyz","inertia_at_com_body_kg_m2","world_position_m","world_quaternion_wxyz","world_linear_velocity_m_s","world_angular_velocity_rad_s"}
        if not isinstance(b,dict) or set(b)!=required: raise EvidenceError(f"body[{i}] schema mismatch")
        if not isinstance(b["name"],str) or not b["name"] or b["name"] in names: raise EvidenceError("body names must be unique")
        names.add(b["name"]); m=_finite_number(b["mass_kg"],"mass_kg")
        if m<=0: raise EvidenceError("body mass must be positive")
        pos=_vec(b["world_position_m"],"world_position_m"); local=_vec(b["local_com_m"],"local_com_m")
        quaternion_rotate(b["local_com_quaternion_wxyz"],(1.0,0.0,0.0))
        world_q=b["world_quaternion_wxyz"]; world_com=_add(pos,quaternion_rotate(world_q,local))
        linear=_vec(b["world_linear_velocity_m_s"],"world_linear_velocity_m_s")
        angular=_vec(b["world_angular_velocity_rad_s"],"world_angular_velocity_rad_s")
        com_velocity=_add(linear,_cross(angular,quaternion_rotate(world_q,local)))
        inertia=_matrix3(b["inertia_at_com_body_kg_m2"],"inertia_at_com_body_kg_m2")
        if any(not math.isclose(inertia[i][j],inertia[j][i],abs_tol=1e-10) for i in range(3) for j in range(3)) or any(inertia[i][i]<=0 for i in range(3)):
            raise EvidenceError("body inertia must be symmetric with positive diagonal")
        total+=m; weighted=_add(weighted,_scale(world_com,m)); momentum=_add(momentum,_scale(com_velocity,m))
        parts.append((m,world_com,com_velocity,angular,inertia,quaternion_matrix(world_q)))
    combined=_scale(weighted,1/total); angular_momentum=(0.0,0.0,0.0)
    for m,world_com,com_velocity,angular,inertia,rotation in parts:
        inertia_world=_mat_mul(_mat_mul(rotation,inertia),_transpose(rotation))
        angular_momentum=_add(angular_momentum,_add(_mat_vec(inertia_world,angular),_cross(_sub(world_com,combined),_scale(com_velocity,m))))
    return {"total_mass_kg":total,"combined_com_world_m":combined,"linear_momentum_kg_m_s":momentum,"angular_momentum_at_com_kg_m2_s":angular_momentum}

def rotor_wrench(rotors: Any, com: Any) -> dict[str, tuple[float,float,float]]:
    if not isinstance(rotors,list) or not rotors: raise EvidenceError("rotors must be non-empty")
    com=_vec(com,"combined_com_world_m"); force=(0.,0.,0.); torque=(0.,0.,0.); names=set()
    for i,r in enumerate(rotors):
        required={"name","command","target_throttle","actual_throttle","thrust_n","reaction_moment_nm","application_point_world_m","direction_world","reaction_body","reaction_direction_world"}
        if not isinstance(r,dict) or set(r)!=required: raise EvidenceError(f"rotor[{i}] schema mismatch")
        if r["name"] in names: raise EvidenceError("rotor names must be unique")
        names.add(r["name"])
        for key in ("command","target_throttle","actual_throttle","thrust_n","reaction_moment_nm"): _finite_number(r[key],key)
        direction=_vec(r["direction_world"],"direction_world")
        if not math.isclose(_norm(direction),1.0,rel_tol=0,abs_tol=1e-5): raise EvidenceError("rotor direction must be unit length")
        reaction_direction=_vec(r["reaction_direction_world"],"reaction_direction_world")
        if not isinstance(r["reaction_body"],str) or not r["reaction_body"] or not math.isclose(_norm(reaction_direction),1.0,rel_tol=0,abs_tol=1e-5):
            raise EvidenceError("reaction body/direction must be explicit and unit length")
        f=_scale(direction,float(r["thrust_n"])); arm=_sub(_vec(r["application_point_world_m"],"application_point_world_m"),com)
        t=_add(_cross(arm,f),_scale(reaction_direction,float(r["reaction_moment_nm"])))
        force=_add(force,f); torque=_add(torque,t)
    return {"force_n":force,"torque_at_com_nm":torque}

def _wrench(value: Any, name: str):
    if value == {"availability": UNAVAILABLE}: return UNAVAILABLE
    if not isinstance(value,dict) or set(value)!={"availability","force_n","torque_at_com_nm"} or value["availability"]!="AVAILABLE": raise EvidenceError(f"{name} wrench schema/availability mismatch")
    return {"force_n":_vec(value["force_n"],name+".force_n"),"torque_at_com_nm":_vec(value["torque_at_com_nm"],name+".torque_at_com_nm")}

def infer_momentum_wrench(previous: Mapping[str,Any], current: Mapping[str,Any], dt: Any) -> dict[str,tuple[float,float,float]]:
    dt=_finite_number(dt,"dt_s")
    if dt<=0: raise EvidenceError("dt_s must be positive")
    return {"force_n":_scale(_sub(_vec(current["linear_momentum_kg_m_s"],"linear momentum"),_vec(previous["linear_momentum_kg_m_s"],"linear momentum")),1/dt),
            "torque_at_com_nm":_scale(_sub(_vec(current["angular_momentum_at_com_kg_m2_s"],"angular momentum"),_vec(previous["angular_momentum_at_com_kg_m2_s"],"angular momentum")),1/dt)}

def close_wrench(momentum: Mapping[str,Any], wrenches: Mapping[str,Any]) -> dict[str,Any]:
    if not isinstance(wrenches,dict) or tuple(wrenches)!=WRENCH_CLASSES: raise EvidenceError("wrench availability schema/order mismatch")
    known_f=(0.,0.,0.); known_t=(0.,0.,0.); unavailable=[]
    normalized={}
    for name in WRENCH_CLASSES:
        value=_wrench(wrenches[name],name); normalized[name]=value
        if value==UNAVAILABLE: unavailable.append(name)
        else: known_f=_add(known_f,value["force_n"]); known_t=_add(known_t,value["torque_at_com_nm"])
    return {"known_force_n":known_f,"known_torque_at_com_nm":known_t,
            "residual_force_n":_sub(_vec(momentum["force_n"],"momentum.force_n"),known_f),
            "residual_torque_at_com_nm":_sub(_vec(momentum["torque_at_com_nm"],"momentum.torque_at_com_nm"),known_t),
            "unavailable_classes":unavailable,"causal_closure_available":not unavailable}

ROW_FIELDS=("schema_version","experiment_id","case","policy_step","physics_step","lane","episode_id","reset","root_state","system","bodies","controller","rotors","wrenches","momentum","response")

def validate_row(row: Any) -> dict[str,Any]:
    if not isinstance(row,dict) or tuple(row)!=ROW_FIELDS: raise EvidenceError("physics row exact schema mismatch")
    if row["schema_version"]!=SCHEMA or row["experiment_id"]!=EXPERIMENT_ID: raise EvidenceError("physics row identity mismatch")
    if row["case"] not in CASES: raise EvidenceError("unknown case")
    for k in ("policy_step","physics_step","lane","episode_id"):
        if type(row[k]) is not int or row[k]<0: raise EvidenceError(f"{k} must be a nonnegative integer")
    if row["policy_step"]>=CASE_POLICY_STEPS[row["case"]] or row["physics_step"]>=DECIMATION or row["lane"]>=LANES: raise EvidenceError("physics row coordinate out of range")
    reset=row["reset"]
    if not isinstance(reset,dict) or set(reset)!={"mask","reason","initial_boundary","last_pre_reset_state","post_reset_states"}: raise EvidenceError("reset schema mismatch")
    if type(reset["mask"]) is not bool or type(reset["initial_boundary"]) is not bool or not isinstance(reset["reason"],str): raise EvidenceError("reset mask/reason mismatch")
    if reset["mask"]:
        if not isinstance(reset["post_reset_states"],list) or len(reset["post_reset_states"])<1: raise EvidenceError("reset boundary lacks a real post-reset controller and rotor state")
        if reset["initial_boundary"]:
            if (reset["last_pre_reset_state"] is not None or row["case"] != "Z" or
                    row["policy_step"] != 0 or row["physics_step"] != 0 or
                    row["episode_id"] != 0 or reset["reason"] != "explicit_case_reset"):
                raise EvidenceError("initial reset boundary must be the exact Z lane-first explicit reset and truthfully omit pre-reset cache")
        elif not isinstance(reset["last_pre_reset_state"],dict):
            raise EvidenceError("later reset boundary lacks valid pre-reset state or exact reason")
        elif reset["reason"] == "explicit_case_reset":
            if (row["case"] not in {"T", "S"} or row["policy_step"] != 0 or
                    row["physics_step"] != 0 or row["episode_id"] != 0):
                raise EvidenceError("later explicit reset must be the exact T/S case-first boundary")
        elif reset["reason"] == "automatic_environment_reset":
            if row["episode_id"] == 0:
                raise EvidenceError("automatic reset cannot precede the explicit case boundary")
        else:
            raise EvidenceError("later reset boundary lacks valid pre-reset state or exact reason")
    elif (reset["reason"] != "" or reset["initial_boundary"] or
          reset["last_pre_reset_state"] is not None or reset["post_reset_states"] != []):
        raise EvidenceError("non-reset row carries reset boundary state or reason")
    system=reconstruct_com(row["bodies"])
    declared=row["system"]
    if not isinstance(declared,dict) or set(declared)!={"total_mass_kg","combined_com_world_m","linear_momentum_kg_m_s","angular_momentum_at_com_kg_m2_s"}: raise EvidenceError("system schema mismatch")
    if abs(_finite_number(declared["total_mass_kg"],"total_mass_kg")-system["total_mass_kg"])>1e-9 or _norm(_sub(_vec(declared["combined_com_world_m"],"combined COM"),system["combined_com_world_m"]))>1e-8: raise EvidenceError("articulated COM reconstruction mismatch")
    if _norm(_sub(_vec(declared["linear_momentum_kg_m_s"],"declared linear momentum"),system["linear_momentum_kg_m_s"]))>1e-8: raise EvidenceError("body-derived linear momentum mismatch")
    if _norm(_sub(_vec(declared["angular_momentum_at_com_kg_m2_s"],"declared angular momentum"),system["angular_momentum_at_com_kg_m2_s"]))>1e-8: raise EvidenceError("body-derived angular momentum mismatch")
    derived=rotor_wrench(row["rotors"],system["combined_com_world_m"])
    body_by_name={b["name"]:b for b in row["bodies"]}
    for rotor in row["rotors"]:
        reaction_body=body_by_name.get(rotor["reaction_body"])
        if reaction_body is None or _norm(_sub(_vec(rotor["reaction_direction_world"],"reaction direction"),quaternion_rotate(reaction_body["world_quaternion_wxyz"],(0.,0.,1.))))>1e-8:
            raise EvidenceError("reaction direction disagrees with declared body local-Z frame")
    wrenchmap=row["wrenches"]
    if not isinstance(wrenchmap,dict) or tuple(wrenchmap)!=WRENCH_CLASSES: raise EvidenceError("wrench class schema mismatch")
    for wrench_name in WRENCH_CLASSES: _wrench(wrenchmap[wrench_name], wrench_name)
    declared_rotor=_wrench(wrenchmap["rotor"],"rotor")
    if declared_rotor==UNAVAILABLE or _norm(_sub(declared_rotor["force_n"],derived["force_n"]))>1e-8 or _norm(_sub(declared_rotor["torque_at_com_nm"],derived["torque_at_com_nm"]))>1e-8: raise EvidenceError("rotor wrench derivation mismatch")
    exact = {
        "root_state": {"pose", "linear_velocity", "angular_velocity"},
        "controller": {"target_velocity_b_m_s", "velocity_error_b_m_s", "desired_rotation_w", "collective_n"},
        "momentum": {"dt_s", "inferred_wrench"},
        "response": {"position_world_m", "velocity_world_m_s", "acceleration_world_m_s2"},
    }
    for name, fields in exact.items():
        if not isinstance(row[name], dict) or set(row[name]) != fields:
            raise EvidenceError(f"{name} exact schema mismatch")
    _vec(row["root_state"]["pose"], "root_state.pose", 7)
    _vec(row["root_state"]["linear_velocity"], "root_state.linear_velocity")
    _vec(row["root_state"]["angular_velocity"], "root_state.angular_velocity")
    _vec(row["controller"]["target_velocity_b_m_s"], "controller.target_velocity")
    _vec(row["controller"]["velocity_error_b_m_s"], "controller.velocity_error")
    rotation = row["controller"]["desired_rotation_w"]
    if not isinstance(rotation, (list, tuple)) or len(rotation) != 3:
        raise EvidenceError("controller desired rotation must have shape [3,3]")
    for i, axis in enumerate(rotation): _vec(axis, f"controller.desired_rotation_w[{i}]")
    _finite_number(row["controller"]["collective_n"], "controller.collective_n")
    _finite_number(row["momentum"]["dt_s"], "momentum.dt_s")
    inferred = row["momentum"]["inferred_wrench"]
    if not isinstance(inferred, dict) or set(inferred) != {"force_n", "torque_at_com_nm"}:
        raise EvidenceError("momentum inferred wrench schema mismatch")
    _vec(inferred["force_n"], "momentum.inferred.force")
    _vec(inferred["torque_at_com_nm"], "momentum.inferred.torque")
    for field in ("position_world_m", "velocity_world_m_s", "acceleration_world_m_s2"):
        _vec(row["response"][field], f"response.{field}")
    if reset["mask"]:
        snapshots = [*reset["post_reset_states"]] if reset["initial_boundary"] else [reset["last_pre_reset_state"], *reset["post_reset_states"]]
        for snapshot in snapshots:
            if set(snapshot) != {"controller", "rotors"} or set(snapshot["controller"]) != exact["controller"] or not isinstance(snapshot["rotors"], list) or not snapshot["rotors"]:
                raise EvidenceError("reset boundary controller/rotor snapshot schema mismatch")
            rotor_wrench(snapshot["rotors"], system["combined_com_world_m"])
    return row

class TraceValidator:
    """Streaming exact-coverage validator; duplicates, gaps and reorder fail now."""
    def __init__(self):
        self.count=0
        self.episodes={(case, lane): 0 for case in CASES for lane in range(LANES)}
        self.reset_rows={(case, lane):0 for case in CASES for lane in range(LANES)}
        self.previous={}
        self.last_cache={}
    @staticmethod
    def coordinate(ordinal:int):
        lane=ordinal%LANES; q=ordinal//LANES; physics=q%DECIMATION; policy_global=q//DECIMATION
        for case in CASES:
            n=CASE_POLICY_STEPS[case]
            if policy_global<n: return case,policy_global,physics,lane
            policy_global-=n
        raise EvidenceError("row count exceeds exact horizon")
    def append(self,row:Any):
        validate_row(row); expected=self.coordinate(self.count); actual=(row["case"],row["policy_step"],row["physics_step"],row["lane"])
        if actual!=expected: raise EvidenceError(f"row continuity mismatch: expected {expected}, got {actual}")
        lane=row["lane"]; key=(row["case"],lane); reset=row["reset"]
        case_first=row["policy_step"] == 0 and row["physics_step"] == 0
        if case_first:
            if row["case"] == "Z":
                if not (reset["mask"] and reset["initial_boundary"] and reset["reason"] == "explicit_case_reset"):
                    raise EvidenceError("each lane must begin Z with the truthful initial reset boundary")
            elif not (reset["mask"] and not reset["initial_boundary"] and reset["reason"] == "explicit_case_reset"):
                raise EvidenceError("each lane must begin T/S with a later explicit reset boundary")
        if row["episode_id"]!=self.episodes[key]: raise EvidenceError("episode continuity mismatch")
        if reset["mask"] and not reset["initial_boundary"]:
            expected_cache=self.last_cache.get(lane)
            if expected_cache is None or reset["last_pre_reset_state"] != expected_cache:
                raise EvidenceError("later reset pre-cache does not match the last real lane cache")
        if key in self.previous and not reset["mask"]:
            inferred=infer_momentum_wrench(self.previous[key]["system"],row["system"],row["momentum"].get("dt_s"))
            supplied=row["momentum"].get("inferred_wrench")
            if not isinstance(supplied,dict) or any(_norm(_sub(_vec(supplied[k],k),inferred[k]))>1e-8 for k in inferred): raise EvidenceError("adjacent momentum residual mismatch")
            close_wrench(inferred,row["wrenches"])
        self.previous[key]=row
        self.last_cache[lane]={"controller":row["controller"],"rotors":row["rotors"]}
        if reset["mask"]:
            self.episodes[key]+=1; self.reset_rows[key]+=1
        self.count+=1
    def finalize(self):
        if self.count!=PHYSICS_ROWS: raise EvidenceError(f"expected {PHYSICS_ROWS} rows, observed {self.count}")
        missing=[key for key,count in self.reset_rows.items() if count < 1]
        if missing: raise EvidenceError("trace lacks observed reset/episode boundaries")
        return {"status":"VALID","physics_rows":self.count,"lanes":LANES,"decimation":DECIMATION,"cases":list(CASES),"reset_rows":sum(self.reset_rows.values())}

def canonical_sha256(value:Any)->str:
    return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()).hexdigest()
