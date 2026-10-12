"""Fail-closed D7-R/B0 diagnostic writing and CPU analysis."""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch

CONDITIONS = ("D7-R", "B0")
CASES = ("Z", "T", "S")
CASE_STEPS = {"Z": 200, "T": 100, "S": 330}
LANES = (0, 1, 2, 3, 4, 5, 6, 7)
TRANSITIONS = 20_160
FIXED_B0_ACTION = (1.0, 0.5, 0.5, 0.5)
SCHEMA = "unified-h1-control-diagnostic.v2"
AGGREGATE_DISK_LIMIT_BYTES = 16 * 1024**3
LEDGER_SCHEMA = "unified-h1-control-diagnostic-disk-ledger.v1"
TIMING_SEMANTICS = "pre_step is captured before environment.step; post_step and reset/cache validity are returned or observed from that same step"


def validate_parameters(value: object) -> dict[str, Any]:
    required = {"condition", "experiment_id", "checkpoint", "checkpoint_sha256",
                "checkpoint_source_identity", "cases", "transitions"}
    if not isinstance(value, dict) or set(value) != required:
        raise PermissionError("control diagnostic parameter schema mismatch")
    if value["condition"] not in CONDITIONS or value["cases"] != list(CASES) or value["transitions"] != TRANSITIONS:
        raise PermissionError("control diagnostic condition/case/transition binding mismatch")
    if value["experiment_id"] != f"UNIFIED-H1-R4-{value['condition']}":
        raise PermissionError("control diagnostic experiment identity mismatch")
    checkpoint = Path(value["checkpoint"])
    if not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file():
        raise PermissionError("control diagnostic checkpoint is not an exact regular file")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != value["checkpoint_sha256"]:
        raise PermissionError("control diagnostic checkpoint hash changed")
    origin = value["checkpoint_source_identity"]
    keys = {"product_commit", "product_tree", "workload_commit", "workload_tree", "config_sha256"}
    if not isinstance(origin, dict) or set(origin) != keys:
        raise PermissionError("control diagnostic checkpoint source identity is incomplete")
    for key, item in origin.items():
        size = 64 if key == "config_sha256" else 40
        if not isinstance(item, str) or len(item) != size or any(c not in "0123456789abcdef" for c in item):
            raise PermissionError("control diagnostic checkpoint source identity is invalid")
    return dict(value)


def fixed_b0_action(batch: int, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if type(batch) is not int or batch <= 0: raise ValueError("batch must be a positive integer")
    return torch.tensor(FIXED_B0_ACTION, device=device, dtype=dtype).expand(batch, -1).clone()


def select_actor_input(policy: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if tuple(policy) != ("lidar_grid", "state_vec"): raise ValueError("diagnostic requires the exact actor input mapping")
    lidar, state = policy["lidar_grid"], policy["state_vec"]
    if lidar.shape != (32, 12, 59, 360) or state.shape != (32, 52): raise ValueError("diagnostic actor input shape mismatch")
    lanes = torch.tensor(LANES, device=lidar.device)
    return {"lanes": torch.tensor(LANES), "lidar_grid": lidar[lanes].detach().cpu().clone(), "state_vec": state[lanes].detach().cpu().clone()}


def cpu_snapshot(value: Any) -> Any:
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, Mapping): return {str(key): cpu_snapshot(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type): return {field.name: cpu_snapshot(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple): return tuple(cpu_snapshot(item) for item in value)
    if isinstance(value, list): return [cpu_snapshot(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)): return value
    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")


def _durable_replace(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


class AggregateDiskLedger:
    """Crash-safe shared reservations; failed writes remain charged fail-closed."""
    def __init__(self, path: Path, maximum_bytes: int = AGGREGATE_DISK_LIMIT_BYTES) -> None:
        self.path = Path(path); self.lock = self.path.with_suffix(self.path.suffix + ".lock")
        if type(maximum_bytes) is not int or maximum_bytes <= 0: raise ValueError("disk ceiling must be positive")
        self.maximum_bytes = maximum_bytes; self.path.parent.mkdir(parents=True, exist_ok=True)

    def reserve(self, *, condition: str, relative_path: str, byte_count: int) -> int:
        if condition not in CONDITIONS or type(byte_count) is not int or byte_count < 0: raise ValueError("invalid disk reservation")
        with self.lock.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                if self.path.is_symlink(): raise RuntimeError("diagnostic disk ledger is a symlink")
                value = json.loads(self.path.read_text())
            else:
                value = {"schema_version":LEDGER_SCHEMA,"maximum_bytes":self.maximum_bytes,"reserved_bytes":0,"reservations":[]}
            if value.get("schema_version") != LEDGER_SCHEMA or value.get("maximum_bytes") != self.maximum_bytes:
                raise RuntimeError("diagnostic disk ledger identity or ceiling mismatch")
            total = value.get("reserved_bytes")
            if type(total) is not int or total < 0 or total != sum(row["bytes"] for row in value.get("reservations", [])):
                raise RuntimeError("diagnostic disk ledger accounting is invalid")
            if total + byte_count > self.maximum_bytes: raise RuntimeError("aggregate 16 GiB diagnostic disk ceiling exhausted before write")
            value["reserved_bytes"] = total + byte_count
            value["reservations"].append({"condition":condition,"path":relative_path,"bytes":byte_count})
            _durable_replace(self.path, (json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode())
            return value["reserved_bytes"]


class DiagnosticChunkWriter:
    def __init__(self, root: Path, *, condition: str, experiment_id: str,
                 aggregate_ledger: Path | None = None, maximum_bytes: int = AGGREGATE_DISK_LIMIT_BYTES) -> None:
        if condition not in CONDITIONS or experiment_id != f"UNIFIED-H1-R4-{condition}": raise ValueError("writer identity mismatch")
        self.root=Path(root); self.condition=condition; self.experiment_id=experiment_id
        self.root.mkdir(parents=True,exist_ok=False); self.rows=[]
        ledger_path = aggregate_ledger or self.root.parent / ".d7_control_diagnostic_disk_ledger.json"
        self.ledger=AggregateDiskLedger(ledger_path, maximum_bytes)

    def _write(self, relative: str, data: bytes) -> int:
        target=self.root/relative; target.parent.mkdir(parents=True,exist_ok=True)
        total=self.ledger.reserve(condition=self.condition,relative_path=relative,byte_count=len(data))
        _durable_replace(target,data)
        return total

    def append(self, case: str, step: int, payload: Mapping[str, Any]) -> None:
        if case not in CASES or type(step) is not int or not 0<=step<CASE_STEPS[case]: raise ValueError("chunk case/step mismatch")
        expected=sum(CASE_STEPS[c] for c in CASES[:CASES.index(case)])+step
        if len(self.rows)!=expected: raise ValueError("chunks must be contiguous in Z/T/S order")
        relative=f"chunks/{case}/{step:04d}.pt"; buffer=io.BytesIO(); torch.save(cpu_snapshot(dict(payload)),buffer)
        data=buffer.getvalue(); cumulative=self._write(relative,data)
        self.rows.append({"case":case,"step":step,"path":relative,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest(),"aggregate_reserved_bytes":cumulative})

    def finalize(self) -> dict[str, Any]:
        if len(self.rows)!=sum(CASE_STEPS.values()): raise RuntimeError("diagnostic trace is incomplete")
        value={"schema_version":SCHEMA,"condition":self.condition,"experiment_id":self.experiment_id,
               "cases":list(CASES),"case_steps":CASE_STEPS,"policy_steps":len(self.rows),"transitions":len(self.rows)*32,
               "lanes":32,"selected_actor_input_lanes":list(LANES),"timing_semantics":TIMING_SEMANTICS,
               "aggregate_disk_limit_bytes":self.ledger.maximum_bytes,"chunks":self.rows}
        data=(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n").encode(); self._write("manifest.json",data)
        return value


def _tensor(value: Any, path: str, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(value,torch.Tensor): raise ValueError(f"missing tensor evidence: {path}")
    if shape is not None and tuple(value.shape)!=shape: raise ValueError(f"tensor shape mismatch: {path}")
    if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()): raise ValueError(f"non-finite tensor evidence: {path}")
    return value


def _lane_tree(value: Any, path: str) -> int:
    count=0
    if isinstance(value,torch.Tensor):
        _tensor(value,path)
        if value.ndim < 1 or value.shape[0] != 32: raise ValueError(f"32-lane shape mismatch: {path}")
        count=1
    elif isinstance(value,Mapping):
        for key,item in value.items(): count += _lane_tree(item,f"{path}.{key}")
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value): count += _lane_tree(item,f"{path}[{index}]")
    elif isinstance(value,float) and not math.isfinite(value): raise ValueError(f"non-finite scalar evidence: {path}")
    return count


def _exact_mapping(value: Any, path: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or tuple(value) != keys:
        raise ValueError(f"exact field identity mismatch: {path}")
    return value


def _tensor_schema(value: Mapping[str, Any], path: str,
                   schema: tuple[tuple[str, tuple[int, ...]], ...]) -> None:
    value = _exact_mapping(value, path, tuple(key for key, _ in schema))
    for key, shape in schema:
        _tensor(value[key], f"{path}.{key}", shape)


def _validate_control(value: Any) -> None:
    control = _exact_mapping(value, "pre_reset.control",
        ("alpha","v_corr_b","v_gov_b","v_final_b","v_final_w","anchor_metrics","ics_metrics"))
    for key, shape in (("alpha",(32,1)),("v_corr_b",(32,3)),("v_gov_b",(32,3)),
                       ("v_final_b",(32,3)),("v_final_w",(32,3))):
        _tensor(control[key], f"pre_reset.control.{key}", shape)
    _tensor_schema(control["anchor_metrics"], "pre_reset.control.anchor_metrics", (
        ("active",(32,1)),("reset_code",(32,1)),("usable_fraction",(32,1)),
        ("mean_absolute_error_m",(32,1)),("huber_loss",(32,1))))
    _tensor_schema(control["ics_metrics"], "pre_reset.control.ics_metrics", (
        ("beta",(32,1)),("active_beam_count",(32,1)),("minimum_clearance_m",(32,1)),
        ("worst_margin_m_s",(32,1)),("emergency",(32,1))))


def _validate_task(value: Any) -> Mapping[str, Any]:
    task = _exact_mapping(value, "pre_reset.task",
        ("root_state_w","actual_velocity_b","exact_clearance_m","collision","termination"))
    for key, shape in (("root_state_w",(32,13)),("actual_velocity_b",(32,3)),
                       ("exact_clearance_m",(32,1)),("collision",(32,1))):
        _tensor(task[key], f"pre_reset.task.{key}", shape)
    termination = _exact_mapping(task["termination"], "pre_reset.task.termination",
        ("terminated","truncated","done","reason_code","masks"))
    for key in ("terminated","truncated","done","reason_code"):
        _tensor(termination[key], f"pre_reset.task.termination.{key}", (32,1))
    _tensor_schema(termination["masks"], "pre_reset.task.termination.masks", (
        ("collision",(32,1)),("below_height",(32,1)),("above_height",(32,1)),("timeout",(32,1))))
    return task


def _validate_termination_consistency(task: Mapping[str, Any], outer_terminated: torch.Tensor,
                                      outer_truncated: torch.Tensor) -> None:
    termination = task["termination"]
    masks = termination["masks"]
    terminated = termination["terminated"]
    truncated = termination["truncated"]
    done = termination["done"]
    reason = termination["reason_code"]
    boolean_tensors = (terminated, truncated, done, *masks.values())
    if any(value.dtype != torch.bool for value in boolean_tensors) or reason.dtype != torch.long:
        raise ValueError("task termination dtype mismatch")
    if task["collision"].dtype != torch.bool or not torch.equal(task["collision"], masks["collision"]):
        raise ValueError("task collision/termination mask mismatch")

    expected_terminated = masks["collision"] | masks["below_height"] | masks["above_height"]
    expected_truncated = masks["timeout"]
    expected_done = expected_terminated | expected_truncated
    if not torch.equal(terminated, expected_terminated):
        raise ValueError("task terminated/mask semantics mismatch")
    if not torch.equal(truncated, expected_truncated):
        raise ValueError("task truncated/timeout semantics mismatch")
    if not torch.equal(done, expected_done):
        raise ValueError("task done semantics mismatch")
    if not torch.equal(terminated.reshape(-1), outer_terminated) or not torch.equal(truncated.reshape(-1), outer_truncated):
        raise ValueError("nested/outer termination consistency mismatch")

    mask_values = {key: value.reshape(-1).tolist() for key, value in masks.items()}
    expected_reason = []
    for lane in range(32):
        # Producer precedence is collision > below > above > timeout > none.
        expected_reason.append(4 if mask_values["collision"][lane] else
                               3 if mask_values["below_height"][lane] else
                               2 if mask_values["above_height"][lane] else
                               1 if mask_values["timeout"][lane] else 0)
    if reason.reshape(-1).tolist() != expected_reason:
        raise ValueError("task termination reason precedence mismatch")


def _validate_lee(value: Any) -> None:
    _tensor_schema(value, "lee", (("normalized_rotor_commands",(32,4)),("held_position_w_m",(32,3)),
        ("held_yaw_rad",(32,1)),("desired_rotation_w",(32,3,3)),("collective_thrust_n",(32,1))))


def _validate_wrench(value: Any) -> None:
    wrench = _exact_mapping(value, "wrench", ("rotor","forces_body_n","torques_body_nm"))
    _tensor_schema(wrench["rotor"], "wrench.rotor", (("normalized_commands",(32,4)),
        ("target_throttle",(32,4)),("throttle",(32,4)),("thrusts_n",(32,4)),
        ("reaction_moments_nm",(32,4))))
    _tensor(wrench["forces_body_n"], "wrench.forces_body_n", (32,5,3))
    _tensor(wrench["torques_body_nm"], "wrench.torques_body_nm", (32,5,3))


def analyze_trace(root: Path, *, condition: str, experiment_id: str) -> dict[str, Any]:
    root=Path(root); manifest_path=root/"manifest.json"
    if condition not in CONDITIONS or experiment_id != f"UNIFIED-H1-R4-{condition}": raise ValueError("analysis identity mismatch")
    if not manifest_path.is_file() or manifest_path.is_symlink(): raise ValueError("manifest evidence is missing")
    manifest=json.loads(manifest_path.read_text())
    expected={"schema_version":SCHEMA,"condition":condition,"experiment_id":experiment_id,"cases":list(CASES),
              "case_steps":CASE_STEPS,"policy_steps":630,"transitions":TRANSITIONS,"lanes":32,
              "selected_actor_input_lanes":list(LANES),"timing_semantics":TIMING_SEMANTICS,
              "aggregate_disk_limit_bytes":AGGREGATE_DISK_LIMIT_BYTES}
    for key,value in expected.items():
        if manifest.get(key)!=value: raise ValueError(f"manifest identity/coverage mismatch: {key}")
    rows=manifest.get("chunks")
    if not isinstance(rows,list) or len(rows)!=630: raise ValueError("manifest chunk coverage is incomplete")
    invalid_cache=0; fully_invalid_downstream_chunks=0
    for ordinal,row in enumerate(rows):
        case=next(c for c in CASES if ordinal < sum(CASE_STEPS[x] for x in CASES[:CASES.index(c)+1]))
        step=ordinal-sum(CASE_STEPS[x] for x in CASES[:CASES.index(case)])
        relative=f"chunks/{case}/{step:04d}.pt"
        if not isinstance(row,dict) or row.get("case")!=case or row.get("step")!=step or row.get("path")!=relative: raise ValueError("manifest Z/T/S order is not exact")
        path=root/relative
        if not path.is_file() or path.is_symlink(): raise ValueError("chunk evidence is missing")
        data=path.read_bytes()
        if row.get("bytes")!=len(data) or row.get("sha256")!=hashlib.sha256(data).hexdigest(): raise ValueError("chunk size/hash mismatch")
        payload=torch.load(io.BytesIO(data),map_location="cpu",weights_only=True)
        keys=("schema_version","condition","case","step","ordinal","timing_semantics","pre_step","post_step")
        if not isinstance(payload,dict) or tuple(payload)!=keys: raise ValueError("chunk evidence schema mismatch")
        identity=(payload["condition"],payload["case"],payload["step"],payload["ordinal"],payload["timing_semantics"])
        if payload["schema_version"]!="unified-h1-control-diagnostic.step.v2" or identity!=(condition,case,step,ordinal,TIMING_SEMANTICS):
            raise ValueError("chunk identity/time semantics mismatch")
        pre=_exact_mapping(payload["pre_step"],"pre_step",("actor_input","actor_alpha","actor_beta","actor_mean","issued_action","command_b","root_state_w"))
        post=_exact_mapping(payload["post_step"],"post_step",("reward","terminated","truncated","pre_reset","lee","wrench","downstream_cache_valid"))
        actor=_exact_mapping(pre["actor_input"],"actor_input",("lanes","lidar_grid","state_vec"))
        if _tensor(actor["lanes"],"actor_input.lanes",(8,)).tolist()!=list(LANES): raise ValueError("selected actor lane identity mismatch")
        _tensor(actor["lidar_grid"],"actor_input.lidar_grid",(8,12,59,360)); _tensor(actor["state_vec"],"actor_input.state_vec",(8,52))
        for key,shape in (("actor_alpha",(32,4)),("actor_beta",(32,4)),("actor_mean",(32,4)),("issued_action",(32,4)),("command_b",(32,3)),("root_state_w",(32,13))): _tensor(pre[key],key,shape)
        if condition=="B0" and not torch.equal(pre["issued_action"],fixed_b0_action(32,dtype=pre["issued_action"].dtype)): raise ValueError("B0 issued action is not exactly fixed")
        _tensor(post["reward"],"reward",(32,)); terminated=_tensor(post["terminated"],"terminated",(32,)); truncated=_tensor(post["truncated"],"truncated",(32,))
        valid=_tensor(post["downstream_cache_valid"],"downstream_cache_valid",(32,))
        reset=terminated|truncated
        if terminated.dtype!=torch.bool or truncated.dtype!=torch.bool or valid.dtype!=torch.bool or not torch.equal(valid,~reset): raise ValueError("reset/cache validity mask mismatch")
        pre_reset=_exact_mapping(post["pre_reset"],"pre_reset",("reset_mask","episode_length","root_state_w","critic_h1","control","reward","task"))
        reset_evidence=_tensor(pre_reset["reset_mask"],"pre_reset.reset_mask",(32,)); _tensor(pre_reset["episode_length"],"pre_reset.episode_length",(32,))
        _tensor(pre_reset["root_state_w"],"pre_reset.root_state_w",(32,13)); _tensor(pre_reset["critic_h1"],"pre_reset.critic_h1",(32,21))
        if reset_evidence.dtype!=torch.bool or not torch.equal(reset_evidence,reset): raise ValueError("pre-reset time/reset identity mismatch")
        # pre.root_state_w is the state captured before environment.step.  The
        # pre_reset state is captured after physics and before reset, so real
        # motion normally makes these tensors differ.  Their individual shape
        # and finiteness checks above are the cross-boundary contract; only the
        # two post-physics/pre-reset views must identify the same state.
        _validate_control(pre_reset["control"]); task=_validate_task(pre_reset["task"])
        _validate_termination_consistency(task, terminated, truncated)
        if not torch.equal(task["root_state_w"],pre_reset["root_state_w"]): raise ValueError("task same-state identity mismatch")
        if _lane_tree(pre_reset["reward"],"pre_reset.reward")==0: raise ValueError("missing tensor evidence: pre_reset.reward")
        if post["lee"] is None or post["wrench"] is None:
            if post["lee"] is not None or post["wrench"] is not None:
                raise ValueError("partial downstream cache evidence")
            if int(valid.sum()) != 0:
                raise ValueError("missing downstream cache evidence for valid lanes")
            fully_invalid_downstream_chunks += 1
        else:
            _validate_lee(post["lee"]); _validate_wrench(post["wrench"])
        invalid_cache += int((~valid).sum())
    return {"schema_version":"unified-h1-control-diagnostic-analysis.v2","status":"COMPLETE_VALID_EVIDENCE",
            "condition":condition,"experiment_id":experiment_id,"policy_steps":630,"lanes":32,"transitions":TRANSITIONS,
            "invalid_downstream_cache_samples":invalid_cache,
            "fully_invalid_downstream_chunks":fully_invalid_downstream_chunks,
            "missing_evidence_filled":False}


__all__=["AGGREGATE_DISK_LIMIT_BYTES","AggregateDiskLedger","CASES","CASE_STEPS","CONDITIONS","DiagnosticChunkWriter","FIXED_B0_ACTION","LANES","SCHEMA","TRANSITIONS","analyze_trace","cpu_snapshot","fixed_b0_action","select_actor_input","validate_parameters"]
