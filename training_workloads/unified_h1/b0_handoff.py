"""Validate-only exact handoff for the single enhanced B0 diagnostic."""
from __future__ import annotations
import hashlib, json, os, subprocess, tempfile, zlib
from pathlib import Path
from typing import Any
from .b0_enhanced import CASES, CASE_POLICY_STEPS, EXPERIMENT_ID, PHYSICS_ROWS

SCHEMA="instinctrl-enhanced-b0-handoff/2"
FIELDS=("schema_version","status","experiment_id","argv","authorization","attempt","checkpoint","source_identity","candidate","frozen","budgets","output_manifest","cleanup")
HEX=lambda value,size: isinstance(value,str) and len(value)==size and all(c in "0123456789abcdef" for c in value)
def _regular(path:Any,label:str)->Path:
    value=Path(path)
    if not value.is_absolute() or value.is_symlink() or not value.is_file(): raise PermissionError(f"{label} is not an exact regular file")
    return value
def _digest(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()
def _git(root:Path,*args:str, binary:bool=False):
    if not root.is_absolute() or root.is_symlink() or not root.is_dir(): raise PermissionError("repository identity path is invalid")
    try: return subprocess.run(["git",*args],cwd=root,check=True,capture_output=True,text=not binary).stdout
    except (OSError,subprocess.CalledProcessError) as error: raise PermissionError("repository identity cannot be recomputed") from error

def _candidate_diff(root:Path, base_commit:str)->bytes:
    """Recompute the orchestrator diff, including every untracked candidate file."""
    with tempfile.TemporaryDirectory(prefix="instinctrl-candidate-") as temporary:
        temporary_path=Path(temporary); index=temporary_path/"index"; objects=temporary_path/"objects"
        objects.mkdir()
        empty=b"blob 0\0"; oid=hashlib.sha1(empty).hexdigest(); target=objects/oid[:2]/oid[2:]
        target.parent.mkdir(); target.write_bytes(zlib.compress(empty))
        env=dict(os.environ, GIT_INDEX_FILE=str(index), GIT_OBJECT_DIRECTORY=str(objects),
                 GIT_ALTERNATE_OBJECT_DIRECTORIES=str(root/".git/objects"))
        subprocess.run(["git","read-tree","HEAD"],cwd=root,env=env,check=True,capture_output=True)
        others=subprocess.run(["git","ls-files","--others","--exclude-standard","-z"],cwd=root,check=True,capture_output=True).stdout.split(b"\0")
        paths=[item.decode() for item in others if item]
        if paths: subprocess.run(["git","add","-N","--",*paths],cwd=root,env=env,check=True,capture_output=True)
        return subprocess.run(["git","diff","--binary",base_commit],cwd=root,env=env,check=True,capture_output=True).stdout

def validate_handoff(value:Any)->dict[str,Any]:
    if not isinstance(value,dict) or tuple(value)!=FIELDS: raise PermissionError("enhanced B0 handoff exact schema mismatch")
    if value["schema_version"]!=SCHEMA or value["status"]!="VALIDATE_ONLY_NOT_AUTHORIZED" or value["experiment_id"]!=EXPERIMENT_ID: raise PermissionError("handoff identity/status mismatch")
    expected_argv=["/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11","-B","-m","training_workloads.unified_h1.formal_unified_h1","--authorization",value["authorization"],"--attempt",value["attempt"],"--mode","enhanced-b0-diagnostic"]
    if value["argv"]!=expected_argv: raise PermissionError("live argv is not exact")
    for name in ("authorization","attempt"):
        if not isinstance(value[name],str) or not Path(value[name]).is_absolute(): raise PermissionError(f"{name} path is not absolute")
    if Path(value["authorization"]) != Path(value["attempt"]) / "commands" / "AUTHORIZATION.json":
        raise PermissionError("authorization is not bound to the exact attempt")
    if value["checkpoint"] is not None:
        raise PermissionError("enhanced zero-target diagnostic must not bind/load a policy checkpoint")
    source=value["source_identity"]
    source_fields={"product_repository","product_commit","product_tree","workload_repository","workload_base_commit","workload_base_tree","runtime_python","runtime_python_sha256","asset_manifest","asset_manifest_sha256"}
    if not isinstance(source,dict) or set(source)!=source_fields: raise PermissionError("source identity is incomplete")
    product=Path(source["product_repository"]); workload=Path(source["workload_repository"])
    for root,prefix in ((product,"product"),(workload,"workload_base")):
        if not HEX(source[prefix+"_commit"],40) or not HEX(source[prefix+"_tree"],40): raise PermissionError(f"{prefix} git identity invalid")
        if _git(root,"rev-parse","HEAD").strip()!=source[prefix+"_commit"] or _git(root,"rev-parse","HEAD^{tree}").strip()!=source[prefix+"_tree"]: raise PermissionError(f"{prefix} git identity changed")
    for path_key,hash_key in (("runtime_python","runtime_python_sha256"),("asset_manifest","asset_manifest_sha256")):
        path=_regular(source[path_key],path_key)
        if not HEX(source[hash_key],64) or _digest(path)!=source[hash_key]: raise PermissionError(f"{path_key} bytes changed")
    candidate=value["candidate"]
    if not isinstance(candidate,dict) or set(candidate)!={"repository","base_commit","commit","diff_sha256"} or not HEX(candidate.get("base_commit"),40) or not HEX(candidate.get("diff_sha256"),64): raise PermissionError("candidate diff binding is incomplete")
    candidate_root=Path(candidate["repository"])
    if not HEX(candidate.get("commit"),40) or _git(candidate_root,"rev-parse",candidate["base_commit"]+"^{commit}").strip()!=candidate["base_commit"]: raise PermissionError("candidate base commit changed")
    if _git(candidate_root,"rev-parse","HEAD").strip()!=candidate["commit"]: raise PermissionError("candidate commit changed")
    actual_diff=hashlib.sha256(_candidate_diff(candidate_root,candidate["base_commit"])).hexdigest()
    if actual_diff!=candidate["diff_sha256"]: raise PermissionError("candidate diff bytes changed")
    frozen={"seed":20260909,"cases":list(CASES),"case_policy_steps":CASE_POLICY_STEPS,"lanes":32,"decimation":20,"target_velocity_b_m_s":[0.0,0.0,0.0],"initial_state":"frozen-R4-B0","curriculum":"frozen-Z/T/S","control_frequency_hz":200,"policy_frequency_hz":10,"safety_standards":"unchanged-R4-B0"}
    if value["frozen"]!=frozen: raise PermissionError("frozen experiment parameters changed")
    budgets={"live_process_starts":1,"simulator_initializations":1,"nontraining_transitions":20160,"physics_rows":403200,"training_transitions":0,"optimizer_updates":0,"checkpoint_writes":0,"p0_plr":0,"policy_evaluations":0,"automatic_retries":0,"wall_seconds":910,"gpu_allocation_seconds":910,"diagnostic_bytes":16*1024**3}
    if value["budgets"]!=budgets: raise PermissionError("enhanced B0 budgets changed")
    outputs=value["output_manifest"]
    expected_outputs={"raw_physics_jsonl":"enhanced_b0_physics.jsonl","raw_manifest":"enhanced_b0_manifest.json","causal_analysis":"enhanced_b0_causal_analysis.json","measurement_failure":"enhanced_b0_measurement_failure.json","budget_ledger":"budget_ledger.json","status":"status.json","cleanup_evidence":"cleanup_evidence.json"}
    if outputs != expected_outputs: raise PermissionError("output manifest is not exact")
    if value["cleanup"]!={"environment_close":True,"app_close":True,"no_retry_on_failure":True}: raise PermissionError("cleanup/no-retry contract changed")
    return {"status":"VALID","live_authorized":False,"experiment_id":EXPERIMENT_ID,"physics_rows":PHYSICS_ROWS,"candidate_diff_sha256":candidate["diff_sha256"]}

def load_and_validate(path:Path):
    path=Path(path)
    if path.is_symlink() or not path.is_file(): raise PermissionError("handoff must be an exact regular file")
    return validate_handoff(json.loads(path.read_text(encoding="utf-8")))
