"""Fail-closed readable boundary for predecessor and future live evidence."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from typing import Any

SCHEMA="instinctrl-evaluator-evidence-boundary/1"
POSTRUN_SCHEMA="instinctrl-enhanced-b0-postrun-evidence/2"
PACKAGE_ID="INSTINCTRL-R5-STAGE1-IDENTITY-BOUNDARY-SUCCESSOR-20260929"
PREDECESSOR_PACKAGE_ID="INSTINCTRL-R5-STAGE1-RESET-AVAILABILITY-CLEANUP-SUCCESSOR-20260929"
EXPERIMENT_ID="UNIFIED-H1-R5-STAGE1-B0-RESET-AVAILABILITY-CLEANUP-R1-20260929"
TASK_ID="T-d219cc10961c75be"
PREDECESSOR_DIFF="e58bfa16aa6ca448fb730e42582369b427a7454daec1fae3e351b0597c185f97"

def _digest(path:Path)->str:
    value=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):value.update(chunk)
    return value.hexdigest()

def _regular(root:Path,relative:str)->Path:
    if not isinstance(relative,str) or not relative or relative.startswith("/") or ".." in Path(relative).parts:raise PermissionError("unsafe evidence path")
    path=root/relative
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):raise PermissionError("evidence byte is absent or not a protected regular file")
    return path

def _load(path:Path)->Any:
    if path.is_symlink() or not path.is_file():raise PermissionError("evidence manifest is absent or not regular")
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,UnicodeError,json.JSONDecodeError) as error:raise PermissionError("evidence manifest is unreadable") from error

def _verify_files(root:Path,rows:Any)->dict[str,str]:
    if not isinstance(rows,list) or not rows:raise PermissionError("evidence manifest file inventory is empty")
    result={}
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"path","sha256","bytes"} or row["path"] in result:raise PermissionError("evidence manifest row is malformed or duplicated")
        path=_regular(root,row["path"])
        if type(row["bytes"]) is not int or row["bytes"]<0 or path.stat().st_size!=row["bytes"] or _digest(path)!=row["sha256"]:raise PermissionError("evidence bytes/hash mismatch: "+row["path"])
        result[row["path"]]=row["sha256"]
    return result

def verify_predecessor(manifest_path:Path,predecessor_root:Path)->dict[str,Any]:
    value=_load(Path(manifest_path))
    if not isinstance(value,dict) or set(value)!={"schema_version","package_id","predecessor","future_postrun"} or value["schema_version"]!=SCHEMA or value["package_id"]!=PREDECESSOR_PACKAGE_ID:raise PermissionError("evidence boundary identity mismatch")
    predecessor=value["predecessor"]
    if not isinstance(predecessor,dict) or set(predecessor)!={"source_package_id","files"}:raise PermissionError("predecessor boundary schema mismatch")
    files=_verify_files(Path(predecessor_root),predecessor["files"])
    required={"SHA256SUMS","live_failure/job/h1-collection.json","live_failure/job/receipt.json","live_failure/report/B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md","live_failure/report/b0_enhanced_live_root_cause.json"}
    if not required<=set(files):raise PermissionError("predecessor live job/report boundary is incomplete")
    return {"status":"VERIFIED","files":files,"future_postrun":value["future_postrun"]}

def verify_postrun(manifest_path:Path,postrun_root:Path|None=None,*,expected_experiment_id:str=EXPERIMENT_ID,expected_package_id:str=PREDECESSOR_PACKAGE_ID)->dict[str,Any]:
    value=_load(Path(manifest_path))
    if not isinstance(value,dict) or set(value)!={"schema_version","package_id","experiment_id","roots","files"} or value["schema_version"]!=POSTRUN_SCHEMA or value["package_id"]!=expected_package_id or value["experiment_id"]!=expected_experiment_id:raise PermissionError("postrun evidence identity mismatch")
    roots=value["roots"]
    if not isinstance(roots,dict) or set(roots)!={"job","report"}:raise PermissionError("postrun evidence roots are incomplete")
    root_paths={kind:Path(path) for kind,path in roots.items()}
    for kind,root in root_paths.items():
        if root.is_symlink() or not root.is_dir():raise PermissionError("postrun evidence root is absent or unsafe: "+kind)
    if postrun_root is not None:
        boundary=Path(postrun_root).resolve()
        if boundary.is_symlink() or not boundary.is_dir():raise PermissionError("postrun evidence boundary root is absent or unsafe")
        if any(not root.resolve().is_relative_to(boundary) for root in root_paths.values()):raise PermissionError("postrun evidence root drift")
    files={}
    rows=value["files"]
    if not isinstance(rows,list) or not rows:raise PermissionError("evidence manifest file inventory is empty")
    for row in rows:
        if not isinstance(row,dict) or set(row)!={"kind","path","sha256","bytes"} or row["kind"] not in roots:raise PermissionError("evidence manifest row is malformed or duplicated")
        key=row["kind"]+"/"+row["path"]
        if key in files:raise PermissionError("evidence manifest row is malformed or duplicated")
        path=_regular(root_paths[row["kind"]],row["path"])
        if type(row["bytes"]) is not int or row["bytes"]<0 or path.stat().st_size!=row["bytes"] or _digest(path)!=row["sha256"]:raise PermissionError("evidence bytes/hash mismatch: "+key)
        files[key]=row["sha256"]
    actual=set()
    for kind,root in root_paths.items():
        for path in root.rglob("*"):
            if path.is_symlink():raise PermissionError("postrun evidence inventory contains symlink")
            canonical_manifest=root_paths["job"]/"postrun-evidence-manifest.json"
            if path.is_file() and path.resolve() != canonical_manifest:
                actual.add(kind+"/"+path.relative_to(root).as_posix())
    if actual != set(files):raise PermissionError("postrun evidence manifest inventory mismatch")
    if not any(path.startswith("job/") for path in files) or not any(path.startswith("report/") for path in files):raise PermissionError("postrun job/report evidence is incomplete")
    return {"status":"VERIFIED","files":files}

def verify_evaluator_boundary(boundary_manifest:Path,predecessor_root:Path,postrun_manifest:Path|None=None,postrun_root:Path|None=None)->dict[str,Any]:
    predecessor=verify_predecessor(boundary_manifest,predecessor_root)
    if postrun_manifest is None or postrun_root is None:raise PermissionError("future live postrun evidence is absent")
    return {"predecessor":predecessor,"postrun":verify_postrun(postrun_manifest,postrun_root)}


def verify_delivery_identity(delivery_root:Path)->dict[str,str]:
    """Bind every active handoff/report representation to this successor."""
    root=Path(delivery_root)
    handoff=_load(root/"handoff_contract.json")
    report=_load(root/"b0_enhanced_live_root_cause.json")
    markdown=_regular(root,"B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md").read_text(encoding="utf-8")
    for name,value in (("handoff",handoff),("report",report)):
        predecessor=value.get("accepted_predecessor_candidate",{})
        if (value.get("package_id")!=PACKAGE_ID or value.get("stage_id")!=PACKAGE_ID or
                value.get("task_id")!=TASK_ID or value.get("experiment_id")!=EXPERIMENT_ID or
                predecessor.get("diff_sha256")!=PREDECESSOR_DIFF):
            raise PermissionError("successor delivery identity mismatch: "+name)
    for expected in (PACKAGE_ID,TASK_ID,EXPERIMENT_ID):
        if expected not in markdown:raise PermissionError("successor markdown identity mismatch")
    if (root/"validate_only_receipt.json").exists():
        raise PermissionError("predecessor validate-only receipt is active in successor delivery")
    return {"package_id":PACKAGE_ID,"task_id":TASK_ID,"experiment_id":EXPERIMENT_ID,"predecessor_diff_sha256":PREDECESSOR_DIFF}
