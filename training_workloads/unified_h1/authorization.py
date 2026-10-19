"""Fail-closed future authorization bound to the validated source manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .budget import required_mode_budget
from .config import CONFIG_PATH, METHOD_SHA256, load_config, sha256_file

MODES = ("research-r6-evaluation", "main", "controlled-failure", "r4-training", "evaluation", "r5-resume", "control-diagnostic", "pilot-training", "pilot-evaluation", "enhanced-b0-diagnostic", "research-training", "research-evaluation", "research-transition", "research-formal-training", "research-formal-evaluation")
BUDGET_FIELDS = {"live_process_starts", "simulator_initializations", "training_transitions",
                 "nontraining_transitions", "optimizer_updates", "checkpoint_writes",
                 "initialization_ticks", "explicit_resets", "automatic_resets", "wall_seconds"}

MODE_ORDERS = {"research-r6-evaluation": None,"research-formal-training": None, "research-formal-evaluation": None,"research-training": None, "research-evaluation": None, "research-transition": None,"main": "UNIFIED-H1-R3", "controlled-failure": "UNIFIED-H1-R3",
               "r4-training": "UNIFIED-H1-R4", "evaluation": None,
               "r5-resume": "UNIFIED-H1-R5", "control-diagnostic": None,
               "pilot-training": None, "pilot-evaluation": None,
               "enhanced-b0-diagnostic": "UNIFIED-H1-R5-READINESS-B0-PLANT-R2-20261012"}


def _parameters(mode: str, value: object) -> dict[str, Any]:
    if mode.startswith("research-"):
        from .research_contract import validate_parameters
        return validate_parameters(mode, value)
    if mode == "control-diagnostic":
        from .control_diagnostic import validate_parameters as validate_control_diagnostic_parameters
        return validate_control_diagnostic_parameters(value)
    if mode == "enhanced-b0-diagnostic":
        if not isinstance(value, dict) or set(value) != {"handoff"}: raise PermissionError("enhanced B0 requires exact handoff parameter")
        from .b0_handoff import load_and_validate
        handoff = Path(value["handoff"])
        load_and_validate(handoff)
        return {"handoff": str(handoff)}
    if not isinstance(value, dict): raise PermissionError("R3 parameters must be an exact mapping")
    expected = {"main": set(), "controlled-failure": set(), "r4-training": set(),
                "evaluation": {"stage", "checkpoints"},
                "r5-resume": {"checkpoint", "checkpoint_sha256", "target_update"},
                "pilot-training": {"group"}, "pilot-evaluation": {"group", "checkpoints"}}[mode]
    if mode in {"pilot-training", "pilot-evaluation"} and value.get("group") in {"C0", "C1"}:
        from .pilot_contract import PILOT_OUTPUT_CAP_BYTES, pilot_residual_weight
        expected |= {"residual_neutrality_weight", "output_cap_bytes"}
        weight = value.get("residual_neutrality_weight")
        if type(weight) not in (int, float) or weight != pilot_residual_weight(value["group"]):
            raise PermissionError("new pilot coefficient differs from preregistration")
        if type(value.get("output_cap_bytes")) is not int or value["output_cap_bytes"] != PILOT_OUTPUT_CAP_BYTES:
            raise PermissionError("new pilot output cap differs from preregistration")
    if mode == "evaluation" and "checkpoint_source_identity" in value:
        expected = expected | {"checkpoint_source_identity"}
        origin = value["checkpoint_source_identity"]
        fields = {"product_commit", "product_tree", "workload_commit", "workload_tree", "config_sha256"}
        if (value.get("stage") != "R4" or not isinstance(origin, dict) or set(origin) != fields
                or any(not isinstance(v, str) or len(v) != (64 if k == "config_sha256" else 40)
                       or any(c not in "0123456789abcdef" for c in v) for k, v in origin.items())):
            raise PermissionError("evaluation checkpoint source identity is invalid")
    if set(value) != expected: raise PermissionError("R3 mode parameter schema mismatch")
    if mode in {"pilot-training", "pilot-evaluation"} and value.get("group") not in {"P0", "PLR", "C0", "C1"}: raise PermissionError("pilot group must be P0 or PLR")
    if mode == "pilot-evaluation":
        rows=value["checkpoints"]
        if not isinstance(rows,list) or len(rows)!=2 or [row.get("update") for row in rows if isinstance(row,dict)] != [16,64]: raise PermissionError("pilot evaluation checkpoints must be u16/u64")
        for row in rows:
            if set(row) != {"path","sha256","update"}: raise PermissionError("pilot checkpoint binding is incomplete")
            checkpoint=Path(row["path"])
            if not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file() or sha256_file(checkpoint)!=row["sha256"]: raise PermissionError("pilot checkpoint input/hash changed")
    if "checkpoint" in value:
        checkpoint = Path(value["checkpoint"])
        if (not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file()
                or sha256_file(checkpoint) != value["checkpoint_sha256"]):
            raise PermissionError("checkpoint input/hash is not exact")
    for key in ("update", "target_update"):
        if key in value and (type(value[key]) is not int or value[key] < 0 or value[key] > 2048): raise PermissionError("checkpoint update parameter is invalid")
    if mode == "evaluation":
        stage = value["stage"]
        expected_updates = {"R4": [0, 64, 128, 192, 256], "R5-512": [448, 512],
                            "R5-1024": [960, 1024], "R5-2048": [1984, 2048]}.get(stage)
        rows = value["checkpoints"]
        if expected_updates is None or not isinstance(rows, list) or len(rows) != len(expected_updates):
            raise PermissionError("evaluation checkpoint stage/set is invalid")
        for row, update in zip(rows, expected_updates, strict=True):
            if not isinstance(row, dict) or set(row) != {"path", "sha256", "update"} or row["update"] != update:
                raise PermissionError("evaluation checkpoint binding is incomplete")
            checkpoint = Path(row["path"])
            if (not checkpoint.is_absolute() or checkpoint.is_symlink() or not checkpoint.is_file()
                    or sha256_file(checkpoint) != row["sha256"]):
                raise PermissionError("evaluation checkpoint input/hash changed")
    return dict(value)


def load_r3_authorization(path: Path, attempt: Path, mode: str) -> dict[str, Any]:
    config = load_config()
    if mode not in MODES or not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise PermissionError("R3 authorization must be an explicit regular file and known mode")
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "status", "order_id", "method_sha256", "config_sha256", "product_commit", "product_tree",
                "workload_commit", "workload_tree", "attempt", "mode", "budgets", "parameters"}
    if not isinstance(value, dict) or set(value) != required or value["schema_version"] != "unified-h1-r3-authorization.v2" or value["status"] != "AUTHORIZED":
        raise PermissionError("missing exact R3 machine authorization")
    identity = config["resolved_identity"]
    expected_identity = {"method_sha256": config["method"]["proposal_sha256"], "config_sha256": sha256_file(CONFIG_PATH),
        "product_commit": identity["product_commit"], "product_tree": identity["product_tree"],
        "workload_commit": identity["workload_commit"], "workload_tree": identity["workload_tree"]}
    if any(value[key] != expected for key, expected in expected_identity.items()): raise PermissionError("R3 method/config/candidate binding mismatch")
    parameters = _parameters(mode, value["parameters"])
    research_method = config["method"]["version"] == "unified-h1-formal-readiness-v1"
    if research_method != mode.startswith("research-"): raise PermissionError("research method/mode mismatch")
    new_method = config["method"]["version"] == "unified-h1-zero-neutrality-v1"
    if new_method != (mode in {"pilot-training", "pilot-evaluation"} and parameters.get("group") in {"C0", "C1"}):
        raise PermissionError("pilot group and method profile disagree")
    if mode == "enhanced-b0-diagnostic":
        handoff_path=Path(parameters["handoff"]); handoff=json.loads(handoff_path.read_text(encoding="utf-8"))
        source=handoff["source_identity"]; candidate=handoff["candidate"]
        if Path(handoff["authorization"]).resolve()!=path.resolve() or Path(handoff["attempt"]).resolve()!=attempt.resolve():
            raise PermissionError("enhanced B0 handoff authorization/attempt differs from current entry")
        if Path(source["product_repository"]).resolve()!=Path(config["product"]["path"]).resolve() or Path(source["workload_repository"]).resolve()!=Path(config["workload"]["path"]).resolve():
            raise PermissionError("enhanced B0 handoff source roots differ from current candidate")
        expected_source={"product_commit":identity["product_commit"],"product_tree":identity["product_tree"],"workload_base_commit":identity["workload_commit"],"workload_base_tree":identity["workload_tree"]}
        if any(source[key]!=expected for key,expected in expected_source.items()):
            raise PermissionError("enhanced B0 checkpoint/source identity differs from current candidate")
        if Path(candidate["repository"]).resolve()!=Path(config["workload"]["path"]).resolve() or candidate["commit"]!=identity["workload_commit"]:
            raise PermissionError("enhanced B0 handoff candidate differs from current workload")
    origin = parameters.get("checkpoint_source_identity")
    if mode == "research-r6-evaluation":
        from .robustness_contract import baseline, validate_compatibility
        validate_compatibility(parameters["checkpoint_compatibility"], baseline(parameters)["source_identity"], config)
    elif mode == "research-transition":
        from .checkpoint_compatibility import validate_transition_compatibility
        validate_transition_compatibility(parameters["checkpoint_compatibility"], origin, config)
    elif mode in {"research-formal-training", "research-formal-evaluation"}:
        if 'w_closure_contract' in parameters:
            from .w_closure import validate_compatibility
            validate_compatibility(parameters["checkpoint_compatibility"], origin, config)
        else:
            from .checkpoint_compatibility import validate_formal_compatibility
            validate_formal_compatibility(parameters["checkpoint_compatibility"], origin, config)
    elif origin is not None and any(origin[k] != identity[k] for k in ("product_commit", "product_tree")):
        raise PermissionError("evaluation checkpoint product identity differs from the evaluated product")
    expected_order = MODE_ORDERS[mode]
    if mode.startswith("research-"): expected_order = parameters["experiment_id"] + "-" + parameters["group"] + "-" + mode.upper()
    if mode == "evaluation": expected_order = "UNIFIED-H1-R4" if parameters["stage"] == "R4" else "UNIFIED-H1-R5"
    if mode == "control-diagnostic": expected_order = parameters["experiment_id"]
    if mode == "pilot-training": expected_order = f"UNIFIED-H1-R5-{parameters['group']}-PILOT-TRAIN"
    if mode == "pilot-evaluation": expected_order = f"UNIFIED-H1-R5-{parameters['group']}-PILOT-EVAL"
    if mode == "enhanced-b0-diagnostic": expected_order = "UNIFIED-H1-R5-READINESS-B0-PLANT-R2-20261012"
    if value["order_id"] != expected_order or value["mode"] != mode or Path(value["attempt"]).resolve() != attempt.resolve():
        raise PermissionError("R3 order/attempt/mode binding mismatch")
    if not attempt.is_absolute() or attempt.is_symlink() or not attempt.is_dir(): raise PermissionError("attempt directory is invalid")
    budgets = value["budgets"]
    if not isinstance(budgets, dict) or set(budgets) != BUDGET_FIELDS or any(type(item) is not int or item < 0 for item in budgets.values()):
        raise PermissionError("R3 budget schema mismatch")
    if budgets != required_mode_budget(mode, parameters):
        raise PermissionError("authorization does not exactly match the frozen per-mode budget")
    result = dict(value); result["parameters"] = parameters
    result["authorization_sha256"] = sha256_file(path); result["authorization_path"] = str(path)
    return result


__all__ = ["BUDGET_FIELDS", "MODES", "load_r3_authorization"]
