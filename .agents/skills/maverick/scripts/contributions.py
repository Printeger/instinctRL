"""Small, factual contribution ledger. References are supplied evidence, not verified proof."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROLES = ("maverick", "rebel", "inventor", "hacker", "scientist", "producer")
MODES = ("direct", "perspective", "native_subagent", "delegated_subagent")
KINDS = {"direction": "方向", "implementation": "实施", "correction": "纠错"}
STATUSES = {"proposed": "待验证建议", "confirmed": "已确认影响", "not_adopted": "未采用", "unknown": "结果未知"}
GRADES = {1: "1 · 辅助", 2: "2 · 实质", 3: "3 · 关键"}
REQUIRED = {"id", "task", "persona", "mode", "actor", "trigger", "kind", "contribution", "status", "outcome", "evidence", "impact"}
IDENTITY = ("persona", "mode", "actor", "kind")


def _path(project: Path) -> Path:
    project = project.resolve()
    path = project / ".maverick/contributions.jsonl"
    if not project.is_dir() or not path.resolve().is_relative_to(project):
        raise ValueError("Contribution ledger must be inside an existing project.")
    if path.exists() and not path.is_file():
        raise ValueError("Contribution ledger is not a regular file.")
    return path


def _nonempty(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate(event: dict) -> None:
    if not isinstance(event, dict):
        raise ValueError("A contribution must be a JSON object.")
    missing = REQUIRED - event.keys()
    unknown = event.keys() - REQUIRED - {"at", "reused", "shift"}
    if missing or unknown:
        raise ValueError(f"Contribution fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    for key in ("id", "task", "persona", "mode", "actor", "trigger", "kind", "contribution", "status"):
        if not _nonempty(event[key]):
            raise ValueError(f"{key} must be a nonempty string.")
    if event["persona"] not in ROLES or event["mode"] not in MODES:
        raise ValueError("Unknown contribution persona or mode.")
    if event["kind"] not in KINDS or event["status"] not in STATUSES:
        raise ValueError("Unknown contribution kind or status.")
    if not isinstance(event["outcome"], str):
        raise ValueError("outcome must be a string; it can be empty while unconfirmed.")
    evidence = event["evidence"]
    if not isinstance(evidence, list) or not all(_nonempty(item) for item in evidence):
        raise ValueError("evidence must be a list of nonempty reference strings.")
    mode, persona, actor = event["mode"], event["persona"], event["actor"]
    if mode in ("direct", "perspective"):
        if actor != "maverick" or (mode == "direct" and persona != "maverick"):
            raise ValueError("Root work uses actor=maverick; direct work also uses persona=maverick.")
        if "reused" in event:
            raise ValueError("reused describes an actual subagent, not a root perspective.")
    elif actor == "maverick" or persona == "maverick":
        raise ValueError("Subagent work needs the actual agent/thread ID and a delegated persona.")
    if "reused" in event and type(event["reused"]) is not bool:
        raise ValueError("reused must be a boolean.")
    impact = event["impact"]
    if event["status"] == "confirmed":
        if type(impact) is not int or impact not in GRADES or not evidence or not _nonempty(event["outcome"]):
            raise ValueError("Confirmed influence needs evidence, a concrete outcome and impact 1, 2 or 3.")
    elif impact is not None:
        raise ValueError("Unconfirmed or unadopted work must use impact=null.")
    if "shift" in event:
        shift = event["shift"]
        if (event["status"] != "confirmed" or not isinstance(shift, dict)
                or set(shift) != {"before", "after"} or not all(_nonempty(v) for v in shift.values())):
            raise ValueError("shift requires a confirmed event and nonempty before/after statements.")
    if "at" in event and not _nonempty(event["at"]):
        raise ValueError("at must be a nonempty timestamp string.")


def _revision(previous: dict | None, event: dict) -> None:
    if previous and any(previous[key] != event[key] for key in IDENTITY):
        raise ValueError("A revision must retain persona, mode, actor and kind; use a new id for another contribution.")


def _load(path: Path) -> tuple[dict, bool]:
    latest = {}
    needs_newline = False
    if path.exists():
        with path.open(encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                needs_newline = not line.endswith("\n")
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    _validate(event)
                    key = (event["task"], event["id"])
                    _revision(latest.get(key), event)
                    latest.pop(key, None)
                    latest[key] = event
                except (ValueError, TypeError) as error:
                    raise ValueError(f"contributions.jsonl line {number}: {error}") from error
    return latest, needs_newline


def record(project: Path, event: dict) -> dict:
    """Validate and append one event or revision. The coordinator is the sole writer."""
    _validate(event)
    event = {**event, "at": event.get("at") or datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = _path(project)
    latest, needs_newline = _load(path)
    previous = latest.get((event["task"], event["id"]))
    _revision(previous, event)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as target:
        target.write(("\n" if needs_newline else "") + json.dumps(event, ensure_ascii=False) + "\n")
    return {"status": "recorded", "task": event["task"], "id": event["id"],
            "revision": previous is not None, "ledger": ".maverick/contributions.jsonl"}


def validate_ledger(project: Path) -> list[str]:
    """Check schema and revision consistency; do not claim to verify referenced evidence."""
    try:
        path = _path(project)
        if not path.exists():
            return ["Missing .maverick/contributions.jsonl"]
        _load(path)
        return []
    except (ValueError, OSError) as error:
        return [str(error)]


def _text(value: str) -> str:
    # Treat ledger strings as text, including in Markdown tables.
    return (" ".join(value.split()).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\\", "\\\\").replace("|", "\\|")
            .replace("`", "\\`").replace("*", "\\*").replace("_", "\\_")
            .replace("[", "\\[").replace("]", "\\]"))


def _participation(event: dict) -> str:
    name = event["persona"].capitalize()
    if event["mode"] == "direct":
        return "Maverick 主对话直接处理"
    if event["mode"] == "perspective":
        return f"Maverick 采用 {name} 视角（主对话，非子 Agent）"
    kind = "原生人格子 Agent" if event["mode"] == "native_subagent" else "提示词委派子 Agent（未证实原生角色加载）"
    reused = "，复用" if event.get("reused") else ""
    return f"{kind}：{_text(event['actor'])}{reused}"


def report(project: Path, task: str) -> str:
    """Render the latest recorded outcomes for one task; never infer missing participation."""
    if not _nonempty(task):
        raise ValueError("task must be a nonempty string.")
    path = _path(project)
    if not path.exists():
        raise ValueError("Missing .maverick/contributions.jsonl; initialize or upgrade the project first.")
    latest, _ = _load(path)
    events = [event for event in latest.values() if event["task"] == task]
    lines = [f"# Maverick 开发贡献报告 · {_text(task)}", "",
             "依据本任务已记录的最新结果汇总；证据为记录者提供的引用，未由脚本独立验证。",
             "等级表示各人格已确认影响的最高定性等级，不累加、不换算百分比。主对话借用视角与真实子 Agent 分开标明。", "",
             "| 人格 | 参与方式 | 实际贡献与结果 | 等级 | 证据 |",
             "| --- | --- | --- | --- | --- |"]
    for role in ROLES:
        selected = [event for event in events if event["persona"] == role]
        if not selected:
            if role == "maverick" and any(event["mode"] == "perspective" for event in events):
                lines.append("| Maverick | 主对话采用了其他人格视角 | 见对应人格行，不重复计分 | — | 见对应人格行 |")
            else:
                lines.append(f"| {role.capitalize()} | 未记录参与 | — | — | — |")
            continue
        participation = list(dict.fromkeys(_participation(event) for event in selected))
        contributions = []
        evidence = []
        impacts = []
        for event in selected:
            label = f"{_text(event['id'])} · {KINDS[event['kind']]} · {STATUSES[event['status']]}"
            outcome = f" → {_text(event['outcome'])}" if event["outcome"].strip() else ""
            contributions.append(f"{label}：{_text(event['contribution'])}{outcome}")
            if event["status"] == "confirmed":
                impacts.append(event["impact"])
            if event["evidence"]:
                evidence.append(f"{_text(event['id'])}：" + "；".join(_text(item) for item in event["evidence"]))
        grade = GRADES[max(impacts)] if impacts else "—"
        references = "；".join(dict.fromkeys(evidence)) or "—"
        lines.append(f"| {role.capitalize()} | {'；'.join(participation)} | {'；'.join(contributions)} | {grade} | {references} |")
    lines.extend(["", "## 关键思维转变", ""])
    shifts = [event for event in events if event.get("shift") and event["status"] == "confirmed"]
    if shifts:
        for event in shifts:
            shift = event["shift"]
            refs = "；".join(_text(item) for item in event["evidence"])
            lines.append(f"- {_text(shift['before'])} → {_text(shift['after'])}。来源：{_participation(event)}；证据：{refs}。")
    else:
        lines.append("无已记录、已确认的关键思维转变。")
    if not events:
        lines.extend(["", "本任务没有贡献记录；这不表示所有人格都未参与，也不代表任务已经完成。"])
    return "\n".join(lines) + "\n"
