"""Evidence-bound migration. Historical files are copied, never edited in place."""
from pathlib import Path
import hashlib
import json
import time
from .store import atomic, canonical, sha, Conflict
from .policy import WORKSPACE, COORDINATION, candidate, new_stage

ENTRIES = ["AGENTS.md", "AGENT.md", "DUAL_AGENT_GUIDE.md", "instinctRL_coordination/AGENTS.md",
           "instinctRL_coordination/ACTIVE_WORKFLOW.json", "instinctRL_coordination/commander/COMMANDER.md",
           "instinctRL_coordination/executor/BUILDER.md"]
SOURCES = [*ENTRIES, "INSTINCTRL_MULTI_AGENT_WORKFLOW.md", "instinctRL_coordination/protocols/DUAL_AGENT_WORKFLOW_V3.md",
           "instinctRL_coordination/commander/CURRENT_ORDER.md", "instinctRL_coordination/commander/COMMANDER_STATE.json",
           "instinctRL_coordination/executor/EXECUTOR_STATE.json", "instinctRL_coordination/executor/LAST_REPORT.md",
           "instinctRL_coordination/commander/reviews/UNIFIED-H1-R3_A0006_REVIEW.md",
           "instinctRL_coordination/commander/authorizations/USER-20260907-WORKFLOW-V3.md"]


def snapshot(store):
    manifest_path = store.root / "migration/before/manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    manifest = {}
    for name in SOURCES:
        p = WORKSPACE / name
        if p.exists():
            data = p.read_bytes()
            atomic(store.root / "migration/before" / name, data)
            manifest[name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    atomic(manifest_path, canonical(manifest))
    return manifest


def import_history(store):
    manifest = snapshot(store)
    source = json.loads((COORDINATION / "commander/COMMANDER_STATE.json").read_text())
    summary = {"order": "UNIFIED-H1-R3", "status": "HOLD", "executable_now": "NONE",
               "training_authorized": False, "main_starts_remaining": 0,
               "cumulative_live_starts": 3, "cumulative_applauncher": 3, "cumulative_init_slots": 3,
               "cumulative_physics_ticks": None, "D0004_measured_ticks": 2,
               "D0001_ticks": None, "D0003_ticks": None, "transitions": 0, "ppo_updates": 0, "checkpoints": 0,
               "unresolved_findings": ["R3-F4", "R3-F5", "G5"],
               "closed_findings": "R2 six findings; R3-F1/F2/F3 nine CPU findings remain closed",
               "D0002": "not created / not released", "R4_R5_authorized": False,
               "scientific_conclusion": "NOT_PROVEN; CPU migration does not requalify historical reports"}
    def mutate(s):
        s["historical"] = {"summary": summary, "source_manifest": manifest, "commander_state": source}
        s["migration"] = {"snapshot_sha256": sha(manifest), "authorization": "User implementation plan in this session, 2026-09-13",
                          "status": "PREPARED"}
    return store.transact("HISTORICAL_STATE_IMPORTED", "migration:history:v1", mutate)


def seed_demo(store):
    from .policy import git
    import subprocess
    repo = store.root / "workspaces/budget-parser-demo"
    if "CPU-DEMO" in store.replay()["stages"]:
        return repo
    if repo.exists():
        raise Conflict("unregistered demo workspace exists; preserve and inspect before adopting")
    repo.mkdir(parents=True)
    atomic(repo / "task_config.py", b'''"""Parse CPU task configuration."""
def parse_task(value):
    return {"name": value["name"], "budget": value["budget"]}
''')
    atomic(repo / "test_task_config.py", b'''import unittest
from task_config import parse_task

class ParserTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(parse_task({"name": "demo", "budget": 2}), {"name": "demo", "budget": 2})

if __name__ == "__main__":
    unittest.main()
''')
    atomic(repo / "CONTRACT.md", "# CPU 示例验收契约\nparse_task 接收 dict，name 为非空 str，budget 必须为非负 int（bool 不合法）。\n合法输入返回 name/budget 两字段，其他输入以 ValueError 或 TypeError 明确拒绝。\n不得改变验收标准。补充必要 unittest。GPU/runtime/网络/包安装关闭。\n".encode())
    subprocess.run(["git", "init", "-b", "cpu-demo", str(repo)], check=True, capture_output=True)
    git(repo, "add", "task_config.py", "test_task_config.py", "CONTRACT.md")
    git(repo, "-c", "user.name=rlflow", "-c", "user.email=rlflow@localhost", "commit", "-m", "Fixture: task parser missing budget validation")
    stage = new_stage("CPU-DEMO", repo,
                      "修复 task_config.py 的 parse_task 配置解析器：合法输入保持正确，负预算、错误类型（含 bool）及无效name明确拒绝；仅CPU实现、自测和独立审核。CONTRACT.md为冻结标准，不得修改。",
                      {"role_calls": 6, "wall_seconds": 3600, "fake_starts": 6}, "user implementation plan: isolated real demo")
    stage["protected_paths"] = ["CONTRACT.md"]
    return store.transact("DEMO_STAGE_AUTHORIZED", "demo:authorization:v1", lambda s: s["stages"].update({"CPU-DEMO": stage}), producer="user-plan")


def activate(store, report_path):
    from .isolation import require_doctor
    from .verification import require_tests
    require_doctor(store.root)
    require_tests(store.root)
    state = store.replay()
    if any(s["status"] in {"RUNNING", "START_INTENT", "RECONCILE"} for s in state["sessions"].values()):
        raise Conflict("reconcile active roles before migration")
    if not any(t["stage_id"] == "CPU-DEMO" and t["status"] == "DONE" for t in state["tasks"].values()):
        raise Conflict("real independent CPU demonstration has not passed")
    report = Path(report_path).read_bytes()
    manifest = snapshot(store)
    for name in ENTRIES:
        if state["migration"].get("status") != "ACTIVATING" and hashlib.sha256((WORKSPACE / name).read_bytes()).hexdigest() != manifest[name]["sha256"]:
            raise Conflict("entry changed since snapshot: " + name)
    pointer = {"schema_version": "multi-agent-workflow-pointer/1", "workflow_version": "0.2-cpu",
               "effective_date": "2026-09-13", "policy_path": "protocols/MULTI_AGENT_WORKFLOW_0_2.md",
               "usage_guide_path": "../INSTINCTRL_MULTI_AGENT_GUIDE.md", "entry": "../rlflow",
               "state_path": "research/state/RESEARCH_STATE.json", "authoritative_events": "research/runtime/research.db",
               "enabled_capabilities": ["isolated CPU development", "independent CPU review", "fake jobs"],
               "real_backend_enabled": False, "role_concurrency": 1, "passive_scout": False,
               "historical_R3": "HOLD; no implementation or real execution authorized",
               "migration_snapshot_sha256": sha(manifest), "acceptance_report_sha256": hashlib.sha256(report).hexdigest()}
    store.transact("MIGRATION_INTENT", "migration:activate:intent", lambda s: s["migration"].update(status="ACTIVATING", pointer=pointer))
    for name in ENTRIES:
        if name.endswith("ACTIVE_WORKFLOW.json"):
            continue
        text = "# instinctRL 工作流入口\n\n当前开发任务统一通过 `/home/mint/rl_dev/rlflow` 交给多 Agent 调度器。\n先读 `/home/mint/rl_dev/instinctRL_coordination/ACTIVE_WORKFLOW.json` 和其指定协议，\n使用说明见 `/home/mint/rl_dev/INSTINCTRL_MULTI_AGENT_GUIDE.md`。\n\n本文件替代旧双 agent 派发入口；Commander / Builder 名称仅供历史兼容，不再派发当前开发任务。\n迁移前原文及哈希保存在 `instinctRL_coordination/research/migration/before/` 对应路径。\n冻结产品、runtime、封存证据与旧指令原文继续只读，actor/方法/资源边界保留。\nR3 保持 HOLD；CPU 工作流启用不构成 R3 修复授权、验收或真实运行许可。\n保留其他用户改动，仅暂存自己负责的文件。\n"
        atomic(WORKSPACE / name, text.encode())
    # The pointer is the single publication boundary; intent supports interrupted completion.
    atomic(COORDINATION / "ACTIVE_WORKFLOW.json", canonical(pointer) + b"\n")
    store.transact("WORKFLOW_ACTIVATED", "migration:activated:v1", lambda s: (s.update(enabled=True), s["migration"].update(status="ACTIVE")))


def rollback(store):
    from .engine import reconcile
    state = reconcile(store)
    if not state["paused"]:
        raise Conflict("pause before rollback")
    if any(s["status"] in {"RUNNING", "START_INTENT", "RECONCILE"} for s in state["sessions"].values()):
        raise Conflict("unresolved active role; inspect and reconcile before rollback")
    if state.get('packages') or state.get('releases') or state['migration'].get('control_state_version') == 2:
        if any(j['status'] not in {'SUCCEEDED','FAILED','STOPPED','HANDSHAKE_TIMEOUT'} for j in state['jobs'].values()):
            raise Conflict('unresolved job before rollback')
        return store.transact('CONTROL_WORKFLOW_DISABLED','control-rollback:'+str(time.time_ns()),
                              lambda s:(s.update(enabled=False,autonomous_tasks=[]),s['migration'].update(status='ROLLED_BACK')))
    from . import jobs
    if any(jobs.reconcile(store.root, jid)["state"] not in jobs.TERMINAL for jid in state["jobs"]):
        raise Conflict("unresolved job before rollback")
    for name in ENTRIES:
        atomic(WORKSPACE / name, (store.root / "migration/before" / name).read_bytes())
    return store.transact("WORKFLOW_ROLLED_BACK", "rollback:" + str(time.time_ns()),
                          lambda s: (s.update(enabled=False), s["migration"].update(status="ROLLED_BACK")))


def upgrade_control(store):
    """Append the new object projection without upgrading any old authority."""
    def change(s):
        for name in ('packages','goals','research','repairs','releases','accounting_corrections'):
            s.setdefault(name,{})
        s['migration']['control_state_version']=2
        return {'legacy_permissions':'unchanged','historical_events':'unchanged','pause_intent':'retained'}
    return store.transact('CONTROL_STATE_UPGRADED','control-state-v2',change)
