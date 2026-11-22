# 当前工作流

任务 `maverick-v11-init`，2026-11-22。上游 `Printeger/maverick-codex@9e1a07f8f9c654b2b45c2782291d190aa3bc8dac`，首次执行 init。Maverick 主持日常开发，按问题进展切换视角；交付时按实际贡献汇报。下一项开发任务可直接开始。项目研究状态以 `docs/PROJECT_STATUS.md` 为准。

## 静态验证

`python3 .agents/skills/maverick/scripts/maverick.py doctor --project .` 返回 PASS，项目知识 READY；跨学科地图目录与上游逐文件相同。项目级配置仅设置 agents 的启用与并发。

## 原生子 Agent 验证

PASS。在项目新开的 Codex CLI 0.171.0 会话 `01a20399-bd60-7102-858d-d5a09ba85612` 指定两个原生角色。已核对父子线程关系、运行时 agent_role、返回标记与仓库原文。

| 角色 | 实际子线程 ID | 返回标记 | 只读结果 |
| --- | --- | --- | --- |
| mav_hacker | 01a2039a-01d2-7671-afa0-9c0a03976de8 | 49c90f24be14 | `tools/r6_direct/analyze.py:44` 定义 main，第 74 行调用。 |
| mav_inventor | 01a2039a-14b6-7230-9afe-7eece3b19894 | f5e336b95f05 | `control-information.md:21` 的同观测机制解释 `FINAL_R6_RESULTS.md:5` 的天花板：成功计数无法区分策略余量。 |

验收转折：当前对话缺少原生角色选择字段，改在新项目会话验证角色绑定与返回值。贡献记录见 `.maverick/contributions.jsonl`。
