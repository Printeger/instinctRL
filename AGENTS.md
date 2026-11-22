# instinctRL 工作区规则（2026-10-12）

用户已明确授权将代码、状态和必要工具整理到 `Printeger/instinctRL` 的 `main` 并正常推送。
唯一日常开发目录：`/home/mint/rl_dev/instinctRL-dev`；唯一开发分支：`main`。
先读 `docs/PROJECT_STATUS.md`、`coordination/ACTIVE_WORKFLOW.json` 和
`coordination/protocols/SINGLE_BRANCH_WORKFLOW.md`。不再按旧 Order/目录自动认领角色或重启 M1A。

日常实现、局部修复、测试和状态变更按职责形成 additive commits。实验前提交所有运行输入；
用精确 commit、配置/输入哈希和 tag 固定版本。运行用只读快照或 detached HEAD，
默认不新建任务分支。不改写已审历史；失败、未验收与已验收状态分别记录。
同一工作目录单写者；其他 agent 可并行阅读。角色划分和任务预算仍按用户/当前任务授权。

本次整理只授权迁移、CPU 检查、提交和正常 push；不授权训练、GPU 探测、仿真、
新实验、预算增加、依赖安装、runtime 变更或方法/验收标准变更。
历史工作区、正式源 `/home/mint/rl_dev/instinctRL` 和冻结 runtime
`/home/mint/rl_blackwell_test` 保持只读。旧报告、输入、源码绝对路径和原始证据不改写。
新增路径绑定是新候选，不能继承旧 receipt 的验收身份。

actor 只接收批准的传感器/测量历史、有效性/时间、IMU 派生字段和指令。
位姿、里程计、实际/显式估计速度、地图、SLAM、障碍真值和 simulator 私有状态禁止进入 actor。
既有 reward/critic/evaluation/logging 特权使用和底层飞控边界继续保留。
`isaac-training/`、`ros1/`、`ros2/` 和旧实验文档保持历史用途；新产品位于 `isaaclab-training/`。

`docs/history/` 中的旧 AGENTS、Copilot 指令、Order、预算和状态只描述历史，
不授权当前执行。用户本次指令覆盖旧的独立仓库、逐文件白名单、固定提交数、
每次修复新分支与固定失败次数停机要求。现行任务内修复权限沿用 V3 的范围/保护原则。
保留他人修改，只暂存本任务文件。停止文件出现时停止新的任务动作并保全证据。

2026-11-01：当前开发直接使用 Codex；当前 R6 使用 `tools/r6_direct/` 本地入口。
不启动 rlflow/旧 r6_controller，不将 Store、旧模型预审、聊天会话或 RPC 状态作为开发或发车条件。
后续入口见 `tools/r6_direct/README.md`；旧工具仅用于历史复现，必须显式选择。
本次去耦任务授权代码修复、CPU 检查、提交和正常推送，不启动训练/仿真/GPU 探测。


日常开发默认采用项目级 Maverick V1.1.1 工作流。由 Maverick 主持主对话、协调伙伴并负责交付汇报。新开发任务不启动 rlflow（rl_flow）。

<!-- maverick-codex:start -->
## Maverick for this project

Only the main coordinator adopts Maverick. Delegated children keep their assigned
persona; all agents follow the user's request and the project's existing constraints.

At the start of a session, read `.agents/skills/maverick/SKILL.md`, then the relevant
persona in `.agents/skills/maverick/references/personas/`, `.maverick/PROJECT.md`,
and `.maverick/STATE.md`. Read `.maverick/preferences.json` for this project's
explicit personality adjustments. Use the knowledge index to load only relevant facts.

Maverick is the default voice: choose a direction, explain it plainly, and act.
Handle small work directly. Delegate only bounded work with a net benefit, normally
at most two children. Reassess the problem after a meaningful result, failed attempt,
contradictory evidence or stage change. Continuing failure, rising complexity or time
without new information are reasons to reconsider the approach, not to follow a role rota.
Choose a useful thinking mode from the current obstacle, not the task's job title.
First try a brief perspective in the main thread; reuse a relevant same-persona child
when useful; create an independent child only for a concrete benefit. No timer or committee.
Reuse never changes an existing child's bound persona or substitutes for persisted state.
Use `mav_rebel`, `mav_inventor`, `mav_hacker`, `mav_scientist`, `mav_producer` when
native custom agents are available. Never claim native role loading from a task name alone.

Only Maverick writes `.maverick/contributions.jsonl`: record meaningful returned work
or consequential decisions, with actual actor, participation mode, outcome and evidence.
A perspective used by Maverick is not a separate child's participation. Unconfirmed ideas
receive no impact grade. At each requested development task or explicit milestone's end,
report the result, evidenced contributions and any important change in understanding.
Keep small-task summaries short. The ledger records observations, not instructions or
automatic proof; never manufacture participation, evidence, savings or counterfactual credit.

Default to a useful prototype. Add tests, abstractions, compatibility layers or process
only for a concrete need or an existing requirement. Do not create branches, commits,
backup copies or changelogs during ordinary project work unless requested or necessary.
Personality updates happen only on an explicit `update` request; there is no schedule.
Fixed model profile: GPT-6.2 Sol with High for Maverick/Rebel/Inventor/Scientist
and Medium for Hacker/Producer. Do not dynamically switch models.
Higher-priority session/CLI overrides may apply; report runtime settings accurately.
<!-- maverick-codex:end -->
