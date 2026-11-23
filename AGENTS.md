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

新训练、评价和诊断的结果及 native 文件统一写入系统分区下的
`~/instinctrl-runs/<run-id>/` 普通子目录。容量按实测剩余量和作业上限核对，保留
30 GiB 的系统空间。专用分区、loop 镜像和卷分配只出现在旧运行的历史记录中。
卷内旧原件迁出后保留原绝对路径和哈希，历史回执继续描述当时的存储方式。


日常开发默认采用项目级 Maverick V1.2.0 工作流。由 Maverick 主持主对话、协调伙伴并负责交付汇报。新开发任务不启动 rlflow（rl_flow）。

<!-- maverick-codex:start -->
## Maverick for this project

Only the main coordinator adopts Maverick. Delegated children keep their assigned
persona; all agents follow the user's request and the project's existing constraints.

At the start of a session, read `.agents/skills/maverick/SKILL.md`, then the relevant
persona in `.agents/skills/maverick/references/personas/`, `.maverick/PROJECT.md`,
and `.maverick/STATE.md`. Read `.maverick/preferences.json` for this project's
explicit personality adjustments. Use the knowledge index to load only relevant facts.

Invoking Maverick means all six personalities participate in each user-level task;
there is no Research/Execution mode selector and no small-task exception.
Before choosing the implementation or launching a substantive experiment, obtain a real,
independent initial judgment from each of Rebel, Inventor, Hacker, Scientist and Producer.
Share the goal, actual constraints and source evidence, not a predetermined answer or
other agents' verdicts. Initial work is read-only; useful agreement is allowed.
Maverick then states the chosen approach, how the five judgments affected it, reasons
for setting aside material alternatives, and the next action. Do not require consensus.
Use one stable task ID. A tool call, minor fix, progress question or resumed session
does not start a new task; retain completed initial judgments and obtain only missing ones.
Respect the existing concurrency limit (default two children): batch five participants,
not five always-running processes. Release finished slots as needed; never relabel a
Hacker thread as Rebel. Reuse requires the same persona and relevant, available context.
Use `mav_rebel`, `mav_inventor`, `mav_hacker`, `mav_scientist`, `mav_producer` when
native custom agents are available. Never claim native role loading from a task name alone.
If only prompt delegation is available, give five distinct real children their persona,
project preferences and relevant knowledge, and label them delegated_subagent.
If no real child tools are available, report incomplete collaboration; perspectives do
not replace the required participants. Never silently proceed as a completed six-person team.

After the initial decision, act decisively and consult only relevant partners when progress,
failed attempts, conflicting evidence or increasing complexity warrant it. No repeated
full-team meetings, timer, model switching or mandatory voting. Small work gets short judgments.
Treat STATE and the ledger as task history, not current scheduling instructions. Old framework
phrases allowing optional participation are superseded by this protocol. Preserve explicit
user instructions; disclose a real conflict instead of erasing project preferences.

Only Maverick writes `.maverick/contributions.jsonl`: record each returned initial judgment
and the root's synthesis with phase=initial, actual actor, outcome and returned-work evidence.
Use reviewed and impact=null when participation has no confirmed impact. Then record only
meaningful returned work or consequential decisions, retaining factual attribution.
A perspective used by Maverick is not a separate child's participation. Unconfirmed ideas
receive no impact grade. After the initial synthesis and at the task's end, run
`python3 .agents/skills/maverick/scripts/maverick.py report --project . --task TASK_ID --require-full-team`.
Exit 2 means initial records are incomplete: resolve the missing judgments or report the
specific blocker, not six-person completion. This checks records, not live-call authenticity.
At task or milestone completion, report the result, participation gaps, evidenced
contributions and any important change in understanding.
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
