# 当前任务

`third-training-anchor`，2026-11-22。Maverick 主持。R6 复盘和开发原件重算完成。新保留集的主指标拟选延迟加丢帧停令后的有效锚 MAE。R6 开发 u640→u768 在此项从 50.13 到 57.51 mm，两者均 32/32。候选只将锚奖励 4→8，拟和共同 u640、三条等量原配方续训比较。见 [计划](../docs/experiments/third-training/PLAN.md)。

已实现直接入口的第三轮独立限额接线和 CPU 指标重算。CPU 检查覆盖锚有效性、checkpoint 格式、进程关闭与 actor 边界。有一项旧飞控跨设备测试要求 CUDA，保留在 CUDA 屏蔽环境下的失败记录并复核其余 CPU 项。原固定大卷草案已撤下；先核对原件的末次消费者和精简包复算，再按单卷峰值确认本轮资源，未启动仿真或新训练。下一步是 u640→u642→u646 的两段真实恢复与关闭预检，之后才冻结正式比较。

# 历史初始化工作流

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
