# 本次执行授权（当前用户消息原文，2026-10-15）

来源：当前会话用户的新任务授权。此记录不是外部签名，也不修改旧授权的历史事实。

请为 InstinctRL 当前已批准的 R5 正式第一阶段实现并启动一个“零模型轮询”的本地无人值守执行流程。

工作目录：/home/mint/rl_dev/instinctRL-dev
首先阅读：
- docs/experiments/r5-formal-readiness-auto/READINESS_REPORT.md
- docs/experiments/r5-formal-readiness-auto/START_FORMAL_R5.md
- docs/experiments/r5-formal-readiness-auto/FORMAL_R5_TRAINING_PLAN.md
- docs/experiments/r5-formal-readiness-auto/formal_r5_launch_manifest.json
- coordination/protocols/SINGLE_BRANCH_WORKFLOW.md

目标：完成 R0/R1 两组正式257–512训练，训练结束后进行现有独立 checkpoint 验收与全部原 Z/T/S/X/W 评估，最后生成一次真实结果分析。

强制要求：
1. 不重写训练器、不调 reward/PPO、不新建调度框架。复用 tools/r5_formal_commander.py 的准确 submit/settle/postrun 和现有全部守卫。
2. 实现可恢复的最小本地批处理器，tmux 后台执行，依据精确 job ID 的 Store/receipt/owned ACK 等待。轮询只能由 Python/shell 执行，不允许 Codex 每隔 N 秒调用工具、询问任务状态或输出进度。
3. R0 训练终态与正常清理之后，自动衔接 R1；绝不同时启动或无授权重试。保留原 native FAILED、collector INCOMPLETE 和保守计账语义。
4. 训练两组均结算后，才启动一次 codex exec 对真正 checkpoint 做独立审核、生成并准确授权已有的评估输入。仅凭训练完成标志不能直接发起评估。
5. 评估两组均完成、所有权清理和原始证据独立核对后，再且只再调用一次 codex exec 进行最终科研分析。每次非交互调用的 prompt、退出码、输出路径与摘要均须持久化。
6. 建立唯一状态 JSON、详细日志、tmux 会话、两次独立模型调用的 prompt 文件、失败停机输出；所有转移原子写入，确保已有 job 永不重复 submit。
7. 复用已签发 exact formal approval 及下一次显式 --start-formal 入口。本消息授权按已冻结输入执行这次正式第一阶段及其独立验收/正式评估，不授权任何新的实验变量、自动扩额或重试。
8. 检查真实 CLI/API/权限后，自主完成代码、CPU 单测、审查、必要的正常 commit/tag/push、一次发车。发车后结束当前 Codex 回合，返回 tmux 会话名、控制进程 PID、唯一状态文件、日志命令和自动分析结果路径。不要在 Codex 回合中阻塞等待整个训练。

如有真实 STOP、资源/所有权/授权异常，请按既有机制停机并记录。不要请求用户参与普通阶段衔接。
