# 无人机策略与 R6 证据

Status: READY

- **可观测性。** Actor 使用获批的传感器及测量历史、时间与有效性、IMU 派生量和指令。位姿、里程计、实际或显式估计速度、地图、SLAM、障碍真值、仿真私有状态属于禁入信息。审查输入改动时沿 actor 路径检查；reward、critic、评价和日志各有既定的特权用途。依据为根 `AGENTS.md`。
- **运行与复核。** R6 使用精确快照与输入绑定；`tools/r6_direct/run.py` 接收运行契约，`tools/r6_direct/analyze.py` 依据已关闭作业的原件复核。CPU 接线、作业回执和独立科学验收回答不同问题。参见 `tools/r6_direct/README.md`。
- **结果与未知项。** R6 基线和三个候选在已测条件都达到了 32/32。现有评价无法区分学习增益，也没有确立等价性或对新条件的泛化；录像来自记录姿态回放。依据为 `docs/experiments/r6-direct-training/FINAL_R6_RESULTS.md` 和 `docs/PROJECT_STATUS.md`。
- **Inventor 的资料入口。** 项目约束与上述领域证据配合跨学科地图使用，见 `.agents/skills/maverick/references/knowledge/INDEX.md`。从卡片借来的机制仍需独立证据验证。
