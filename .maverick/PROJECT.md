# instinctRL

Status: READY

本仓库维护 Isaac Lab 无人机强化学习产品、运行输入、验证工具和实验证据。产品开发目录为 `isaaclab-training/`。日常开发在 `/home/mint/rl_dev/instinctRL-dev` 的 `main` 上进行，由 Maverick 主持对话和协调；同一工作目录只设一名写入者。

进入项目先读 `docs/PROJECT_STATUS.md`、`coordination/ACTIVE_WORKFLOW.json` 和 `coordination/protocols/SINGLE_BRANCH_WORKFLOW.md`。根 `AGENTS.md` 规定 actor 边界、只读历史源与冻结 runtime、提交及停止规则。按本次用户授权确定执行范围；安装工作流只做 CPU 验证。

日常 CPU 检查入口为 `python -m tools.run_cpu_checks --output NEW_DIR`。R6 的运行契约、执行和关闭后分析入口见 `tools/r6_direct/README.md`。新开发默认走 Maverick，旧 rlflow 仅供历史复现。实验前冻结源码、配置、输入哈希和 tag，区分失败、候选与验收证据。
