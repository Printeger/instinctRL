# instinctRL

当前开发使用 `main`。代码、workload、必要工具和项目状态统一在本仓库。

先读 [当前项目状态](docs/PROJECT_STATUS.md) 和 [开发流程](coordination/protocols/SINGLE_BRANCH_WORKFLOW.md)。
产品代码位于 `isaaclab-training/`；`isaac-training/`、ROS 和旧文档保留历史用途。
固定工作目录为 `/home/mint/rl_dev/instinctRL-dev`，运行版本用精确 commit/tag 和只读快照冻结。

迁移后的代码是待独立验收的开发候选。没有新的仿真/训练授权，不能直接恢复历史执行包。

CPU 工作负载检查使用已有解释器，逐模块隔离运行：

```sh
python tools/run_cpu_checks.py --output artifacts/cpu-checks-001
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python tools/check_migration.py
```

解释器路径、完整测试结果、GPU 用例范围和工具 CLI 限制见项目状态。
原项目说明见 [历史 README](docs/history/consolidation-20261012/legacy-root-README.md)。
