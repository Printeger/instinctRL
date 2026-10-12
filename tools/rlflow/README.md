# 项目随附的 rlflow 工具

来源为 R5 最新工作区的 control_snapshot；通用 orchestrator 仓库的其他脏改动未合入。
源码及测试迁到此目录，Python 包仍为 rlflow。CPU 环境使用已有
`/home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python`，不安装依赖。
从仓库根运行时设置 `PYTHONPATH=.:tools/rlflow:tools/rlflow/tests`。

这是迁移后的开发候选，未被认证为原历史 release。旧 release.json 已归档；
需要正式运行前必须固定并审核当前工具与输入版本。默认不恢复旧 research state。
