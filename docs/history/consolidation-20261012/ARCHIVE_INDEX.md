# 历史归档目录索引

这些是逻辑归档，原目录与分支均保留，未删除或改写。

| 来源 | 用途 |
| --- | --- |
| `/home/mint/rl_dev/instinctRL` / main `6f6dee3` | 原正式源与旧远端基线 |
| `/home/mint/rl_dev/worktrees/instinctRL-blackwell` / blackwell-migration `e438c4c` | 旧批准冻结产品 |
| `/home/mint/rl_dev/worktrees/instinctRL-unified-h1-r3-d0011-product` / `903afc5` | 本次完整产品底座 |
| `/home/mint/rl_dev/instinctRL_coordination` | 历史 Order、审核、运行原始证据与 checkpoint |
| `/home/mint/rl_dev/instinctRL_orchestrator` | 独立控制工具开发；未泛化合入其所有脏改动 |
| `.../state/workspaces/instinctrl-r5-stage1-product-pin-fixture-successor-20260929` | 最新 R5 未提交候选来源 |
| 其他 worktrees 与 state/workspaces | 历史冻结源码和任务证据，详见完整 inventory |

旧远端 `a2-r5j-default-off-residual`、`ros2-foxy` 保留；旧 R5J 的 Isaac Training 实现没有混入 Blackwell 产品。
原始证据/checkpoint 文件哈希见本地完整 inventory；旧 ledger 的预整理 SHA 见 BACKUP_INDEX.json。
日常新开发只进入 `/home/mint/rl_dev/instinctRL-dev` 的 main。
