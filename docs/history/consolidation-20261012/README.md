# 2026-10-12 单分支整理记录

用户授权按既定六步方案整理并正常推送 main。本次不构成独立产品验收。

## 来源与冲突处理

完整产品以 `903afc5abd0fbf82297527e5790c2cd40827b0b2` 为底座，保留从旧 main 开始的 124 个提交。
最新 R5 候选来自 `instinctrl-r5-stage1-product-pin-fixture-successor-20260929`，其 HEAD 是 `a081614`，
实际改动主要未提交。仅从它的产品 env、workload 和 control_snapshot 移入所需代码。
通用 orchestrator 仓库的其他脏改动不合入。

R5 产品快照缺少 81 个基线文件，未整目录覆盖。保留 float32 ray 数值修复，
更新两个过时产品测试预期；方法、actor 和物理参数不变。旧期望原文保留在本目录。
工具迁到 `tools/rlflow`，product/workload 共用同一 Git 根。生成新的路径/源码哈希绑定，
复制并重绑本地资产 manifest；旧配置/manifest bytes 保留。旧认证 release 不继承。

历史一次性 host runner、receipt 与 CP2-VAL-001 包测试按原身份归档。
通用 reset、清理、证据、source identity 和 workload 功能测试继续运行；归档原因见 ARCHIVED_TESTS.json。

## 备份与验证

完整整理前备份：`/home/mint/rl_dev/consolidation_backups/20261012-main/`，约 527 MiB，
包含 72 个仓库/工作区的 refs、HEAD/分支、tracked/untracked 文件哈希、staged/unstaged binary diff，
未跟踪源码压缩副本、主要仓库 Git bundles 与原工作区入口。
BACKUP_INDEX.json 指向全部原始 inventory；REPOSITORY_INVENTORY.json.gz 是完整 refs/status 的压缩副本。
PRESERVATION_CHECK.json 证明旧文件 bytes 未变；根工作区入口的更新单独列为授权内例外。

VERIFICATION_SUMMARY.json 汇总最终范围；各轮 stdout/stderr/XML 保留初始失败。
workload-isolated-v1/v2 的 COMMANDS.json 保存准确 argv、解释器、CPU 环境和退出码。
工具的两个 CLI 版本检查失败仍未关闭。产品最终只跑 CPU；初始广回归包含 CUDA 单测的事实已记录。
没有训练或仿真启动，没有新模型调用或依赖安装。

大型原始输出、checkpoint 和旧 research ledger 不导入活动 Git 状态；它们的位置/哈希已保全。
历史报告中的绝对路径仅用于追溯。不得据旧路径 receipt 声称新目录获得验收。
