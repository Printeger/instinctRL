# 本地开发与 R6 后续入口

2026-11-01：日常开发直接使用 Codex。运行由普通 Python 执行，等待不调用模型。
不使用 rlflow、Store、旧 controller、逐作业模型预审或聊天会话/RPC 发车条件。
历史代码、原件和预算保留；不得使用旧入口给新结果补造历史验收身份。

## 当前运行链

| 步骤 | 本地入口 | 保留的检查 |
| --- | --- | --- |
| 基线、训练、候选评价、最终评价 | `python -m tools.r6_direct.run --inputs INPUTS.json` | 精确快照、输入/资产哈希、预算、STOP、单实例、资源准入、限时清理 |
| 关闭作业的 CPU 分析 | `python -m tools.r6_direct.analyze --inputs INPUTS.json --destination NEW_DIR` | 真实回执、进程退出、完整计数、原件重算、checkpoint/RNG/概率 gate |
| 构造运行契约 | `python -m tools.r6_direct.prepare contract --request REQUEST.json --destination NEW_CONTRACT.json` | 明确的 `baseline/train/evaluate/final/render` 类型；不发行旧 authorization |
| 三流确认计划 | 同上，将 `contract` 换成 `confirmation-plans` | 原配方冻结、独立科学审核、三独立随机流、共同 u640、完整终点 |
| 最终模型冻结、最终测试批次 | 同上，使用 `final-freeze`、`final-batch` | 三条完整训练证据、终点选择规则、真实 Git 发布及原 holdout commitment/reveal |
| 最终汇总与录像源选择 | `python -m tools.r6_direct.collect --request REQUEST.json --destination NEW_DIR` | 四模型共同测试流、全条件、原评分、完整原件；保留失败结果 |
| 录像轨迹提取 | `python -m tools.prepare_r6_video_trajectories SELECTION.json --sha256 SHA OUTPUT_DIR` | 实际保存的姿态/时序、完整轨迹、输入哈希；不生成新物理结果 |

这些命令是接口说明，不是本次启动许可。`INPUTS.json` 仍使用原 direct-run schema，
必须绑定已提交的准确 source/contract/快照、物理输出卷与前一次关闭回执；不能直接用
旧快照运行本轮新增代码。`run` 使用支持 pidfd 的 `/usr/bin/python3` 监督，
worker/科学 CPU 分析使用已有科学环境 Python；不安装依赖。
`PYTHONPATH` 仅包含仓库根及 `isaaclab-training`，不包含 `tools/rlflow`。

`contract` 的 request 为 `{"operation": "evaluate", "parameters": {...}}`。
`evaluate` 参数沿用原基线参数，增加 `evaluation_selection`，使用
`r6-local-evaluation-selection/1`：角色、固定比较、checkpoint、method、training_plan、
training_analysis、baseline_bindings；不含逐节点 `independent_review`。
训练候选的 `training_analysis` 指向 `analyze` 生成的 `binding.json`。
严格模型/Adam 恢复和原评分共用既有科学实现；本地分析不能代替最终独立科学验收。

`confirmation-plans` request 字段为 `freeze_pin, baseline_bindings, experiment_ids,
wall_seconds, storage`。freeze 仍含原科学审核和配对评价证据；计划使用
`r6-local-training-plan/1`，不要求旧 recipe_review 作业签发。
所有将要发车的 experiment_id 使用 `R6-DIRECT-` 前缀。
`final-freeze` request 为 `recipe_freeze, baseline_bindings, analyses`，analyses 为按原
confirmation seed 顺序排列的三个本地训练 binding pin。
`final-batch` request 为 `freeze, freeze_review, publication, publication_commit, realization`。
先冻结配方/模型并发布承诺，再揭晓最终随机流；沿用
`robustness_holdout.commitment/reveal`，不使用最终测试调参。
最终选择为 `{"schema_version":"r6-local-final-selection/1","batch":PIN,"assignment_index":0}`；
0–3 依次是 baseline、seed1、seed2、seed3。最终 `collect` request 为 `batch_pin, analyses`，
analyses 按相同顺序排列四份最终评价 binding pin。汇总仍标记独审待补。

## 已解除的依赖与保留的边界

本地运行使用自己的 writer.lock 和已导入的预算；不再每次读取旧 controller state、
current_job/current_model、审核 hash 或旧账本原件。已存在的旧 writer.lock 仅用于
防止历史程序与新程序同时运行；不要求旧目录或旧状态文件存在。历史预算、未释放
预留和全部 STOP 继续有效，不以去耦获得新的实验额度。

入口阻止导入 rlflow；真实 worker 仍保存加载模块审计。分析、最终汇总、录像解码与
底层渲染函数已改用独立证据工具，不再间接导入旧恢复/调度器。
旧 `tools/r6_controller*`、`r6_cpu_analysis`、`r6_video_extraction/validation`、
`run_r6_render` 等控制入口只用于历史复现，不能作为当前直接链的后续步骤。
录像渲染通过同一 `run` 入口的 `render` 契约调用共享渲染核心，复用同一资源准入、
输出限额与有界清理；不创建训练环境、不调用优化器。render 参数为
`experiment_id, extraction, wall_seconds, output_cap_bytes`，extraction 指向轨迹提取的 result.json。
同一 `analyze` 入口执行已有视频完整解码检查。本轮仅检查 CPU 接线，未启动渲染或 GPU。

日常验证使用 `python -m tools.run_cpu_checks --output NEW_DIR`：默认检查本地直接链，
按系统进程测试/科学测试选择现有 Python，不注入 rlflow 路径、不使用旧 orchestrator 虚拟环境。
pytest 默认收集范围也改为直接链；其他科学模块可显式指定路径。
确需复现历史调度器测试时显式指定 `tools/run_cpu_checks.py --legacy-rlflow`。
测试的合成回执/模型、纯 CPU 检查不代表真实训练、仿真或科学验收。

## 接续已有训练之前

本地 `R6-DIRECT-TRAIN-01` 回执已关闭并记录 cleanup timeout；旧首页“未启动”已过时。
先恢复结果卷的只读可见性，核验完整计数与 checkpoint，再运行 CPU 分析。
本轮未挂载卷、未修改旧 ledger/raw/receipt、未消费旧实验预算，也未创建新的实验输入或发车。
