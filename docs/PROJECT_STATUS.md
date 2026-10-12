# instinctRL 当前状态

更新：2026-10-12。唯一日常开发目录 `/home/mint/rl_dev/instinctRL-dev`，唯一开发分支 `main`，
远端 [Printeger/instinctRL](https://github.com/Printeger/instinctRL)。当前精确版本以 `git rev-parse HEAD` 为准。
代码、状态、workload、必要工具和测试已整理到本仓库。源历史完整继承到 `903afc5`，
其后产品集成提交为 `91c6997`，workload/工具集成为 `e7d58d2`。

## 当前 Goal（2026-10-12 新授权）

诊断闭环、最小修复与短程训练准入已完成，终态 **`READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`**。需求见 [需求](goals/R5_DIAGNOSTIC_CLOSURE_REQUIREMENTS.md)，唯一证据见 [索引](diagnostics/r5-readiness/STATE_AND_EVIDENCE.md)，[启动包](diagnostics/r5-readiness/PILOT_START_READINESS.md)。独立 Role15 [最终审核](diagnostics/r5-readiness/reviews/review-entry6-final-readiness.json) PASS，无阻塞性 MUST_FIX，当前／历史状态同步已完成。

实测五体总质量／真实 rotor allocation 的最小修复、公开 reset、observer 与诊断持久性均有测试和真实回归。零目标403200物理行已独立验收；独立命令直通 ENTRY6 原 Z/T/S 各32/32、各方向子组4/4通过，保留 sensor-T0、所有启动／安全与原门槛。

ENTRY6 进程仍 FAILED/-9：app.close9.5s超时，原 PASS_PENDING_CLOSE 与 INCOMPLETE/null collector 保留。630完成steps/20160现场记录不替代正式结算，20160仍全额预留；正常 owned ACK 无残留，667原件独立核验。原第四次超时仍原样计账。当前6start/init扣账、cap8；80640nontraining全额保守预留，15角色调用。训练 transitions、optimizer updates、训练 checkpoint writes 均0；本 Goal 停止。

下一轮两组共享 `ef68ebb5682b85aa83044fdea1c13b7d0edbf325`，只比较 actor LR3e-4/1e-4。真实首次512样本概率 gate 留作下一轮任一 optimizer 前的强制门槛；未来训练资源授权和 Commander release 未签发。历史 R4 全部失败原因与未记录惯量仍未知，已界定为不阻塞本次有限准入。

以下历史状态、迁移验证和续接记录只作历史，不继承旧 receipt 验收或执行授权。

## 历史阶段及结论

R5 Stage 1 的 B0 诊断工程修复候选，尚未获得最终独立验收；水平漂移根因仍未证明。
历史 receipt 记录 132 项 CPU 测试通过；这份记录不是迁移后新路径和新候选的验收。
最新独立 Evaluator 输出因缺少完整 MUST_FIX 字段被协议拒绝，任务为 `BUDGET_EXHAUSTED`，
completion 标记仍未通过。本次整理不重置该状态，也不签发新的运行或训练授权。

9 月 28 日真实 enhanced B0 诊断在第一次 reset 边界失败：controller/rotor 状态不可用，
physics rows 和 transitions 均为 0。`app.close` 超时及脱离进程组的 telemetry 子进程问题
也已记录；此次真实运行未证明漂移原因，未开始后续 Stage 2/3。

迁入的 R5 修复包括观测协议、初始 reset 缓存可用性与真实 reset 计账、证据可读边界及
进程后代清理。保留 R3 `903afc5` 的 float32 ray 重建舍入修复，避免旧 R5 快照将其回退。
actor、reward、控制器、质量/COM、物理参数及方法不因整理改变。

## 历史迁移验证

| 检查 | 结果 |
| --- | --- |
| 完整产品树和源码/配置迁移检查 | PASS：3,476 个基线文件保留，328 个 Python 文件可解析，52 个配置源码哈希匹配 |
| 产品 CPU 单测 | 1,264 passed，40 subtests passed；8 项 GPU 用例 skipped，2 项 GPU 必需用例 deselected |
| workload 分模块 CPU 单测 | 239 passed，22 个模块，无失败或跳过 |
| 项目随附 rlflow 工具检查 | 310 passed；2 项安装环境检查失败，CLI 版本与历史固定契约不一致 |
| 旧工作区保全 | 72 个仓库/工作区、219,011 条文件记录核对通过，旧文件无变更 |

产品最终测试强制 `CUDA_VISIBLE_DEVICES=''`。初始广回归包含 CUDA 相关单测，随后改为
强制 CPU 范围；全部初始失败、命令和输出保留。本次未启动仿真或训练，未安装依赖。

rlflow 的隔离协议固定 Codex CLI `0.153.4`，本机为 `0.161.0`，因此不能宣称真实
orchestrator 后端已认证或可直接恢复。未通过修改固定版本或跳过实际安装检查伪造 PASS。

## 历史迁移后的下一步（已由当前 Goal 续接）

先对统一目录上的精确候选补齐独立审核；如需真实诊断，再固定提交/配置/工具/输入哈希，
核对新预算与启动授权。历史失败、未验收和源路径均保留，不继承旧 receipt 的通过状态。
新的常规修复直接在 main 提交；实验使用精确 commit/tag 和只读快照。

当前配置是 CPU 开发候选，runtime 权限为零。旧 SCALE checkpoint 和冻结系统依赖保留
外部路径与哈希用于追溯，不能据此恢复旧训练。`offline_supervision` 保留历史 release 校验，
它不会把迁入且已修改的工具冒充原认证 release。

## 入口与证据

- [现行流程](../coordination/protocols/SINGLE_BRANCH_WORKFLOW.md)、[机器状态](../coordination/PROJECT_STATE.json)、[当前任务](../coordination/CURRENT_TASK.md)。
- [整理报告与索引](history/consolidation-20261012/README.md)、[验证摘要](history/consolidation-20261012/VERIFICATION_SUMMARY.json)。
- [真实诊断失败报告](history/consolidation-20261012/r5-live-failure/B0_ENHANCED_LIVE_ROOT_CAUSE_REPORT.md)。
- [归档清单](history/consolidation-20261012/ARCHIVE_INDEX.md)。旧目录保留，不删除。

2026-10-12 R5 actual first-live review: MUST_FIX LIVE-R1/R2. New measurement-R1
candidate preserves failed snapshot inputs and corrects float32 symmetry contract;
CPU passes, fresh exact source/input prereview pending. First live remains failed,
full20160 reservation retained; no owned survivors. Goal IN_PROGRESS, training0.

当前 Goal 实测更新：measurement-r1b 的 128,640 条物理行及 6,400 条已完成 nontraining transitions 已保全并离线验证。质量来源与实际 rotor allocation 机制有定量证据，最小修复尚待独立预审和真实回归；T 首步 reward coverage 失败及强制 app.close 清理仍为 FAILED，不作为准入。完整失败额度保留，剩余最多两次 live／初始化、40,320 transitions；训练为零。详见当前控制诊断与 Goal 账本。

当前续接（第四次运行中）：最小实测质量／实际 allocation、公开 reset 与精确 actor digest 修复已通过独立预审。第三次入口因继承旧 SCALE 资源产品哈希而在 coexistence／AppLauncher 前失败，现有 collector 正式结算其零 transitions；启动和初始化保守预留仍扣账。资源绑定兼容修复后的第四次 J-afe4bda51f5206e5 已通过真实 coexistence，正在收集 Z/T/S 零目标回归。全部 4 次启动槽已占用，当前保守 transitions 账为 60,480；命令直通的第五次精确提案和未来配对 pilot argv 已生成但未授权。完整轨迹、清理及独立科学验收仍待完成，训练为零。唯一最新索引为 docs/diagnostics/r5-readiness/STATE_AND_EVIDENCE.md。

最新续接：第四次在完整采集后CPU分析期间被910秒正式watchdog终止，FAILED/137；实际wall910.5478907090146秒原样计账。原manifest仍PREPARED、最终reward计数缺失，20160全额预留。全403200物理行与96reset段已被离线和独立Reviewer复核，接受修复后的零目标物理回归；命令直通原Z/T/S尚待验证。ef68ebb新候选保持同一物理基线，仅补分析前持久化和pilot-evaluation评估seed分类，32CPU回归通过。全部4次启动槽已用完；准确第五次提案和仅start/init4→5的申请包已生成，未签发许可。未来两组pilot精确argv位于pilot_future_proposals.r3.json，首次rollout概率gate强制保留。Goal仍IN_PROGRESS，训练为零。
