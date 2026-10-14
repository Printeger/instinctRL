# instinctRL 当前状态

更新：2026-10-15。唯一日常开发目录 `/home/mint/rl_dev/instinctRL-dev`，唯一开发分支 `main`，
远端 [Printeger/instinctRL](https://github.com/Printeger/instinctRL)。当前精确版本以 `git rev-parse HEAD` 为准。
代码、状态、workload、必要工具和测试已整理到本仓库。源历史完整继承到 `903afc5`，
其后产品集成提交为 `91c6997`，workload/工具集成为 `e7d58d2`。

## 当前 Goal（2026-10-15 自主正式训练准入）

**IN_PROGRESS，非READY。** 当前唯一LIVE fresh同一新确认seed18主256原Z J-a8dc9cb1b63aa810/A-11c66f5f5fb086ec，准确inputs2b3962e/tag/push/actualAUTH256f5476；同一实际u25647796b6c/source617/config2e9/controlc468，仅6400nontraining、0训练/optimizer/写checkpoint。前资源拒绝保留FINAL0/start/init/output/time，D018有界恢复，≥79.78s退避实录，原门槛不改。旧15训练Role30已独立接受且准确Z A-4f24d896680295b3未提交；首17原Z32/32已接受。只poll本句柄，终态normalcleanup后继续；整体G4未授，formal0。

唯一续接：[GOAL_STATE](experiments/r5-formal-readiness-auto/GOAL_STATE.md)、[总账](experiments/r5-formal-readiness-auto/goal_budget.json)、[证据索引](experiments/r5-formal-readiness-auto/EVIDENCE_INDEX.md)。

## 已完成 Goal（2026-10-14 零命令残差单变量）

**Goal已完成：`VALID_EXPERIMENT_POSITIVE`。** 独立Role6最终PASS，无阻塞性MUST_FIX；全需求审计通过。两组各64updates，真实首512gate先于两类optimizer通过；实际初始化和首输入/动作/oldBeta/logp逐位一致。新方法只比较零命令残差软代价w0/.05，两组actorLR均1e-4。

预注册u64最大lane指标：C0→C1速度RMS **.102369525→.020412797m/s**、水平漂移 **1.959831357→.347792268m**、高度偏差 **.471305609→.095134258m**。在这一个配对seed内可信改善，但C1所有32lane漂移仍>.30m，四个原Z节点均0/32；**未达到原Z准入，不扩256updates**，不宣称泛化或完整R5通过。

用户单次例外后的freshC1补充评估已完成且仅增加12800eval、0训练更新/写checkpoint。五个native FAILED保留（四close/-9，一beforeApp/1），四INCOMPLETE/null保留65536train+25600eval+128updates+14writes，旧FINAL0只结算其未用eval；5start/init、80GiB输出预留、正常ownedACK无残留及各两次postrun不变。Role6/12，0自动重试。旧诊断和P0/PLR历史不改写。

[完整报告](experiments/r5-zero-correction-20261014/PAIR_ANALYSIS.md)、[机器结果](experiments/r5-zero-correction-20261014/pair_analysis.json)、[独立审核](experiments/r5-zero-correction-20261014/reviews/review-zero-final-scientific.json)、[完成审计](experiments/r5-zero-correction-20261014/COMPLETION_AUDIT.json)、[研究决定](experiments/r5-zero-correction-20261014/NEXT_RESEARCH_DECISION.md)。本Goal停止；后续ray-anchor单变量仅提案，未授权或实施。

## 最新完成 Goal（2026-10-13 短程配对实验）

**PAIR_GOAL_SCIENTIFICALLY_COMPLETE_NEGATIVE_Z_RESULTS**。独立Role4最终PASS，无阻塞性MUST_FIX，用户完整目标17项核对通过，已停止。两组各64updates与u16/u64原Z真实完成；共享ef68且只actorLR3e-4/1e-4不同，init和首次512采样逐位相同，真实首次gate均先于两类optimizer通过。

首更新mb8后analyticBetaKL均值P0=15.012778、PLR=0.276192；原Z四节点全部0/32。较低LR大幅减少本对首更新震荡，PLRu64速度/高度满足原门槛但水平漂移1.088824m超0.30m，仍不支持原Z准入或256updates扩展。一对seed不证明一般优劣。

[报告](experiments/r5-paired-pilot-20261013/PAIR_REPORT.md)、[机器结果](experiments/r5-paired-pilot-20261013/pair_results.json)、[唯一证据索引](experiments/r5-paired-pilot-20261013/STATE_AND_EVIDENCE.md)、[最终独立审核](experiments/r5-paired-pilot-20261013/reviews/review-pair-final-scientific.json)。观察训练65536、评估25600、128logicalupdates、14writes；四个nativeclose仍FAILED/-9，normalownedACK无残留，H1INCOMPLETE/null与全部reservation保留。4freshentries/4roles已用，未重做控制诊断，未改方法；不再live或探测。下一阶段CPU科学分解仅为提案，未开始。

## 已完成诊断 Goal（2026-10-12 新授权）

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



最新H256：R0原入口J-98c4c0af48d17d1a在App前原GPU门槛4/10/16%>5失败，FINAL0/无训练数据，正常ACK/settle/2postrun并保留启动/init/output收费。无live。下一R1原准确A-56b453060fda4459退避≥60秒已实录，提交当前状态/输入并push后单次release。R0 fresh RECOVERY1packet1e38a94b.../expectedAUTHb00a62ee...独立Role11预审待回，只能R1终止/正常清理后发行。D010 stage12slots/112GiB，其余资源和master/confirmation/formal保护不变；当前未解决事件1/3，下一R1为attempt2/3，若仍拒第三次至少180秒且不第四次。

当前唯一LIVE：H256R1 J-a4a37087b4d84996，准确A-56b453060fda4459，input44bc785已push后release；原coexistence2/1/2%及真实first512gate PASS，65updates lastobserved。仅poll此handle，终止+正常ACK后settle/postrun/CPU。R0freshRECOVERY1独立Role11条件PASS，实际AUTH仍未发行，须R1终态/清理/结算后才能issue；不重开旧失败。main-only G6/G7核心CPU6+6PASS，尚无真实入口或完成证据。

最新：H256R1 J-a4a37087b4d84996完成256updates/131072train/2048Adam每类/11writes、全4096policy步零命令，CPU训练完整覆盖PASS；nativeFAILED/-9/rawPASS_PENDING_CLOSE/INCOMPLETE fullreservation保留，正常ACK/settle/2postrun完成。无live。R0freshRECOVERY1准确A-6c257ca91ac2a1a0已发行，须提交/tag/push后一次release，不能重开旧R0。G6/G7 main-only CPU核心8/6PASS、warmfirstgate11PASS，未接入新真实入口，不能替代准入。H256R1与H128R1初始化/首512完全相同，但u1actor有1.19e-7差异、后段权重不相等，不宣称跨入口精确训练前缀；原因未由本有限核查定论。

DEV-01仅CPU配置路径绑定拒绝，旧snapshot/packet和2GiB预留保留；DEV-02修正准确method路径，科学方法不变，无旧live失败或预算重置。

最新唯一live mean10 DEV02 R1 J-b8ec45f94a245f6a，真实coexistence和首512gate PASS；53updates已观察。source617af4d/config2e9a7284/tag7b18755，R0准确许可已冻结待串行。无当前方法原Z结果，旧G4失败不变。

Current ONLYLIVE R0 J-b959e2d9ded9ebb3 / A-cb56dae76162565d, inputaa9664b/push; actualfirst512gatePASS. Samefrozen617af4d/config2e9a7284. R1 scientifictrainingcomplete256/normalACK/settle/2postrun/CPUcoveragePASS; Role22actualcp/fullspecfixclosurepending. NoactualZyet; noformalupdates. MainCPUprepares newcompatibleintegration snapshot only, noG6AUTH orrelease.
