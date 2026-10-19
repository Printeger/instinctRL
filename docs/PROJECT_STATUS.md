# 当前 Goal：R6 鲁棒性与课程学习闭环（控制组件独审通过，集成未完成）

2026-10-20：完整目标未完成，R6完整controller尚未部署，没有R6 live作业。
冻结u640 SHA256为 `aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429`；
基线config/源码/method/runtime可执行文件及地面/UAV资产已完成CPU哈希审计。
主委托、预算和证据位于 `docs/experiments/r6-robustness-auto/`。

自动审核会话已完成5次，剩余27次。首次CLI认证/schema通过但沙箱核查失败，第二次独审提出5项MUST_FIX；
第三次已核验准确修复与新增模型交接路径，返回PASS_CPU_SCOPE/noMUST_FIX。
`reviews/cpu-fix-handoff-review.json`、三次原始receipt/usage/事件完整保留。
独审覆盖31项控制/worker、36项张量/评分/归档和7项交接测试；确认配对噪声、预算保护、
输出写入限额、后代清理、隔离测试账本以及控制器重启接回同一worker，不代表完整无人值守或live准入。
此前误用缺少jsonschema的仿真Python导致的66项合并测试失败记录仍保留；未安装依赖。

新增默认关闭的R6测量接口已接到actor/ICS共用history前，干净测量单独交审计sink，
原critic边界保留。14项CPU接线/测量测试通过（含实际观察方法AST与CPU替身）；
见 `CPU_MEASUREMENT_ADAPTER_TESTS.json`。该新接口未包含在上述独审范围内，尚未live验证。

新增R6物理接口已接到base reset之后与每步旋翼施力之后，默认关闭；按实测刚体质量
分配40tick世界系脉冲，保留旋翼力矩，并检查实际sim时钟/质量漂移及最后一步完成。
实际传感器刷新后核对reset，审计浮点世界坐标精度容差，不改几何评分。
`CPU_PHYSICS_ADAPTER_TESTS.json`记录20项CPU检查通过；这是施力暂存/接线证据，
不是物理积分、轨迹恢复或独审通过。新增物理与测量接口均待合并独审和live验证。

新评价block workload已串联reset、测量归档、6600物理步审计与完整32lane运动评分；
新增独立R6 backend入口拒绝旧R5授权和未冻结协议。5项CPU替身workload测试通过，
另15项实际观察接线/物理接口回归通过，准确输入见 `CPU_EVALUATION_TESTS.json`。
world指令在实际观察边界转换后进入history，保持内外观察与过去command帧一致；
丢帧索引明确定义为观察边界0..steps。真实research-mode发行尚未完成，不能把CPU替身数据称为真实基线结果。

新增 `research-r6-evaluation` 基线入口已接入CLI、参数/预算验证、源码兼容检查和lifecycle
分派；实际u640加载与原五任务→16新块顺序已实现，尚未live。新主委托机器记录
`MASTER_AUTHORITY.json`不代替Store许可，capacity仍false。61项CPU回归通过，见
`CPU_CONTRACT_TESTS.json`；此前2项回归失败保留并修复。入口要求独审冻结协议、容量
报告及精确兼容review，固定193600非训练transitions/21显式reset，拒绝原seed重标、
删域、训练和提前final揭晓。候选训练与最终确认入口仍是后续完整Goal工作，未缩减目标。

协议仍为12目标块及4压力块草案，最终三模型加基线774400非训练transitions。
延迟感知CPU合成估计：独立帧约60.2GiB，新增上一帧无损引用后约54.0GiB；均非最坏界。
23项归档/测量/完整评价替身回归通过，见 `CPU_TEMPORAL_ARCHIVE_TESTS.json`，新codec未独审。
旧训练固定诊断与逐步日志已分开统计；内部最终保护80GiB/24000秒仍待完整容量审核。
完整force sink的18条CPU归档样本精确重建；最终四模型额外估计3.96GiB，测距+force约57.92GiB。
见 `FORCE_ARCHIVE_CAPACITY_CPU.json`；不含其余日志/训练/native/快照/视频，也不是最坏界或live吞吐。
新增 `tools/r6_job_handoff.py` 复用rlflow关闭已提交的准确job：本地等待、STOP单次取消、
normal ACK/所属进程核验、原生失败与全额预留保留。新增19项CPU检查通过，原恢复/模型交接
17项回归通过，见 `CPU_JOB_HANDOFF_TESTS.json`；尚未独审或部署。两份早期模型receipt缺少
新版cleanup字段，仍需单独核对；新入口不会静默当作已清理。资源/submit、分析及
下一计划衔接仍未闭合，本轮未改唯一batch状态或操作Store作业。
新增准确job/attempt原件清单及保守预算观察：全文件哈希、失败原件保留、扫描变化拒绝，
实际消耗超过预留时计入较大值，低于预留不退款。63项CPU回归通过，见
`CPU_JOB_INVENTORY_TESTS.json`；支持清单写入后中断恢复，未改操作账本或Store。
清单不包含全部controller元数据；postrun另按查询事件预留，整体元数据容量仍待补齐；尚未独审。
新增固定两类postrun查询的有界worker/receipt衔接：准确ordinal去重、控制器重启接回、
15秒查询超时、1MiB写前输出限额及所属后代清理。14项CPU假查询检查通过，见
`CPU_POSTRUN_TESTS.json`；没有真实GPU查询。postrun成功不覆盖原件缺失/原生失败。
每查询另预留1MiB+64KiB；该模块尚待独审、准确部署输入冻结及完整下一计划衔接。
第四次独审针对新控制组件提出3项MUST_FIX，原件见 `reviews/cpu-control-review.json`：
job事件别名可重复postrun、超额后已预留事件仍可派发、postrun控制器没有独立截止时间。
已形成修复候选：job唯一绑定、统一硬/保护额度检查、45秒派发截止及有界准确身份清理；
96项CPU回归通过（含实际SIGSTOP进程和无关进程保护），见 `CPU_CONTROL_REPAIR_TESTS.json`。
这不是独审闭环；原MUST_FIX与原始receipt/usage保留，待独立复核，不增加live准入。
第五次独立复核已关闭上述三项缺陷：`reviews/cpu-control-fix-review.json`为PASS_CPU_SCOPE/noMUST_FIX，
准确source/test哈希及实际receipt/usage已核验。此结论只覆盖受审控制组件，不授予完整控制器或live准入。
物理reset/force的实际验证、完整job/resource控制器、容量/协议冻结、exact Store许可及科学验证均未完成。
聊天Goal自动续推理与零模型等待的交接限制尚未解决；不能称后台运行或科学完成。
新增离线运动重放组件：沿原评分器重放原五任务，并逐块重算新增32lane运动评分；保存评分不一致、缺件、身份错误与性能失败分别记录。决策摘要写前限48KiB，完整指标/原件hash另存。CPU替身验证不授予科学验收；准确job/inventory/model绑定、测量/物理/RNG原件核验与controller接线仍未完成。见 `CPU_ANALYSIS_TESTS.json`。
新增准确作业分析绑定入口：`r6_batch --analysis-report ... --analysis-store ...` 可消费已生成的CPU重放报告，核对spec/authorization/attempt/protocol、原件清单及关闭/postrun记录；记录保持幂等和原生失败/控制状态。54项CPU检查通过（含16项新增绑定测试），见 `CPU_ANALYSIS_BINDING_TESTS.json`。模型绑定缺失明确记为缺口；该入口不等于自动分析worker或科学接受，完整controller与下一决策派发仍未部署。
新增流式测量重放：归档clean测量、时间和audit reset mask重建实际送达帧，逐字段bit-exact核对噪声、延迟、丢帧与有效性，并接入CPU分析及作业绑定。发现原实现设备相关RNG无法纯CPU重放，新增 `R6/range-noise/cpu-v1` 候选固定CPU随机生成/加噪，保留独立lane流与无效beam抽样规则；旧身份不继承。完整32×21240beam CPU检查通过，详见 `CPU_MEASUREMENT_REPLAY_TESTS.json`。真实GPU传输/开销、物理传感器与时钟真实性、独审和容量仍未通过；未启动R6 live。
新增物理暂存/reset原件重算：保存每tick姿态、原旋翼wrench、实际dt与末步完成时钟，以及reset世界坐标/dtype；NumPy独立重算世界/局部附加力和叠加，检查撤力、力矩保留、seed/reset/tolerance与最终tick。结果明确 `physical_integration_verified=false`，已接入分析/绑定但未独审或live。见 `CPU_PHYSICS_REPLAY_TESTS.json`。新增字段使最终四模型force归档CPU样本估计增至7,048,886,240bytes（约6.56GiB）；`FORCE_ARCHIVE_CAPACITY_CPU_V2.json`仍非上界，capacity_pass=false，旧估计/原件保留。
完整容量补算与force无损二进制候选：45项CPU检查通过，旧JSON审计仍可读、逐record哈希/fsync/配额保留；`FORCE_ARCHIVE_CAPACITY_CPU_V3.json`最终四模型force样本估计2.54GiB。全control/reward step样本和新版CPU RNG测距已补测；旧训练1.958GB只读无损压缩估计0.411GB，原字节/冻结u640哈希核验，训练压缩writer未实现。`CAPACITY_COMPOSED_DRAFT_V2.json`保留最终774400非训练transitions，提出最终92GiB/27000秒（现80GiB/24000秒不足）；开发示例与最终共127.516GiB，未修改操作账本。原生/训练/元数据/视频精确配额和独审仍未闭合，capacity_pass=false。
下方历史状态不覆盖上述事实。

# 当前完成：R5 W 全原门槛通过（2026-10-19）

**R5_ALL_ORIGINAL_GATES_PASS，最终独审PASS/noMUST_FIX。** 选定R1/u640 SHA `aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429`；u640/u768原seed20260909五任务均32/32，独立seed20261020的u640完整160case/lane重放通过、全部方向4/4、W两墙16/16及真实运动32/32。W最差几何.861944675m、停车2.4s。

E1固定u512+.15m已通过，收益归因控制工程；L1只支持有限q-positive训练和能力保留，未识别独立RL安全增益。独立执行固定初态，所有step数据与原u640相同、T0仅seed不同；原primary保留160个CASE_SEED_MISMATCH，不能称新初态/扰动覆盖、广泛泛化或硬件安全。

七作业原生失败原样保留（四FAILED/-9 INCOMPLETE/null，三prelaunch FINAL0），各正常owned ACK/settle/恰好两postrun；131072训练、131840非训练、256updates、2writes的观察证据不退款全reserve。7start/init、5snapshot、66GiB全输出/快照reserve（含metadata68GiB）在原caps内。资源V2本次live窗口通过，仍只是临时调度规则。

最终报告：`docs/experiments/r5-w-closure-auto/FINAL_R5_W_RESULTS.md/.json`；精确模型/方法/复现manifest、全原件索引、资源与需求审计、W时序图及`reviews/final-independent.json`均已交付。无后续实验计划。唯一controller的`batch-resource-v2/state.json`仍由controller在本次最终回复后记录COMPLETE；本写者未修改它或旧已停止`batch/state.json`。

以下内容保留为历史阶段记录，当时的IN_PROGRESS/尚未执行表述不覆盖上述最终证据。

# instinctRL 当前状态

## 当前 Goal：R5 W 安全训练闭环（2026-10-19）

2026-10-19 用户追加授权修订资源准入并继续原Goal。单次61样本/300.008秒只读观测完成；新V2窗口准入、补充Store授权和W-V2-001阶段已独审登记。保持旧5%入口及旧失败原件，新增exactID仅做u640/seed20261020完整五任务验证。源码bc9f623/configec64b14e冻结为只读新快照，CPU相关60项通过；精确packet独审/发行中，尚未live。旧batch终态保持只读，续接唯一运行目录为 `docs/experiments/r5-w-closure-auto/batch-resource-v2/`，共享原writer.lock及STOP。完整Goal未完成。

用户一次性授权完整诊断、工程修正、近墙训练与原五任务验收，总包络见 [新主委托](experiments/r5-w-closure-auto/GOAL_SCOPE_AND_MASTER_AUTHORITY.md)。旧首段及旧预算记录保持历史身份。

320lane W 原件诊断后，E1固定原R1/u512及ICS额外.15m真实回归已独立核验：原Z/T/S/X/W各32/32、W两墙各16/16、全部lane连续11样本接近达标；这是控制工程收益，不是RL学习。native FAILED/-9、INCOMPLETE/null及32960全额预留保持。

L1原入口 `R5-W-L1-NEARWALL-01` / `J-06a095a1c78d517e` 在prelaunch共存门因GPU2/6/3%中6%超过原5%限制而FAILED/1、FINAL0：训练/更新/Adam/checkpoint均0，无u640/u768。27份原件与归档已独立核验，正常owned ACK、settle、恰好两postrun完成；失败start/init及8GiB输出预留保留。详见 `experiments/r5-w-closure-auto/reviews/l1-failed-actual-independent.json` 与 `L1_RECOVERY_DECISION.json`。

恢复入口 `R5-W-L1-NEARWALL-RECOVERY1` / `J-8bdfaca0488e524d` 已独立接受实际训练与准确u640/u768：原u512全trainer/Adam/RNG恢复，first512 gate先于两optimizer，绝对u513–768、131072训练transitions、256updates、两类各2048Adam、2writes。实际覆盖center98304/x16384/y16384，危险q2004（x1342/y662），ICS干预x1946/y993；全部reward有限。覆盖不是学习效果。53项原件/100原件归档哈希核验，正常owned ACK/settle/恰好2postrun；native FAILED/-9、INCOMPLETE/null、131072训练及8GiB全reserve保持。见 [实际独审](experiments/r5-w-closure-auto/reviews/l1-recovery1-actual-independent.json)。

原两节点评价 `R5-W-L1-EVAL-01` / `J-3db27ce7c40e9542` 已实际完成并独立接受：u640/u768各原Z/T/S/X/W32/32、每方向4/4、两墙16/16、每W lane连续11样本真实接近通过。冻结evaluate_checkpoint逐节点原raw重放完整字典一致。nativeFAILED/-9/INCOMPLETE/null/full65920reserve保留，正常ACK/settle/恰好2postrun。首通过u640，E1固定原u512+.15m已经全通过，因此不宣称L1额外RL安全增益。见 [评价与消融报告](experiments/r5-w-closure-auto/L1_EVALUATION_REPORT.md)。

独立验证原入口 `R5-W-L1-INDEPENDENT-01` / `J-fd7e0e132dca6e7d` 已prelaunch FAILED/1、FINAL0：GPU三样本2/12/3%，12%超过原5%共存门；postrun3%，未建立持续外部阻塞证据。27项原件/归档及packet pins独审PASS，normal owned ACK、settle、恰好两postrun完成；无评价raw，不能报告独立seed通过或科学阴性。失败start/init和8GiB输出预留保留，旧INCOMPLETE/null/full reserves不退。见 [失败独审](experiments/r5-w-closure-auto/reviews/independent01-failure-actual.json)。

同Goal内准备一次有依据的 `R5-W-L1-INDEPENDENT-RECOVERY1`：固定首通过u640、seed20261020、32960非训练/5reset、0训练/写入。仅独立模块增加明确RECOVERY1身份白名单；primary/backend/evaluator/产品与评分门槛不变。首次CPU快照method路径绑定拒绝，原snapshot/config及2GiB预留保全；新r2候选 source032412b/config6a65b6ae已CPU冻结，32项回归PASS。W-V1-002已独审登记，6start/init、64GiB输出封顶在原master内；准确packet独审PASS，许可 `A-35403dcbfe28c1b1` 已合法签发，尚未submit；输入提交/tag/push和计划独审完成后由原controller执行。**完整Goal仍IN_PROGRESS**：独立seed实际全raw重放、最终交付和final_review尚待完成；不能把E1工程收益记为RL增益。唯一controller状态 `docs/experiments/r5-w-closure-auto/batch/state.json` 只读。下文旧任务仅作历史。

最新实际终态：`R5-W-L1-INDEPENDENT-RECOVERY1` / `J-4c3d85af4438c260` 再次prelaunch FAILED/1、FINAL0，GPU6/5/4%，没有评价raw。27项原件及归档、40处pins独审PASS；normal owned ACK、settle、恰好2postrun完成，postrun3%。与原独立入口相隔484.72秒的两次完整窗口均拒绝，支持重复/间歇外部共存阻塞；不证明持续>5%或daemon因果。无资源改善证据时不再盲RECOVERY链，交原controller `STOPPED_EXCEPTION`；**完整Goal未完成，非科学阴性**。旧FAILED/-9、INCOMPLETE/null和全reserve保留，累计6start/init、四snapshot预留；master未耗尽。见 `INDEPENDENT_RECOVERY1_STOP_RESULT.json`、`reviews/independent-recovery1-failure-actual.json`。原独立seed实测、全raw重放和最终科学验收仍待完成。

更新：2026-10-17。唯一日常开发目录 `/home/mint/rl_dev/instinctRL-dev`，唯一开发分支 `main`，
远端 [Printeger/instinctRL](https://github.com/Printeger/instinctRL)。当前精确版本以 `git rev-parse HEAD` 为准。
代码、状态、workload、必要工具和测试已整理到本仓库。源历史完整继承到 `903afc5`，
其后产品集成提交为 `91c6997`，workload/工具集成为 `e7d58d2`。

## 当前任务：固定正式首段已完成，本次结束（2026-10-17）

**独立科学证据审核PASS/noMUST_FIX；完整R5未通过。** R0/R1各完成绝对257–512、131072训练transitions、256更新和5个checkpoint；两组全部五节点Z/T/S/X/W已按原函数从50case×32lane完整重放。全部节点Z32/32；两组u384/u512的T/S/X32/32、各方向4/4；所有节点W0/32、两墙子组0/16，安全距离全部未达标。两组原Z均未回退，但不因此获得追加段授权。R1晚期漂移和跟踪较好，早期学习进展较慢且部分高度/W指标更差；仅限共同origin的单seed aux0/10配对解释。

四个job均已终态、normal owned ACK、两次稳定空扫描无残留、settle和各恰好2postrun。原生FAILED/-9、app.close9.5s、raw PASS_PENDING_CLOSE、collector INCOMPLETE/null及全额保守工作量/输出预留保留，goal_budget.json原样提交。当前无live，不继续训练、评价、模型调用或重试。本次模型上限3、实际调用3（原STOP、恢复独立复审PASS、本次最终分析），历史调用和缺失binding事实不改写。

[最终分析](experiments/r5-formal-first-segment-20261015/FINAL_ANALYSIS.md)、[机器结果](experiments/r5-formal-first-segment-20261015/EXPERIMENT_RESULTS.json)、[独立审核](experiments/r5-formal-first-segment-20261015/reviews/final-scientific-review.json)。唯一controller状态 `coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json` 本次只读：分析时为MODEL:final-analysis，由controller在本次退出并验证提交/push后原子转COMPLETE。完成的是固定首段，非完整非零任务收敛。下一阶段仅建议先分析既有W触发/制动/几何及训练覆盖差异，未执行或改变方法。

### 本段授权与历史进度（保留当时事实）

2026-10-16有限恢复：用户选择优先验收现有成果并完成原定评价。[恢复包](../coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015/recovery-20261016/README.md)已准备两组CPU补充证据及一次额外独立复审，累计模型上限3含旧STOP和原最终分析。修复未来正式入口漏写binding的缺陷，原冻结评价source不改。只有独立接受原始证据、10checkpoint及精确恢复代码后才可发行/执行两组原评价；无新增训练、作业重试或实验扩额。准备状态不等于独立接受；实际进度继续以原唯一state.json为准。

2026-10-16结果补记：R0/R1各完成绝对257–512更新、131072个观察到的训练transitions及5个新checkpoint，均已结算和正常owned ACK清理；原生FAILED/-9、collector INCOMPLETE/null及保守预留保持。独立checkpoint审核因两组原件缺少 `actor_mean_prior_binding.json` 给出STOP；批处理停在 `MODEL:checkpoint-review`，未接受checkpoint、未发行或执行Z/T/S/X/W评价，最终独立模型分析未调用。用户另行要求的[离线训练效果报告](experiments/r5-formal-first-segment-20261015/offline-analysis-20261016/REPORT.md)已生成：非零跟踪改善、R1末段较好但全段略差，均值约束效果不等于悬停达标；另记录末尾episode重复日志问题。该报告不改变原审核与唯一运行状态、不恢复批处理。

用户已明确授权固定R0/R1正式257–512、两组训练后一次独立checkpoint验收/准确评价发行、全部原Z/T/S/X/W评价后一次最终独立科研分析。复用已有准确许可和原Commander守卫，不扩额、不重试、不改方法。运行由tmux中的本地Python等待，Codex不定时轮询。唯一运行状态：`coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json`；[执行说明](../coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015/README.md)、[本轮实际结果目录](experiments/r5-formal-first-segment-20261015/README.md)。同一目录仅controller或它等待的独立模型写入。

## 已完成 Goal（2026-10-15 自主正式训练准入）

**READY_FOR_FORMAL_R5_TRAINING。** 新mean10候选开发预定192/256原Z均32/32，冻结后未用seed20261017/18主256均32/32并独立接受；旧失败seed15/16新方法回归及真实G6有限零/正/反/零、G7同设备原容差重算/重载均已独立核验。最终审核PASS/noMUST_FIX，准确formal首段manifest/两组条件许可/完整启动命令已固定且CPU入口核验通过。当前无live，本Goal未提交任何formal训练，至此停止。完整非零任务性能仍待正式训练评价；旧FAILED/INCOMPLETE及全额保守收费保留。

[最终报告](experiments/r5-formal-readiness-auto/READINESS_REPORT.md)、[READY](experiments/r5-formal-readiness-auto/formal_r5_readiness.json)、[独立审核](experiments/r5-formal-readiness-auto/reviews/final-readiness.json)、[下一次显式入口](experiments/r5-formal-readiness-auto/START_FORMAL_R5.md)、[完成审计](experiments/r5-formal-readiness-auto/COMPLETION_AUDIT.json)。

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
