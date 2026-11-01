# R6 接续 Goal 启动 prompt

状态：供用户复制后启动；保存此文件不等于已启动Goal，不授予新预算。
先处理方案中管理员只读访问条件及预算缺口。下面整段可作为新的Codex任务消息；不需要rlflow或旧controller。

```text
请创建并执行一个Goal：接续完成InstinctRL R6，从查清已关闭的R6-DIRECT-TRAIN-01开始，完成与u640的既定评价、按证据决定后续分段训练、冻结配方后的三随机流续训确认、最终保留集、独立审核以及模型/录像交付。

必须先读：
AGENTS.md
 docs/PROJECT_STATUS.md
 coordination/ACTIVE_WORKFLOW.json
 coordination/protocols/SINGLE_BRANCH_WORKFLOW.md
 docs/goals/InstinctRL_R6_Continuation_Requirements.md
 docs/goals/InstinctRL_R6_Continuation_Plan.md
 docs/goals/InstinctRL_R6_Standalone_Training_Requirements.md
 tools/r6_direct/README.md
并核对这些文件的最新版本。旧文件的“尚未启动/PPO为0”是历史状态，不得据此重跑TRAIN-01。

本消息授权在上述任务和原剩余预算内，执行必要只读查证、CPU检查、局部工程修复、提交及正常push；在首段核验、预算及精确输入全部通过后，执行本地训练、评价、录像渲染和最终交付。普通阶段衔接不反复请求确认。管理员权限不足、STOP、预算或方法范围改变仍须明确处理，不绕过。不安装依赖、不修改冻结runtime、不改变actor/reward/PPO/ICS和科学验收标准。本消息不增加任何资源上限，也不允许动用不足的最终保护额度。

按以下顺序执行：
1. 查证原输入/receipt及准确进程退出。只读恢复原始结果可见性，核对实际update/transition、raw、模型、Adam、RNG、counter、课程和概率gate。分别记录运行退出异常与科学数据完整性。完整则转评价；部分完整则仅从可验证恢复点创建新attempt；无恢复点则记录损失及重算费用，不复用原run ID。禁止执行旧TRAIN_RUN_INPUTS_01.json来重启训练。
2. 核对RESOURCE_LEDGER、MODEL_USAGE_PENDING_TRAIN_01与MODEL_BUDGET_BOUNDARY；待记费用只结算一次，保留未知消耗及未释放预留。若506/512与保护8的缺口仍成立，报告所需外部预算决定，不自行改cap、不用模型循环等待。
3. 与冻结u640按既定协议/条件/随机流做配对评价，保存原五任务回归及16开发条件结果，补基线科学审核。确认提升、无收益或退化，不以loss或录像替代实测。
4. 依评价预登记必要的有界追加段或课程，逐段评价；先保留最终确认/交付额度。首个可执行作业准备好时生成NEXT_ACTION.json与准确START_NEXT_RUN.sh，报告READY_TO_LAUNCH里程碑及启动/查看/停止方式；继续本Goal已授权的执行，不能把M0当整个Goal完成。
5. 冻结最终配方，进行共同u640起点的三个实际不同随机流续训确认；模型冻结和承诺发布后才揭晓保留集；完成基线+三个模型共同最终评价，不能用最终测试调参。
6. 完成独立科学审核、需求逐项审计、模型/配置/源码hash、训练曲线、原始证据索引和录像交付，核对预算及全部所属进程清理，提交并正常push。

只用tools/r6_direct本地直接链。每次新运行先提交精确输入/source/config/hash/tag并建立只读快照或detached HEAD，不把当前main代码冒充旧快照。保留旧报告、raw、模型和receipt原件以及他人修改。仅main工作区一个写者；独立科学审核可使用只读审阅者，审核不能自签，缺失时明确待补。

等待由普通Python监督，不调用模型轮询。不依赖旧Store、旧审核hash、聊天RPC或rlflow状态。遇到需要新科学判断才做一次有用分析；服务不可用不伪造审核通过。任何运行失败保全证据，定位并针对性修复后以新身份继续，不原参数盲重试。

Goal完成仅限全部必要训练/评价、三流确认、保留集、独审与模型录像交付齐全。允许真实科学负结论，不允许用预算耗尽、仅CPU通过、后台启动或READY_TO_LAUNCH代替完成。阻塞时据实列出已完成、剩余事项及具体外部动作；遵循Goal工具的状态规则，不伪造完成/暂停。
```
