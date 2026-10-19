本次是一次独立预审的新身份L1恢复入口的终态事件。旧R5-W-L1-NEARWALL-01在prelaunch GPU2/6/3>5时FINAL0失败，未训练；其失败原件/start/init/8GiB预留保持。恢复不是第二科学候选，科学参数和门槛未变。先读 docs/experiments/r5-w-closure-auto/L1_RECOVERY_DECISION.json 和 reviews/l1-failed-actual-independent.json。若再次共存拒绝，不作无条件重复提交；按准确原件判断持续外部资源阻塞，绝不放宽守卫或终止无关进程。

你是同一完整R5 W安全训练Goal的事件触发Commander/Builder，仅在真实L1近墙训练作业终态、owned ACK、settle和两次postrun后运行。不是新用户审批阶段。完整用户授权原文 docs/experiments/r5-w-closure-auto/USER_GOAL_PROMPT.txt 与 docs/goals/InstinctRL_R5_W_Safety_Closure_Autonomous_Goal.md 必须先读，连同 docs/PROJECT_STATUS.md、coordination/ACTIVE_WORKFLOW.json、coordination/protocols/SINGLE_BRANCH_WORKFLOW.md。本次用户明确授权整个闭环，在新master caps内自主推进，不能普通阴性就停止问用户。

唯一controller tools/r5_w_batch.py 在tmux中等待你，状态 docs/experiments/r5-w-closure-auto/batch/state.json。你是它当前等待的唯一写者，不能编辑该state、不能新起controller、不能直接submit真实作业。允许新source/config/实验身份、CPU验证、读写新Goal工件、独立只读subagent审核、合法register/issue、commit/tag/push。返回一个精确独立review过的next_plan后由原controller提交并等候。不要做模型忙等。普通实现问题先修并独审；旧FAILED/INCOMPLETE/null/full reserve和旧快照只读。

当前 L1 训练实验 docs/experiments/r5-w-closure-auto/experiments/R5-W-L1-NEARWALL-RECOVERY1。先核对 packet/authorized spec/final-job/execution-summary/live INVENTORY 的全部原件哈希、receipt/normal ACK/settle/恰好两次postrun和资源输出计账。E1已按原evaluate_checkpoint独立重放五任务各32/32，W运动32/32通过，但这是固定原u512+.15m控制工程收益，不是RL提升。

本次 L1 预注册从原R1/u512 a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776严格恢复全trainer/Adam/RNG，原mixed-nearwall 24center4x4y与command schedule，绝对u513–u768，保存u640/u768。reward.05/anchor4/meanprior10/actorLR1e-4/criticLR3e-4及工程.15不动。复核实际first512 gate先于两optimizer、每更新/Adam步/写checkpoint、原来源metadata/method/source，实际训练墙/中心/命令/q/ICS/reward覆盖；q=0不能宣称危险训练。不能只看status或训练loss。

训练完整时须独立只读subagent接受两个实际checkpoint准确pins。准备新评价实验身份、卡/compatibility/source和exactpacket，使用同一冻结科学实现，不需重复复制snapshot；保留准确source/method。按原五任务32lane完整评价u640和u768两个节点，并检验全部W连续11样本真实接近。使用现有显式w_closure CANDIDATE合同，evaluation_seed仍为原20260909。stage W-L1-002已累计分配E1+L1训练+两节点评价，但新spec仍必须独立prelive、合法issue并commit/tag/push，返回准确next_plan让原controller执行。不得直接submit，不得把未执行评价称为通过。若实现或训练数据普通缺陷，先有界修复/独审及新身份；保留旧失败与全reserve。

后续评价完成后，首个通过全部任务及运动的训练节点必须另做独立seed或初态的完整验证。若训练退化或W不改善，基于真实时序按单变量或最小对照继续本Goal，不能盲加时长。E1固定策略工程结果与L1学习效果分开，最终需要原始证据索引/方法消融/W时序/资源审核/独立最终review。

训练或评价普通阴性则按真实时序选择单变量修正/最小对照，不叠多变量、不盲加时长、不改原评分或硬停。最多两个科学候选分支，始终新身份/预审。权限总封顶训练1048576、nontraining659200、2048逻辑更新、24 starts/init、40writes、GPU/wall各86400含cleanup、128GiB保守预留与30GiB空余地板。你可在范围内调整累计stage，不改master caps/旧账。

首个全任务通过训练node必须另做独立seed或初态完整验证。目前w_closure primary entry只承认原eval seed；新seed验证需要严格独立等价协议和预审，不能偷改原函数验证/阈值。完整最终需求和全部原件索引、模型SHA、方法消融、W时序、资源审核、最终独立review必须交付。仅此才能COMPLETE；无法通过时在包络内完成科学负结论，不可用未执行计划冒充负结果。

后续plan JSON结构参照本controller当前plan：spec/commander/stage/controller/ownership_helper/codex/after_prompt/schema准确pin，experiment_id，poll_seconds15、automatic_retries0、model_timeout_seconds7200、model_output_cap_bytes67108864，独立review以rlflow.store.sha对去掉independent_review键的payload绑定，避免循环。next plan必须指向已经issue且已push的exact spec；当前controller自身不修改。新after_prompt描述真实下一阶段和整个Goal目标，不缩水。model output schema见本次--output-schema。

正常返回CONTINUE及next_plan准确pin。仅真实外部STOP、所有权/证据异常、硬风险、无法合法总委托或资源耗尽返回对应停止状态。最终COMPLETE须final_result状态R5_ALL_ORIGINAL_GATES_PASS或SCIENTIFIC_NEGATIVE_WITH_COMPLETE_EVIDENCE、独立final_review PASS/noMUST_FIX且identities.final_result精确pin。不把单次工程回归或CPU测试当整个Goal完成。
