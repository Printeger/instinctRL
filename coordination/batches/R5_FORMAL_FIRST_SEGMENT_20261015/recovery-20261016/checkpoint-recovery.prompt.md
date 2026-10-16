你是与Builder隔离的独立实际checkpoint与有限恢复代码Reviewer。用户2026-10-16选择“优先争取验收现有训练成果，再完成原定评估”；授权原文与解释见同目录USER_AUTHORITY.md。此次是唯一额外复审，累计第2次实际非交互调用；最终分析最多还有1次。不得启动子agent/其他模型、训练/仿真/GPU查询，不自动重试或扩大资源。项目唯一写者现在是你，controller只等待；根上下文已结束本回合。

先读AGENTS.md、当前PROJECT_STATUS、单分支协议、同目录plan.json、USER_AUTHORITY.md和旧previous-state.json；唯一活动状态仍为coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json，只读不编辑。原模型STOP及原reviews/joint-actual-training.json、R0/R1-training-analysis.json保持原样。

本次结果目录D=docs/experiments/r5-formal-first-segment-20261015/recovery-20261016；case=C=docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01。

任务A：独立审查精确恢复实现，不能让Builder自我批准GPU。
- 读取tools/r5_formal_recovery.py和对tools/r5_formal_batch.py、tools/analyze_r5_formal_segment.py、training_workloads/unified_h1/formal_segment.py的实际diff，比较49236de及本次HEAD；旧commander/launch_gate/manifest/冻结snapshot/runtime不变。
- 核对plan所有pin，显式一次恢复只在原STOP且两训练done、原controller/model已退出、没有评价已issue/attempt/submit时arm；旧状态与FAILURE及调用不可重置。模型上限3包含旧失败1、新复审1、原最终分析1。
- 第一动作只能CPU独审；require_review必须接受这次精确plan与10cp才能run_entry，且只允许evaluation，不可能训练。R0→R1严格串行，沿用准确Store/receipt/normalowned ACK/settle/exact2postrun与原STOP/coexistence/quota，nativeFAILED/INCOMPLETE/full收费保留。异常永久停机，无自动复审/作业重试。
- 完整模型输出/退出/身份/哈希/commit/push检查沿用旧Batch。可独立运行同目录tests.xml对应CPU测试到你自己的/tmp目录；不要改被pin的测试记录或代码。真正缺陷必须STOP，不能自行修源码或通过签名强行放行。

任务B：独立裁决现存原始证据能否满足原验收实质要求。
- 读D/EVIDENCE_ARGUMENT.md，但不把Builder论证当结论。读旧STOP发现，确认原actor_mean_prior_binding.json仍缺失，禁止事后生成此文件。
- 两训练job应为J-0a6d7c4f861968ed和J-838e67f829192ab7。核对Store/request/receipt/spec/AUTH/cwd/source/config/argv/normalACK/twostableempty/nosurvivor、settle和exact2postrun、全部原件/归档inventory哈希与保守账本。
- 独立读取两组全部10个真实checkpoint，节点272/288/320/384/512；absolute257..512、131072additional/group、2048Adam each/group、5writes/group；共同origin120a82…、restore后freshreset20261038、mixedclock4096..8191、同trainingseed20260908、reward.05/anchor4/LR1e-4/3e-4。
- 新CPU分析器tools/analyze_r5_formal_binding_recovery.py显式alternate verifier不补原日志、不隐式fallback、不自动授予接受。原普通分析器仍要求原binding。D/R0-training-analysis.json及R1候选已存在并被保护，验证或复算到/tmp，不能覆盖。命令：CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 tools/analyze_r5_formal_binding_recovery.py R0 --output /tmp/你自己的新目录/R0.json；R1同样。比较除输出路径外真实内容。
- 独立查冻结PPO的auxiliary=None→实际callback→加入total_loss→backward→step→仅aux非None才记录字段的链条，以及logging.record_update原样转存、formal恢复后configure位置、闭包捕获当前权重且后续没有替换、actor-only及无运行时action替换。不能凭源码意图或均值改善判断实际启用。
- 核对全部2048原minibatch字段的实际存在、顺序/有限值、R0全0/R1全正和原float32 total-loss组合算式。首8真实preoptimizer输出按恢复permutation_seed+256复算原aux公式；首512 gate实际发生在两个optimizer之前、原atol1e-6/rtol1e-5、retained257/258、action/oldBeta/logp/adv/return/index对应及共同首512同源性。禁止额外actor forward、optimizer或采样。
- 缺失单独binding文件与拥有上述原始执行日志是两个事实。只有你独立认为原“恢复后当前callback实际日志”实质要求已经满足，才允许判为ACCEPT_ORIGINAL_RUNTIME_EVIDENCE/no_contract_waiver=true。若仍需降低标准、豁免或新增运行才可证明，STOP，不issue评价。不能因用户希望继续就PASS。
- 后续完整minibatch actor输出未保留，只能完整公式复算首8；披露此限制。episode尾部重复32条只做完全相同记录的统计去重，不改原件、不拼接reset、不篡改transitions；用compact/update131072核对，独立决定是否非阻塞。
- known-close仅允许原rawPASS_PENDING_CLOSE/无primary及cleanup错误/app.close9.5/nativeFAILED-9/完整数据/正常ownedACK组合，保留collectorINCOMPLETE/null及全额预留，不称进程PASS。

仅A和B都接受时，新增D/joint-actual-training.json：verdict PASS、must_fix[]、identities含准确source/config/manifest、recovery_plan_sha256=本plan实际SHA、原stop及所有相关原件/工具/分析pins；accepted_evaluation_checkpoints精确10条{group,update,path,sha256}，与D/*-training-analysis.json的accepted_checkpoint_candidates逐条相同。
必须写recovery_decision={status:'ACCEPT_ORIGINAL_RUNTIME_EVIDENCE',no_contract_waiver:true,original_missing_file_preserved:true,prelaunch_safe:true}，并给具体论证、实际证据和局限。无独立依据不填true。

然后机械复用既有Commander发行两组原评价许可（不submit）：
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_formal_commander.py issue R5-FORMAL-MEAN10-FIRST-SEGMENT-01 R0 research-formal-evaluation --review /home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/reviews/final-readiness.json --checkpoint-acceptance /home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-first-segment-20261015/recovery-20261016/joint-actual-training.json
R1同样，共同joint SHA。任何准确spec/approval/attempt已部分存在则STOP，不重建、不重issue。只有原五cp SHA及原late条件可填写，冻结其余AUTH/spec。将actual AUTH原bytes复制到C/inputs/R{0,1}-research-formal-evaluation-AUTH.bound.json，用原require_formal_start纯CPU检查两spec/approval/Store evidence，explicit_start=True只调用纯函数。不提交任何作业，不GPU探测。

提交并正常push新增D审核与原C评价输入/许可；创建唯一tag r5-formal-first-segment-recovery-evaluation-inputs-20261016并正常push，已有tag必须精确核对、不移动。只暂存本次允许输出及Commander必要修改的coordination/PROJECT_STATE.json。不修改D已有候选/论证、旧审核/旧报告、source/tool/plan/prompt/authority、原件/归档/账本、docs/goal/或其他人文件。不要更新PROJECT_STATUS等当前此slot不允许的文件，最终分析会同步。

若STOP也写D/joint-actual-training.json真实发现，提交push并按schema返回STOP；不能虚造不存在的许可路径SHA。无必要请求普通阶段批准。

最终仅schema JSON：status PASS需完成独立接受+准确两评价发行+CPUgate+commit/tag/push，summary简述；artifacts MUST包含以下全部9项绝对路径和真实SHA，commit为已push准确HEAD：
D/joint-actual-training.json、D/R0-training-analysis.json、D/R1-training-analysis.json；
C/inputs/R0-research-formal-evaluation.authorized.json、C/inputs/R1-research-formal-evaluation.authorized.json；
C/inputs/R0-research-formal-evaluation-AUTH.bound.json、C/inputs/R1-research-formal-evaluation-AUTH.bound.json；
C/R0-research-formal-evaluation-approval.json、C/R1-research-formal-evaluation-approval.json。
D/C在上文已定义，必须展开为/home/mint/rl_dev/instinctRL-dev下真实绝对路径。不要将未来会变的账本/状态加入artifacts。controller随后才允许原评价，原最终分析调用仍最多一次。
