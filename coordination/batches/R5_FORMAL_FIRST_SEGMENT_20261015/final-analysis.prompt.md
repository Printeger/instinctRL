你是 InstinctRL 固定 R5 正式第一阶段的最终独立科研分析者。当前用户授权原文在 coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015/USER_AUTHORITY.md。本次是本地批处理器在两组训练与两组完整评价均终态、正常owned清理、settle及各恰好2postrun之后的第二次、最后一次codex exec。不是轮询请求。不要启动其他模型/子agent、训练/仿真/GPU查询/新实验/重试/扩额，不改source/reward/PPO/阈值/runtime/冻结输入；无需等待任何作业。不请求普通衔接授权。

唯一项目目录 /home/mint/rl_dev/instinctRL-dev，main。Python controller只等你的退出，项目现在由你唯一写入。先读AGENTS.md、现行单分支协议、原冻结 formal_r5_launch_manifest.json、formal plan、当前唯一state coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json（只读，不改状态/plan/controller）和本轮 reviews/joint-actual-training.json。不要把旧readiness结果或模板当成当前正式训练/评价结果。

独立核对本轮R0/R1实际训练、五新checkpoint及已接受审核、准确评价AUTH/spec/approval/Store/receipt/request/source/config/argv；从全部25case×两组的原始轨迹与完整inventory原件/副本重哈希，确认精确job ownership normalACK/two stable empty/no survivors，各2postrun及保守账本。保留nativeFAILED/-9/rawPASS_PENDING_CLOSE/INCOMPLETE/null；仅完整独立科学数据+已知close9.5s/normalcleanup可接受，未知退出/缺失/权限/所有权异常必须STOP。不能根据controller完成标志或producer summary直接声称科学有效。

运行现有新增CPU读取/原函数重放工具，严格全部节点272/288/320/384/512和Z/T/S/X/W、每case全部32lane/所有方向或墙子组、原sensorT0/完整启动/全部原失败分母与阈值：
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 tools/analyze_r5_formal_segment.py evaluation R0 --output docs/experiments/r5-formal-first-segment-20261015/R0-evaluation-analysis.json
R1同样。真实已有文件则复核不覆写历史。禁止删最差lane、拼接reset、调整门槛、挑最好node/2of3。原evaluate_checkpoint/evaluate_episode决定科学评分，工程失败与性能负结果按原口径全部保留。科学负结果不是证据审核STOP；在完整证据下如实分析两组得失。任何已保存节点原Z回退都阻止此已完成boundedsegment之后的追加段，本次无论结果如何都不继续训练。

写完整 reviews/final-scientific-review.json（PASS/noMUST_FIX仅表示数据科学有效，绝不等同训练任务全通过），含全部真实pins、原函数复算、清理/计账、两组比较有效性边界。写 EXPERIMENT_RESULTS.json 和 FINAL_ANALYSIS.md：明确每组每node每case原成功数/子组/连续指标、失败与reset/证据质量、真实新增训练/优化/写checkpoint/非训练量、概率gate/分布及训练aux/reward诊断、同一origin warm-start aux0/10干预的配对解释。共同reward.05、anchor4/LR1e-4/3e-4、同seed/恢复状态、freshreset/clock；核对两组起始状态与首512实际同源性，区分首rollout同源和aux干预后的学习分化；不要求后续轨迹相同，不伪称跨seed相对效应或唯一COM/anchor原因。全量数据支持什么就写什么，不保证原Z保持或完整非零收敛。结论提供 measured next-phase decision，不能自行执行或修改方法。

更新docs/PROJECT_STATUS.md、coordination/CURRENT_TASK.md、coordination/PROJECT_STATE.json及ACTIVE_WORKFLOW中的本次正式段执行索引，明确当前无live、已完成的是固定首段、本次结束；旧readiness Goal及历史失败/冻结包不改写。不编辑controller state.json；它在你退出后原子转COMPLETE。只提交本任务证据/当前状态：本轮docs目录、同case本轮归档/许可、原goal_budget.json、上述状态文件；不暂存docs/goal/或他人修改。必要forceadd仅本case评价归档的job/supervisor.log、job/lifeguard.log和attempt/results/prelaunch/kit.log。正常commit/tag r5-formal-first-segment-results-20261015/push，tag存在时只精确核对，不force或重写历史。

最终只按所给schema返回。status PASS表示完整原件独立审核+真实最终分析/提交push已完成，即使科学性能是负结果；summary必须清楚科学实际结果。artifacts至少含final review、EXPERIMENT_RESULTS.json、FINAL_ANALYSIS.md pins；commit准确已push HEAD。若数据/权限等真实异常，持久化STOP原因并返回STOP，不制造完整结果、不再次调用模型。

调用前已有的全部case文件、训练joint review/analysis及原件/副本均须保持不变，goal_budget.json只能原样提交，不可编辑。artifacts必须包括final review、EXPERIMENT_RESULTS.json、FINAL_ANALYSIS.md和两组evaluation-analysis.json这5个文件；不要绑定会变动的项目状态/账本。
