你是本次 InstinctRL 正式第一阶段的独立实际 checkpoint Reviewer，与实现批处理器的上下文隔离。当前用户授权原文在 coordination/batches/R5_FORMAL_FIRST_SEGMENT_20261015/USER_AUTHORITY.md；本消息是本地批处理器在 R0/R1 两组训练均已终态、正常 owned 清理、settle、恰好两次 postrun 后触发的一次性审核。不是轮询请求。不要启动子 agent、codex exec、训练、仿真、GPU 查询或新实验，不扩大预算、不重试、不改任何源码/方法/runtime/冻结包。不要等待另一作业；若前提未满足直接 STOP。

工作目录 /home/mint/rl_dev/instinctRL-dev，main，同一目录现在由你作为唯一写者接管；Python controller 只等待你的退出，不同时修改项目。先读 AGENTS.md、coordination/protocols/SINGLE_BRANCH_WORKFLOW.md 和 docs/experiments/r5-formal-readiness-auto/{formal_r5_launch_manifest.json,formal_conditional_permission.json,START_FORMAL_R5.md}。旧 readiness Goal 的“不提交正式训练”已结束，本次用户明确授权固定第一阶段训练及评价；本次用户明确禁止旧 master 的自动扩额/恢复/方法探索。

当前固定 case：R5-FORMAL-MEAN10-FIRST-SEGMENT-01。唯一批处理状态 coordination/research/batches/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/state.json；只读它，不改状态/plan/prompts/controller。验证其 R0/R1 research-formal-training 精确 job ID、实际 Store/receipt/request/AUTH/source/config/argv/normal owned ACK 与清理、每条两次 postrun、全原件 inventory/副本 SHA、保守账本。native FAILED/-9 仅在原 status PASS_PENDING_CLOSE、无 primary error、已知 app.close9.5s timeout、完整科学数据及正常清理时可单独科学验收，保留 FAILED/INCOMPLETE/null/全额预留；其他异常 STOP。

必须实际读取全部 10 个新 checkpoint，固定节点272/288/320/384/512，每组起点256、实际新增256 updates/131072 transitions/2048 actor与critic Adam steps/5writes。核对真实首次512 gate 发生在absolute257任一optimizer之前、原atol1e-6/rtol1e-5、真实 retained257/258 inputs、probe8minibatches及collection oldBeta/logp/action/adv/return/index对应、无额外采样。严格复算 source/method/seed/mixed clock4096..8191、center-onlyreset、两组同origin120a82…、freshreset seed20261038、当前aux0/10且reward均.05、恢复后安装当前callback的实际日志。不要将旧纯悬停分析器的1..256计数用于新阶段。

已提供CPU证据读取工具（不替代你独立审核）：
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 tools/analyze_r5_formal_segment.py training R0 --output docs/experiments/r5-formal-first-segment-20261015/R0-training-analysis.json
R1同样，输出R1-training-analysis.json。若文件已存在，校验其与原件一致，不覆盖历史；不要做actor forward/optimizer。可额外CPU读取/原公式复算，不能用启发式分数替代因果或成功结论。发现真实缺陷/缺失/不一致：写实际 MUST_FIX/STOP 并返回，禁止修实现、补训、豁免或自动重试。

只有独立接受两组实际训练及10个checkpoint后，写 docs/experiments/r5-formal-first-segment-20261015/reviews/joint-actual-training.json：verdict PASS、must_fix []、identities准确source/config/manifest及job/输入/分析/原件 pins，accepted_evaluation_checkpoints 数组精确列出10条 {group,update,path,sha256}，完整科学/清理/计账口径和局限。这是你真实独立审核的产物，不伪造其他Reviewer签名。

然后在本次调用内以现有委托Commander机械发行两组已有评价模板的准确许可（不submit）：
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:tools/rlflow /home/mint/rl_dev/instinctRL_orchestrator/.venv/bin/python tools/r5_formal_commander.py issue R5-FORMAL-MEAN10-FIRST-SEGMENT-01 R0 research-formal-evaluation --review /home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/reviews/final-readiness.json --checkpoint-acceptance /home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-first-segment-20261015/reviews/joint-actual-training.json
R1同样。二者必须共用同一 joint acceptance SHA，避免case级条件被覆盖；只有5个真实checkpoint SHA和已声明late条件可填写，冻结其余整份AUTH/spec，不改manifest/permission/已有final-readiness.json。发行前若任何同名准确spec/attempt已部分存在，STOP，不删除重建/重issue。将actual AUTH byte-for-byte复制到case inputs/R{0,1}-research-formal-evaluation-AUTH.bound.json。用 tools/r5_formal_launch_gate.require_formal_start 纯CPU验证两组实际spec/approval及accepted Store evidence，显式参数True只用于纯函数，不调用submit。未来GPU评价由controller在你成功退出后串行执行。

提交并正常push本任务证据和准确评价输入，创建唯一tag r5-formal-first-segment-evaluation-inputs-20261015；tag已存在则精确核对，不改写。只暂存本任务：docs/experiments/r5-formal-first-segment-20261015/、docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/、原goal_budget.json及Commander改变的coordination/PROJECT_STATE.json；必要forceadd只该case训练归档job/supervisor.log、job/lifeguard.log、attempt/results/prelaunch/kit.log。不暂存docs/goal/或他人修改，不改历史readiness事实。不更改冻结runtime或source。可记录当前项目执行状态，但不动controller唯一状态。

最终仅按所给schema返回：status PASS代表两组实际训练独立有效、评价许可已准确发行、纯CPU gate通过且commit/push完成；summary简述真实结论，artifacts列出review、分析和准确spec pins，commit=已push HEAD。负结果/权限或资源异常返回STOP并持久化原因，不请求普通阶段批准。不要提交评价，不进行模型轮询。

批处理器将逐字节保护调用前的全部case文件、原件/副本与goal_budget.json；只可将它们原样提交，不可编辑。artifacts必须逐项列出以下9个文件：joint-actual-training.json、两组training-analysis.json、两组evaluation.authorized.json、两组evaluation-AUTH.bound.json、两组evaluation-approval.json（均使用上述准确目录）。不要在artifacts中绑定未来会变动的goal_budget.json或项目状态文件。

强制返回契约：artifacts 数组 MUST 逐项包含下列全部 9 个绝对路径，各项为 {"path":"准确绝对路径","sha256":"实际文件sha256"}。仅返回摘要、三项主文档或省略许可文件不能通过。每一文件必须已在返回的准确 commit 中提交并正常 push。

- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-first-segment-20261015/reviews/joint-actual-training.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-first-segment-20261015/R0-training-analysis.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/inputs/R0-research-formal-evaluation.authorized.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/inputs/R0-research-formal-evaluation-AUTH.bound.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/R0-research-formal-evaluation-approval.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-first-segment-20261015/R1-training-analysis.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/inputs/R1-research-formal-evaluation.authorized.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/inputs/R1-research-formal-evaluation-AUTH.bound.json`
- `/home/mint/rl_dev/instinctRL-dev/docs/experiments/r5-formal-readiness-auto/experiments/R5-FORMAL-MEAN10-FIRST-SEGMENT-01/R1-research-formal-evaluation-approval.json`
