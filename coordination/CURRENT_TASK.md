# 当前任务

当前进行 R5-FORMAL-READINESS-AUTO-01，目标 READY_FOR_FORMAL_R5_TRAINING。完整新master委托见 docs/experiments/r5-formal-readiness-auto/MASTER_AUTHORITY.json，运行/恢复/阶段扩额由现有Commander在总范围内办理。旧Goal和账本保持历史，不启动正式完整训练。

两组128纯悬停训练均完整、独立接受（Role6/7）：各65536train、128updates、1024Adam/类、9writes；真实first512gate先于两类optimizer PASS。实际init和首次512 policy/privileged/actions/oldBeta/logp逐位一致。reward单变量w0/.05，actorLR1e-4、critic3e-4、anchor4，training20260908。冻结source a13c5a4、config73aeb4ff，原Z预定u96/u128、primary128。

R0原Z两节点完整12800eval、均0/32；R1原Z尚无轨迹。R1两个评价入口在App前因原5%coexistence分别1/1/9、1/1/6被拒，FINAL0、无App/动作/updates/checkpoint，启动/init/output收费保留。全部已终止/正常ACK2stable无残留/结算及每次2postrun，无live。保守计两次R1拒绝2/3，D008只扩阶段start/init6→7；准确RECOVERY2提案2f9f87c7已Role9条件PASS，A-8ffd68bb4789ae5a已发行，退避≥180秒已实录，提交/tag/push后单次release，至少180秒退避后第三次也是最后一次尝试，再拒则STOP持续资源阻塞。禁止额外GPU探测、无关进程终止、放宽门槛或重开失败job。

已知native close FAILED/-9与INCOMPLETE/null全额逻辑/输出预留保留。R1原episode16个近墙标签属旧调度metadata缺陷；实际64resetlane首返回均在中心，原件不改，未来局部日志修复7CPU PASS。配对分析器明确训练与评价分别绑定恢复身份，6CPU PASS。旧C1u64和新R1u64权重不完全相同，不宣称历史训练前缀等价。

唯一续接入口 docs/experiments/r5-formal-readiness-auto/{GOAL_STATE.md,EVIDENCE_INDEX.md,goal_budget.json}。尚无候选资格，G2/G3、两未见训练seed、有限非零衔接、checkpoint同设备复算和准确正式包/最终独立审核待完成；预算内科学负结果后继续，不以CPU通过或首实验结束代替Goal。
