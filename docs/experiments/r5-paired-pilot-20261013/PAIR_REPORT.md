# R5 P0 / PLR 配对实验报告

当前：两组64updates及四次原Z评估已真实完成，完整科学结果待Role4最终独立审核。结论为 **降低actor LR显著减少本对首更新的分布偏移，但两组u16/u64均未通过原Z；不扩至256updates**。负结果按原门槛保留，没有删启动段、失败lane或重跑结果。

共同只读生产基线 ef68ebb5682b85aa83044fdea1c13b7d0edbf325，tree b735eafac3888415eded7c6684ed385706871acf；config SHA d9e61f95a65d424e5d0d2cbba7dee24ab8788209519bc90862a03f42f675cf7c；工具74a6042b0a1c92bda01efc5cd62da7f59948db4ff2be667eb9e34765c1a76c60。P0 actorLR3e-4、PLR1e-4，critic均3e-4；32env/16rollout/64minibatch/1epoch，训练seed20260908、评估seed20260909，其余reward/network/controller/curriculum/算法/原metric-h1-v1全部不变。没有重做E0、质量补偿或控制ZTS诊断。

两组实际init actor/critic参数逐位相等，PPO配置唯一差异为actor LR。首次512样本的policy lidar_grid/state_vec、privileged、action、old Beta alpha/beta及logp逐位相等。两组真实首次gate按原atol1e-6/rtol1e-5在trainer.update及两类optimizer前通过。实际捕获probe在更新前logp与oldlogp逐位相等、ratio=1；Role2/3独立复核。Role2的跨CPU/原设备logp前向不满足同一容差，这只是跨设备重演边界，未用CPU取代真实同设备gate，也未放宽容差。

| 实际训练 | P0 | PLR |
| --- | --- | --- |
| 更新 / training transitions | 64 / 32768 | 64 / 32768 |
| actor / critic Adam steps | 512 / 512 | 512 / 512 |
| checkpoint writes | 7 | 7 |
| 首更新mb1后analytic Beta KL均值 | 3.328310 | 0.399873 |
| 首更新mb8后analytic Beta KL均值 | 15.012778 | 0.276192 |
| mb8后固定真实512输入的mean v_corr_b (m/s) | [0.090704, 0.001835, -0.183290] | [0.011450, -0.017609, 0.073493] |

analytic Beta KL是对同一首次采样分布和512固定输入的实际pre/post probe统计，八个minibatch轨迹、近似KL、梯度和参数变化都在训练分析和原PT内。所有64updates的approximate KL、1024个返回step×32lane的实际v_corr_b/v_final_b/velocity、reset与episode失败完整保留。速度修正是实际governor heads1:4的body-frame m/s；四维sample-minus-mean仍是无量纲action字段。

| 原Z节点 | 成功 / 原分母 | 最大lane speed RMS (m/s) | 最大水平漂移 (m) | 最大高度偏差 (m) | 原失败 |
| --- | --- | --- | --- | --- | --- |
| P0 u16 | 0 / 32 | 0.198704 | 1.832576 | 1.508342 | 32lane高度failure与reset interruption |
| P0 u64 | 0 / 32 | 0.137930 | 2.629815 | 0.696651 | 原Z阈值失败 |
| PLR u16 | 0 / 32 | 0.051569 | 0.548090 | 0.826962 | 漂移与高度超原门槛 |
| PLR u64 | 0 / 32 | 0.056532 | 1.088824 | 0.046659 | 漂移超原门槛 |

每节点原sensor-T0、完整200steps×32lanes、启动及安全样本保留；原speed0.08、drift0.30、height0.20门槛未改，方向d0–d7全部0/4。128个episode均在原分母。P0u16的原metrics覆盖reset-interrupted窗口，不能作为连续稳态漂移估计；原fail不splice、不删除。其32个engineering evidence_valid=false的原因是实际reset interruption，不是缺失raw或伪造样本；另96个episode evidence_valid=true，全部节点missing=0。完整原raw为每节点T0+200STEP，原evaluator已从原件重放，与现场producer逐字段匹配。

PLR的较低首更新KL是受控首次输入下的本对结果；它在u64改善速度和高度，却仍有1.088824m水平漂移。不能把首更新较小、平均指标较好或零collision替换原成功门槛，也不能因两组均失败断言LR完全无影响。这是一对训练seed的开发Z比较，不证明多seed泛化，也不构成完整策略验收。

所有运行均经新Commander精确审批，先提交实际AUTH/spec再release，评估仅加载Role2/3独立接受并绑定SHA/source的真实u16/u64；无占位checkpoint许可。两组7节点init0000/u1/u2/u8/u16/u32/u64身份、update/transition/Adam/curriculum/seed/LR已独立验证。原checkpoint和大raw留本地只读证据存储，以全量inventory索引，不上传大文件。

四个原进程均为FAILED/-9，因native app.close9.5s超时；operation的PASS_PENDING_CLOSE不改写成process PASS。各正常owned ACK为两次稳定空扫描/无survivors，按准确PID/boot/start身份确认owner/guard/child消失。每入口原六次prelaunch查询后coexistence PASS才创建App；训练各1225、评估各403个runtime memory observations保留；各两次授权postrun审计仅见既有GNOME164MiB。共4starts/4初始化预留、8个postrun查询，没有自动重试。

本轮观测训练65536、评估25600、128logical updates、14训练checkpoint writes，actor与critic分别1024Adam steps。实际outer wall815.8738663156983s、allocatedGPU813.7702312869951s，原输出与归档拷贝4288720528bytes。四个generic H1 collector均INCOMPLETE/null，故训练65536、nontraining25600、128updates和14writes全额保守reservation继续保留；现场完整计数是其子集，不双扣、不伪造FINAL、不退款。旧诊断80640reservation及所有失败完全保留。本轮4entry包络已使用，不再live。

下一阶段科学决策：停止于本轮负结果，不自动256updates。保留修复后控制基线，先用已有u1/u2输入、首更新probe和u16/u64 checkpoint做CPU离线的actor/critic梯度、目标信号与持续零命令修正分解，解释PLRu64水平偏置及u16垂直偏置，再提出一个有依据的单变量实验。LR1e-4可作首更新较平缓的研究候选，不能称已经修复了训练问题；不在本Goal改变reward、actor输入、network、controller、curriculum或算法。

机器结果 [pair_results.json](pair_results.json)，完整 [证据索引](STATE_AND_EVIDENCE.md)，[账本](goal_budget.json)。原训练/评估逐lane结果、梯度、参数和全部失败记录通过四个live inventory与独立评审定位。最终独立Role4应分别验收两组原Z结果、配对科学范围、关闭/计账以及本Goal完成条件；通过后只同步状态和正常提交/push，不启动下一阶段。
