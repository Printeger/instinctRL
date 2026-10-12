# Paired pilot start readiness

当前：`READY_FOR_PAIRED_PILOT_WITH_FIRST_ROLLOUT_GATE`。本 Goal 已完成技术准入并停止；独立 Role15 最终审核 PASS，无阻塞性 MUST_FIX。[最终独立审核](reviews/review-entry6-final-readiness.json)，SHA `5fd1902b08ce2a968a3ceb07043ccc63a2bfc65606106f417358c6f9a9ede620`。当前／历史状态同步条件已满足；本 Goal 未启动训练，未来训练许可仍为 UNISSUED。

[最终状态核对](FINAL_READINESS_VERIFICATION.json) 已验证13项需求索引、48个证据路径、共同源码／配置和未来包哈希、零训练、无活动任务／所属进程；[现有 ledger 验收记录](final-readiness-store-evidence.json) 只登记独立审核，不改变 job、proposal 或 approval。

修复后的零目标增强 B0 已由 Role12 独立复核全部403200物理行、96段及实测质量／真实rotor allocation中间量，接受物理机制回归。独立的命令直通 ENTRY6 使用同一修复后基线，经原 governor／ICS／Lee／rotor 路径执行固定动作；原sensor-T0、32lanes、所有启动与安全样本及原metric-h1-v1门槛均保留。

| 实测条件 | 原gate结果 | 连续指标（32lane最大值） |
| --- | --- | --- |
| Z，200steps | 32/32；d0–d7各4/4 | speed RMS0.010481975m/s；drift0.006700954m；height0.048525572m |
| T，100steps | 32/32；d0–d7各4/4 | 全窗RMSE0.037160032m/s；steady RMSE0.000572826m/s；vz RMS0.000323263m/s；height0.053401828m |
| S，330steps | 32/32；d0–d7各4/4 | stop_time1.5s；drift0.007121895m；刹车位移0.095374950m；zero-block总位移0.099505298m |

逐lane原始评价在 `entry6-command-original-gates.json`，来源／范围在 `entry6-command-regression-summary.json`。这是实际仿真控制验证，不是训练成功或全部旧R4失败的解释。控制基线通过后，没有证据要求额外D7-R才能准入本次全零warmup pilot；旧R4策略只能作旧系统历史参考。

真实进程仍为 FAILED/-9：`app.close`在9.5s超时后被终止；原status为PASS_PENDING_CLOSE，H1 collector为INCOMPLETE/transitions=null。观察到630完成steps／20160nontraining，但20160全额预留仍保留，不写成正式FINAL。正常lifeguard ACK确认两次稳定空扫描、无所属残留；两次授权postrun审计仅见原GNOME服务。完整原job和整个attempt/results的667对象在 `live-entry6-command-regression/INVENTORY.json`，大chunk保留原路径和SHA。ENTRY6使用generic H1 collection加完整独立归档，没有声称获得specialized seal或graceful close。独立 Role15 已分别接受真实控制结论、失败分类和所有权清理；native close 失败是明确披露的非阻塞限制。

两组共同基线：`ef68ebb5682b85aa83044fdea1c13b7d0edbf325`，只读snapshot `coordination/research/workspaces/R5-pilot-r3-20261012`；tree `b735eafac3888415eded7c6684ed385706871acf`，config SHA `d9e61f95a65d424e5d0d2cbba7dee24ab8788209519bc90862a03f42f675cf7c`，tool release `74a6042b0a1c92bda01efc5cd62da7f59948db4ff2be667eb9e34765c1a76c60`。物理源码与已实测f726回归逐字节一致；ef68只补分析前耐久证据，工具补pilot-evaluation原评估seed分类。

准确下一轮包：`pilot_future_proposals.r3.json`，SHA `0154b4b2c9a693c99bf590e91256e092a22ba8fb07599dd66ca71d564b03113a`；准入与入口索引 `pilot_launch_manifest.json`。四个未来attempt、实际AUTH和训练checkpoint均未创建。P0 actorLR3e-4、PLR actorLR1e-4；两组criticLR3e-4及其余设置相同：32env／16rollout／64minibatch／1epoch，原1024policy-step全零warmup，64updates／32768training transitions每组，原七个保存节点，u16/u64的Z评价。降低actor LR仍是待测试假设。

冻结 proposal 中保留生成时的旧预审／cap元数据，准确argv、pins和未签发AUTH模板未改变；当前基线验收以本 manifest 的最终独立审核为准，旧元数据不提供执行许可。

首次真实rollout的概率gate必须留给下一轮：512sample的采样时old Beta alpha/beta、logp、action、输入和索引须在任一actor／critic optimizer之前按固定atol1e-6／rtol1e-5一致；不一致保全证据且optimizer steps为0。已有实际HistoryActor／PrivilegedCriticH1 CPU张量与生命周期／异常保存测试及独立审核支持实现，不能替代这道真实gate。xyz_correction是governor heads1:4的实际v_corr（m/s），四维sample-minus-mean保留为无量纲字段。

本 Goal累计start/init cap8，已用6；旧失败与ENTRY6未最终结算的80640nontraining全额保守预留，原3640s／64GiB／24角色总帽未扩大。当前15角色调用包含已完成Role15，6次关键审核预留仍保留；实际wall1608.3564260550775s、GPU1592.547116729198s，输出9761277960bytes。第四次910.5478907090146s越过单次910s的0.547890709s仍原样记录计账。一次资源拒绝重试例外已消费，不再启动live、探测GPU或自动重试。

最终独立 PASS 及状态同步已完成，本 Goal 结束于技术准入。下一轮训练仍须自己的明确训练资源包络、准确Commander proposal／approval、输入与checkpoint绑定、STOP／coexistence和所属清理检查；native close已知失败及保守计账规则必须保留。技术准入不签发训练权限。
