# R5 W 最终科学结果

**R5_ALL_ORIGINAL_GATES_PASS**。首通过训练模型为 R1/u640，原 seed20260909 与独立 seed20261020 的完整五任务均通过。最终独审与提交交付见 `reviews/final-independent.json`；运行控制器终态仍由唯一 controller 在收到最终决定后记录。

| 实际模型/协议 | Z | T | S | X | W | W最小几何(m) | 最慢停车(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1 u512 margin0 | 32/32 | 32/32 | 32/32 | 32/32 | 0/32 | 0.711548 | 3.2 |
| E1 u512 margin.15 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 0.864166 | 2.3 |
| L1 u640 margin.15 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 0.861945 | 2.4 |
| L1 u768 margin.15 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 0.855339 | 2.3 |
| Independent u640 seed20261020 | 32/32 | 32/32 | 32/32 | 32/32 | 32/32 | 0.861945 | 2.4 |

所有通过节点 Z/T/S/X 每个方向4/4、W两墙各16/16；所有W lanes真实世界法向速度均连续至少11样本≥0.1m/s，不能靠静止通过。完整分组和每lane时序在 manifest、各CPU replay与原结果中。

选定模型 SHA256：`aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429`。训练source `0c6363bbc0d2947a272f533c02e73d2995ac225a`，config `df96f98d0d55f997d20ace98468bfa4f4a6a614b1284d6f5b320ee34b9b2adbb`。准确原件路径、u512/u768 SHA、完整method字典与所有复现输入见 [manifest](TRAINING_AND_EVALUATION_MANIFEST.json)。本交付模型必须连同已验证的ICS .15m裕度及冻结控制/方法使用；不是可脱离安全层的actor安全声明。

训练从原u512完整trainer/Adam/RNG恢复、fresh physical reset，真实首512 gate先于两个optimizer；u513–768共131072 transitions、256updates、两类各2048 Adam steps、2writes。中心98304、x/y近墙各16384；q=2004（x1342/y662），ICS干预x1946/y993。有限危险样本覆盖及能力保留真实成立，但不等于学习增益。原prelaunch失败和RECOVERY1是同一科学候选的调度恢复。

[W时序对照](FINAL_W_TIMING.png) 展示原R1、E1、两L1节点和独立seed的几何、世界法向速度、q与beta。几何/速度为控制后的观测时钟，q/beta为发令时钟；组平均只用于图示，验收要求每lane通过。原R1 W失败；固定u512仅增加.15m裕度的E1已全部通过，构成控制工程收益。其后近墙训练保留五任务能力，无法从这些实验识别独立RL安全增益。没有通过叠变量、修改评分、关闭ICS、actor加入位姿/速度/地图真值或硬编码停住获得通过。

独立评价实际T0 seed为20261020，原固定suite初态不变；冻结evaluate_independent只在内存评分视图将seed替换为20260909，再调用原evaluate_checkpoint。160case/lane完整重放与现场独立结果逐字典一致。直接原primary评分对全部160项报CASE_SEED_MISMATCH，原backend单lane诊断保留。此为明确标注的独立等价协议复现，不能冒称原primary seed执行、独立初态分布或广泛泛化。独立核对五任务每个step字典与原primary u640完全相同，T0唯一差异为seed。确定性策略及固定初态下同样的轨迹不提供额外随机扰动覆盖；即使全通过也不证明硬件飞行安全。

资源V2是临时调度规则。live13样本跨度60.4627s、median1%、mean2.15385%、peak9%，>10计0，真实通过新门；61样本诊断不是live receipt，也不证明无性能干扰。显存/进程/身份/runtime内存/STOP/磁盘/timeout原保护保留，原scale_003入口不改。

七个作业原生全部FAILED：四个完整采集在app.close超时后FAILED/-9、INCOMPLETE/null；三个prelaunch FAILED/1、FINAL0且无raw。正常owned ACK、settle及各恰好两postrun均完成。科学raw验收不改写这些终态。累计观察训练131072/非训练131840/256updates/2writes；全额逻辑预留仍在，7starts/init、7×8GiB输出+5×2GiB快照=66GiB，含2GiB metadata预算共68GiB≤stage72GiB≤master128GiB。GPU1805.5978s、wall1827.2660s含清理；原master所有caps不变，磁盘地板30GiB保留。详见[资源审计](FINAL_RESOURCE_AUDIT.json)。旧batch/state及master只读，旧停止不改为成功。

完整原件、归档与哈希见[原件索引](FINAL_ORIGINALS_INDEX.json)，逐条需求见[需求审计](FINAL_REQUIREMENTS_AUDIT.json)，方法消融见[机器对照](FINAL_METHOD_COMPARISON.json)。无新增训练或实验需求；最终审查如发现实质缺陷，先修复交付再完成。
