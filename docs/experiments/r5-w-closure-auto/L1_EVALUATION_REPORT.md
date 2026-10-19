# L1 原五任务评价与下一项独立验证

状态：两节点原评价实际完成并独立接受；完整 Goal 仍 IN_PROGRESS。

`R5-W-L1-EVAL-01` / `J-3db27ce7c40e9542` 的 u640/u768 均从各自原 evaluation_raw 调用冻结 `evaluate_checkpoint` 重放，完整字典与现场结果一致。58 项 inventory、106 份可用原件/归档哈希及独审25个packet/spec pins一致。原seed20260909、source0c6363bbc0d2947a272f533c02e73d2995ac225a、configdf96f98d0d55f997d20ace98468bfa4f4a6a614b1284d6f5b320ee34b9b2adbb不变。

| 节点 | Z/T/S/X/W | 方向子组 | 两墙 | W世界法向运动 |
|---|---|---|---|---|
| u640 | 各32/32 | Z/T/S/X各d0–d7均4/4 | 各16/16 | 每lane最长30连续样本≥.1m/s |
| u768 | 各32/32 | Z/T/S/X各d0–d7均4/4 | 各16/16 | 每lane最长28连续样本≥.1m/s |

u640 SHA `aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429` 为首通过节点；u768 SHA `0c4a8b486280a3bb8dbc91778ed1949c085c4dedd3c6ff75ac7194a957111e41`。两者均为 RECOVERY1 实际保存的完整 trainer/Adam/RNG checkpoint，来源和方法见 `reviews/l1-recovery1-actual-independent.json`，本次接受见 `reviews/l1-evaluation-actual-independent.json`。

| 对照 | x墙最小几何/最大停车时间 | y墙最小几何/最大停车时间 | 解释 |
|---|---|---|---|
| E1固定原u512+.15m | .864304m / 2.3s | .864166m / 2.3s | 工程控制收益，原五任务及运动已全过 |
| L1 u640+.15m | .867889m / 2.3s | .861945m / 2.4s | 有限近墙训练后保留能力，无一致新增安全优势 |
| L1 u768+.15m | .856094m / 2.3s | .855339m / 2.3s | 仍达原门槛，几何裕度低于E1 |

原R1无margin的W0/32负基线及320lane诊断仍在 `W_CAUSAL_DIAGNOSIS.json`，不改写原结果。工程与学习分开：E1仅改变ICS .15m已使五任务通过；L1实际131072样本/256更新、24center4x4y课程、2004个q-positive（x1342/y662）说明暴露覆盖，不能单凭覆盖或loss声称RL额外安全收益。保持当前候选并进行规定的首通过节点独立验证，无依据继续长训。

`L1_W_TIMING.png` 从三组原W数据绘制每墙平均实际几何、世界法向速度、q、ICS beta；`L1_METHOD_COMPARISON.json` 精确pin原始数据与逐lane评分。图中几何/速度为观测时刻，q/beta为发令时刻，不能跨时刻直接相减。完整每lane事件在 `L1_U640_CPU_REPLAY.json`、`L1_U768_CPU_REPLAY.json`。均出现有效接近和制动；原.8m、3s及全部lane评分未放宽。均值图不替代逐lane验收。

当前评价实际65920 nontraining、0train/更新/write；正常owned ACK含两次稳定空扫描，无残留，已settle和恰好两次postrun。native FAILED/-9、PASS_PENDING_CLOSE、INCOMPLETE/null及65920全reserve保持。当前累计观察train131072/nontraining98880/256updates/2048Adam每类/2writes，4start/init，GPU1520.7087300447747s、wall1532.4889966598712s，四个8GiB输出reserve不退款。结算快照 `L1_EVALUATION_SETTLED_BUDGET_SNAPSHOT.json`；后续snapshot收费另记当前goal_budget。

下一真实阶段 `R5-W-L1-INDEPENDENT-01` 验证固定u640、独立reset seed20261020，完整五任务各32lanes；新source4386730b1ac026dd08e1fe628961bf4fcbc9c76c/configf702a6f8cce4ada89a81d34077d893504b2cd84958f07776ba799d0f6d3c203c。原primary入口继续仅认20260909。独立协议先核对真实20261020，随后仅对内存评分视图的seed元数据canonicalize调用不变原evaluate_checkpoint；raw保持真实seed，原backend单lane primary seed mismatch诊断也保留。所有物理、初态、命令、时序、完整性和评分阈值原样执行。新seed不表示初态分布变化或广泛泛化。当前该独立验证尚未执行，最终交付及独立final_review均待实际结果；不可COMPLETE。

发行后CPU授权加载及当前账本复核见 `experiments/R5-W-L1-INDEPENDENT-01/ISSUED_CPU_CHECK.json`。实际新snapshot登记后 prospective output为46GiB，≤48GiB stage/128GiB master；先前stage独审44GiB估算少计新snapshot2GiB，原review保留并以当前准确审计纠正。累计prospective nontraining131840、5start/init、GPU3330.7087300447747s/wall3342.488996659871s，其余训练计数不增加。
