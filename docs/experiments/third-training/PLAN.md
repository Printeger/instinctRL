# instinctRL 第三轮训练

当前处于 CPU 准备。资源待确认，真实恢复与关闭预检待运行。以下设计及额度为草案，正式冻结在预检通过之后。

## R6 复盘

- TRAIN-01 的 107 次更新止于 `EVIDENCE_RAW_QUOTA`，01B 的 122 次更新止于作业超时。两次都没有可验证的 checkpoint，合计 117,856 条 transition 和 6,354.5 秒监督器墙钟。前者应先用证据流的实测增长率检验每文件和总量上限，后者应在首次完整训练前核对吞吐量和截止时间。两个失败都可以更早留下已完成更新对应的 checkpoint。TRAIN-01 的 107 次以工作负载状态为依据，完整科学分析无法核对该段的全部计数。[TRAIN-01 恢复记录](../r6-direct-training/continuation-20261101/TRAIN_01_RECOVERY.json) · [TRAIN-01 回执](../r6-direct-training/runs/R6-DIRECT-TRAIN-01/receipt.json) · [01B 分析](../r6-direct-training/continuation-20261101/confirmation-128/seed-01b-analysis/summary.json) · [01B 回执](../r6-direct-training/runs/R6-DIRECT-CONFIRM-128-SEED-01B/receipt.json)
- 现存回执中 19 次标记 `DIRECT_CLEANUP_TIMEOUT`。完整工作负载、退出码和所属进程清理各自需要核对。先在短仿真上验证有界关闭，再决定是否需要针对性修复。CPU 与 CUDA 概率回放有实测差异，保留 CPU 原报告和同设备精确复核。 [R6 最终结果](../r6-direct-training/FINAL_R6_RESULTS.md) · [资源与关闭审计](../r6-direct-training/continuation-20261101/delivery/FINAL_RESOURCE_AND_CLEANUP_AUDIT.json)
- 录像前八次作业合计约 26.9 分钟。录像是记录姿态回放，渲染返工属于交付。十一月二日已收齐四模型最终评价，十一月十二日发布了 R6 总交付。两日期之间的总等待、人工投入和工具切换返工无法从现有回执分别归因。训练关键路径只安排模型、逐条件数据和复现证据。 [R6 录像清单](../r6-direct-training/videos/VIDEO_MANIFEST.json) · [发布记录](../r6-direct-training/continuation-20261101/delivery/FINAL_PUBLICATION.json)

## 假设与最小改动

主假设是，停令后当冻结锚持续有效、测距历史能分辨相关方向的运动时，将现有 PPO 零指令锚 Huber 奖励的系数从 4 提到 8，会使保留集延迟加丢帧条件下的测距锚平均绝对误差，相对冻结 u640 和同训练量原配方各降低至少 5 mm，并通过行为和安全护栏。现有奖励将该项门控在零指令、锚激活且至少 10% 射线可用的时刻。单独增加这项的相对权重，使 PPO 在已有观测可辨认的偏差上承担更大的长期代价。优势归一化、误差噪声或其它奖励项仍可能让这项改动没有可测效果。 [Paper-1 方法](../../paper1_vel_ctrl.tex) · [候选训练方法](ANCHOR_WEIGHT_8.json) · [原配方](../r6-direct-training/TRAIN_METHOD_01.json)

该预测适用于静态局部表面、小姿态变化和有效回波高于噪声的情形。单平面的切向恒速漂移可以产生相同测距和 IMU 历史。测量锚误差小也可能伴随不可观测方向的物理漂移。训练把锚损失变大，也可能学到更慢的接近、更小的运动或更敏感的噪声响应。用停令前速度跟踪、停令时速度、锚有效覆盖、清晰条件及离线位移检查这些解释。保留既有 actor 字段、历史、归一化、ICS 干预、机体系速度接口和底层飞控。新指标只读取关闭后的评价日志。

## 指标与比较

主指标为 `S-delay-drop` 停令后 [1, 11) 秒的每 lane 锚 MAE，单位 mm。取控制审计里的 `mean_absolute_error_m`，它按有效射线计数归一化，并按当前及冻结锚的可靠度给误差加权。按固定 100 个控制步取均值，再汇总同 seed、同 block、同 lane 的差值。要求停令边沿捕获且在整个检查期保持同一有效锚。重锚、episode 重置、有效比例小于 0.1、安全失败或缺帧导致相应 endpoint 无效，32 个 lane 的分母保留。`S-clean-low` 同指标为清晰条件对照。

现存开发数据中两个模型在两条件都 32/32。对于 `S-delay-drop`，u640 为 50.13 mm，R6 开发 u768 为 57.51 mm，32 个配对差为 +2.20 到 +10.77 mm。清晰停令为 24.32 到 29.27 mm。每组每模型 3,200 个计分样本齐全，窗口内无重锚。延迟加丢帧时两模型平均可用射线比例均约 56.9%。这是一个历史开发续训流上的可区分性证据，无法估计独立重新运行的误差，也无法证明第三轮候选会受益。5 mm 是预先提出的工程门槛，量级约等于干净停令下已观察到的退化。 [逐 lane CPU 重算](R6_DEVELOPMENT_ANCHOR_ENDPOINT.json)

正式比较拟用三个配对训练 seed `2026112201–03`。每对从同一 u640 恢复模型和 Adam，沿用 R6 的 64 原始 + 64 鲁棒更新、同课程和随机流种子。两个方法各 65,536 条 transition、128 次 PPO 更新，每条 1,024 次 actor 和 critic 优化步。冻结 u640 仅评价一次。R6 历史 u768 仅作开发参考。候选和原配方各取预登记的 u768 终点，七个模型共享一次新的评价随机种子。先留存随机承诺，在方案和模型哈希入 Git 后揭晓；保留 R6 的五任务和 16 个块。现有 `evaluate` 入口可消费新 seed 和两个方法的各自来源 pin，内部 `development` 是运行契约字段，本轮保留集身份另由同一承诺和七模型清单证明。评价仍使用同一控制配置，训练锚系数只改变记录的 reward。

计划接受规则为两项比较的三对点估计方向均为改善，跨训练流平均各达到 5 mm，且条件 95% 区间上界均小于 0。区间对八个指令方向做配对分组重采样，保留每方向四个 lane 与七个模型的配对。单列三个训练流的效果及范围，区间只反映这一评价分布下的样本差异。硬护栏保留原五任务与 16 块的原阈值和全部 lane，核查碰撞、越界、间隙、漂移、停令和避让。附加实用护栏为清晰停令 MAE 至多增加 5 mm、安全非零指令稳定段速度 RMSE 至多增加 0.01 m/s、主指标窗口有效射线比例至多下降 2 个百分点，各自对两个对照。按预期方向重复但不足 5 mm 时记为未达到工程门槛。若点估计或区间仍不能辨认有效改善，或出现明确退化，保留结果和相应证据。

## 发车前工程验证

先用候选独立预检 seed `2026112200` 在原始课程上验证 u640→u642 写盘、哈希、关闭和准确所属进程退出，再把 u642 载入一个新作业，在短鲁棒课程上跑到 u646。第二段一并测量高负载证据流的增长率。严格核对模型、Adam、计数器和设备 RNG。恢复使用新物理 reset，预检 checkpoint 和样本只用于工程验证。关停需有完整原件、`CLOSED` 回执、空 `remaining`、准确 boot/PID/starttime 已退出。有界强制关闭须保留原退出异常。超预算、无进展或进程残留不得用作通过。正式训练预登记 u642、u672、u704、u736、u768 写盘，保留完整个体证据；中断后若需要新物理 reset，就不可把该段当成原计划的同一连续训练流。

沿用 `tools/r6_direct`、已锁 runtime 和本机历史登记的 RTX 5070 单卡串行。R6 四条完成的 128 次更新耗时约 61–69 分钟；最终评价单模型约 38–50 分钟。建议 18 GPU 小时累计、自首个 live 起 36 小时、最多 22 次 live 启动和 300 GiB 新证据总占用。监督器预算含准入和清理，预检每段最多 20 分钟，正式训练每条最多 80 分钟，评价每模型最多 65 分钟。额度含必要同设备概率复核及一次有证据支撑的恢复。到 STOP、任一上限、预检严格恢复失败、不可恢复证据损坏、归属进程残留或硬护栏失败时停新作业并保全原件。

现有两张可写结果卷共余约 65 GiB。仅六次训练和七次评价的 R6 同类原件合计预计约 73 GiB。建议新建 112 GiB 结果卷和 1.5 GiB native 卷，按现有管理员入口挂载，保留既有卷和原件。历史资源授权属于 R6。第三轮没有运行授权记录，管理员非交互 sudo 也不可用。批准后的限额、挂载、准确 Git 版本和文件哈希须在真实作业前有新运行绑定。 [本地入口](../../../tools/r6_direct/README.md)

卷身份、目标路径、容量及现有 provisioner 哈希见 [卷容量草案](STORAGE_PROPOSAL.json)。

## 复核开发指标

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONPATH=.:isaaclab-training \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B -m tools.r6_direct.anchor_endpoint \
  docs/experiments/r6-direct-training/continuation-20261101/u640-03-analysis-fixed/report.json \
  docs/experiments/r6-direct-training/continuation-20261101/u768-01-analysis/report.json
```
