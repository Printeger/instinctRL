# 第一轮全量 GPU 训练与完整评估

Isaac Lab 使用 32 个环境完成 u640 至 u768 的 128 次 PPO 更新、65,536 条物理训练转换和五次权重写入。两个阶段均使用成对动作监督，1,024 个小批次记录了 2,959 对可见的闭合与远离历史。CUDA 同设备概率复核通过。训练和评估分别用时约 60 分钟和 35 分钟，监督器保留清理超时回执，所属进程均已退出。 [冻结训练输入](round-1/R6-DIRECT-INSTINCT-GPU-R1-TRAIN-inputs.json) · [完整结果](round-1/RESULTS.json) · [资源账本](round-1/RESOURCE_LEDGER.json)

原五项和 16 个扰动条件完成 193,600 条 GPU 评估转换，每项均为 32/32，独立复核通过。相对冻结的旧 GPU 第二轮，延迟加丢帧的停令锚 MAE 从 51.38 降到 49.06 mm，改善 2.32 mm，未达 5 mm 工程参考。最差条件 `stress-W-high` 的速度跟踪 RMSE 增加 0.0178 m/s，配对护栏未通过。旧 GPU 第二轮继续作为对照。 [完整配对指标](round-1/RESULTS.json)

固定同一最新扫描的 CPU 对照冻结了三个实际 GPU 权重。32 组有纵向测距历史变化的样本中，第一轮平均远离减逼近的指令方向速度差为 −0.008 mm/s，静止障碍样本的平均请求速度比旧 GPU 第二轮低 19.33 mm/s。成对合成回波在这些未见夹具上还没有形成有用的方向选择，静态回归也看到更慢的跟踪。 [固定扫描对照](/home/mint/instinctrl-runs/R6-DIRECT-INSTINCT-GPU-R1-TRAIN/matched-history-probe.json)

下一轮继续从共同 u640 开始，用真实 PhysX 物体产生逼近、远离和横穿的闭环训练样本。局部未来运动只用于训练奖励的方向目标，M4 历史门控保护正常悬停。评估同时复用原 R6 全量和冻结对照的动态场景。 [短验证](DYNAMIC_TRAINING_CHECK.md) · [动态验证设计](DYNAMIC_DESIGN.md)
