# 悬停方向增益的接线验证

第二轮 `u736` 的独立诊断保持最新扫描相同。32 组纵向历史下，零指令的平均逼近反应差仍低于 0.001 mm/s，静止时平均水平动作约 0.36 mm/s。同一模型在 0.20 m/s 指令下的平均历史反应差也约为 0.001 mm/s。这是训练中途的权重。 [零指令诊断](/home/mint/instinctrl-runs/INSTINCT-GPU-R2-U736-MATCHED-HOVER/hover.json) · [低速指令诊断](/home/mint/instinctrl-runs/INSTINCT-GPU-R2-U736-MATCHED-HOVER/low_command.json)

可选反射头只增加一个训练标量，M4 两个输入字段和原权重保持原形。增益从零开始，Adam 的新标量一阶和二阶矩也从零开始，使用同一个既有更新时钟。继承的是时钟值 5,120，新增标量从接入后开始获得真实梯度。已有 checkpoint 未带方法标记时保留原结构。 [权重和优化器接线](../../../isaaclab-training/instinctlab/training/checkpoint_h1.py)

在留存的 512 个真实首次 PPO 输入上，对比 `u640` 安装零增益前后的 CPU 计算。把偏置加在完整 Beta logits 上再分割，可以保留原数值布局。四项结果逐位相同，包括两个 Beta 浓度、平均动作与动作对数概率，CPU RNG 也保持一致。先按两片分别相加的版本曾产生最高 3.81e−6 的浓度差，已修正。 [真实输入的逐位复核](/home/mint/instinctrl-runs/INSTINCT-GPU-REFLEX-ORIGIN-EXACT/result.json) · [初版误差定位](/home/mint/instinctrl-runs/INSTINCT-GPU-REFLEX-ORIGIN-DETAIL/result.json)

用第二轮重跑留存的 512 个同批输入，以及冻结 u640 的权重和 Adam 状态，在 CPU 完成两臂各八个小批次的单次 PPO 更新。该批没有触发方向判据，更新后旧有 actor 参数逐位相同，新增增益为零，其 Adam 时钟从 5,120 进到 5,128。旧概率用 CPU 重算，复核未申请运行仿真，也未移植 CUDA 随机状态。 [一次 PPO 更新的兼容复核](/home/mint/instinctrl-runs/INSTINCT-GPU-REFLEX-REAL-MINIBATCH-CHECK/result.json)

使用原训练的 1e−4 Adam 学习率，在解析短验证中只更新新标量。128 个小批次后，未见端面上逼近相对远离的退向差达 147.00 mm/s，静止水平动作维持 0.58 mm/s。该配置给出在既定优化时钟内可学到的动作幅度。物理碰撞、操作者跟踪及跨场景迁移仍由闭环数据评估。 [学习率短验证](/home/mint/instinctrl-runs/INSTINCT-GPU-REFLEX-ADAM-SCALE-PROBE/result.json) · [静态旋转控制](HOVER_ROTATION_CHECK.md)
