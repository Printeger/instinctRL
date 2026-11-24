# Isaac CUDA 动态球刚体接线

第一次冻结预检在首个 GPU 物理步退出。CUDA 后端报告 `GPU Rigid Body View kinematic target setting not implemented`，没有完成环境转换。保留该次的输入、日志、回执和资源账本。第二次预检用新的源码快照、输出目录和前驱关闭回执复测。 [首次失败证据](pilot-01/FAILURE.json)

球改用禁重力、零阻尼、有实测质量的 PhysX 动态刚体。每个物理步给质心施加至多 0.95 m/s 的世界线速度，由 PhysX 推进并计算接触。重置和停放时同时写入位姿与零速度。球面回波使用推进后的实测球心，预期轨迹、写入速度和 PhysX 实测速度分别留存。无人机原 ContactSensor 继续逐物理步计入接触。 [动态验证设计](DYNAMIC_DESIGN.md)

球是可重复控制的移动障碍物，质量与轨迹用于降低无人机接触后的场景漂移。研究覆及实际总体接触、回波送达和闭环指标。actor 的 M4 输入仍限于 LiDAR、IMU、时间和操作者指令。
