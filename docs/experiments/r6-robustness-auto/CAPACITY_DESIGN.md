# 容量设计草案：尚未准入

本文件不冻结测试标准，不代表完整容量核算 PASS。测试实现、每作业物理步/reset/ticks、
实际输出上限和独审仍待完成，`goal_budget.json.capacity_pass=false`。

已核验 R5 原审计：256 updates / 131072 transitions 的训练输出 1,969,675,045 bytes，
单节点原五任务 32960 transitions 输出约 29.7 MB。七次旧作业总 GPU 1805.60 秒、
live wall 1827.27 秒；这不是 R6 吞吐保证，也不能用该汇总推算单训练作业的运行时间。
来源 `../r5-w-closure-auto/FINAL_RESOURCE_AUDIT.json`。当前可用磁盘约 474 GiB，
精确采样值在 STARTUP_PREFLIGHT；实际准入仍须重新检查至少 30 GiB 地板。

内部首先保护 3 × 1024 × 512 = 1,572,864 训练 transitions（不是强制训练配方），
800,000 非训练 transitions、18,000 秒 GPU 和 live wall、16 次启动/初始化、24 次写入、
40 GiB 输出与快照。三种子最终配方必须适配此保护或在首次使用前经独审调整，不能让探索借走。
开发候选可先按每个最多 768 updates 规划：三个共 1,179,648 transitions；连同上述确认
共 2,752,512，距总上限余 247,488。初段 128/256 updates 与后续晋退须事前登记。

旧训练输出线性外推全部 3,000,000 transitions 约 45.1 GB，尚未加新字段、快照、视频
和失败预留，因此只能用于起始估计，不能替代实际字段/行数核算。拟按准确日志规模分配
配额，禁止机械地给 64 个作业各预留 8 GiB。

32 次自动会话中至少保护最终审核 4 次、异常修复 4 次；初始用户会话另列。
tools/r6_budget.py 对每次预留绑定 goal/event/job/input hash/role，重复一致预留幂等，
漂移拒绝；失败不退款。当前仅实现 CPU 预算检查，不是已经验证的事件调度器。

首次 GPU 前仍需：冻结目标/压力域与 split、确定最少最终样本及物理可行性、拆出每作业
policy/physics/reset/init/Adam/checkpoint/cleanup 成本、提供时间和磁盘裕量、独立审核，
并完成 CPU 替身重启/死亡/半写 receipt/STOP/去重/单写者测试和一次真实非交互审核。


## 逐帧测量归档尚待核算

新增全量 clean/delivered 测距归档不能按旧 29.7MB 评价日志线性外推。每lane有21240 beams；
160640 新评价 transitions ×21240×4bytes×2份 =27,295,948,800 bytes（以实现的精确计算为准），
还不含 validity/timestamps。必须采用可逐字节重建的去重/无损编码，并实测压缩和输出配额，
否则当前40GiB最终输出保护不足以容纳四模型全量原始双份数组。不能丢负结果或降采样来
掩盖此缺口；容量状态仍为false。

旧训练目录实查发现两个首更新 full-input 诊断文件各522177865 bytes，是每次作业固定成本，
两个checkpoint各约1.51MB。此前45.1GB训练线性外推只是粗略上界，不能直接拿来分配新作业。
需要分离每作业诊断、每transition日志、每checkpoint和快照成本后再独审冻结。
