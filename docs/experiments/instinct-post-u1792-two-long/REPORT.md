# 接续 u1792 的两次有效正式长训

B、C 通过全部冻结门槛，任务达到 **2/2**。每轮均从冻结的 `u1792` 出发，经两个 512 更新分段到达 `u2816`，新增 1,024 次 PPO 更新和 524,288 个真实训练步。A 同样完成训练和完整 R6，独立物理留出新增一次接触，判为未通过。 [A、B 完整审计](study/COMPLETION-AUDIT.json) · [C 完整审计](study-front/COMPLETION-AUDIT.json)

| 轮次 | 训练种子 | 物理留出与同执行起点 | R6 最差跟踪增量对同执行起点 | 验收 |
| --- | --- | --- | --- | --- |
| A | `1547007899` | 4/192 对 3/192，新增一条树干前向停令接触 | +0.000961 m/s | 未通过 |
| B | `637683454` | 3/192 对 3/192，三个接触案例一致 | +0.000088 m/s | 通过 |
| C | `428032978` | 5/192 对 5/192，五个接触案例和时刻一致 | +0.000180 m/s | 通过 |

B 与 A 使用同一物理留出，C 使用提前锁定的新种子和场景。R6 扰动流事先冻结，C 沿用三个已关闭的确切控制证据，额外执行完整终点臂。合计三轮独立物理与六个实际 R6 臂完成 1,438,080 个评价步。六臂的原五项及 16 扰动块均逐项通过，B、C 相对同执行起点及旧合格的工程护栏均通过。冻结执行、存储、记账和概率证据的 35 项 CPU 测试通过。 [共享评价](study/EVALUATIONS.json) · [C 冻结协议](study-front/INDEPENDENT-PROTOCOL.json) · [C 评价原件](study-front/EVALUATIONS.json)

B、C 的静止悬停、普通通行和管道恢复均保住原有水平。B 的同权重执行对照救回一条后向接触，训练终点维持同执行起点的接触结果。C 的新留出也保住同执行起点的结果。现有证据支持两个独立续训保住已验证的执行表现，尚未显示正式续训权重进一步减少接触。 [执行对照](study/MECHANISM-CLEARANCE-AUDIT.json) · [B 物理留出](study/HOLDOUT-B.json) · [C 物理留出](study-front/HOLDOUT-C.json)

A 的前向新增接触触发了后续验证。同权重新镜像树干的八条首拍均无活跃束，两臂初始下发一致。按提前冻结的分支条件，C 复验 B 的后向首拍选宽侧和参考在线训练配方。试验中另外五条开管轨迹实际换侧，记录为开发证据。 [前向固定权重复算](study-front/FRONT-CLEARANCE-01-AUDIT.json) · [C 训练方案](study-front/FORMAL-C-SPEC.json)

## 代表性权重与关闭证据

- B 终点 `u2816`，SHA-256 `841a0a7c7926a14584f257667c5dffe0077129c33458898026ae37405ce9e231`。 [权重路径与训练证据](study/FORMAL-B/RESULTS.json)
- C 终点 `u2816`，SHA-256 `450b8675161f132178eb345f85c3dd6c27dffe9f3de6a33c792c3fabcf6ba172`。 [权重路径与训练证据](study-front/FORMAL-C/RESULTS.json)
- A 终点、原接触轨迹和开发重发记录保留供复算。训练、评价及概率复算的两份审计共锁定 38 份关闭回执。保留模拟器写完原件后清理超时的 `-15` 原回执，登记进程均核对退出。模型和原件位于本机 `~/instinctrl-runs/` 的原始路径。 [A 训练证据](study/FORMAL-A/RESULTS.json) · [A、B 账本](study/RESOURCE_LEDGER.json) · [C 账本](study-front/RESOURCE_LEDGER.json)

Maverick 六方首轮齐全。Scientist 独立复算 A、B、C 留出和镜像诊断。Producer 独立复核三条训练链、同设备概率复算及进程关闭。 [贡献账本](../../../.maverick/contributions.jsonl)
