# B/C 连续三臂开发基线

PAIR-04 完成 34,560 个真实物理评价步、192 条 18 s 测试槽位。首次碰撞或重置截断后，实际首飞合计 3,265.9 s。街道出现 32 个实测进入横穿区域的首飞事件。两份权重分别沿用原 25° LiDAR 和后向宽侧执行配置。 [配对协议](PAIR.md) · [冻结输入](R6-DIRECT-INSTINCT-GPU-COMMAND-PAIR-04-inputs.json) · [复算绑定及原件索引](BASELINE-RESULT.json)

| 权重 | 无横穿接触 | 横穿提前接触 | 名义重合接触 | 名义重合组实际带风险换向 |
| --- | ---: | ---: | ---: | ---: |
| B | 0/32 | 0/32 | 8/32，街道 8/8 | 0/8 |
| C | 0/32 | 0/32 | 8/32，街道 8/8 | 0/8 |

名义重合组均在第 60–61 拍发生接触并重置，换向在第 64 拍。两条最小根位置到球表面间距分别为 B lane 18 的 0.1155 m 和 C lane 16 的 0.1387 m。该量与逐拍任务射线极小值均是诊断量。原件保留接触阈值和每拍 20 次采样，接触记录没有刚体身份。 [B 重合原件](/home/mint/instinctrl-runs/R6-DIRECT-INSTINCT-GPU-COMMAND-PAIR-04/results/research-r6-dynamic-assay/b-crossing_at_reversal.jsonl) · [C 重合原件](/home/mint/instinctrl-runs/R6-DIRECT-INSTINCT-GPU-COMMAND-PAIR-04/results/research-r6-dynamic-assay/c-crossing_at_reversal.jsonl)

Scientist 复算 B 时，actor 实收指令回到世界系后的最大差为 4.52e−8 m/s。B 街道重合组已有可见回波，第 30 拍逃逸介入，第 46 拍 ICS 将下发裁为零，第 56 拍进入 emergency。示例 lane 16 触发时纵向下发由 0.214 提至 0.334 m/s，人令为 0.25 m/s。碰撞前真实速度只有 0.0024–0.0051 m/s，机体停在距横穿平面约 0.28–0.30 m 处。错开组也曾零下发，停位约在平面外 0.64 m，并在发出反向人令后 0.6 s 达到连续跟踪判据。证据指向逃逸使机体向交叉面推进、之后的零下发停位不安全。闭环输入已交付，底层也已实现停速。

关闭横穿时，街道反向跟踪在 0.7–0.8 s 达到开发判据。仍有首次松杆观测的轨迹，其最大漂移不超过 0.070 m。室内、树林、开放管道在本轮仅覆盖连续指令基本响应，仍需含门洞和转角通行、障碍绕行及管内受限动作的开发路线。

PAIR-01 因系统目录编号检查在监督前拒绝。PAIR-02 在准入采样前因模式名拒绝。PAIR-03 初始化后缺共享后端字段，实际评价步为零。PAIR-04 工作负载完整，模拟器清理超时的原始 `-15` 回执保留，登记的 7 个进程身份均退出。冻结后重放核验了 6 条轨迹的时序、指令和历史。CPU 完整入口通过 27 模块、174 项。 [本轮账本](RESOURCE_LEDGER.json) · [CPU 检查记录](/home/mint/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-02/COMMANDS.json)
