# 同输入街道折返加速度配对

`NOVEL-101` 复用 `NOVEL-100` 的 B `u2816`、`v45`、几何、32 路 Crazy 指令、发令窗口、球体半径、空间范围、轴向最高速度和初始空间相位。沿每轴在端点前一秒减至零速，转向后用一秒回到轴向最高速度，复位速度取该相位的解析速度。半周期因此增加一秒，球到达机体附近的时间随之变化。 [原输入和失败](../independent_street_sweep/REPORT.md)

相同的两组独立编排预检中，八路街道的四类假设响应逐路预计产生 16–33 和 11–32 个合格动态事件。用原实测机体轨迹和射线作配对，第 17 路在原最晚可避让拍的可见回波增加，旧轨迹最低净空约 0.136 米；到达时刻对齐的急折返复现该值。第 20、23 路在相同旧机体轨迹上余量更低。原来物可见性和避让失败继续作为原版本证据。 [编排预检](/home/mint/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-FREEZE/acceleration-bounded-street-preflight/PLAN-ONLY.jsonl) · [到达时刻对齐复算](/home/mint/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-FREEZE/rebel-n100-sweep/compare.jsonl)

原生测试 120 秒、32 路、38,400 个评价步、一次显式初始化、零训练。逐路核验每 5 毫秒的球位置和每次复位速度。保留接触前后的原件，按原统计分别看首次回波、5 毫秒净空、停杆、危险介入、恢复和真实覆盖。墙钟上限 1,200 秒，载荷 512 MiB，监督余量 256 MiB，系统盘至少保留 30 GiB。正式独立 30 分钟使用新的命令和几何种子，并使用原验收门槛。 [原门槛](../long/GATES.json)
