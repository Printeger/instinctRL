# 街道第 19 路的有限转零后备 · 后半段横穿仍失守

两组 CPU 配对在切换前逐拍复现原交回动作和机体状态。困难组第 1066 拍证书的动态下界为 0.459 米，独立可见回波检查为 0.607 米。第 1067–1071 拍经原 governor 和 ICS 实发零。第 1072 拍 v60 拦截接管并下发 0.30 米/秒，早于证书在 107.5 秒到期。原来物组第 1069–1074 拍实发零，第 1075 拍由原停驻分支接管并继续下发零。所有零速度都经最终流水线回读。 [冻结方案](PLAN.md) · [原件与结果](RESULT.json)

| CPU 配对 | 原交回 → 转零时段最低净空（米） | 原交回 → 后段最低净空（米） | 后备臂松杆诊断 | 后备臂可行跟踪与恢复 |
| --- | ---: | ---: | --- | --- |
| 原来物、记录姿态 | 0.565 → 0.582 | 0.389 → 0.389 | 未持续停稳，漂移 0.227 米 | RMSE 0.070 米/秒，41/41 拍达到沿向进展，恢复 0.5 秒 |
| 延迟正扰动、固定姿态 | 0.534 → 0.557 | −0.161 → **−0.142** | 未持续停稳，漂移 0.219 米 | RMSE 0.102 米/秒，15/20 拍达到沿向进展，恢复 0.5 秒 |

困难组低于 0.15 米的首拍从 1085 提前到 1084，109.390 秒达到最低。为区分拦截提前接管与停车位置的影响，另外从第 1067 拍队列和实测支路状态离线积分持续下发零。它与实际零动作期间的平移轨迹相合，在原证书到期前包络最低 **0.545 米**，108.225 秒开始低于 0.15 米，109.160 秒最低 **−0.180 米**。持续零采用固定姿态和简化跟随，计分量为移动体包络净空。该停车位置本身会被后续横穿扫过。 [持续零离线输入](HELD-ZERO-INPUTS.json)

零速度在原证书覆盖时段仍可作为局部后备。困难组要守住后续横穿，需要在证书到期前完成有净空证据的位置转移。v60 正式收口仍止于第一关。原生评价与训练新增步数为零。 [v60 收口](../../../mover_passage_reproof/REPORT.md)

脚本在 v60 的 `a346bea8` detached HEAD 下运行。CPU 配对分析冻结在 `dfcd694b`，持续零分析冻结在 `ce25e19f`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/closed/backup
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-BACKUP"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/replay_backup.py" hard_positive "$output/HARD-POSITIVE.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/replay_backup.py" nominal "$output/NOMINAL.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/held_zero_bound.py" "$output/HARD-HELD-ZERO.json"
```
