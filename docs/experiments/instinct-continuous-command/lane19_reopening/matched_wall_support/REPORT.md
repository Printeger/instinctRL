# 街道第 19 路的同区域墙面配准与重新接续

第 1061 拍用相同墙面区域独立计算两半，仍缺支撑。保持臂其中一侧只有 4 个有效点，另一侧有 789 个。引导臂的稀疏侧同样只有 4 个点。两臂该拍都保留否决。第 1063 拍，引导臂的相同墙面区域恢复了三面墙和独立半区支撑，两半与整体拟合的最大差都不超过 **0.012 毫米**。注入 6 毫米单半区偏移后，原 1 毫米门槛仍拒绝。用已记录真实位移核对该拍，最大误差低于 0.008 毫米。原来物和正扰动两条原件的结果一致。 [同区域方案](PLAN.md) · [原件及结果](RESULT.json)

把重新接续限定在已有转零后备仍有效的时段，且先复核最新扫描、原保护、完整 governor 和 ICS。两臂从第 1052 至 1062 拍的实发与上轮引导臂完全一致。只用原配准时，在两组第 **1064** 拍实际接续。允许同区域复核时，两组均提前到 **1063** 拍，世界系实发复核误差小于 2×10⁻⁸ 米/秒。 [闭环方案](REARM-PLAN.md)

| CPU 配对 | 原配准接续净空 | 同区域接续净空 | 同区域松杆诊断 | 同区域可行跟踪与恢复 |
| --- | ---: | ---: | --- | --- |
| 原来物 | 0.409 米 | 0.409 米 | 0.6 秒停稳，漂移 0.093 米 | RMSE 0.089 米/秒，恢复 0.5 秒 |
| 早分岔正扰动 | −0.113 米 | **−0.067 米** | 未持续停稳，漂移 0.213 米 | RMSE 0.053 米/秒，恢复 0.9 秒 |

第 1066 拍，正扰动下同区域接续臂的 84 对 ICS 可发动作中，短段最佳下界为 0.608 米，能同时满足远端预计净空的动作数为零。控制仍执行转零后备。第二次交回后的最低净空在 **109.940 秒**。墙面复核让接续提前一拍，正扰动的完整轨迹仍未守住 0.15 米净空及 0.20 米停杆漂移门槛。

结果适用于同一固定姿态、0.45 秒简化跟随及两条已登记来物。原生评价与训练新增均为零，v60 正式收口仍止于第一关。 [v60 收口](../../mover_passage_reproof/REPORT.md)

复现用 v60 `a346bea8` detached HEAD，同区域复核与重新接续分别冻结于 `c70a343b`、`0ab60306`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/matched_wall_support
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-MATCHED-WALL"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/check_wall_support.py" "$output/SUPPORT.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/rearm_from_backup.py" "$output/REARM.json"
```
