# 街道第 19 路的分支时钟

在上一轮过线的正扰动动作上冻结共用前缀。原来物及负扰动在指定拍起允许使用另一组动作，仍保持同三个跟随响应、松杆全程下发零和原漂移门槛。每个分支用八个起始方向，按 5 毫秒时钟复算五个来物档位。第 1064 拍是本次测试中最晚过线的分支拍。它需要在该拍直接改发一个与正扰动分支相差 0.30 米/秒的动作。 [冻结输入与原件](RESULT.json)

| 允许分支的拍位 | 五种轨迹合并最低净空 |
| ---: | ---: |
| 1061 | 0.152 米 |
| 1064 | **0.152 米** |
| 1065 | 0.140 米 |
| 1066 | 0.105 米 |
| 1067 | 0.072 米 |
| 1068 | 0.041 米 |

同一初始姿态及执行前缀下，三个响应时间的困难正扰动与原来物在第 1064 拍最近四帧逐元素相同。第 1065 拍实收量才出现差异。按原配准，最小配对中心差约 **0.179 毫米**，两侧各留 1 毫米误差后仍有重叠，按扰动轴投影的组间速度区间也相交。第 1066 拍有一档响应配准失败，第 1067 拍三档在这组配对均未得到成功配准。扰动轴用于离线比对，区间复算未使用未来观测。 [相邻拍与传感器方案](CLOCK-PLAN.md)

按这次共用前缀，能过线的第 1064 拍离线分支需要提前知道迟分岔来物将怎样变化。等第一帧测量差异送达再分支，本次动作搜索的净空已经不足。该结果约束冻结前缀和所测动作族，其他共用前缀及松杆期间经验证的保护运动仍有待核对。新增原生评价与训练均为零，v60 正式收口仍停在第一关。 [v60 收口](../../mover_passage_reproof/REPORT.md)

复现用 v60 `a346bea8` detached HEAD，主分支搜索冻结于 `20295dca`，相邻拍与传感器复算冻结于 `241b3753`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/earlier_split
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-EARLY-SPLIT"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/find_split.py" "$output/SPLITS.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/check_clock.py" "$output/CLOCK.json"
```
