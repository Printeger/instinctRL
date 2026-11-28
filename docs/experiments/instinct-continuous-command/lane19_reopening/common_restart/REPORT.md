# 街道第 19 路快速恢复的开环配对

对 17 个仍有理论空间的首拍分别尝试三个方向，共 51 次优化。每个候选用同一速度序列覆盖五种来物及三种响应时间，松杆窗全程下发零。第 1084 拍允许直接下发 0.45 米/秒，其他活动拍仍限每拍变化 0.30 米/秒。最好的共同序列在困难正扰动上有 **0.152 米**净空，但在原来物上降至 **−0.046 米**，出现在 109.875 秒、0.45 秒响应下。最小静态净空为 1.491 米。 [共同搜索方案](PLAN.md) · [数据与原件哈希](RESULT.json)

保留共同序列的最佳首拍及直到第 1083 拍的全部下发，仅从第 1084 拍按离线来物分组优化两套恢复动作。每组做九次搜索，均覆盖三个响应时间。两组数据实际存在相同的停车轨迹，因此对照只衡量后续恢复动作可用的范围。

| 恢复安排 | 最低净空 | 最差对象 |
| --- | ---: | --- |
| 五种来物共用一条序列 | −0.046 米 | 原来物 |
| 同一停车轨迹，原来物及负扰动单独优化 | −0.0047 米 | 原来物 |
| 同一停车轨迹，两个正扰动单独优化 | 0.152 米 | 延迟正扰动 |

最佳各臂的松杆诊断漂移均为 **0.199999 米**，松杆一秒后的最高速度为 **0.0515 米/秒**。第 1110 拍起沿最新指令达到连续五拍 60% 实际进展用时均为 0.5 秒，计分依据为开环执行的速度。正式恢复还需要逐拍可行性和原控制闭环证据。分组搜索首跑没有返回候选。复算把与分支变量无关的前缀漂移改为直接校验，并按原 0.20 米门槛核对。

两组继续共用这一停车轨迹时，单独改变恢复动作仍未让原来物通过。搜索采用近障分离平面及有限初值，结果覆盖本次试过的动作。两组联合未过，原 ICS 后续逐拍可发性未开始验证。新增原生评价与训练均为零，v60 正式收口仍停在第一关。 [分组方案](BRANCH-PLAN.md) · [v60 收口](../../mover_passage_reproof/REPORT.md)

复现用 v60 `a346bea8` detached HEAD。共同搜索冻结在 `9beb15bc`，分组复算冻结在 `0f6b41e8`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/common_restart
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-COMMON-RESTART"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/find_restart.py" "$output/SEARCH.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/find_observed_branches.py" "$output/BRANCHES.json"
```
