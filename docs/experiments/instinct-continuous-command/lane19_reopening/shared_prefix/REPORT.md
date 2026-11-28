# 街道第 19 路的共用前缀与松杆保护位移

两组共同优化第 1067 拍之前必须相同的动作。所有序列逐 5 毫秒计分至第 1149 拍，覆盖五种来物与三个速度响应时间。三组均未同时达到物理净空和诊断停稳要求。 [固定方案](PLAN.md) · [逐组结果和原件](RESULT.json)

| 动作条件 | 首拍数量 | 最佳序列最低净空 | 松杆最大漂移 | 松杆一秒后的最高速度 |
| --- | ---: | ---: | ---: | ---: |
| 窗内全零，第 1067 拍可离线分支，恢复首拍可发 0.45 米/秒 | 17 | 0.14326 米 | 0.1999 米 | 0.0517 米/秒 |
| 松杆初期可保护位移 0.4 秒，全场景同一序列，限幅恢复 | 142 | −0.05344 米 | 0.1999 米 | 0.0699 米/秒 |
| 松杆初期可保护位移 0.4 秒，第 1067 拍可离线分支，限幅恢复 | 142 | **0.14754 米** | 0.1999 米 | 0.0699 米/秒 |

全零组将上一轮第 1067 拍固定前缀的净空 0.072 米提高到 0.143 米。保护位移并允许分支的最佳序列距离 0.15 米门槛差 **2.46 毫米**。最差是困难正扰动、0.45 秒响应，发生在 108.895 秒。该候选的求解器没有报告收敛，表中物理值为输出序列的独立全采样复算。这是有限搜索结果，其他路径仍待验证。两种分支组采用了不同的恢复首拍限制，表中差值同时包含动作和限幅差异。

使用全零组最佳共同前缀重新渲染批准的扫描历史。第 1067 拍只有 0.45 秒响应一档能同时配准原来物与困难正扰动。组间速度区间因此没有覆盖全部响应。第 1067 拍仍只作为离线分支时间，原 ICS 下发和回退验证尚未启动。原生评价和训练新增均为零，v60 正式收口仍止于第一关。 [新前缀观测复核](OBSERVATION-PLAN.md) · [v60 收口](../../mover_passage_reproof/REPORT.md)

从 v60 `a346bea8` detached HEAD 执行，联合搜索冻结于 `0e6924f7`，观测复核冻结于 `d81bff00`。两个脚本之间可在 `SEARCH-strict_joint.json` 写完后启动传感器复核。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/shared_prefix
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-SHARED-PREFIX"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/find_shared_prefix.py" "$output/SEARCH.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/audit_prefix_observation.py" "$output/OBSERVATION.json"
```
