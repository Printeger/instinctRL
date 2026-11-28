# 街道第 19 路提前避让的逐拍续证

保留第 1052 拍的同一原始动作，改由每拍实收扫描重选续行动作。引导臂按原离线序列给获证动作排序，保持臂只用上一拍获证方向和 IMU。两臂均在第 1052–1060 拍连续实发 **9 拍**，随后在原 0.90 秒后备时限内实际下发零 **7 拍**，第 1068 拍交回冻结 v60。第 1053 拍，原离线续行动作的 ICS β 为零，当前菜单仍有 67 条引导臂的合格动作配对。选中第一动作精确实发，动态保持下界 0.427 米，移动体转零后备下界 0.443 米。 [冻结方案](PLAN.md) · [数据与原件](RESULT.json)

| CPU 臂 | 来物 | 最低净空 | 晚段松杆漂移 | 可行跟踪 RMSE | 登记恢复 |
| --- | --- | ---: | ---: | ---: | ---: |
| 引导 | 原来物 | 0.398 米 | 0.079 米 | 0.276 米/秒 | 0.5 秒 |
| 引导 | 早分岔正扰动 | −0.147 米 | 0.127 米 | 0.086 米/秒 | 0.5 秒 |
| 保持 | 原来物 | 0.449 米 | 0.085 米 | 0.279 米/秒 | 0.5 秒 |
| 保持 | 早分岔正扰动 | −0.198 米 | 0.060 米 | 0.091 米/秒 | 0.5 秒 |

两臂四条轨迹的晚段诊断松杆均未持续停稳。引导和保持的原来物最低净空均高于原 v60 的 −0.163 米。正扰动中两臂均低于原 v60 的 0.046 米及发一拍后交回的 0.004 米。两组都没有资格进入其他来物和响应时间验证。

第 1061 拍先于早分岔来物的第 1062 拍。此时两种来物在各自配对中的物理状态与下发仍一致，移动体在四帧中的识别、可见度和计时均通过复核。失效发生在墙面平移配准。三个墙面中位数的投影残差低于 **0.001 毫米**，但分半独立平移与整体拟合最大差达到 **1.63 毫米**和 **8.36 毫米**，均超过 1 毫米上限。这个失证使两臂在危险分岔发生之前就放弃了前移。 [配准复核方案](REGISTRATION-PLAN.md)

开环引导和固定起步仅用于诊断。原始生产控制、验收口径和优先级继续按冻结状态使用。新增原生评价和训练均为零，v60 正式收口仍停在第一关。 [v60 收口](../../mover_passage_reproof/REPORT.md)

复现使用 v60 `a346bea8` detached HEAD，逐拍搜索与配准复核分别冻结于 `ecaf810b`、`172000ba`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/adaptive_reserve
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-ADAPTIVE-RESERVE"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/replay_reproof.py" "$output/REPROOF.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/audit_registration.py" "$output/REGISTRATION.json"
```
