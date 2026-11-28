# 街道第 19 路的提前共用余量

第 1052 拍起存在一条五种来物和三档响应共用的开环序列。其松杆窗全程下发零，最低净空 **0.235 米**，最大漂移 **0.1999 米**，松杆一秒后最高速度 **0.0508 米/秒**。第 1055、1058、1059 拍起的同类搜索均低于 0.15 米，分别只有 0.121、0.022、0.018 米。第 1052 拍开环可行让问题从能否留出位置转为能否持续发出动作。 [固定方案](PLAN.md) · [原件及结果](RESULT.json)

第 1052、1055、1058、1059 拍分别有 69、154、167、167 个满足短时绝对证书的非零首拍，均未达到原排序的 0.08 米改善条件。第 1059 拍裸 ICS 下发为零，原生产下游墙停复核则下发约 0.246 米/秒，两者单独归档。

原来物和早分岔正扰动的配对使用相同固定姿态和 0.45 秒跟随。第一次回放在第 1052 拍验证了首个目标的裸 ICS 输出，计划中的下一动作却无法从同一拍通过 ICS，动作未下发。将第二动作锁为首拍原短证已覆盖的续行后，共用开环净空为 **0.228 米**。两组均在第 1052 拍通过全链实发复核，β 为 1，实发误差小于 (2×10^{-8}) 米/秒。该拍保持净空下界 **0.442 米**，转零后备 **0.402 米**。第 1053 拍当拍目标仍可发，计划中的下一动作又被当拍 ICS 否决，试验控制退回原 v60。 [下发方案](READBACK-PLAN.md) · [固定续行方案](CONTINUATION-PLAN.md)

| CPU 实发配对 | 原 v60 最低净空 | 发一拍后交回 | 发一拍后松杆漂移 | 登记恢复 |
| --- | ---: | ---: | ---: | ---: |
| 原来物 | −0.163 米 | −0.137 米 | 0.161 米，未持续停稳 | 0.5 秒 |
| 早分岔正扰动 | 0.046 米 | 0.004 米 | 0.191 米，未持续停稳 | 0.5 秒 |

受扰臂净空恶化，单拍提前动作后交回不足以采用。未来轨迹用于开环搜索，测试动作尚无通用触发依据。第一版物理复算误用了归档字段名称，改用实际存在的下游保持时长后执行。原生评价和训练均新增零步，v60 正式收口仍停在第一关。 [v60 收口](../../mover_passage_reproof/REPORT.md)

复现从冻结 v60 `a346bea8` detached HEAD 执行。物理搜索、第一次下发核对及固定续行分别冻结于 `92c6d1bc`、`9d6b18e5`、`7572441d`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/earlier_reserve
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-EARLIER-RESERVE"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/find_reserve.py" "$output/SEARCH.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/readback_reserve.py" "$output/READBACK.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/pin_first_continuation.py" "$output/CONTINUATION.json"
```
