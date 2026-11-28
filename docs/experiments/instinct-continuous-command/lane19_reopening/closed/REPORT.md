# 街道第 19 路 CPU 闭环复放 · 困难扰动未达标

第 1060 拍复现冻结 v60 原动作、机体与世界下发、ICS β 和有限边缘状态。替代动作的原指令经完整生产流水线下发，与登记证书相差不到 (3×10^{-9}) 米/秒。两组配对均使用同一冻结权重、原 CCG 指令和 0.35 秒近似跟随响应。困难组的两臂共同固定分支姿态，来物在第 1064 拍之后沿东北向叠加平滑加速度。最低净空按每 5 毫秒计分。 [冻结输入](INPUTS.json) · [结果与原件哈希](RESULT.json)

| CPU 配对 | 对照 v60 最低净空 | 早启替代最低净空 | 晚段零指令最大漂移 | 最新指令恢复 | 替代可行跟踪 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原来物、记录姿态 | −0.080 米 | **0.389 米**（108.535 秒） | 0.261 → 0.231 米 | 1.0 → 0.5 秒 | RMSE 0.074 米/秒，41/41 拍达到沿向进展 |
| 延迟正扰动、固定姿态 | 0.004 米 | **−0.161 米**（109.605 秒） | 0.203 → 0.227 米 | 0.5 → 0.5 秒 | RMSE 0.059 米/秒，15/16 拍达到沿向进展 |

两组的 1.4 秒诊断松杆期均未持续停稳。原来物组在第 1069 拍失去配准，第 1076 拍零指令内原保护重新发出约 0.30 米/秒。困难组在第 1067 拍仍有 0.444 米的最佳局部证书，但没有动作同时守住远端排序净空 0.15 米，控制此时返回 v60。此后净空低于 0.15 米并进入包络交叠。困难组同时改变来物和姿态处理，配对内两臂共享这两个条件。

原来物组达到该路 CPU 净空门槛，困难组否决了早启后失证交回 v60 的方案，未扩到另外三个扰动。原 v60 收口仍停在第一关。正式 v60 评价与本 CPU 简化响应结果分开，新种子、完整 R6 及 32 路长测未启动，新增训练为零。 [v60 收口结果](../../mover_passage_reproof/REPORT.md)

复放从冻结的 v60 源码 detached HEAD 执行。分析脚本及输入对应 `d34bfc19`，标签 `instinctrl-ccg-lane19-reopening-closed-cpu-20261129`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B \
  /home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/closed/replay.py \
  hard_positive "$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-REOPENING/HARD-POSITIVE.json"
```
