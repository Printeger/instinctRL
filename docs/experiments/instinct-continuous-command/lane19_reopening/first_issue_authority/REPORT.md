# 街道第 19 路替代首拍复算 · 限幅恢复仍低于净空门槛

第 1060 拍复现 142 条替代首拍证书。归档扫描重建误差最高 4.81 微米，重复运行原 governor 和 ICS 后，142 个机体系下发向量逐元素一致。原动作、首拍及续行证书均保存在原件。 [输入及结果](RESULT.json)

| 松杆窗全程下发零，第 1084 拍起的动作 | 不加漂移时未被排除的首拍 | 加 0.20 米窗首尾漂移后 | 最佳净空上界 |
| --- | ---: | ---: | ---: |
| 按原 CCG 和原试验变化限幅 | 0/142 | 已由更宽的上界排除 | **0.055 米** |
| 任意水平恢复，正速度变化每拍至多 0.30 米/秒 | 17/142 | **0/142** | **0.148 米** |
| 任意水平恢复，可直接达到 0.45 米/秒 | 45/142 | 17/142 | 0.154 米 |

限幅恢复最紧的是固定姿态困难正扰动，响应时间 0.45 秒、时刻 108.860 秒。用逐方向线性规划独立校验最佳菜单项，该采样的上界为 0.14777 米，低于网格对偶的 0.14802 米。复算允许每个采样点分别挑选最有利的后续动作，并放开后续 ICS、墙面和感知约束。0.148 米是这一动作类的上界。直接全速恢复的 17 条只通过上界筛选，实际可发性、停稳、跟踪与恢复仍待闭环证明。松杆期间的经验证保护动作、菜单以外的首拍和后续垂向机动均超出这项上界的适用范围。 [上界方案](PLAN.md) · [漂移复算](DRIFT-PLAN.md)

采用第 1070–1083 拍的 1.4 秒诊断松杆窗。v60 的正式登记停杆、原生收口和训练预算沿用原结果，本轮 CPU 复算新增原生评价和训练均为零。 [v60 收口结果](../../mover_passage_reproof/REPORT.md)

从 v60 源码 `a346bea8` detached HEAD 运行。首拍复算与漂移收紧分别冻结在 `d25c2f6b`、`f7eef00d`。

```bash
cd ~/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-SOURCE-135
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/isaaclab-training"
analysis=/home/mint/rl_dev/instinctRL-dev/docs/experiments/instinct-continuous-command/lane19_reopening/first_issue_authority
output="$HOME/instinctrl-runs/INSTINCT-CONTINUOUS-COMMAND-CPU-LANE19-FIRST-ISSUE"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/screen_first_issue.py" "$output/SCREEN.json"
~/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B "$analysis/tighten_drift.py" "$output/DRIFT.json"
```
