# instinctRL 第三轮训练结项

两段恢复预检、六条正式续训和七模型配对评价均已完成。将锚奖励系数从 4 提到 8 后，主指标相对冻结 u640 和等量原配方均小幅变差。两项比较都未达到预登记的 5 mm 改善门槛。全部安全与行为护栏通过。 [冻结输入](FREEZE.json) · [完整结果](RESULTS.json) · [结项复核](COMPLETION_AUDIT.json)

## 正式结果

主指标取 `S-delay-drop` 停令后 [1, 11) 秒的有效测距锚 MAE，单位 mm，越低越好。续训组取三个同种子流的均值，每个模型均保留完整 32 个 lane。差值为候选减去该行对照，正值表示候选误差增加。

| 模型或组 | 主指标（mm） | 候选减该行（mm） | 护栏 | 5 mm 改善门槛 |
| --- | ---: | ---: | --- | --- |
| 冻结 u640 | 50.35 | +0.81 | 通过 | 未达到，条件 95% 区间 +0.31 至 +1.31 mm |
| 等量原配方 u768，系数 4 | 50.79 | +0.37 | 通过 | 未达到，条件 95% 区间 +0.15 至 +0.59 mm |
| 候选 u768，系数 8 | 51.16 | | 通过 | 两项比较均未达标 |

按种子 `2026112201–03` 排列，候选相对冻结 u640 的差为 +1.27、+3.22、−2.07 mm，相对等量原配方为 +0.07、+0.50、+0.55 mm。配对区间按八个指令方向分组重采样，保留每方向四条 lane 和跨模型配对。七模型共用一次保留集实现，续训共用 u640 起点。区间只量化本次评价条件下的方向样本差异，无法外推到新场景或训练种子总体。 [逐 lane 指标与差值](RESULTS.json) · [保留集揭晓](HOLDOUT.json)

七模型在原五项任务和 16 个登记块均为 32/32，包含原有碰撞、越界、间隙、漂移、停令和避让门槛。候选相对两个对照的清晰停令锚误差、射线覆盖和非零指令稳定段速度 RMSE 均在预登记容差内。非零稳定跟踪在 `S-delay-drop` 下的 RMSE 分别为 0.02877、0.03007、0.02571 m/s（冻结基线、等量原配方、候选）。跨全部计入的跟踪条件，候选相对基线的最大退化为 0.00789 m/s，相对等量原配方的稳定跟踪 RMSE 在每个配对条件均下降。最大的清晰停令锚误差增加为 1.62 mm，最大的有效射线比例下降为 0.028 个百分点，均出现在与冻结基线的比较中。 [原任务、登记块与三项附加护栏](RESULTS.json)

## 执行、异常和结算

- 每条正式训练从冻结 u640 严格载入，共 128 次 PPO 更新和 65,536 条转换。两组共享种子、课程和续训量，唯一学习配方差异为锚权重 4→8。共 393,216 条正式训练转换。七个 u640/u768 checkpoint 全部按模型清单重哈希。两段预检另外验证了 u642 写盘和新进程严格恢复至 u646。 [模型清单](MODELS.json) · [训练冻结](FREEZE.json) · [恢复预检](../PREFLIGHT_RESULT_AND_FORMAL_CAPACITY.json)
- 零指令继续在测量空间驻留，非零指令继续调节机体系速度。actor 字段、现有 reward/critic/evaluation 特权边界、底层飞控和冻结 runtime 沿用原协议。评价主指标只消费关闭后的日志。正式输入冻结后先发布全部模型哈希，再揭晓保留集 seed `569431406`。 [冻结方案](../PLAN.md) · [保留集](HOLDOUT.json)
- 首条正式训练的首次 CUDA 概率诊断因 native 卷绑定错误在 live 准入前退出。保留失败输入，按实际绑定重建诊断，原训练随同设备复核完成交接。六条训练均有独立 CPU 分析和同设备诊断，无需重跑训练。两次预检、六次正式训练和七次评价在模拟器清理超时后记录 `CLOSED/-15`，六次 CUDA 诊断正常关闭。查验全部 21 份回执和准确的所属进程身份，无残留。 [准入失败记录](CONTROL-01-DIAGNOSTIC-PREFLIGHT-FAULT.json) · [关闭复核](COMPLETION_AUDIT.json)
- 单卡串行共 21/22 次 live，包含两次预检、六次正式训练、六次 CUDA 诊断和七次评价。按监督器完整生命周期计费 37,216 秒，即 10.34/18 GPU 小时，含预检的 563 秒。首个 live 到末份回执约 13 小时 12 分钟，低于 36 小时。账本计证据 76.14/300 GiB，复核时系统盘余量 282.84 GiB。新增卷分配为零，继续复用 native 临时卷，保留原始证据。 [独立账本](../RESOURCE_LEDGER.json) · [存储绑定](STORAGE_BINDING.json) · [计量口径与资源上限复核](COMPLETION_AUDIT.json)

## 模型和复算

模型路径、SHA-256 和训练分析绑定见模型清单。六个正式 u768 位于 `~/instinctrl-runs/R6-DIRECT-THIRD-FORMAL-{CONTROL,CANDIDATE}-0{1,2,3}/results/research-r6-training/checkpoints/u768.pt`，相同目录保留中途 checkpoint。冻结 u640 引用已验证的历史模型。七个评价目录路径和报告哈希见 `RESULTS.json`。每次启动的回执、输入绑定、分析与工作负载原件均可从结项复核按作业找到。 [模型清单](MODELS.json) · [原件、回执与账目索引](COMPLETION_AUDIT.json)

在本地项目根目录，用冻结协调代码 `40dda6f1` 读取已经保存的报告，可以复算每个主指标、配对差和护栏。

```bash
third_snapshot=docs/experiments/r6-direct-training/snapshots/third-study-v2
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  PYTHONPATH="$third_snapshot:$third_snapshot/isaaclab-training" \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B -P - <<'PY'
import json
from pathlib import Path
from tools.r6_direct.study_report import summarize

formal = Path('docs/experiments/third-training/formal')
freeze = json.loads((formal / 'FREEZE.json').read_bytes())
saved = json.loads((formal / 'RESULTS.json').read_bytes())
recomputed = summarize(freeze, saved['models'])
assert recomputed == saved
print(recomputed['comparisons'].keys(), recomputed['accepted'])
PY
```

从原始记录重建单次评价报告，可用 `python -m tools.r6_direct.analyze --inputs INPUTS.json --destination NEW_DIR`，输入取本目录对应的七份评价 `*-inputs.json`，目标用独立的新目录。正式运行版本、各素材输入和哈希见冻结文件与模型清单。 [直接链入口](../../../../tools/r6_direct/README.md) · [运行输入冻结](FREEZE.json)

## 判断与后续

R6 的两次长训练因证据配额和超时中断，约 117,856 条转换未留下可验证的 checkpoint。提前实测证据增长和吞吐量、写入可恢复的中途点，可减少这类重算。录像返工与训练验收分开结算。本轮则用两个短预检先验证恢复、关闭和证据写量。基线与候选均在硬门槛达到 32/32，但连续停令锚误差区分了两组，且显示原假设预测的改善方向与本次均值相反。 [R6 复盘与理论适用条件](../PLAN.md)

下一步优先用已保留的清晰与延迟丢帧日志，按观测年龄、有效射线覆盖和停令时速度复算配对差，检查退化集中在观测断续还是停止瞬态。清晰条件下候选比等量原配方好，延迟丢帧下三个配对均相反，现有结果无法识别原因。若日志区分力不足，可固定停令前速度，仅改变是否施加延迟与丢帧，先检验这两个解释的不同预测，再确定训练改动。静态单平面的切向漂移可以保持相同测距和 IMU 历史，在该反例下较低的测距锚误差也不保证较小的物理位移。 [条件与逐 lane 结果](RESULTS.json) · [Contribution Ledger](../../../../.maverick/contributions.jsonl)
