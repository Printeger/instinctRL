# R5 Control Chain Diagnosis

结论：`MIXED_CAUSES`；训练层结论：`BLOCKED_BY_CONTROL_BASELINE`。

## 证据有效性

- 复用 D7-R `J-44f25963c38ad089`，没有重跑；新 B0 仅 `J-ea67539f3baab11d` 一次。
- 两者均有 Z/T/S=200/100/330、32 lanes、630 steps、20,160 transitions 的完整、有限且 hash-valid 原始轨迹。
- D7-R invalid downstream samples=19、fully invalid chunks=0；B0 invalid samples=64、fully invalid chunks=1。B0 Z/0166 的 Lee/wrench 缺失仅对应全 32 lanes cache invalid，没有补零。
- 三个 case 的 initial root state 最大绝对差均为 0。

## 因果链

D7-R 的 deterministic actor mean 被精确映射为持续非零 `v_corr`，且 `v_final==v_gov`；Z 的平均 `v_final_z=+0.154103`，实际平均 z 速度 `+0.061127`，终点高度中位数增加 `1.214620 m`。这支持上游 residual 驱动爬升。

B0 issued action 始终精确为 `[1,.5,.5,.5]`。Z 中 `v_corr=v_final=0`，但实际平均 body velocity 为 `[0.185976,-0.052585,-0.090423]`，终点高度中位数下降 `0.307523 m`，step 166 全 32 lanes below-height reset。S 在 step 165 同样全 lane below-height reset。这证明下层闭环/plant baseline 本身不稳定。

因此：不是纯上游，不是由初始状态差异造成，也不能把全部问题归到 reset。上游 residual 和下层 baseline 分别造成方向相反的不安全行为，选择 `MIXED_CAUSES`。由于 B0 是 actor-LR 对照的控制基线且它本身不安全，P0/PLR 不具备科学准入。

完整机器数据：[pair analysis](../r5_d7_b0_offline_analysis_20260927.json)。
