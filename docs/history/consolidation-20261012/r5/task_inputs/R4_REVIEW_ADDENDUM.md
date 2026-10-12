# R4 Review Addendum — E0 与 D7-R/B0 实现准备

日期：2026-09-23（Asia/Shanghai）  
结论：**E0 已完成；D7-R/B0 已完成 CPU 层实现准备；未启动仿真、GPU、训练、正式评估或 rlflow 作业；未进入 R5。**

## 1. 权限、角色与状态边界

本次先读取了工作区 `AGENTS.md`、`ACTIVE_WORKFLOW.json`（SHA256 `9da87dca…`）、公共规则 V3（`6ff98175…`）、Builder 入口及当前 Commander/Executor 状态。为避免自我审批，本次涉及源码的写入明确按 **Builder/Executor** 执行；没有写 Commander 状态、批准文件、历史 Order、冻结证据或验收标准。

最新 rlflow 状态文件为 `state/state/RESEARCH_STATE.json`，version 592、`enabled=false`、`paused=false`，SHA256 `e42cc983…`。本次没有调用 `pilot`，也没有注册、批准、释放或提交提案。历史 `CURRENT_ORDER` 仍是 R3 HOLD；用户本次直接授权仅覆盖离线补证与隔离候选的 CPU 实现准备，不把旧 coordination 状态改写成新批准，也不撤销已经存在的批准记录。

授权链事实保持分层：R4 训练作业 `J-3c846ac2069d010a` 已成功；第一次评估所在旧链的失败状态没有被改写；恢复评估 E0002 作业 `J-880cd798ee89d6eb` 由独立授权成功完成。这些事实不等于 D7-R/B0 已获批准。

## 2. 冻结身份复核

| 对象 | 绑定身份 |
|---|---|
| R4 product | commit `903afc5abd0fbf82297527e5790c2cd40827b0b2`; tree `a887c20a900e65039b1b761adf0a623b45f61269` |
| R4 training workload | commit `eb3f9460a914f4a7bf65d894f73b6df30d128d51`; tree `4f37d60c43a82fcb8f04ca22692660d4db518531`; config `74ac41ad…` |
| E0002 evaluation workload | commit `df87b279e7cace6b7589c5bef7359d96f5769f47`; tree `f33c0e586804053d5cd5e90e2a1c0e1013bfb206`; config `fc75a8aa…` |
| R4 u0256 | SHA256 `39ad6806332376b655ee2b37bcf573997a804ef6f9324a2fe0c367fcaaa9886c` |

冻结 product 与 E0002 worktree 在工作前后均为 clean。本次候选从 `df87b279…` 独立派生，没有修改上述冻结源码、checkpoint 或历史证据。

## 3. E0-1：minibatch、update 与 optimizer

**OFFLINE_TEST — PASS。** `minibatches.jsonl` 恰有 2,048 条，256 个 update 各 8 条；每个 update 内 `minibatch_ordinal` 均为 0–7。`updates.jsonl` 的每个 `[1,512]` coverage 全为 1，说明每个样本恰覆盖一次；actor/critic optimizer 累计计数均严格等于 `8 × completed_updates`。

旧诊断把 `minibatch_ordinal` 误当作全局 0–2047 编号。源码实际在每个 update 内对 records 重新 `enumerate`，因此该旧结论分类为 **DIAGNOSTIC_ASSUMPTION_ERROR**，不是训练实现缺陷，也不改变 R4 结果。

## 4. E0-2：同轨迹动作映射与时序

**OFFLINE_TEST — PASS。** 对 4,096 个 policy rows、131,072 个 lane samples 全量复算：

- `alpha = action_sample[:,0]`，最大绝对误差 `0`；
- `v_corr = (2*a[1:]-1)*0.5`，最大误差 `1.49e-8`；
- `v_gov = clip_norm(alpha*command + v_corr, 2.0)`，最大误差 `4.88e-8`；
- `v_final = clip_norm(ICS_beta*v_gov, 2.0)`，最大误差 `4.88e-8`。

ICS 分类为 normal 120,154、intervened 6,702、emergency 4,216。调用顺序由源码与日志共同确定为：actor 先产生 `action_sample`，随后 `environment.step(action)`，该 step 返回的 `pre_reset.control` 和 `pre_reset.task` 写入同一日志行。因此这证明同一 action→governor→ICS 映射，不证明未保存的 Lee/rotor 完整历史因果链。实际速度与同/前一行 `v_final` 都不是代数恒等（平均最大分量误差分别约 0.2373/0.2320 m/s），这是受控对象动力学响应，不应被解释为一行移位即可修正。

## 5. E0-3：early KL top 10

**RUN_FACT + OFFLINE_TEST。** PPO 的 `approximate_kl = mean(old_log_prob-new_log_prob)`；同一 update 第一个 minibatch 在 optimizer 变更前应为 KL 0，后续 minibatch 使用已经被本 update 前序 step 更新过的网络。前 64 updates 的最大十条为：

| rank | update | local minibatch | KL | ratio mean | clip fraction |
|---:|---:|---:|---:|---:|---:|
|1|1|7|20.454645|0.000134|1.000000|
|2|1|6|20.124577|0.003833|1.000000|
|3|1|4|17.097183|0.015015|1.000000|
|4|1|5|15.290251|0.029965|0.984375|
|5|1|3|14.783306|0.079928|1.000000|
|6|1|2|8.097852|0.203909|1.000000|
|7|1|1|3.792447|0.334185|0.984375|
|8|10|5|0.471186|0.896492|0.843750|
|9|2|1|0.341707|0.967939|0.828125|
|10|10|3|0.338009|0.833649|0.703125|

机器结果按 `seed=1701+(update-1)` 重建每条 minibatch 的 64 个 flat indices，并绑定到相应 `action_sample`、`action_mean` 和 command。**NOT PROVEN：** 历史日志没有 old/new log probability 和 Beta concentration，故不能重放每个样本的概率对应关系；机器结果明确写入此限制，没有用合成输入补洞。

## 6. E0-4：真实 actor 输入可恢复性

**NOT PROVEN。** `compact_rollouts` 保存动作、命令和 control，但不保存 `lidar_grid/state_vec`；`partial_rollout_steps` 不保存完整 policy input；measurement windows 仅保存 lanes 0/15/16/31 的部分原始 ray source；评估 raw trace 也没有 actor input。u0256 checkpoint 顶层保存 actor/critic、optimizer、RNG、config、counter 等，但没有环境观测或 actor 历史。

因此无法从历史 artifact 完整恢复真实 actor 输入与概率状态，未执行冻结 checkpoint 历史 CPU 前向。这不是 actor 算法缺陷；它是历史 artifact schema 的诊断可观测性缺口。最小补齐方式是未来获批的诊断在 action 前无损保存实际 `lidar_grid/state_vec`、Beta alpha/beta、mean 和 issued action。

## 7. E0-5：GAE、timeout、tail 与 terminal observation

**OFFLINE_TEST — PASS。** 纯 CPU fixture 覆盖并通过：非零 value 与普通 rollout-tail bootstrap；pure timeout 使用显式 terminal bootstrap；terminated+truncated 时 terminated 优先、bootstrap 为零；reset boundary 截断 GAE 递归；timeout critic 从 `extras.pre_reset.critic_h1` 取值；返回的 reset observation 只作为 `next_value`。辨别 fixture 中 initial/terminal/reset value 分别为 2/7/11，证明 terminal observation 选择不是偶然相等。未导入或初始化模拟器。

## 8. E0 总体判断

E0 未发现阻止 D7-R/B0 诊断准备的训练实现缺陷。确认的真实问题是：历史完整 actor 输入/概率状态不可恢复，以及冻结 unified product 的 `pre_reset` 未保留 Lee/wrench。前者标为 `NOT PROVEN`；后者作为信号覆盖限制单列，不自动扩大成产品或方法修改。

E0 可复算脚本两次运行得到完全相同 JSON SHA256 `4210375fac4a16dd83a023e5fc4c3750474bfe80355634b969088ec411b6ea1f`。

## 9. D7-R/B0 隔离候选

候选 worktree：`/home/mint/rl_dev/worktrees/instinctrl-r4-d7-control-diagnostic-20260923`  
branch：`r4-d7-control-diagnostic-20260923`  
commit：`c7a4d79e35916ce2711085674b9cd37f1054a1ab`  
tree：`1e06553c95d9f34725179919165b28086d81778d`  
config SHA256：`97e2345cc9cd5637659eea3218b5c68fd914fbf8cea0c24f2fdbc2d8b0fe47c0`

候选新增 fail-closed `control-diagnostic` mode：condition、实验身份、u0256 路径/hash/source identity、Z/T/S 和 20,160 transitions 均是精确授权字段；每条件固定一个 start、一个 initialization、3 个 explicit resets、0 training、0 optimizer、0 checkpoint、900 action seconds。实现见 `control_diagnostic.py:17-50`、`authorization.py:63-91`、`budget.py:153-159`。

D7-R 使用 u0256 deterministic Beta mean；B0 固定 `[1,.5,.5,.5]`，但两者都调用 `prepare_h1_step` 并走相同 governor→ICS→Lee→rotor。没有 Bz，没有 direct-B0 旁路，没有自动重试（`backend.py:363-427`）。

每步为全部 32 lanes 保存 command、action、actor alpha/beta/mean、root state、reward、termination、task/control；为 lanes 0–7 无损保存 action 前完整 `[12,59,360]` lidar 与 52-D state，逐 chunk SHA256。Lee/wrench 是 step 后 cache；reset lane 因冻结产品在 reset 时清零而标 `downstream_cache_valid=false`，该 lane 的下游信号仍为 `NOT PROVEN`，不得作完整链证据。

CPU 测试：新增/相关测试 `8 passed in 0.59s`；`--validate-only` 返回 `VALID` 且 `live_authorized=False`。没有运行 simulator/GPU/environment transition。

## 10. rlflow 支持性与缺口

现有 rlflow 能表达精确 proposal/approval、独立 experiment identity、hash pins、GPU/start/transition/wall 资源和零重试；候选 workload 也能 fail-closed 校验新 mode。当前仍缺少的、必须在运行前由授权流程产生的对象是：

1. 注册后的精确 rlflow proposal ID/SHA 与用户/Commander 批准；
2. D7-R、B0 两份独立 machine authorization 与两个 fresh attempt；
3. 基于获批 proposal 的 package/stage/experiment materialization；
4. 显式启用 rlflow、STOP/唯一性/coexistence preflight 与一次性 release。

这些缺口没有被绕过。当前 `enabled=false`，提案没有注册，authorization/attempt/job 均未创建。

## 11. 建议硬上限

聚合上限：2 次启动、2 次 simulator initialization、20 initialization ticks、40,320 nontraining transitions、0 training/optimizer/checkpoint、1,800 action GPU seconds、1,820 秒含清理外层墙钟、16 GiB 磁盘、0 retry。

依据：E0002 的 164,800 transitions 实测为 825.336833 allocated GPU seconds/862.959676 wall seconds；每条件 900+10 秒对小规模诊断及大量无损 I/O有保守余量。完整保存 8 lanes × 630 steps × `12×59×360 float32` 单条件约 4.79 GiB，两个条件加 state/control/manifest 后以 16 GiB fail-closed；它是上限，不是预分配目标。

## 12. 后续命令与批准前置条件

机器提案中给出了两个条件的精确 entry argv 模板。批准前只允许查看：

```bash
./pilot proposal show <P-ID>
```

只有注册出的 proposal bytes 与本报告机器提案一致、用户/Commander 明确批准 exact SHA 后，才可执行：

```bash
./pilot approve <P-ID> --sha256 <EXACT-PROPOSAL-SHA256>
./pilot resume --package <APPROVED-PACKAGE-ID>
./pilot package-status <APPROVED-PACKAGE-ID>
./pilot job-status <JOB-ID>
```

每个获批 experiment 的 workload argv 必须是：

```bash
timeout --signal=TERM --kill-after=10s 910s \
  /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B \
  -m training_workloads.unified_h1.formal_unified_h1 \
  --authorization <D7-R-or-B0-exact-authorization.json> \
  --attempt <matching-fresh-attempt> \
  --mode control-diagnostic
```

前置条件包括 exact proposal approval、两份独立授权、fresh attempts、候选/product/checkpoint/config hashes unchanged and clean、rlflow authorized enable、无 STOP、无 active/uncertain job、coexistence gate PASS。任一失败即停止且不重试。

## 交付索引

- `r4_review_e0.py` — SHA256 `3c2e67fd…`
- `r4_review_e0.json` — SHA256 `4210375f…`
- `r4_d7_candidate_cpu_results.json` — SHA256 `07806511…`
- `R4_D7_CONTROL_DIAGNOSTIC_PROPOSAL.json` — SHA256 `62992b22…`
- 本 addendum 与隔离候选 commit `c7a4d79…`

最终状态：**E0_COMPLETE / D7_CPU_PREPARED / LIVE_NOT_AUTHORIZED / R5_NOT_ENTERED**。
