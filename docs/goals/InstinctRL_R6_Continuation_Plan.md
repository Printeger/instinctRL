# R6 接续执行方案

日期：2026-11-01。文档已设计；真实启动条件尚未全部满足。
配套：[任务需求书](InstinctRL_R6_Continuation_Requirements.md)、[Goal 启动 prompt](InstinctRL_R6_Continuation_Prompt.md)。
本方案接续 [原本地训练契约](InstinctRL_R6_Standalone_Training_Requirements.md)，不重启旧调度器，不重建基线项目。

## 当前事实与首要决策

- baseline04 已有 193600 非训练 transitions，原五任务与16开发条件保存评分各32/32，CPU回放通过；独立科学审核待补。
- TRAIN-01 已 CLOSED，运行2571.18秒，DIRECT_CLEANUP_TIMEOUT、退出-15、remaining=[]。原计划 u640→u768、128 updates、65536 transitions、一个checkpoint。这些是计划值，尚非确认实绩。
- 当前原始结果卷未挂载，镜像 root:root 0600；本会话 `sudo -n -l` 返回需要密码。不能据日志尾部猜测模型完整，更不能重复运行原 TRAIN_RUN_INPUTS_01.json。
- 去 rlflow 的直接链已完成 CPU 接线；新分析可以读旧证据，新 live 作业必须使用本轮代码的全新准确快照及输入身份。
- 模型调用边界记录为496已入账+10待结算=506/512，最终保护额度8，差额2。核实账目不能擅自抹去真实调用或获得新额度。

## 执行顺序和阶段交付

| 阶段 | 工作及判断 | 交付/出口 |
| --- | --- | --- |
| A 原件恢复与核验 | 恢复只读访问；核对原输入、回执、工作负载状态、raw、计数、checkpoint、Adam/RNG及概率gate；核对进程退出 | TRAIN_01_RECOVERY.json、CPU分析report/summary/binding；完整/可恢复/不可恢复三分支 |
| B 资源与运行准备 | 一次性结算待记调用及已关闭作业；核对所有维度和最终预留；补齐新快照、契约与输入 | RESOURCE_RECONCILIATION.json、准确CPU预检、下一作业命令 |
| C 配对开发评价 | 完整首段模型与u640按同一协议、条件、随机流评价；保存逐条件及原五任务结果 | POST_TRAIN_COMPARISON.json、下一步决策；基线独审 |
| D 有界迭代 | 只依据C决定是否追加训练/课程；每段先登记假设、变量、终点、资源、停止条件，再训练和评价 | 全部候选证据、DECISIONS.jsonl；冻结配方或科学负结论候选 |
| E 最终确认 | 审核冻结配方，从共同u640运行三条真实不同随机流；冻结模型后揭晓最终保留集；四模型共同评价 | 三流训练binding、冻结承诺、最终评价与汇总 |
| F 审核交付 | 独立审核、模型与配置/哈希、训练曲线、真实轨迹录像、复现说明、账目结算与清理 | FINAL_R6_RESULTS、manifest、video manifest、独审及需求对照 |

A 的恢复分支：

1. 计数/raw/checkpoint齐全且科学检查通过：接受该段实际结果，保留cleanup异常，下一live作业是评价。
2. 部分完成且存在可验证恢复点：记录已完成与丢失区间；恢复必须包含模型、优化器、RNG、counter及课程，遵循既有fresh reset契约并重新过first-rollout gate。新run ID，只登记剩余部分；不能声称逐位接续原物理轨迹。
3. 无可验证恢复点：记为不可恢复；在资源允许时提出从已接受节点重算的全新attempt，列明不可避免的重复计算和重复费用。不按原run ID重试，不把重算称作无损恢复。

C/D 的决策不改原验收线。报告逐任务、逐条件、逐seed的原指标、失败分布及配对差值；32/32的天花板不能证明学习带来收益。原指标无法分辨时结论为未证实改善，不增设事后“更容易赢”的指标。loss下降、控制参数收益和录像观感均不能代替RL收益证据。

默认先评已完成的+128节点；追加段长从已有128/256更新方案中依据结果和预算预登记，不能现在虚构需要多少步。每个新段结束立即评价，存在安全/能力退化、数据错误或保护预算不足时停止追加，先定位原因。

## 启动前操作

以下只读挂载由有本机sudo权限的人执行。先确认原target未挂载、镜像没有其他挂载/写者；若已挂载则核对设备和选项，不叠加挂载。不要运行volume.json里的历史mkfs/fallocate命令，不chmod镜像，不执行fsck写修复。

```bash
findmnt --mountpoint /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-BASELINE-ROBUSTNESS-01/results
sudo losetup -j /var/lib/instinctrl-r6-results/J-d737390980235144/results.ext4
sudo mount -t ext4 -o loop,ro,noload,nodev,nosuid,noexec /var/lib/instinctrl-r6-results/J-d737390980235144/results.ext4 /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-BASELINE-ROBUSTNESS-01/results
findmnt --mountpoint /home/mint/rl_dev/instinctRL-dev/coordination/research/attempts/R6-BASELINE-ROBUSTNESS-01/results
```

`ro,noload`避免写入原镜像及日志回放；如果因此看到不一致，保全原镜像，在足额计账的副本上诊断，不能在原件上修复。不从不可读推断数据丢失。此只读挂载只用于查证，不能作为新训练写入卷。

原路径恢复可见后，先检查原件及hash，再执行CPU分析（目标目录必须不存在）：

```bash
cd /home/mint/rl_dev/instinctRL-dev
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD:$PWD/isaaclab-training" /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B -m tools.r6_direct.analyze --inputs "$PWD/docs/experiments/r6-direct-training/TRAIN_RUN_INPUTS_01.json" --destination "$PWD/docs/experiments/r6-direct-training/continuation-20261101/train-01-analysis"
```

分析失败先诊断，不降低校验强度。分析命令只读取旧训练，不会启动训练。

## 新训练/评价作业的命令交付标准

目前不能发行准确的下一训练命令：恢复节点和新模型hash未知，预算保护缺口未解决，下一作业可能应为评价。下列只是已有接口模板，INPUTS_ABSOLUTE_PATH 必须由A–D的实际决策生成，不能指向TRAIN_RUN_INPUTS_01.json：

```bash
cd /home/mint/rl_dev/instinctRL-dev
PYTHONNOUSERSITE=1 PYTHONPATH="$PWD:$PWD/isaaclab-training" /usr/bin/python3 -B -m tools.r6_direct.run --inputs INPUTS_ABSOLUTE_PATH
```

Goal执行者须在发车前生成 `NEXT_ACTION.json` 和 `START_NEXT_RUN.sh`，明确操作类型、准确绝对输入路径与hash、source commit、只读/detached快照、契约、前驱receipt、新run ID、CPU预检证据及预算；脚本拒绝缺失/漂移输入、已存在run输出和非READY状态。不要为未核验数据提前制造“可启动”文件。

新作业需可写的已批准受限native/results输出空间；原TRAIN-01证据继续只读保护，不能对原镜像直接整体改成可写再混写旧证据。空间/路径变更按新候选绑定并重新核验物理配额，不继承旧receipt的身份。确切存储方案在核验原件和剩余容量后确定；所需管理员挂载权限属于外部条件。

代码、配置、准确输入先提交/tag/正常push；源清单使用已提交实现，快照准确匹配。真实发车时资源准入、STOP和进程检查必须重新执行；CPU PASS不等于GPU资源已就绪。可以先交付READY_TO_LAUNCH里程碑，不能把它写成完整R6完成。
