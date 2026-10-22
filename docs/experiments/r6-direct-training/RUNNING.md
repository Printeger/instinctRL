# 本地直接基线操作

本轮单次入口为 `tools.r6_direct.run`，实际科学worker为 `tools.r6_direct.worker`。
不调用模型，不使用rlflow。当前队列只有一次完整基线；结束进入 WAITING_FOR_ANALYSIS，
之后根据真实分条件结果冻结首段PPO课程。此基线不代表P1完成。

准确源码快照为 `snapshots/baseline-01`（detached 9d5268193934357539ceb8e738079ba29259b2a7），
配置为 `BASELINE_RUN_INPUTS_01.json`。监督器必须使用支持pidfd的 `/usr/bin/python3`；
科学worker使用原冻结Conda Python。CPU --check已实际核验源、模型、资产、扩展输入、两个kernel配额卷，无GPU探测或初始化。

启动（从仓库根目录；不可重复提交同一run）：

```sh
cd docs/experiments/r6-direct-training/snapshots/baseline-01
/usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/BASELINE_RUN_INPUTS_01.json
```

实际以tmux会话 `instinctrl-r6-direct-baseline-01` 执行上述命令。
查看 `docs/experiments/r6-direct-training/runs/R6-DIRECT-BASELINE-01/state.json`、
`stdout.log`、`stderr.log` 和终态 `receipt.json`。父进程在普通Python中等待并清理自己的后代。

停止：创建 `docs/experiments/r6-direct-training/STOP` 或该run目录中的 `STOP`。
STOP不会自动清除；不使用宽泛pkill。超时、STOP、日志限额导致的原生退出与科学原件分别保留。
恢复：先查看receipt及准确PID/starttime/boot身份；同一run目录存在即拒绝重发。
需要修复时创建新输入和新run，重新核对预算与剩余卷容量，不覆写旧结果。

物理输出使用原已预留但从未发车的两个空kernel配额卷中的新 `R6-DIRECT-BASELINE-01` 子目录。
旧卷清单、receipt和账本只读，旧20,250,099,712 bytes容量继续全额计入历史消耗；
新子目录获得新的运行身份，绝不继承旧Store验收。新ledger仅绑定一次该容量，避免重复支出和退款。
源码快照另预留2GiB；当前实际约663MB。原最终确认/交付保护额度保留。

真实结果位置见RUN_INPUTS.plan.attempt_dir，文件在其 `results/research-r6-evaluation/` 下。
该目录保存完整原五任务、16开发blocks的raw/测量/物理审计和原评分。
CPU模块清单位于实际attempt目录；最终NO_RLFLOW审计还需真实运行结束后的模块及进程记录。

## 初始化失败后的准确第二attempt

01准入通过后因共享信号量权限原生-6退出，未产生policy transitions，准确清理已验证。
修复源码 `23d7f5fa`，快照 `snapshots/baseline-02-ipc-final`，输入 `BASELINE_RUN_INPUTS_02.json`；
命令同上但使用该快照和02输入，tmux名称 `instinctrl-r6-direct-baseline-02`。
运行状态在 `runs/R6-DIRECT-BASELINE-02/`。
新IPC命名空间不修改host共享内存；源runtime不变。六项定向CPU测试和namespace内无GPU预检通过。
整个未使用中间快照也保留计账，不删除、不继承执行身份。

## CUDA 初始化诊断

第二 attempt 已清理；CUDA error 304，不是科学结果。准确输入为 `diagnostics/cuda-01/plan.json`，只读单文件源码复用精确 `baseline-02-ipc-final` helper 快照。
使用系统 Python 启动该源码，`--plan` 指向准确输入；`--check` 仅做 CPU 预检。PYTHONPATH 仅含该快照和其 isaaclab-training。
后台 tmux 名称为 `instinctrl-r6-cuda-diagnostic-01`；状态及 receipt 在 `runs/R6-DIRECT-CUDA-DIAGNOSTIC-01/`，系统调用日志和 CUDA 结果在输入的 native_dir。
保留相同 STOP、单写锁、写保护、私有 IPC、全额失败预留；诊断仅一次 cuInit，原资源窗口后最多30秒子进程，预留240秒。

## 第三次完整基线

有界诊断02返回 CUDA_SUCCESS，进程已清理。直接 worker 仅增加当前进程 task 子树的写入/截断权限以允许 CUDA 命名线程，磁盘原件与冻结 runtime 仍受保护。
运行快照 `snapshots/baseline-03`（a1dd492c），输入 `BASELINE_RUN_INPUTS_03.json`，tmux `instinctrl-r6-direct-baseline-03`。
启动方式仍为在准确快照下执行 `/usr/bin/python3 -B -m tools.r6_direct.run --inputs` 加该输入绝对路径。
状态、日志和 receipt 在 `runs/R6-DIRECT-BASELINE-03/`；完整科学输出位于输入的 attempt_dir。
此前失败和 CPU 草稿拒绝均保存；同一 run 不重复启动。

## 第四次基线（延迟初帧评价修复）

第三次结果不完整，原件保留。新源码 `d10ce5b0`，detached 快照 `snapshots/baseline-04`，准确输入 `BASELINE_RUN_INPUTS_04.json`；tag `r6-direct-baseline-04-inputs`。
仅非训练评价中已审计的延迟初帧允许显式 reward unavailable，PPO 接收端拒绝该占位值；actor 测量、评分与五任务/16块不变。
使用原 prepaid 结果卷的新目录，新的16GiB聚合写入上限由当前剩余容量支持；旧数据、旧配额和全部历史计账保留。
从准确快照启动 `/usr/bin/python3 -B -m tools.r6_direct.run --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/BASELINE_RUN_INPUTS_04.json`。
后台 tmux `instinctrl-r6-direct-baseline-04`；状态和 receipt 位于 `runs/R6-DIRECT-BASELINE-04/`。查看 `state.json` 和输入 attempt_dir 下的 `state.json`；STOP 路径沿本文前述规则，停止文件不得自动清除。
时间结算仅针对已准确清理的新基线，按完整 supervisor 含准入/清理 wall 向上取整；原预留记录保存，未知其他量不释放。
