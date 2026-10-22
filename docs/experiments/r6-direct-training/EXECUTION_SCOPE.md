# R6 本地直接训练执行范围

2026-10-22 生效。准确用户启动指令保存于 `USER_AUTHORITY.txt`；执行契约为
`docs/goals/InstinctRL_R6_Standalone_Training_Requirements.md`。
这次明确授权真实基线、PPO训练与评价，覆盖旧整理阶段的 CPU-only 限制。
仅替换运行编排：不再使用 rlflow、Store、逐作业模型预审、旧 review hash、聊天会话/RPC发车门。
旧证据、账本、失败和所有未释放预留保持不变；科学标准、actor边界、冻结runtime及资源上限保留。

按 P1真实基线→有界训练→真实评价，P2证据驱动改进，P3三共同u640起点随机流确认、holdout、独审和视频交付推进。
当前为 PREPARING_NO_LIVE_RUN；预算导入和 CPU 检查不构成 P1 或科学完成。

开发在 main，单写者。运行前提交/tag/正常push全部输入，使用准确只读快照或detached HEAD。
运行入口必须与旧 batch/writer.lock 互斥并检查准确进程身份；任何 STOP 均保留并停止新动作。
新输出位于本目录，每run独立且不可覆盖；普通Python负责运行等待，不启动模型轮询。

预算导入使用原账本 `used + Σ max(预留, 各观测)`，每一项保存原JSON指针、完整原记录和计入值。
`tools.r6_budget` 是仅依赖标准库的纯数值工具，复用它不执行旧控制器或导入 rlflow。
导入器只读旧state，获取原writer flock并核对无在途job/model及旧PID身份；不清除旧锁文件。
最终确认/交付预留原样继承，不能用于探索。自动模型会话不等于token；未知平台活动保持UNKNOWN。

已完成的本地检查：旧controller记录PID已不存在、旧current_job/current_model均空、原writer锁可获取、STOP未出现；
u640文件SHA256实际核对为 aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429。
另一个工作区的 rlflow CPU测试不属于本任务，不终止、不修改。

当前命令（尚无live启动命令）：

```sh
python -m tools.r6_direct.import_legacy          # 只读重新核对；不覆盖既有导入
python -m unittest tools.r6_direct.test_import_legacy -v
```

后续：完成单次baseline/train/evaluate入口、串行批处理、实际依赖审计及定向CPU检查；
冻结基线输入后运行首次真实非训练评价。运行前核对资源窗口和预算，不能将本说明当作已发车记录。
