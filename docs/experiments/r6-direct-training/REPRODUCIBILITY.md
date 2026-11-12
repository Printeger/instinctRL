# R6 证据复核与复现

先读 FINAL_R6_RESULTS.json、TRAINING_AND_EVALUATION_MANIFEST.json、videos/VIDEO_MANIFEST.json。模型包位于 continuation-20261101/delivery/frozen-models-01.zip，ZIP成员的原路径/hash见 FROZEN_MODEL_BUNDLE.json。原始绝对路径和历史输入保持原样；移动副本必须新增绑定，不能继承旧验收身份。

本机唯一开发目录为 /home/mint/rl_dev/instinctRL-dev，main。运行输入中的 snapshot 是实际执行的 detached HEAD；源码、科学配置及资产hash来自对应 source 清单。CPU与CUDA概率复核须使用原报告指定的准确环境和已封存补充证据；不能用另一模型或任意当前代码替代。

在原证据可见、本机科学环境保持冻结时，可重新进行录像完整CPU核验。以下只启动CPU分析，目标目录必须不存在；再次复核需选择另一个新目录：

```bash
cd /home/mint/rl_dev/instinctRL-dev
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:isaaclab-training /home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11 -B -m tools.r6_direct.analyze --inputs /home/mint/rl_dev/instinctRL-dev/docs/experiments/r6-direct-training/continuation-20261101/delivery/render-09/inputs.json --destination /tmp/r6-video-independent-recheck-20261112
```

查看真实运行状态：读取 inputs.json 指定 output_root 下的 receipt.json，以及 attempt_dir 下的 state.json。CLOSED回执还须核对remaining=[]与准确boot_id/PID/starttime退出，不能只凭旧PID判断。所有已关闭输入只用于复核，不重新运行 START_NEXT_RUN.sh 或旧TRAIN01输入。

新训练、评价或渲染须按 tools/r6_direct/README.md 创建新身份和准确契约、先commit/tag/push、再使用只读快照；原科学终点、STOP和容量检查保持有效。没有无损物理恢复声明；TRAIN01和01B无终点checkpoint，只能按已审新attempt从u640重算。

将来若有新活动作业，触发该输入output_root下的STOP文件可由直接监督器执行有界清理；不要清除已有STOP，不停止RustDesk或无关进程。当前交付的原件无需恢复运行。

录像为保存姿态的Isaac Sim回放，无新动力学/策略推理。视频独审为完整解码、全部遥测及首中末画面抽样，不是每像素姿态数学证明。训练曲线为更新统计，不证明RL改善。全部模型同样通过的天花板条件不能证明等价或泛化。
