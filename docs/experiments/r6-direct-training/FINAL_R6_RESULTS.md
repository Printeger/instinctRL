# R6 接续结果：未证实学习增益

状态：训练、最终评价与录像已完成，最终独立需求审核待完成；Goal 尚未完成。

冻结 u640 与三个续训 u768 模型均通过原五任务及16个最终条件，每条件32/32。基线同样达到天花板，不能据此声称强化学习改善、统计等价或泛化。最终科学结论候选为 `TRAINING_COMPLETE_SCIENTIFIC_NEGATIVE`。

首段 TRAIN01 实测107更新、54912 transitions，无可验证训练checkpoint；从已接受u640以新身份TRAIN02重算。失败01B实测122更新且无终点checkpoint，01C重新从u640开始，失败原件和费用保留。

| 训练段 | seed | 更新 | transitions | actor/critic steps | checkpoint |
| --- | --- | --- | --- | --- | --- |
| R6-DIRECT-CONFIRM-128-SEED-01C | 2026102202 | 640→768 | 65536 | 1024/1024 | cbd1eda09169439e28586200f0d67e5fc626c862f5a5335f329fe139ccc13cf4 |
| R6-DIRECT-CONFIRM-128-SEED-02 | 2026102203 | 640→768 | 65536 | 1024/1024 | f7632803deaa9f3cabc3cfee3158e3836cee1f64ee1b735b5c9351707dab20f7 |
| R6-DIRECT-CONFIRM-128-SEED-03 | 2026102204 | 640→768 | 65536 | 1024/1024 | 6a616a2582190acdf037bf454b5ae3d921120da3ef85916dff5763de9137e07a |
| R6-DIRECT-TRAIN-02 | 2026102201 | 640→768 | 65536 | 1024/1024 | 8f6bb79650218332159e24d552f17ea88498d015fde293cc5733e2f8df1b3bf6 |

三条确认流共同从u640起步，seed为2026102202/03/04。它们是续训随机流，不是从零独立训练。模型冻结及承诺提交 `74f4dcbe35842f07ad7a38d2bfd4ca319ccf42a3` 发布后揭晓保留集，四模型共同最终seed1812183990；没有用最终测试调参。

模型与配置索引见 [FROZEN_MODEL_BUNDLE](continuation-20261101/delivery/FROZEN_MODEL_BUNDLE.json)，可下载包为 [frozen-models-01.zip](continuation-20261101/delivery/frozen-models-01.zip)。训练统计曲线保留512行连续更新，不是独立episode收益或学习增益证据。

四段录像各330帧、10fps、960×640，共1320帧，完整解码和遥测核验及独立抽样画面审核通过。播放路径和原件hash见 [VIDEO_MANIFEST](videos/VIDEO_MANIFEST.json)。它们呈现保存的评估轨迹，没有新动力学测试；旋翼相位不可得。

科学与工程证据的精确路径/hash、每模型逐条件结果和R01–R10对照见 [FINAL_R6_RESULTS.json](FINAL_R6_RESULTS.json) 与 [训练评价总清单](TRAINING_AND_EVALUATION_MANIFEST.json)。历史CPU概率差异、CUDA补充核验范围、cleanup timeout/-15、失败的渲染/分析记录均保留。

用户已取消累计预算及保护额度。历史账本、未知平台调用和未释放预留不清零；资源自查见 [最终资源记录](continuation-20261101/delivery/FINAL_RESOURCE_AND_CLEANUP_AUDIT.json)。RustDesk保持连接。

复核与复现入口见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

已知局限：

- No demonstrated learning gain: baseline and all candidates pass all registered gates at ceiling; no equivalence or generalization proof.
- Three confirmation streams are shared-u640 continuations, not independent training from scratch.
- Original CPU probability mismatches are retained; exact CUDA supplements have limited binding scope. Full CUDA RNG semantics and independent physical reintegration are not established.
- Cleanup timeout/-15 receipts are retained separately from scientifically complete workload evidence.
- Video is recorded-pose replay with no new policy/physics; visual review sampled first/middle/last; no mathematical per-pixel pose proof.
- Raw index covers1223versions from18analyses, not every historical raw or allvideo records. Absolute paths and frozen environment are required; model ZIP is not a portable runtime.
- Module snapshots do not prove lifetime import history; no complete filesystem mutation audit.
- Current platform model usage UNKNOWN; accounting includes unreleased reservations and is not actual workload consumption.
