# 原始训练证据的补充验收论证（待独立裁决）

本文件是Builder提出的可核验论证，不是独立接受或事后运行时日志。旧joint-actual-training.json仍为STOP，两组原件缺少actor_mean_prior_binding.json的事实不变。

原实质要求是：共同u256起点恢复后、首次新更新前安装**当前组**0/10辅助损失；只影响actor loss，不替换运行时动作、不使用privileged输入；同seed/共同完整首512、原概率gate、全部新checkpoint/计数/评价标准不变。

## 可核验的原始证据链

1. 新分析核对每组完整inventory原件/副本，对冻结snapshot的formal_segment、mean_prior、PPO、driver、logging、backend逐文件与精确提交feb1e96f97e44c7d872cec44268fc2d8b352bc93比较。运行request/spec/AUTH/cwd与该source/config对应。
2. 冻结formal_segment在_load_evaluation_checkpoint恢复后调用configure_trainer(p)，随后才创建loop并run_update。冻结mean_prior以当前p['mean_prior_weight']作为闭包捕获值；在这一训练路径没有后续替换hook的代码。共同origin是mean10，但当前R0=0/R1=10，不能从origin推断当前值。
3. 冻结PPO每个minibatch以auxiliary=None起步，只有self.actor_auxiliary非None才调用它并加到total_loss；仅当实际auxiliary非None才写actor_auxiliary_loss。该字段不是默认0占位。logging.record_update原样复制每个PPO record，不生成此字段。两组所有2048条原minibatch均实际含字段，R0全0，R1全正。
4. 所有2048条total_loss都按原float32算式与actor_loss/value_loss/entropy/aux重新核对，原atol1e-6/rtol1e-5保持。它支持辅助项被纳入实际优化loss，不只是在旁边打印一个数。
5. 首更新8个真实preoptimizer probe与真实512输入、actions/oldlogp/advantages/returns/indices对应；用恢复后的原permutation_seed+256、批准命令state_vec[45:48]、输出mean[1:4]逐minibatch复算原0/10公式。另读取真实update258输入。无actor forward、optimizer、采样、仿真或新checkpoint写入。
6. 冻结PPO只把policy和已有actor output传给hook；mean_prior只读已批准command与mean；无critic/privileged输入、无动作覆盖。这个边界由精确实际运行源码与日志路径共同支持，不能仅以训练成绩推断。
7. 原分析器其余完整检查保持执行：准确job/owned ACK、首次gate、131072additional transitions/256updates、所有10个真实checkpoint的源/方法/seed/LR/Adam计数/有限值、mixedclock/reward公式及全部原件哈希。新增报告显式为CPU_RECOVERY_CANDIDATE_REQUIRES_INDEPENDENT_ADJUDICATION，不能自行授权评价。

## 限制与必须独立裁决的事项

- 缺少单独的原始binding文件是真实缺陷。本次不生成该文件、不修改attempt或历史审核。
- 以后更新没有保存每个minibatch完整actor输出，不能声称复算全部2048次aux公式；全部标量计数与total-loss纳入关系已复算，完整aux公式只对首8次复算。既有验收器同样只对首8次完整复算。
- 原显式“恢复后当前callback实际日志”可否由这些原始执行日志共同满足，必须由Reviewer独立裁决。若认为仍需契约豁免，返回STOP，不给no_contract_waiver=true，不issue评价。
- 现有episodes尾部有32条完全相同的截断记录重复写入，每组raw192/unique160，直接累加134144而unique/compact/update为131072。只在统计视图按相同(env,episode,reset)且内容完全一致去重，原件不动；不得掩盖失败/reset。这不影响compact训练计数，评价仍用原case逐步raw/原函数，不能用episode汇总代替。
- nativeFAILED/-9/app.close9.5秒与collector INCOMPLETE/null保持，全额保守预留不变。不能将科学接受改称进程SUCCEEDED。

## 本次实现边界

main上的formal_segment只增加恢复后、reset/更新前原子写binding文件，CPU测试覆盖R0/R1权重、阶段、授权SHA及更新前存在。修复不会部署进已冻结source；已有checkpoint继续绑定原feb1e96source，原评价也使用该冻结source。

普通training_analysis仍严格要求原binding；只有显式指定original_log_binding验证器的新离线路径会检验上述替代原始证据，且输出待审候选。没有隐式fallback或条件跳过。

恢复器继承原Batch的submit/wait/settle/postrun/owned-model-worker；Commander/launch_gate不改。只允许额外一个checkpoint-recovery模型调用，须接受精确原checkpoint及恢复代码/plan后才可运行两个原评价，随后一次final-analysis。禁止训练入口，禁止清除外部STOP，禁止重置旧调用/失败或自动重试。独立Reviewer同时审核恢复代码安全，不接受则停在CPU审核阶段。
