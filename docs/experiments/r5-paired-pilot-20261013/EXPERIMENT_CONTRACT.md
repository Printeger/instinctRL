# R5 P0 / PLR paired pilot

本轮用户新 Goal 授权准备并在新的准确 Commander 审批后执行短程配对实验。旧诊断 Goal 已完成；其历史额度、失败和验收原件保持不变。本轮不重新做 E0、质量补偿或 Z/T/S 控制基线诊断。

共同只读基线 ef68ebb5682b85aa83044fdea1c13b7d0edbf325，沿用已验收 config、工具、runtime 和输入边界。仅 actor LR 不同：P0=3e-4、PLR=1e-4；critic LR=3e-4。每组原32env/16rollout/64minibatch/1epoch、64updates/32768 training transitions；原1024 policy steps零命令 warmup，init0000/u1/u2/u8/u16/u32/u64七个保存节点。两组使用相同训练seed20260908、评估seed20260909。

各组首次真实512-sample rollout的采样时 old Beta alpha/beta、logp、action、输入和索引必须按 atol1e-6/rtol1e-5在任一 actor或critic optimizer之前一致；失败零参数更新并保存实际故障证据。CPU通过不代替真实gate。原模型诊断记录第一次更新八个minibatch的analytic Beta KL、近似KL、ratio、实际body-frame速度修正m/s、action统计、梯度与参数变化，保留u1/u2完整输入；后续更新沿用实际learning log记录。

每组另有一个独立评估入口，对真实u16/u64 checkpoint做原Z：200steps×32lanes×2=12800 nontraining transitions。沿用原sensor-T0、startup、安全、成功阈值和方向子组；失败样本不删除。记录全部lane水平漂移、速度RMS、高度偏差、速度修正和失效原因。Z只是开发评估，不称完整策略验收。

本轮四个fresh入口合计最多4starts和4初始化预留；训练65536、评估25600、128updates、14个训练checkpoint writes。现有准确mode预算给每个训练3600s行动+10s清理，每个评估1800s行动+10s清理，共10840s outer wall和allocated-GPU上限。每次最多10初始化ticks；训练1显式reset/32768自动reset lanes，评估2显式reset/12800自动reset lanes。每次原coexistence三样本/至少10秒/六个nvidia-smi调用先于AppLauncher；训练最多1225、评估403个runtime CUDA-memory观察。每次terminal且所属清理确认后最多2个postrun nvidia-smi审计调用，无额外探测，无自动重试。磁盘容量在release前核对，实际输出完整计账；没有伪造不存在的写入前输出硬帽。

现有 Commander 使用 accepted readiness 与本轮独立预审 SHA predicates 绑定准确 spec、实际AUTH、config/source/tool/pins、剩余预算。先提交所有运行输入，再从唯一 readonly snapshot release；同一目录单写者，Reviewer只读。评估只能在真实训练gate、完整数据和实际checkpoint源身份／SHA被独立接受后签发。不签发checkpoint占位哈希。

native app.close9.5s失败仍为FAILED，正常owned lifeguard必须证明两次稳定空扫描/无所属残留；记录generic H1 INCOMPLETE/null和全额reservation，不能把现场完成数伪造为正式FINAL。完整job和整个attempt/results按原bytes/SHA建立索引，raw/checkpoints留现有本地存储，大文件不提交GitHub。

最终交付 PAIR_REPORT.md、pair_results.json、STATE_AND_EVIDENCE.md、独立实际数据审核与下一阶段科学决策。比较首次更新幅度/KL/修正和u16/u64原Z指标，明确LR效应范围和只有一对seed的限制，不预设PLR更好，不自动扩至256updates，不改方法。完整负结果也是本轮科学结果；资源或实现阻塞必须明确，不能以提案或未执行模板宣布实验完成。
