# R1：统一停止驻留与安全速度跟踪训练契约

日期：2026-09-08。方案版本：`unified-h1-v1`。状态：**方案完成，集中等待用户确认；未授权实现或运行**。

唯一推荐：保留一个四帧 actor 和现有飞控/ICS，冷启动；32,768 transitions 零指令预热，随后持续 50/50 混合；统一实际速度主项，独立于 actor 动作的原指令风险门控，危险指令下奖励制动，碰撞/高度失败统一一次扣 300；critic 改为局部位置加当前指令的 21 维。首轮新训练 131,072 transitions，以分项行为决定是否扩大。

本文件是用户授权的 Commander R1 设计交付，不是 Builder Order，也不是训练授权。所有数值是本次选定的可证伪工程方案，尚未经新方法实验验证。用户确认本方案后，Commander 才签发 R2 CPU 实现任务；R3 起各运行包另有精确输入和预算授权。

## 1. 基线、已审交付与旧任务处置

已读取 ACTIVE_WORKFLOW、V3、Commander 入口、双方 canonical state、CURRENT_ORDER、LAST_REPORT、A0004 完整报告和审批，以及 roadmap 全文。A0004 APPROVE 已覆盖最新交付；Executor 的 READY_FOR_REVIEW 是历史交付状态，不构成新的待审任务。F1–F4 全部保持关闭，不重启 bootstrap，不重复审核或重跑旧实验。

| 用途 | 只读源码根 | Commit / Tree |
| --- | --- | --- |
| 新方法产品基线 | `/home/mint/rl_dev/worktrees/instinctRL-scale-004-f1-product` | `de9d6b8c03274a11760a3f06ca68d38cc0058f10` / `afb386b0ccc0a023d77afaf20aab0f39c888fb46` |
| 新方法 workload 基线 | `/home/mint/rl_dev/worktrees/instinctRL-scale-004-f1-workload` | `ee6956b463eb70549f21c4348310a1d483ccbc3a` / `caa54a30a65a54cd187fb918613486c53aea5ee8` |

本轮核对两根 clean、branch/HEAD/tree 与审批一致。R2 在上述两根的隔离 additive 后继实现，不能直接修改它们。原 blackwell 产品、正式源、冻结 runtime、A0001–A0004、旧 checkpoint、失败 D0001 与所有证据绑定源码保持只读。

SCALE-004 处置为“旧执行路线停止推进，由新路线接续；训练目标未完成”，不是 APPROVED 训练成功。累计仍为一次失败启动，0 transitions、0 updates、0 checkpoints；旧 station/tracking 额度仍受不存在的 final1954 handoff 限制，不挪用、不执行。CURRENT_ORDER 与归档 SCALE-004 bytes 相同且被历史输入绑定，本轮保留原字节；本规划及 Commander 的 `training_roadmap` 状态是当前规划入口，旧 Order 的历史 EXECUTABLE_NOW 文本不恢复已耗尽权限。

审批依据：[A0004 APPROVAL](../reviews/SCALE-004_A0004_APPROVAL.md)。本轮源码/旧轨迹核对入口：[R1_READ_ONLY_AUDIT.json](R1_READ_ONLY_AUDIT.json)。这不是 A0004 复审，也没有产生新 MUST_FIX。

## 2. 目标与不变边界

首个目标为 H1：显式有效零指令下停止并在有限窗口驻留；安全水平非零指令下跟踪；朝静态墙的指令下减速/停止；同一 checkpoint 完成运动到零、零到运动和反向转换。

H2（受扰后返回旧参考位置）不纳入本轮承诺。四帧在 0.1s 间隔下只跨约 0.3s，anchor 历史参考只在控制内部，不能用加大 anchor 权重补足 actor 缺失的旧参考信息。R2 加信息不可区分性 CPU 检查用于记录能力范围，不因此扩 actor。

保留 actor 输入 `lidar_grid [N,12,59,360]`、`state_vec [N,52]`，四帧获准测量/有效性/可靠性/时间、IMU 派生量、当前和 previous-issued 指令；四维 Beta 动作、governor correction limit 0.5m/s、command limit 2m/s、anchor、实际 ICS 和 Lee/rotor 路径不变。位姿、真值/估计速度、origin、env id、课程时钟、地图和障碍真值不能进入 actor。

物理 dt=0.005s、decimation=20、policy dt=0.1s；MID360 40m、挂载与射线定义不变。初步仅水平八方向，vz_command=0；实际三维速度与高度始终纳入奖励和评估。没有消息与显式零指令不同，命令超时策略留给 R7。

## 3. 源码审查导致的明确选择

以下是新方法设计依据，不重开已关闭的 F1–F4。

| 源码事实 | 本方案处理 / 理由 |
| --- | --- |
| `task/rewards.py` 的 ordinary gate 使用 actor 输出后的 beta；零指令旧系数为 2 | 新 gate 使用 pre-action 测量和原始 command；统一速度系数 1，额外零速度项 0，明确不等价于旧奖励 |
| ICS 在 0.2m/s、d_safe=0.8 时约到射线距离 0.81m 才衰减 | reward-only 风险计算用 d_safe=1.0m，提前留下 0.2m 学习制动缓冲；实际 ICS 仍 0.8m，不把它称为物理制动保证 |
| 关闭危险指令的 tracking 后缺少明确实际速度制动目标 | 加 `-q*||v_actual||`；避免危险区域只剩弱 clearance/proxy 而无停止信号 |
| `termination.py` 高度 <0.5 或 >4.0 为真正失败，旧 reward 无高度 terminal cost | 碰撞/上下越界 union 一次扣 300；替换旧 collision 10，保留终止分类和 masks |
| `critic.py` 为 18 维绝对位置、无 command；network/rollout/checkpoint 也有严格维度检查 | 新显式 21 维版本贯穿全部接线；不能只改首层或静默放宽 loader |
| `checkpoint.py` 要求 Adam 全参数 state；冷启动 Adam state 尚为空 | v2 只在 update=0 且两个 optimizer 均未 step 时允许精确空 state；不用假 optimizer step“初始化” |
| `live_training.update()` 丢弃 PPO records | 原样持久化现有每个 minibatch records，加行为/target 分项日志 |
| 原始 fixture 固定墙 x/y=6m，中心起点短低速 block 未必遇险 | 预热仍中心；混合 reset 分布加入 25% 近墙起点，几何不变，明确这是训练分布变化 |
| `exact_clearance_m = ranges_m.min()` | 保留旧字段；新独立几何距离评估不以 ray 最小值冒充机体净空 |

已只读查看 EVAL-002 两份 partial_rows：每份 256 行、2 env；各 4 次上界终止，episode 长度约119–121步，终止前 correction_z约0.271–0.273m/s、实际世界 vz约0.182–0.184m/s。station 平均旧 reward=-2.53909，其中 anchor=-1.71481、零速度=-0.82378；tracking 平均=-0.98123。完整分项和终止行见 audit。旧行有动作/飞控/终止/分项 reward，**没有 pre-action critic value、训练 advantage/target/KL 或完整 LiDAR 历史**；不补造、不声称能回放新 risk gate。观测说明持续向上和终止风险，不证明 reward hacking 的因果关系。

## 4. Curriculum 与时序：`curriculum-h1-v1`

32 env，horizon16，512 transitions/update；每 update 1 epoch、minibatch64，因此 actor Adam和critic Adam各8 steps。固定 PPO：gamma .99、GAE lambda .95、clip .2、value coefficient .5、entropy .01、双 lr 3e-4、grad clip .5、permutation seed1701；网络主体/初始化算法沿用基线。新训练 seed=20260908，环境/课程随机生成器分离并持久化身份。

| 阶段 | 新 transitions | updates | command |
| --- | ---: | ---: | --- |
| 预热 | 32,768 | 64 | 全部精确 `[0,0,0]` |
| 首轮混合 | 98,304 | 192 | 每步16 zero +16 nonzero |
| R4 合计 | 131,072 | 256 | 新 counters 0→256 |
| R5 增量上限 | 917,504 | 1,792 | 持续50/50，不退火 |

混合时令 k 为已完成的新实验 policy steps，j=k-1024，block b=floor(j/128)。env e 的组 g=floor(e/16)；g=b%2 为非零，否则为零。每128步（12.8s、8 rollouts）交换。方向集合保持原八项顺序 `[+.2,0,0],[-.2,0,0],[0,+.2,0],[0,-.2,0],[+.15,+.15,0],[-.15,+.15,0],[+.15,-.15,0],[-.15,-.15,0]`；该 env 第 m=floor(b/2) 个非零 block 取 index=(e%8+m)%8。无需依据真值/成败挑选方向。每个非零组每个方向恰有2 env。

每个完整混合 rollout 必须实际计数256 zero、256 nonzero，包含 terminal samples；预热结束不按 hover是否达标延长。阶段边界不换网络、不清 optimizer、不重置课程 clock。R4 中两组各有12个非零 block；后续一直轮换。block被reset截断独立标记，不把“从reset开始零指令”当作真实运动后停止。

Reset：预热全部局部 `(0,0,2)`、identity quaternion、零初速度。混合开始不强制reset/瞬移；之后每个自然reset，e%8<6仍中心，=6为 `(4.6,0,2)`，=7为 `(0,4.6,2)`，即配置上24中心/4近x墙/4近y墙。保留1000步timeout、原高度和碰撞终止。近墙首个完整非零块通过固定方向序列覆盖朝墙；真实近墙、风险与转换样本数必须记录。无随机扰动/噪声/动力学随机化。实际reset分布与配置分布分别报告，失衡不改变全局50/50 clock。

单步契约：obs_t 只更新一次历史且含 c_t；actor采样a_t、真实控制及反事实risk都读这份历史；物理推进；reward用c_t与pre-reset实际 v_{t+1}，anchor使用现有控制本步指标；terminal记录完成后，obs_{t+1}才装入c_{t+1}。当前critic=V(s_t,c_t)，普通next value=V(s_{t+1},c_{t+1})。timeout bootstrap必须是截断前物理状态配下一指令的critic-only构造，不多读传感器、不插入actor历史，也不用reset后的状态。

Reset不重启k、方向序列或剩余hold；清观测/anchor/低层原有缓存，保留首次anchor不可用的真实状态。warm boundary发生在完整rollout边界；CPU检查k=1023/1024、1151/1152、timeout与command同时切换、局部reset和续训边界。

## 5. Reward：`reward-h1-v2`

所有向量在本步对应body frame，速度单位m/s；reward逐0.1s transition记一次，无额外dt缩放、clip、归一化或alive bonus。

### 5.1 精确的原指令风险门控

以同一pre-action `lidar_grid`、同一mounted rays、原始c，纯调用现有 `attenuate_command` 数学函数，使用**独立reward配置**：d_safe=1.0、emergency=.25、a_max=2、active_horizon_margin=.5、min_reliability=.1、velocity_limit=2、latency=0、clearance_margin=0、range_rate_filter=false、empty_active_set_beta=1，history_dt=.1。不写实际ICS cache、不发出其返回command、不新增sensor采样。实际ICS继续基线配置。

等价定义：最新有效且reliability≥.1的beam为reliable；a_i=max(ray_i·c,0)，d_i为最新range，s_i=sqrt(4*max(d_i-1.0,0))；active_i=reliable AND a_i>0 AND d_i≤1.5。beta_cmd取active beams中min(clamp(s_i/a_i,0,1))，空active取1；任一reliable d_i<.25则beta_cmd=0。除法仅对a_i>0的项计算。

令 h=1[||c||≤.001]，q=(1-h)*1[beta_cmd<.999 OR emergency_cmd]，g=(1-h)*(1-q)。因此所有零指令q=0；q与当前候选action无关（既往动作改变未来物理状态是正常MDP因果，不在此排除）。判定用pre-action观测，不能用actor本步产生的beta。

可靠性边界：无active但存在有效可靠beam时按1处理，不把空active称作已证明安全。最新reliable beams占比<.1或观测时间/finite/validity违反契约，标记 `MEASUREMENT_COVERAGE_UNAVAILABLE` 并安全停止本次开发运行，保留原始失败；不能按q=1消除跟踪损失、不能伪造task终止或继续学习缺失数据。这是R3–R5证据覆盖gate，非部署安全策略。新近墙起点的真实覆盖由R3检查；局部覆盖通过也不保证所有方向可见。

必要反例：同obs/c而两action让实际ICS一项介入一项不介入，q必须相同；零c在近墙时q仍0；朝墙与离墙q可不同；对齐beam d=1.005、c=.2产生beta≈.7071且q=1，d=1.02产生beta=1且q=0；同beam d=.8、离墙无closing时q=0（无emergency）；无可靠beam走缺失分支。新reward不使用actor自己的v_final作为跟踪目标。

### 5.2 唯一完整公式

记 v=实际v_{t+1}，u=v_gov，f=v_final，f_prev=本步之前issued；A为旧anchor Huber loss，a_use=旧anchor active且usable_fraction≥.1，d_ray为旧exact_clearance_m；B为实际ICS beta。

```text
r_velocity     = -(1-q)*norm(v-c) - q*norm(v)
r_zero_extra   = 0
r_anchor       = -4*h*a_use*A
r_clearance    = -0.5*max(1.0-d_ray, 0)
r_intervention = -0.05*((1-B)+norm(u-f))
r_smooth       = -0.05*norm(f-f_prev)
r_gov_proxy    = -0.1*g*norm(u-c)
r_final_proxy  = -0.25*g*norm(f-c)
r_preservation = -0.5*g*(max(.75-ratio,0)+max(ratio-1.05,0))
ratio          = norm(f)/norm(c) 仅在nonzero下计算；zero日志UNAVAILABLE
failure        = collision OR below_height OR above_height
r_failure      = -300*failure
r_total        = 上述所有项的和
```

实际奖励按finite张量计算，不能把日志的UNAVAILABLE塞入数值路径。危险非零指令q=1时实际速度目标为停止，保留clearance、intervention、smooth；安全非零保持原速度范数主项和proxy权重。零指令速度总系数从2降为1；无“correction非零即罚”项，制动可用非零correction。anchor保留eps_enter=.05/eps_exit=.15、Huber delta=.25以及原缓存/时序，不声称增加长期位置反馈。

Terminal union只扣一次300，旧collision=-10在v2停用；同时碰撞/越界保留所有masks，reason优先级仍collision>below>above>timeout。高度边界仍world z<.5或>4（本fixture origin_z=0），等号不终止。timeout1000步仍truncated；collision/height即使同timeout也为真正terminated。真正失败bootstrap=0，纯timeout用pre-reset terminal value，GAE不跨reset递推。

### 5.3 失败代价量级与离线可反驳检查

gamma=.99时300对应每步成本3的无限折扣尾部。旧station均值2.539的尾部约253.9；将零速度系数降至1后，按旧样本已知项重算的量级约2.127/步（不是新方法轨迹），选择300留出量级余量。奖励最坏值无此上界，**300不保证所有可能状态都无终止捷径**。

R1已做纯算术反例：安全驻留持续cost=.3，带bootstrap折扣return=-30；缓慢上升120步每步cost=.2后失败、下降80步、碰撞40步，return按 `-.2*(1-.99^N)/.01 - 300*.99^(N-1)` 均小于-30（具体值见audit）。旧零高度代价的对应上升/下降return反而可高于-30。这是激励量级检查，不是物理可达性或真实reward实现通过。

R2必须调用新reward实际函数构造安全驻留、安全非零、上升、下降、碰撞、碰撞+越界、纯timeout的张量轨迹，将各项及GAE与手算核对；terminal step用同一prefix、同一决策时刻和明确bootstrap尾部比较，不能靠不同截断长度造结论。R3测实际B0分项量级与critic target；若合理安全轨迹的持续成本逼近/超过3或出现更优提前失败反例，暂停训练，记录明确假设，集中提交方法修订，不能由Builder临时调300。

PPO本身的采样/更新关系依据[原论文](https://arxiv.org/abs/1707.06347)；任务终止与外部截断的区别依据[Farama说明](https://farama.org/Gymnasium-Terminated-Truncated-Step-API)。本项目比例、风险阈值和300均由源码及目标作工程选择，非文献结论。

## 6. Critic与初始化/checkpoint

`critic-h1-v2`固定21维，原18字段顺序保留，仅前三位置替换为 `(position_world-env_origin)/10m`，追加c_t三分量（m/s，缩放1）。姿态、world/body线速度、world角速度、旧ray-clearance/collision字段和单位保留；不做在线normalization或clipping。隐藏层仍64/64。平移测试用float32可表达一致状态及声明容差，不能强求大坐标浮点减法任意bitwise相等；同local状态在0/500m/原32-env origin下，位置缩放后abs差≤1e-5，command变化必须体现在末3维，actor字节不受critic输入影响。

这仍是近似value输入：anchor参考、测量历史、风险q、指令剩余hold和控制内部状态未全包含。不在第一轮扩critic到完整Markov状态；记录zero/nonzero、q分组value residual。相同21维但不同anchor/clock导致不同target的样本应披露，不能把value误差都归为训练步数不够。

`checkpoint-h1-v2`：新版本入口保存actor/critic/schema/field order及scales、network/PPO config、reward/curriculum/metric/scene/method hashes、双方模型和Adam状态、update/transition/global-policy-step及两个optimizer counters、Torch CPU/CUDA RNG、实际使用的Python/NumPy/课程生成器状态、训练seed、reset编号、完整source身份。使用restricted loader、finite/schema/hash检查，原子写入、禁止覆盖既有checkpoint。

R4冷启动新actor和新critic，随机seed20260908，默认PyTorch初始化，不额外改Beta偏置或方差；初始化权重和mean/sample统计存证。旧update250不加载，R3 smoke权重不加载。新初始化update0、transition0、k0、Adam counters0；合法空Adam state是v2唯一特例，step>0需所有参数严格覆盖且step与8*update相符。旧v1继续由旧入口只读加载，新训练入口拒绝v1，不截断/补零权重或optimizer。

保存：R4 `init0000` 加u64/128/192/256，共**5份**（比roadmap的4个训练后checkpoint多1份真实可评估初始化）；到u2048共33份。每份hash/bytes和CPU结构检查；R3最终、R4最终、每R5节点做实际加载一致性验证。初始化cold状态和resume状态分别测试。

同方法resume仅在完整update/checkpoint边界，恢复全部训练状态及k，环境fresh reset，绝不声称恢复旧物理轨迹。环境创建/reset采用隔离generator，完成后恢复训练RNG；新物理reset编号写日志。课程不重新预热，不以任意clock override伪称resume。评估另进程，用deterministic mean、只读权重和独立seed，不改变训练RNG。

## 7. 日志契约：`learning-log-h1-v1`

| 频率 | 必需记录 |
| --- | --- |
| 每个minibatch / update | PPO现有完整records；两Adam实际step、coverage、transition/update计数；returns/advantage/value mean/std/min/max、explained variance和zero/nonzero/q分组残差；梯度/finite、吞吐、显存 |
| 每rollout，zero/nonzero及q分组 | 各reward项mean、cost P95、sum；速度误差XYZ/范数；alpha、correctionXYZ、issuedXYZ、实际ICS beta/介入/紧急率、beta_cmd/q、anchor active/usable/error；实际样本数 |
| 每episode | env/episode/reset身份、起点类型、长度/return/平均reward、分项累计、termination全部mask/reason、速度/漂移/高度/距离；运行末未完成episode标right-censored |
| 每command block | 命令前后值、生效step、原始固定reference、运动后停止是否成立、reset打断、停止时间/制动位移/停止后漂移；不自动更新停止参考 |
| 紧凑逐步记录 | command、action mean/sample、governor/ICS/anchor标量、pre-reset root/actual velocity、reward各项、flags、value；每rollout成块写入，足以离线重算行为指标 |
| 原始测量诊断窗口 | 每个新进程前16步的env0/15/16/31，以及首次任务失败前后至多16步这4个env中的可用窗口；ring buffer，所有failure仍保留上行紧凑记录；初始32-env原始ray/preflight证据沿用A0004 |

P95对cost=-reward_component报告，另保留有符号均值，避免把负reward的P95误当最差惩罚。explained variance零target方差、zero的preservation ratio和未完成停止时间保留UNAVAILABLE及原因。不要复制每步全部dense LiDAR；记录诊断窗之外的新q不能事后完整重算这一限制。

每rollout flush；每update结束fsync索引/计账，checkpoint原子落盘；终止/异常进入已批准A0003生命周期，在关闭Kit前保全已有日志/partial counters。失败可能丢失未完成小块，须明确实际完成/可证明计数差异。attempted env.step先记录，optimizer局部失败记录各自实际counter；不以最后完整update掩盖已发生的预算消耗。训练紧凑记录与评估全episode trace分别版本化。

## 8. 评估：`metric-h1-v1`

固定验证32-env同一A0004 combined geometry，spacing100，独立seed20260909；无噪声/扰动，零初速度、identity quaternion、高度2。中心case的4个重复起点按e//8取xy=(-.1,-.1),(-.1,.1),(.1,-.1),(.1,.1)，方向由e%8取上述八方向。各检查点同样本，无挑选。近墙case e<16沿+x、起点x=4.6；其余沿+y、起点y=4.6，切向坐标按(e%4-.5*3)*.05m。不得沿用旧2-env global corner的轨迹数值作配对结果。

| Case | 每env指令序列 | steps/env | 32-env transitions |
| --- | --- | ---: | ---: |
| Z 驻留 | 精确zero 20s | 200 | 6,400 |
| T 跟踪 | 固定安全八方向之一10s | 100 | 3,200 |
| S 运动后停止 | 非零10s，zero23s | 330 | 10,560 |
| X 起步与反向 | zero5s，c7.5s，-c7.5s | 200 | 6,400 |
| W 朝墙 | 近墙向墙0.2m/s持续20s | 200 | 6,400 |
| 每checkpoint合计 | 5 case，32 env，repeat=1 | 1,030 | **32,960** |

每case只把首次episode作为指定试验。提前碰撞/越界记失败，reset后行不能拼接；如向量后端继续step，全部仍计预算且标post-failure-unused，不增加成功分母。若可无额外物理推进地结束某lane，实际少于上限照实记录。终止、缺失和reset打断均列入32个原定样本分母；不得删成“有效episode成功率”。

冻结目标：

1. Z：完整20s，3D speed RMS≤.08m/s、相对t0最大水平漂移≤.30m、最大高度偏差≤.20m。
2. T：全10s 3D tracking RMSE≤.15m/s、稳定段[2s,10s]≤.10；稳定段vz RMS≤.08、高度最大偏差≤.20；报告方向角误差，低速时角度UNAVAILABLE。增加全窗口门槛是为了不靠丢掉启动段通过。
3. S：零生效后≤3s完成停止确认：||v||≤.08连续至少1.0s（dt=.1，包含端点需11个样本），确认时间不回溯。确认后固定200步/20s按Z的速度/漂移/高度阈值；reference只取一次。停止前10s必须满足T，否则“没动所以停得快”不通过。制动位移相对松杆位置单列，停止后reference取确认时位置；同时披露整个23s相对松杆漂移。
4. X：两个非零段各在≤3s内达到||v-c||≤.10并持续1s，稳定段（每段前2s以外）RMSE≤.10，全非零窗口≤.15；沿指令投影的峰值正向速度≤1.5*||c||，全程高度偏差≤.20。起步前zero5s按同Z速度/漂移限，动作增量单列。
5. W：无碰撞/高度失败，首次几何墙距≤1.1m后≤3s完成上述停止确认并保持至结束，若始终未到1.1m，则需先证明确实朝墙运动（前安全段至少连续1s实际指令方向速度≥.10m/s）并在20s内完成停止；否则不能靠永久不动通过。不设朝墙raw command RMSE门槛。整个case几何距离≥.8m、高度偏差≤.20；报告全部q、ICS和速度轨迹。若在1.1m外已停止，保留触发前主动停止的独立标签。

每宏case成功≥95%=至少31/32，且八方向子组分别≥95%（每组4个时等价4/4）；W的x/y子组各16个时等价16/16。所有必需case合计固定集**无碰撞、无高度越界**；样本小不声称95%总体可靠度或零风险。给出成功分子/分母、最差episode、分位数和episode级Wilson 95%区间。重复确定性样本不是独立训练重复。

距离定义：旧`exact_clearance_m`仍是sensor-origin ray-hit最小range，用于既有reward并单独记录。新增 `sensor_origin_surface_distance_m` 使用pre-reset挂载后sensor位置到当前有限fixture三角面的最短欧氏距离（floor/x-wall/y-wall及surface id，场景内可用已核对的有限平面解析等价式）；它是evaluation-only，不送actor。W的.8m门槛采用此几何距离，不采用稀疏ray最小值；初始sensor几何位置和mount hash绑定。这个.8m是**传感器原点到表面的距离，不是机体外壳余量**；不从arm length猜碰撞半径。真实contact label仍独立按原1N阈值记录，未知外壳余量不宣称已证明。

B0采用已审 `execute_b0_direct_velocity`，经同frame/Lee/rotor，完全绕开governor/anchor/ICS，不能用`use_b0`组合代替。R3对同5case执行一次B0；W的碰撞结果如实保留，不要求无ICS的B0达到安全门槛。Z/T/S/X必须达到相应基础指标且没有持续异常升高，否则R4不启动，先定位低层可达性。若物理上不能达标，训练前修改需求需单独记录决策，不能事后放宽。

### 8.1 R4/R5继续规则与checkpoint选择

R4同一次训练完成后，在独立评估进程顺序评估u0/64/128/192/256；不穿插污染训练物理状态。预热处仅以在线工程/失败信号决定异常停止，不要求预热已收敛。所有5份均公开，旧update250仅历史参考，本轮不额外评估它。

每checkpoint构造8个越小越好误差：Z速度RMS/.08、Z最大水平漂移/.30、Z高度偏差/.20、T全窗RMSE/.15、S停止确认时间/3、S停止后最大水平漂移/.30、X全非零RMSE/.15、W的max(0,.8-min距离)/.8。先逐episode算再等权平均；失败/缺失对应error取max(已观察归一化误差,5)，未停止的time项至少5。L为8项均值；另保存真实未截断值、各case失败数F和高度/碰撞数，不以惩罚填值冒充物理测量。

若u256全部行为gates通过，可直接转R6设计。否则“可扩到下一节点”必须同时：工程gates全通过；u192与u256均无碰撞/高度越界；u256至少一个Z/S指标较u0改善≥10%且一个T/X指标改善≥10%；L256≤.9*L0；8个单项较u0均未恶化超过max(基线值的10%,.1归一化单位)；各宏case失败数不增加。混合后Z相对u64若退化超过max(20%,.1归一化单位)，先诊断，不追加步数掩盖遗忘。未达这些规则归为“无足够继续证据”，不是产品实现失败。

R5每节点沿用同规则，相对上个获准节点比较L改善≥10%与单项容忍；不再强求已接近地板的Z/T各下降10%，但须至少一项未达标能力改善≥10%、已达标能力不跌出标准；最新两个已评估checkpoint无碰撞/越界。每个节点还评估其前64-update checkpoint，避免只看单一好结果。达到目标立即停在已评估完整节点，不消耗余量。

可选部署候选仅从已注册评估checkpoint中选：先必须全部gates通过，再L最小，再同分取最早update；同时报告最后checkpoint。R4/R5若没有全部通过的checkpoint，输出“没有合格候选”，不能用best-return代替。R6 holdout只测试冻结规则选择出的候选，每个训练seed一份；查看后不回头调参仍称holdout。

## 9. 分阶段预算提案（本轮所有可执行额度仍为0）

以下均为未来硬上限提案，失败初始化、失败step及重试消耗不因换attempt重置。无隐含retry，出现工程失败先离线修复；额外live启动需有真实授权。每个env.step=32 transitions、20次physics substep；重置/初始化物理推进次数必须单列，不能藏在env.step=0后面。R2只允许设计CPU替身，无真实optimizer训练。

| 阶段 | live启动上限及wall time | 训练transitions / updates | actor、critic各Adam steps | 非训练transitions上限 | 新checkpoint上限 |
| --- | --- | ---: | ---: | ---: | ---: |
| R1 设计 | 0 | 0 / 0 | 0 | 0 | 0 |
| R2 CPU实现 | 0 | 0 / 0 | 0真实训练 | 0 | 0真实训练；测试临时fixture单列 |
| R3真实核验 | 1主进程≤3600s +1受控失败退出进程≤600s | 2,048 / 4 | 32 | B0 32,960 + reload rollout512 =33,472 | smoke init+final=2 |
| R4短试训 | 1训练≤7200s +1评估≤7200s | 131,072 / 256 | 2,048 | 5*32,960=164,800 | init+4=5 |
| R5到262,144 | 1续训≤7200s +1评估≤3600s | +131,072 /256 | +2,048 | u448、512：65,920 | +4 |
| R5到524,288 | 1续训≤7200s +1评估≤3600s | +262,144 /512 | +4,096 | u960、1024：65,920 | +8 |
| R5到1,048,576 | 1续训≤7200s +1评估≤3600s | +524,288 /1024 | +8,192 | u1984、2048：65,920 | +16 |

R3总transitions≤35,520：2,048训练 +33,472非训练；主进程顺序复用32-env场景完成preflight/B0/learned smoke/reload，阶段间fresh reset。smoke 4 updates依次为全zero一份、16/16混合一份、角色交换两份，使用标记为diagnostic的专用短clock，不冒充生产1024/128步课程；R4完全独立冷启动。reload后16步mixed rollout无update用于核对actor/critic/action/log接线；随后丢弃。第二进程在初始raw-ray记录后、首个policy step前受控注入异常，验证A0003保全/非成功退出，0 transitions/updates/checkpoints。不故意制造真实碰撞测试退出。

R3主进程AppLauncher一次、受控失败一次；每个进程simulation initialization最多1次，主进程显式环境reset最多8次（B0五case+smoke+reload+必要初始preflight，能复用则少用），失败进程最多1次。R4训练只一次初始化reset，评估最多25次case reset；R5每训练一次reset，每评估10次。自动task reset另按发生次数记账，不推进全局课程额外step。

已只读核对固定IsaacLab的DirectRLEnv.reset调用write_data/forward而非sim.step，constructor会调用sim.reset；Isaac Sim底层play还存在初始化更新。因此预算另设每live进程初始化物理推进硬上限10 ticks（每tick=.005s、32个环境），正常显式/自动reset的额外物理推进上限0；不能用初始化额度做warmup rollout或optimizer。R3–R5最多100个全场景初始化ticks，独立于上表policy transitions计账。R2设计进度计数接线，R3实测time/step delta验证；未能证明真实计数或超过10则停止、保全，不能宣称初始化无物理代价。

到R5全上限：新方法训练1,048,576（不含R3 smoke），2048updates，双Adam各16384steps；R3–R5合计live starts≤10、训练含smoke1,050,624、非训练396,032、全部policy transitions1,446,656；checkpoint含smoke共35。wall上限合计51,000s（14h10m），是硬上限而非预计耗时。评估消耗已展开，不能把“一套评估”算成零成本。实际成功早停或失败后的未用额度不用于补跑。

资源原边界保留：固定RTX5070/driver身份、三采样跨≥10s、utilization≤5%、used≤1024MiB、free≥10240MiB、原compute-app白名单；运行期free≥1GiB。每新rollout/update/eval step前检查，超线保全退出，不杀他人进程/改包驱动。不新增GPU探测权限到R1/R2。Python继续 `/home/mint/miniconda3/envs/navrl_blackwell_test/bin/python3.11`，原hash `efa5468dc4ff3850e950cb68c0e430b3a1621e2c6bd9e0ef48f784ba90a9667a`；runtime `/home/mint/rl_blackwell_test`只读。旧certified公共工具commit `eff00ea9d339c66913ea472a30142406888ca1ca` 保持固定；新STANDARD workload不伪装已支持旧formal manifest。R2核对现有runner接口能否表达新预算；若公共核心需改，单独界定工具范围并在正式使用前审核。

R6预算占位也给出量级，**不作为当前运行请求**：三独立训练seeds20260908/20260918/20260928，已有一条成功训练可复用；另外两条每条最多1,048,576训练，课程/PPO不变，每条4个训练阶段启动，5+2+2+2份开发评估=362,560非训练，含init33checkpoint。两条重复合计2,097,152训练、725,120开发评估、16启动、66checkpoint。方法/选择规则冻结后，另3个holdout进程每个≤7200s：Z1、T8、S8、X8、W2独立case，各32env，steps分别200/100/330/200/200，单seed180,480、三seed541,440。holdout以不同固定局部起点（中心xy±.2、近墙距1.3/1.5m）作静态位置变化，源几何相同；完整case JSON在R6签发前固定。本量级不含噪声/延迟/扰动/幅值变化和消融；这些按roadmap顺序单独设计并计预算，不暗含全B1–B8。R7当前预算0，不安排硬件或实飞。

## 10. R2预定范围与必要验收（待确认后签发）

产品后继涉及 `isaaclab-training/instinctlab/task/`、`training/`、必要 `envs/` 接线及对应tests；只增本方法的显式版本，保留v1可读/测试路径。控制数学可复用现有纯函数，不改变 `control/`实际管线、observations actor字段、资产、传感器几何或runtime。workload后继新增 `training_workloads/unified_h1/`及对应tests，复用/适配A0003生命周期与A0004 geometry/preflight helper；旧scale_003不改历史语义。具体helper和文件拆分由Builder决定。

必要CPU验收一次交付：

- 课程全R4/R5计数、50/50、八方向、真实转换、reset/续训clock、各边界单步command/obs/reward/critic配对。
- 风险动作独立反例、零指令不关速度项、朝/离墙与可靠性边界；新reward逐项精确值、制动correction反例、terminal union/timeout与GAE。
- 21维全链与平移/command条件化、actor正负来源审计、critic梯度不进入actor输入；H1/H2信息边界记录。
- cold init空Adam roundtrip；已更新checkpoint counters/RNG/课程一致性；v1/v2拒绝矩阵；load失败不部分污染trainer。
- logger每minibatch不丢records、partial失败计账；S停止reference不漂移、11样本1s定义、未停止/提前reset不能成功、W不动不能成功、case/方向分母、距离独立定义与非有限值。
- 新init/reset分布的CPU几何/源接线、B0Direct真实adapter静态来源，评估隔离、所有预算乘法和退出上限、A0003/A0004仅受影响回归。

R2可做CPU合成张量update/临时checkpoint的必要单元验证（在后续Order里明确），但不包含env rollout训练、AppLauncher/CUDA/GPU探测。R1本轮没有运行这些实现测试，仅完成源码、只读旧张量和算术审查。

R3实际进入门槛：R2审过精确候选；新的运行授权/输入hash/argv/env、实际初始化/reset底层推进计账、预算与资源gate全部固定。R4进入门槛：R3真实forward residual/hit identity/rank3、actor/reward/clock/log/reload/退出通过，B0 Z/T/S/X达标且实际reward尾部量级可接受。若任何缺口尚未证实，标NOT PROVEN，不从A0004 CPU通过外推。

## 11. 一次集中决定与本轮交接

**请整体确认 `unified-h1-v1` 方法方案，并允许 Commander 随后签发上述 R2 CPU 实现与离线验证任务。** 该确认涵盖H1范围、50/50与近墙reset分布、reward-v2（含独立风险计算、制动主项、失败300）、critic21维、冷启动/v2 checkpoint、日志和冻结评估口径；无需逐个选择权重或helper。R3–R6预算表仅供审核规划，确认方法不自动授权新增训练/运行；R2审核后再依据实际候选申请第一份R3精确运行包。

需要确认的来源是用户本轮“方法变化集中提交一次”“方案确认后，再签发R2”，以及V3第2节保留研究方法/训练授权；不是旧入口的逐文件审批、固定提交次数或新设流程障碍。

当前阶段：R1方案完成、待用户方法决定。已完成：基线/审批/两方状态核对，源码及旧轨迹审查，唯一方案、risk/terminal精确定义、日志、评价、预算与R2验收设计。未解决：新方法可学习性、B0阈值可达性、真实GPU/关闭/reload路径需R3；不属于已关闭修复的新返工。下一步：用户整体确认后签发R2；当前Builder无新执行任务。需要用户决定：**是，1个整体方法/R2授权决定**。

本R1交接计数1（Commander→用户）；本轮已完成新用户方法决定0、待决定1；证据问题导致仿真重跑0、新MUST_FIX0、审核后实质遗漏0（仅指本R1自审，不覆盖未来独立审核）。后续每次交接继续维护roadmap：当前阶段、完成事项、未解决问题、下一步、是否需要用户决定。
