# 容量设计草案：尚未准入

本文件不冻结测试标准，不代表完整容量核算 PASS。测试实现、每作业物理步/reset/ticks、
实际输出上限和独审仍待完成，`goal_budget.json.capacity_pass=false`。

已核验 R5 原审计：256 updates / 131072 transitions 的训练输出 1,969,675,045 bytes，
单节点原五任务 32960 transitions 输出约 29.7 MB。七次旧作业总 GPU 1805.60 秒、
live wall 1827.27 秒；这不是 R6 吞吐保证，也不能用该汇总推算单训练作业的运行时间。
来源 `../r5-w-closure-auto/FINAL_RESOURCE_AUDIT.json`。当前可用磁盘约 474 GiB，
精确采样值在 STARTUP_PREFLIGHT；实际准入仍须重新检查至少 30 GiB 地板。

内部首先保护 3 × 1024 × 512 = 1,572,864 训练 transitions（不是强制训练配方），
800,000 非训练 transitions、24,000 秒 GPU 和 live wall、16 次启动/初始化、24 次写入、
80 GiB 输出与快照。三种子最终配方必须适配此保护或在首次使用前经独审调整，不能让探索借走。
开发候选可先按每个最多 768 updates 规划：三个共 1,179,648 transitions；连同上述确认
共 2,752,512，距总上限余 247,488。初段 128/256 updates 与后续晋退须事前登记。

旧训练输出线性外推全部 3,000,000 transitions 约 45.1 GB，尚未加新字段、快照、视频
和失败预留，因此只能用于起始估计，不能替代实际字段/行数核算。拟按准确日志规模分配
配额，禁止机械地给 64 个作业各预留 8 GiB。

32 次自动会话中至少保护最终审核 4 次、异常修复 4 次；初始用户会话另列。
tools/r6_budget.py 对每次预留绑定 goal/event/job/input hash/role，重复一致预留幂等，
漂移拒绝；失败不退款。当前仅实现 CPU 预算检查，不是已经验证的事件调度器。

首次 GPU 前仍需：冻结目标/压力域与 split、确定最少最终样本及物理可行性、拆出每作业
policy/physics/reset/init/Adam/checkpoint/cleanup 成本、提供时间和磁盘裕量、独立审核，
并完成 CPU 替身重启/死亡/半写 receipt/STOP/去重/单写者测试和一次真实非交互审核。


## 逐帧测量归档尚待核算

新增全量 clean/delivered 测距归档不能按旧 29.7MB 评价日志线性外推。每lane有21240 beams；
160640 新评价 transitions ×21240×4bytes×2份 =27,295,948,800 bytes（以实现的精确计算为准），
还不含 validity/timestamps。必须采用可逐字节重建的去重/无损编码，并实测压缩和输出配额，
否则当前40GiB最终输出保护不足以容纳四模型全量原始双份数组。不能丢负结果或降采样来
掩盖此缺口；容量状态仍为false。

旧训练目录实查发现两个首更新 full-input 诊断文件各522177865 bytes，是每次作业固定成本，
两个checkpoint各约1.51MB。此前45.1GB训练线性外推只是粗略上界，不能直接拿来分配新作业。
需要分离每作业诊断、每transition日志、每checkpoint和快照成本后再独审冻结。


## 无损编码 CPU 测量与内部额度调整

`ARCHIVE_CAPACITY_CPU.json` 记录完整32×21240帧的四类合成输入测量，全部逐字节重建。
算法保留原dtype/shape/hash，用帧内相同数组引用、字节平面、可选XOR和zlib压缩；每帧独立
校验并fsync，截断/额外帧/错误身份不被接受。对独立随机测距样本按草案5020步（2180噪声步）
外推，四模型约58,552,076,320 bytes（54.5GiB）。此值不包括reset、原回归、训练、视频，
不是实际RayCaster数据或压缩最坏界。

原训练J-8bdfaca0488e524d的GPU908.632s/wall911.635s（256updates），原评价
J-c2412eb2e9c23e5a的GPU263.218s/wall266.579s（32960transitions）。每条真实receipt已pin，
不再使用七作业汇总去猜单训练耗时。粗略按段长外推再加编码/初始化/清理，应将最终时间
保护从18000提高到24000秒、输出保护从40提高到80GiB，总caps保持不变，开发最多48GiB。
调整是未发车的内部设计，待独审；没有退款旧记录。减少开发中无必要的完整16块反复评价，
不减最终四模型覆盖。精确配额、训练有限字段策略和失败窗口仍待实现与审查。


## 延迟归档容量修正（2026-10-20，仍未准入）

原same-frame合成估计未充分覆盖延迟。按实际RangeDelivery的0/1周期延迟、三个sigma、
初始无效帧及登记丢帧次数重测，独立帧编码四模型测量约60.2GiB。新v2无损编码只缓存
上一解码帧，允许精确引用或XOR，逐数组哈希重建，旧v1仍可读；同一合成输入降到约54.0GiB。
完整32×21240CPU样本全部bitwise重建，23项归档/测量/评价替身回归通过，尚待独审。
这不是物理数据的压缩保证，也不代表80GiB最终输出保护足够。

`TRAINING_STORAGE_CPU.json`将旧256update训练目录的约1.958GB拆为两份full-input诊断
约1.044GB、其他诊断、measurement windows、checkpoint和逐步日志，避免将固定开销
线性乘到所有transitions。只读取目录大小，未改变旧证据；原生日志不在该目录总量中。
R6训练窗口/字段上限、全体native预留、视频/快照和完整时间裕量仍未闭合，capacity_pass保持false。


## 全物理步施力日志补算（仍未准入）

`FORCE_ARCHIVE_CAPACITY_CPU.json`通过实际R6PhysicsMixin的force sink和AuditLog，
以32lane×5body完整字段、随机float32旋翼力/力矩与归一化姿态，采样脉冲前、
两端及撤除后的记录。18条记录全部按原JSON值重建；临时归档自动清理。
fixture质量为1.16kg，明确不是基线质量或真实动力学验证，不调用GPU或simulator。

按草案100400物理tick/模型、脉冲40tick逐块加权，估计1,062,041,200 bytes/模型，
最终四模型4,248,164,800 bytes（约3.96GiB）。与测距合计约57.92GiB；仍未包括
reset、step_attempt/step_complete、motion、训练、native、快照、视频及失败裕量。
此前微型workload替身的force记录只有tick与fixture标记，不能用于这项容量推断。
本工具改用实际完整sink字段，但随机样本最大值仍不是最坏界，不改变任何日志字段。

每条计时含序列化、压缩、write、flush及fsync；样本存在计时尖峰且数量少，
不能将其称为live吞吐，更未计入GPU到CPU同步。完整运行时间必须同时考虑测距编码
与逐tick审计成本；原R5吞吐线性外推不足以批准R6。80GiB/24000秒内部保护不变，
capacity_pass继续为false。


## 全组件合并与无损 force 编码候选（仍未准入）

`CAPACITY_COMPOSED_DRAFT_V2.json`将实际CPU样本、每文件配额提案及未实现部分放在同一份表中。
每次完整评价仍为193600 transitions，最终三模型与配对基线仍为774400；目标/压力块不删减。
全量force张量使用独立二进制record，保留dtype/shape、所有数值与外层哈希/fsync。
18条真实mixin CPU样本精确回放，最终四模型force估计从6.56GiB降至2.54GiB，非最坏界。
新版CPU RNG测距约54.0GiB；完整control/reward step额外约0.196GiB。

按样本增加15%并向上取整、保留每作业3GiB原生输出等现有余量后，完整评价提案20.203125GiB。
四次最终评价、三条确认训练以及快照/视频/控制器/清理合计91.8125GiB，较现80GiB多11.8125GiB。
提案将最终保护调到92GiB，开发池36GiB；两个256update开发候选及各一次“原套件+预登记受影响块”
的容量示例共35.703125GiB。开发子集须在基线揭晓前登记，长训/控制对照/第三候选仍按实际剩余池准入，
不能把三个候选硬上限理解为任何三个最大组合都已装入。最终覆盖完全不变。

旧R5训练文件只读流式压缩并核验逐字节SHA，约1.958GB降至0.411GB；未加载pickle或改写原件。
该结果仅支持开发无损训练writer/reader的可行性，不表示R6新噪声字段具有相同压缩率。
确认作业每条2GiB是未实现配额提案，必须先验证概率gate/full-input诊断的读取兼容。
原生路径覆盖、reset/header/motion上界、快照/视频限额、controller元数据和新settlement仍待闭合。

历史吞吐加当前CPU编码样本外推，最终约24601秒（已加7×180秒清理和1200秒视频），
超过现24000秒；建议保护27000秒，开发剩16200秒。既非真实GPU吞吐，也非最坏时延保证。
磁盘/时间总caps不变，所有方案仍capacity_pass=false，操作账本尚未调整。
最初step样本漏了lane的policy_step/command字段，保留原文件但合并计算只用V2；
最初合并草案开发池超出0.203125GiB，V2只将小评价整GiB向上取整改为256MiB取整，
原字段/15%余量/分母不减。未隐式退款旧或新预留，也未发车。

2026-10-20实际分块训练存储补证：`TRAINING_CHUNK_STORE_CAPACITY_CPU.json` 对历史原字节采用新codec，JSONL逐record flush/fsync，压缩与小文件合计415,677,048bytes，原1,958,329,318bytes。所有测量流重新核验原SHA，临时档验证后删除；无pickle加载。12项新CPU测试及11项原诊断回归通过，见 `CPU_LOSSLESS_TRAINING_TESTS.json`。新增适配层为opt-in，概率gate计算/顺序未改；默认R5保存方式保留。V3合并草案采用实际codec估计，最终92GiB/27000秒仍仅提案。训练lifecycle及消费者接线、独审、新噪声字段压缩率与live吞吐未验证，capacity_pass=false。


实际R6 prelaunch新增独立13样本/至少60秒窗口，最大采集180秒；队列观测不能替代。下一版完整时间预算必须显式计入该窗口、初始化与清理，不能直接沿用旧R5吞吐外推或假定旧receipt已有相同窗口成本。`TEST_PROTOCOL.json`新增资源policy使其身份改变；已有容量草案及review保持历史身份，不视为重新接受。当前80GiB/24000秒保护、92GiB/27000秒提案和capacity_pass=false均不变。CPU证据见 `CPU_R6_PRELAUNCH_TESTS.json`，未发车。


V4容量草案纠正快照与启动成本：当前工具每身份全额预留2GiB，实际第一份约0.922GiB且不退款；开发快照项包含这份既有预留。最终共享源码快照也按2GiB预留，最终总量92.8125GiB，保护提案93GiB、开发余35GiB。原两候选示例不再适配，V4以一个256update候选及一次“原套件+预登记受影响块”示例计算约30.953GiB；其余候选/控制需按真实剩余池准入，不能替代必要实验或删减最终覆盖。此示例不冻结方法或科学选择，操作保护仍80GiB/24000秒。

最终七作业各增加完整180秒独立prelaunch采集上限，原历史外推未扣未知旧准入成本；最终样本外推25861.2秒，27000秒仍仅提案。完整32lane、全部16块的motion字段以长有限float64序列化，最大4,249,464bytes（约4.05MiB）；新增motion+result合计16MiB写前拒绝，保留全局累计配额。7项CPU检查通过，见 `CPU_R6_METADATA_QUOTA_TESTS.json`。样本是非物理序列化证据，不能证明科学有效性或原生日志跨路径上界；capacity_pass=false，原只读快照未改，新代码需新身份。


原生控制台路径核验：rlflow的held-child原先继承supervisor.log，Kit另写结果目录kit.log；单文件2GiB限制不能证明二者合计≤3GiB。新R6专用分派选项将console合并到独占native-output.log并在256MiB处拒绝后续写入，保存前缀/SHA/原因，既有Kit单文件上限2GiB保留。名义余量768MiB仍需覆盖控制元数据、旋转/额外原生日志与其他路径，不能据此声明总native容量PASS。56项CPU检查通过，见 `CPU_R6_NATIVE_OUTPUT_TESTS.json`；旧控制release/快照不变，真实R6发行和独审尚未完成。
