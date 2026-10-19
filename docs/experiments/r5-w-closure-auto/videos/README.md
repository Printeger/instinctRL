# R5 Isaac Sim 三维轨迹回放

这些 MP4 是 **Isaac Sim 5.1 RTX 渲染的真实 TASLAB 无人机资产与 H1 场景**。机体位姿来自已冻结的原始评估日志；本次没有重新运行策略或物理积分，也没有训练。它们是原始仿真评估的可视化回放，不是新增的闭环仿真验证或新的验收结果。

优先看两个 W 对比视频。三栏从左到右分别为原 u512 / margin=0、同一个 u512 / margin=0.15m、新 u640 / margin=0.15m。三栏使用相同视角、相同 lane 和相同原始采样时间。**固定 u512 只改控制器 margin 已经通过 W；不能把这个改善归因于 RL 学习。**

| 视频 | 文件 | 时长 |
|---|---|---|
| W 接近 x 墙，三配置对比 | [W_x_wall_comparison.mp4](W_x_wall_comparison.mp4) | 20 秒 |
| W 接近 y 墙，三配置对比 | [W_y_wall_comparison.mp4](W_y_wall_comparison.mp4) | 20 秒 |
| Z 悬停，u640 | [Z_u640_m015.mp4](render-v2/Z_u640_m015.mp4) | 20 秒 |
| T 速度跟踪，u640 | [T_u640_m015.mp4](render-v2/T_u640_m015.mp4) | 10 秒 |
| S 停止保持，u640 | [S_u640_m015.mp4](render-v2/S_u640_m015.mp4) | 33 秒 |
| X 反向，u640 | [X_u640_m015.mp4](render-v2/X_u640_m015.mp4) | 20 秒 |
| W x 墙，u640 / 0.15m | [W_x_wall_u640_m015.mp4](render-v2/W_x_wall_u640_m015.mp4) | 20 秒 |
| W y 墙，u640 / 0.15m | [W_y_wall_u640_m015.mp4](render-v2/W_y_wall_u640_m015.mp4) | 20 秒 |
| W x 墙，u512 / 0m | [W_x_wall_u512_m000.mp4](render-v2/W_x_wall_u512_m000.mp4) | 20 秒 |
| W y 墙，u512 / 0m | [W_y_wall_u512_m000.mp4](render-v2/W_y_wall_u512_m000.mp4) | 20 秒 |
| W x 墙，u512 / 0.15m | [W_x_wall_u512_m015.mp4](render-v2/W_x_wall_u512_m015.mp4) | 20 秒 |
| W y 墙，u512 / 0.15m | [W_y_wall_u512_m015.mp4](render-v2/W_y_wall_u512_m015.mp4) | 20 秒 |

画面为第三人称视角；Z/T/S/X 跟随机体，W 使用固定相机便于比较接近和停住的位置。黄色线连接传感器原点与最近有限墙面/地面；橙色线表示距墙 0.80m 的原验收距离，**不是** 0.15m ICS clearance margin。蓝色线表示机体轨迹。地面网格间距 1m，机体未放大。画面显示机体系实际速度和目标速度、几何距离、q 与 ICS beta。

原始采样为 10Hz，视频同为 10fps、1 倍速，未插值。第 i 帧的 state 时间为 `(i+1)*0.1s`，control 时间为 `i*0.1s`：原日志记录该步积分前的指令/q/beta 与积分后的位姿/速度/几何。两种时间均显示在画面上。没有将下一步控制量移到当前步。

选择规则预先固定：Z/T/S/X 和 x_wall 使用 lane 0，y_wall 使用 lane 16；不是全部 32 lane。原始世界位置仅减去该 lane 的环境原点。机体姿态完整保留，转子相位没有记录，因此转子网格保持资产默认姿态，不代表转子停转。材质、照明、相机、网格及辅助线仅用于展示；H1 有限墙面位置和尺寸来自冻结源码。actor 输入没有变更。

[VIDEO_MANIFEST.json](VIDEO_MANIFEST.json) 保存每条视频、原始轨迹、checkpoint、配置与来源文件的实际绝对路径和 SHA256。[REPLAY_INPUTS.json](REPLAY_INPUTS.json) 固定输入；[render-v2/RUN_MANIFEST.json](render-v2/RUN_MANIFEST.json) 保存运行 PID、时间、渲染源码 commit、资产闭包及零训练/零物理步计数。每段录像同名 JSON 保存逐帧数值，可直接与原始日志对照。

u640 SHA256：`aaef7d40de50032cf2c61e0bc7a6e6f277a2f65e252284f3af58c0f25bae5429`。
u512 SHA256：`a3b9f9115268270ec000750035085b03980de2aed31987b7b08ae97bf26d8776`。

完整输出验证包括全部 MP4 解码帧数、逐帧速度/指令/q/beta/位置/姿态比对、距离与原评估 policy record 比对，以及冻结输入的前后 SHA256。原 R5 科学报告、raw、checkpoint、Reward、验收门槛均未覆盖。

预览目录 `preview-v1/` 仅为 2.1 秒制作检查：渲染成功但清理接口错误导致进程异常退出，证据保留于 `PREVIEW_STATUS.json` 和日志；不属于交付视频清单。完整录像使用修复后的 `render-v2/`。
