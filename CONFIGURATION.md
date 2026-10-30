# G1 跟踪任务配置

标准任务的配置入口是 [`supermimic/tasks/tracking/config/g1/flat_env_cfg.py`](supermimic/tasks/tracking/config/g1/flat_env_cfg.py) 中的 `G1FlatEnvCfg`。它继承共享 `TrackingEnvCfg`，显式列出模型、PD、随机化、奖励及网络观测配置。训练算法仍由 `training/trainer.py` 实现。

| 配置块 | 内容 |
|---|---|
| `robot_model`、`self_collisions` | 一致的 URDF/MJCF/USD 资产及物理自碰撞开关；默认 False |
| `control` | PD、动作缩放与按关节名覆盖 |
| `randomization` | 物理参数、PD 和 reset 扰动 |
| `analytic_reward` | 共享解析奖励、终止阈值 |
| `policy_observation` | 本体观测组、scale、noise 与历史长度 |
| `critic_observation`、`world_model_observation` | 特权网络输入约定，`reference_orientation=None` |
| `scene.num_envs`、`episode_length_s` | 并行环境数、训练 episode 长度 |
| `commands.motion.goal_frames` | 未来参考帧数 |

修改配置后调用 `cfg.prepare()` 会校验并重新生成机器人场景配置；训练、评估和环境入口也会调用它。命令行显式传入的 `--num_envs`、`--hist`、`--goal_frames` 覆盖对应默认值。

## PD 和动作

编辑任务的 `control = PDControlCfg(...)`，例如：

```python
control = PDControlCfg(
    natural_frequency_hz=10.0,
    damping_ratio=2.0,
    stiffness_scale=1.0,
    damping_scale=1.0,
    action_scale_multiplier=0.25,
    stiffness_overrides={".*_knee_joint": 100.0},
    damping_overrides={".*_knee_joint": 4.0},
)
```

膝关节覆盖仅为示例，当前默认覆盖字典为空。先按 motor armature 计算
`Kp = armature × (2πf)²`、`Kd = 2ζ × armature × 2πf`，再乘全局 stiffness/damping scale，最后应用正则覆盖。正则对完整关节名做 `fullmatch`，多项匹配时最后一项生效，覆盖值是绝对增益。

动作缩放为 `action_scale_multiplier × effort_limit / nominal_Kp`，与场景执行器共享 PD 配置。episode 内随机 Kp 不重新计算动作缩放，保留随机增益对控制的影响。

## 随机化

| 子项 | 默认配置 |
|---|---|
| `friction` | U[0.3, 1.2]，startup；nominal=1 |
| `joint_offsets` | U[−0.01, 0.01]，reset；nominal=0 |
| `torso_com` | X±0.025、Y/Z±0.05，startup；nominal=(0, 0, 0) |
| `pd_gains` | 关闭；启用时 Kp/Kd 比例默认均为 [0.8, 1.2]，reset |
| `reset` | 根位置/旋转/线速度/角速度和关节位置扰动；范围见任务文件 |

`mode="startup"` 在环境创建时抽样一次，`mode="reset"` 在对应环境重置时重抽。
`enabled=False` 使用 nominal 值。评估关闭随机抽样和 reset 噪声，采用保存配置的 nominal 参数。

例如修改已有的 PD 子项：

```python
pd_gains=PDGainRandomizationCfg(
    enabled=True,
    kp_scale=(0.9, 1.1),
    kd_scale=(0.9, 1.1),
    mode="reset",
),
```

启用后实际增益写入 Isaac Lab implicit actuator，context 从 33 维扩展为 91 维：摩擦 1、关节偏置 29、躯干 CoM 偏移 3，加上 29 个 Kp 比例和 29 个 Kd 比例。回放、世界模型与 critic 同步适配，已有权重不能直接用于新的输入尺寸。

## 奖励和终止

| 参数 | 默认值 |
|---|---:|
| `w_body_pos / w_body_ori / w_body_linvel / w_body_angvel` | 各 1.0 |
| `w_root_pos / w_root_ori` | 各 0.5 |
| `w_action_rate / w_joint_limit` | −0.1 / −2.0 |
| `body_pos_std / body_ori_std` | 0.3 / 0.4 |
| `body_linvel_std / body_angvel_std` | 1.0 / 3.14 |
| `root_pos_std / root_ori_std` | 0.3 / 0.4 |
| `soft_limit_factor` | 0.9 |
| `term_height / term_tilt` | 0.25 / 0.8 |
| `soft_term_height_temperature / soft_term_tilt_temperature` | 0.02 / 0.04 |

在 `analytic_reward` 中设置权重；0 关闭对应贡献。tracking 权重必须非负且至少一项为正，惩罚权重不能为正。`std` 越小，对误差越严格。所有配置同时用于仿真和想象轨迹中的解析 MDP。

奖励归一化上限随六项 tracking 权重总和调整。默认每步最大 tracking reward 为 0.1。`rewards.tracking.weight` 必须保持 1.0；只修改 Manager 总权重会使仿真和想象奖励不一致，因此配置校验会拒绝它。新增奖励类型需在 [`mdp.py`](supermimic/mdp.py) 中实现对应的可微公式。

## 网络输入与归一化

默认 actor 每帧观测按以下顺序连接：

| 观测组 | 维数 | 默认 scale | 默认均匀噪声半宽 |
|---|---:|---:|---:|
| `joint_pos`：相对默认姿态关节位置 | 29 | 1 | 0.01 rad |
| `joint_vel` | 29 | 1 | 0.5 rad/s |
| `base_ang_vel` | 3 | 1 | 0.2 rad/s |
| `projected_gravity` | 3 | 1 | 0.05 |
| `previous_action` | 29 | 1 | 0 |

每帧共 93 维，默认 `history=10`。`reference_orientation=None`，不参与默认观测计算；该兼容字段可读取部分历史 checkpoint 的 99 维接口，不能通过裁剪旧 decoder 权重转换成 93 维策略。

每组支持 `enabled / scale / noise`。关闭一组会改变网络输入宽度；scale 同时作用于信号与噪声。`noise_enabled=False` 关闭采集和想象中的观测噪声，评估本来就不加噪。不能关闭所有观测组，也不要在 `observations.policy.enable_corruption` 再加一层噪声。

| 网络 | 输入顺序 | 默认宽度 |
|---|---|---:|
| Actor encoder | 未来 10 帧 reference goal | 730 |
| Actor decoder | latent；按帧展平的历史观测 | 32 + 930 = 962 |
| Critic | 完整 state；goal；归一化 context ξ | 71 + 730 + 33 = 834 |
| World model | 完整 state；执行动作；归一化 context ξ | 71 + 29 + 33 = 133 |

完整 state 为 `[root_position(3), root_quaternion_wxyz(4), root_linear_velocity(3), root_angular_velocity(3), joint_position(29), joint_velocity(29)]`。根位置、旋转和速度采用环境世界坐标系，关节使用既定的 29 关节顺序。73 维单帧 goal 由 `MotionLib.goal` 构造。开启 PD 随机化后 critic/world model 输入分别变为 892/191 维。

actor 使用 `BatchNorm1d(affine=False, eps=1e-4, momentum=0.1)`，仅仿真采集中的显式 `update()` 更新统计量；训练前向、想象 rollout、播放和导出均使用保存统计。历史缓冲为 `[B, H, D]`，调用端按帧展平为 `[B, H*D]`。critic/world model 使用原有 RunningNorm。

## 配置快照和兼容性

训练将受支持的配置项保存到 `task_settings.json` 和 checkpoint 的 `task_settings` 中。评估优先恢复 checkpoint 配置；watch 模式读取运行目录快照并检查兼容性。播放默认也恢复配置，可显式覆盖物理自碰撞开关。

快照包括 PD、随机化、奖励、观测、机器人资产、自碰撞、episode 长度和 goal 帧数；它不是完整 Isaac Lab 配置序列化。数据目录由 `--motion_dir` 指定，未存入任务快照；迁移数据后应显式传入新路径。

新 checkpoint 使用 `model_input_version=2`，world model/critic 输入语义已与旧实现不同。观测维数、history、goal 帧数或 context 改变时需要新训练；即使尺寸相同，改变 scale 也会改变输入语义。旧 actor 的 `mean/var/count` 归一化键不能直接加载到当前 BatchNorm 策略。

训练入口目前拒绝 `--resume`，因为 replay 恢复尚未实现。控制时钟固定为 50 Hz（`sim.dt=0.02, decimation=1`），Newton 内部 4 个子步；任务校验会拒绝单独更改控制周期。
