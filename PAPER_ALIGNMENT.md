# SuperMimic 论文与实现范围

本项目重构自 [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed)，论文核对对象为 [*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*，arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480v1)。以下记录代码中采用的实现和已知差异，区分算法结构、具体工程选择与实验结果。

当前实现覆盖单一 reference-motion 接口的 G1 跟踪：仿真采集、世界模型拟合、想象轨迹 BPTT 和 critic 更新。Manager 环境的重构不等于论文全部实验或性能已经复现。

## 主要结构

| 论文部分 | 当前实现 |
|---|---|
| §3.1 / Algorithm A.1：采集、世界模型、策略、critic | `Trainer.collect / wm_update / policy_update / run`；使用世界模型与 BPTT |
| §3.2：以 context ξ 条件化的确定性动力学 | 世界模型输入 state、action、归一化 ξ；默认 ξ 为 33 维 |
| §3.2：解析或学习的 reward/observation/termination heads | 奖励、观测和终止共用解析 MDP，世界模型拟合动力学 |
| Eq. 1：logged actions 上的 k=8 自回归预测 | 回放采样连续窗口，沿真实执行动作展开；跨 episode 边界做 mask |
| Eq. 2：H=32 的策略 BPTT | 固定世界模型/critic 参数，保留输入导数；生存概率加权奖励与末端值 bootstrap |
| Eq. 3：λ-return | 从末端 critic 值反向递推，固定目标后执行 critic 更新 |
| 表 A.4：网络 | 动力学 2048×3，motion encoder/decoder 2048/1024/512/256，critic 256/128，32 维 tanh latent |
| 表 A.1：actor 观测 | 默认 29+29+3+3+29=93 维，10 帧历史共 930 维 |
| Appendix A：未来参考 | 使用 f+1…f+10 共 10 帧 goal；每帧 73 维为此实现的编码选择 |
| 表 A.2：奖励 | 六项 tracking、action rate 和 joint limit；仿真与想象共用公式 |
| Appendix A：终止 | root 高度差 0.25 / tilt 差 0.8；仿真 hard done，想象 soft termination |
| D.1：回报驱动起点采样 | 1 秒 bins，50% uniform / 40% current softmax / 10% historical softmax |
| Fig. 5：仿真时钟 | Newton/MuJoCo-Warp，50 Hz 控制，4 个 5 ms 子步 |
| F.1：完整片段评估 | 从每段首帧开始，无观测噪声和随机物理采样，失败后不随机重启 |

具体默认超参数见 [`supermimic_cfg.py`](supermimic/tasks/tracking/config/g1/agents/supermimic_cfg.py)，配置和输入布局见 [CONFIGURATION.md](CONFIGURATION.md)。表中的一致性指结构或列明参数，不延伸为未公开细节或结果一致。

## 输入约定

actor 默认不拼接额外 6D 参考相对姿态，`reference_orientation=None`。历史 checkpoint 中保存的可选 99 维观测仍是另一种接口，不能裁掉旧 decoder 的输入后继续使用。

critic 输入为完整 state + goal + ξ，world model 输入为完整 state + action + ξ。state 采用 replay 的 71 维表示：世界坐标根位置、WXYZ 四元数、线速度、角速度、29 个关节位置和速度。论文未逐维规定 state 编码，此表示属于项目选择。

当前 checkpoint 以 `model_input_version=2` 标识。旧 world model 即使输入宽度相同，特征语义也可能不同；仅检查张量尺寸不能证明兼容。actor 使用显式采集更新的 BatchNorm，旧 RunningNorm actor 的归一化键不兼容。

## 明确差异与工程选择

| 项目 | 实现与影响 |
|---|---|
| 世界模型损失 | 当前将局部表达的 70 维状态误差除以动态输出标准差及各分量 floor，再对分量取均值、时间求和。它不等同于 Eq. 1 写出的状态平方范数和，也改变了误差分量之间的相对权重。见 `Trainer._wm_err`、`WorldModel.state_error`。 |
| 训练精度 | 默认 `--amp bf16` 对 MLP 使用 CUDA autocast；论文 Fig. 5 标注 TF32。`--amp none` 关闭 BF16 路径，但不能证明其他数值细节一致。 |
| 初期调度 | 当前先采集 64 步，前 500 轮仅训练世界模型；这些额外 warmup 未在 Algorithm A.1 中列明。 |
| 物理自碰撞 | 标准 G1 默认关闭。论文移除 self-collision reward penalty 不等于要求关闭碰撞求解，两者应分别理解。 |
| 随机化时机 | friction 和 torso CoM 默认在 startup 抽样，joint offsets 每 episode 抽样；范围一致不代表各项重抽时机一致。 |
| 状态/目标表示 | frame-local increments、70 维动力学误差、73 维单帧 goal、30→50 Hz 插值与片段尾部处理均属于具体实现选择。 |
| 终止和平滑 | 10 秒训练 episode 限制、clip 尾部 timeout、soft survival 温度 0.02/0.04 及 reset 噪声并非论文全部明确规定的参数。 |
| 额外配置能力 | PD 增益随机化默认关闭；启用后 context 由 33 维变为 91 维，这是项目扩展配置。 |

这些选择会影响优化和评估，变更时应独立验证。上游代码与本项目数值相近也不能替代与论文的核对。

## 已知限制

- **训练恢复**：checkpoint 不保存 replay buffer。重新建立的缓冲区可能不足以提供下一次世界模型更新需要的连续窗口，因此当前训练入口明确拒绝 `--resume`。checkpoint 中有优化器状态不等于完整续训可用。
- **重复 bin 更新**：`TrackingEnv.finish_transition()` 使用 `index_put_` 更新起点回报估计，同批多个环境命中同一 bin 时未归约。重复索引且 `accumulate=False` 的行为不可靠，应在后续修复中定义 batch 聚合方式。
- **短动作采样**：`Trainer.sample_valid_starts()` 要求起点之后有足够长的 reference；若回放中不存在合格起点，当前循环没有失败退出保护。自定义数据必须提供足够长的动作。
- **片段边界**：重采样后的片段长度若满足 `length % 50 == 1`，最后一个 bin 可能仅包含 terminal frame，当前 sampler 存在越过片段边界的风险。`tools/audit_paper.py` 可以扫描本地标准数据是否触及此边界。

## 未覆盖的论文实验

当前仓库不提供 Appendix B/G 的多接口联合训练、独立 velocity encoder、每接口 critic、混合 rollout 或 latent alignment，也未复现完整 BONES-SEED 大规模 benchmark、多 GPU 梯度同步和实机实验。

默认 40 个 G1/LAFAN1 动作面向受控参考跟踪实验范围。CPU 单元测试、短 GPU 冒烟运行和上游性能报告均不能证明本重构的长期收敛、论文训练时间或成功率。

## 验证入口

```bash
python -m unittest discover -s tests -v
python tools/audit_paper.py
python tools/test_env.py --num_envs 8
```

单元测试包含真实 Isaac Lab Managers 与确定性 CPU 物理替身，覆盖 reset/step 生命周期、配置、单次学习更新、评估录制、输入约定和 TorchScript 导出。`test_env.py` 另外启动真实 GPU solver 检查机器人状态与 FK。

`audit_paper.py` 在本仓库内输出默认观测宽度、受控世界模型损失、warmup/precision、空 replay 恢复窗口和重复 bin 更新示例，并扫描数据边界。单位误差和单位标准差下，当前八步 WM loss 为 8；同一 70 分量的原始平方和为 560。这只说明 reduction 差异，不能推断论文作者实际使用的未公开状态坐标和损失尺度。
