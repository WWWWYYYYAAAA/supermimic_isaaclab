# Current implementation alignment

本项目实现单 reference-motion 接口的仿真采集、世界模型拟合、想象 BPTT 和 critic 更新。
网络接口 v7 按本地 `SuperMimic-Reproduced` 对齐，区别是所有网络输入改为
EmpiricalNormalization，anchor 默认 true，termination 可选解析或神经网络，默认后者。
完整逐维输入、输出、数据集时序和损失定义见 [NETWORK_INPUTS.md](NETWORK_INPUTS.md)。
这不是论文实验结果或所有算法细节已经复现的声明。

## 已对齐的代码语义

- 71D canonical state、133D dynamics 输入，70D 速度增量和显式积分修正，SiLU。
- motion goal：f..f+K-1，每帧 68D，窗口末帧到达动作末尾即 timeout；可选 torso 相对位姿 9D。
- actor 93D/frame，term-major 历史，编码器/解码器 2048/1024/512/256，32D tanh latent。
- critic 256/128，输入包含额外 4D reference-heading 位置误差和 wrapped yaw 误差。
- termination 4→64→64→1，BCE 监督真实失败标签；不把 timeout 当失败，保留预测状态梯度。
- 状态损失按固定分量尺度归一化，70 项求和、batch/time 平均，再加 masked BCE。

## 保留的差异与限制

- 输入归一化有意采用 RSL-RL 风格 EmpiricalNormalization；Reproduced 使用固定缩放。
- 本项目 simulator 的下一状态对应 reference f+1；没有复制 Reproduced 的 reference 时序差异。
- reward/FK 的参考重新锚定方式、原有 PD offset、随机化与采样策略仍为本项目实现。
- BPTT 目标保持原缩放；warmup 64 环境步、500 次 WM-only 更新，默认 BF16。这些不能等同于
  论文未公开的实现细节，也不意味着复现了论文 TF32 训练速度。
- 原始 replay 仍为世界坐标状态；网络边界显式转换 canonical 特征和 body angular velocity。
- 同批重复 bin 的 index_put_ 更新未归约；短动作缺少有效 imagination 起点时仍有采样限制。
- reset 采样排除没有有效完整参考窗口的尾部 bin；结束条件按 goal 窗口末帧计算。
- checkpoint 不保存 replay；训练入口的 resume 限制保留。主 Trainer 的恢复路径会重做 warmup，
  但它不代表精确恢复整个随机过程和 replay。
- 本地测试使用真实 Isaac Lab Managers 和 CPU 物理替身，不能证明 GPU 长期稳定性或收敛。
- 未实现多接口联合训练、独立 velocity encoder、多 GPU 训练和完整 BONES-SEED/实机实验。

## 验证

```bash
python -m unittest discover -s tests -v
python -m unittest tools.test_motion_goal -v
python tools/audit_paper.py
```

测试包含归一化累计统计/冻结、goal 边界和帧序、yaw 不变性、critic yaw 误差、torso anchor
梯度、动力学积分、termination 梯度和 timeout 标签、CPU 学习更新、评估录制及新旧 TorchScript 导出。
