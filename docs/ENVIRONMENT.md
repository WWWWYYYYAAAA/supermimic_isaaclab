# 环境配置

本项目使用 Isaac Lab 3 的 kit-less Newton / MuJoCo-Warp 后端。请使用固定版本源码与锁定依赖，不要用 Isaac Lab 的其他版本或单独安装 `mujoco` 来替代完整环境。

## 固定版本与系统需求

| 项目 | 配置 |
|---|---|
| Isaac Lab | `46d6cc3d7b115906fad12efd372fbf69f4e3d3ed`，该提交 `VERSION=3.0.0` |
| 依赖锁 | 固定提交的 `IsaacLab/uv.lock` |
| Python | 3.12 |
| 安装工具 | uv 0.12.9 |
| Docker 基础系统 | Ubuntu 24.04，来自固定提交的 `Dockerfile.kitless` |
| 物理 | Newton / MuJoCo-Warp，50 Hz 控制，4 个物理子步 |
| 执行设备 | NVIDIA CUDA GPU；导出和部分测试可使用 CPU |

本地验证环境中主要包为 PyTorch `2.11.0+cu130`、Warp `1.17.0`、MuJoCo / MuJoCo-Warp `3.15.0`、Newton `1.8.0.dev0`、USD Exchange `3.0.1`。Newton 等开发包还由 `uv.lock` 记录源提交；只匹配版本字符串不足以重建相同环境。

宿主机需安装 NVIDIA 驱动。先检查 `nvidia-smi`，安装后再检查 `torch.cuda.is_available()`；该锁使用 CUDA 13 构建，驱动必须兼容相应运行时。完整安装还需访问 GitHub、Python 包源、容器源和 Hugging Face。建议为镜像、依赖缓存及训练输出预留数十 GB 磁盘空间。

本次整理采用已有环境执行验证：59 项 CPU 回归测试、3 项 motion-goal 测试和源码/资产独立性检查通过；RTX 5090 上完成 2 个环境、2 次迭代的短训练及 checkpoint 保存、TorchScript 导出、checkpoint 评估/轨迹录制和 20 步 headless 播放。短训练使用截取的 300 帧源动作并关闭 `torch.compile`，不代表完整编译路径或长期收敛验证。未从空环境重新完成 Docker 镜像或原生依赖构建。以下脚本固定了依赖来源；构建仍依赖外部源可用性及本机驱动条件。

## Docker 安装

宿主机预先配置 Docker、NVIDIA Container Toolkit，并确保当前用户可以运行 Docker。

```bash
cd /path/to/supermimic_isaaclab
nvidia-smi
./tools/setup.sh
./docker/run_container.sh 0 128g
./tools/download_data.sh
```

`setup.sh` 获取固定 Isaac Lab 提交，按其 `Dockerfile.kitless` 构建 `isaac-lab-kitless:main-46d6cc3`，再构建 `supermimic_isaaclab:latest`。项目镜像补充 `ffmpeg`、编译工具和 NVIDIA EGL vendor 配置。

`run_container.sh` 的两个位置参数分别为 GPU 编号和容器内存上限。启动容器名为 `supermimic_isaaclab`，仓库挂载到 `/workspace/supermimic_isaaclab`。容器内日志、生成资产和缓存因此保存在宿主仓库中。默认脚本不转发桌面显示，适合训练、评估和 headless 播放；交互播放使用原生桌面环境。

`download_data.sh` 在宿主机运行 Python 标准库下载器，下载标准动作、URDF 和网格并校验哈希，然后在容器中执行 MJCF/USD 转换。宿主机需要可用 `python3`。

```bash
docker exec supermimic_isaaclab python -c "import torch; print(torch.__version__, torch.version.cuda); print('CUDA:', torch.cuda.is_available())"
docker exec supermimic_isaaclab python tools/check_independence.py
docker exec supermimic_isaaclab python -m unittest discover -s tests -v
```

已有同名容器时先用 `docker inspect supermimic_isaaclab` 确认它挂载的是此仓库；不要把旧项目的容器当作当前项目环境。停止和恢复当前容器可分别使用 `docker stop supermimic_isaaclab`、`docker start supermimic_isaaclab`。

## 原生环境

原生安装使用同一固定提交和 `uv.lock`，在仓库中创建 `.venv`。先安装 Git、C/C++ 编译工具、Python 开发头文件、OpenGL/EGL 系统库及 `ffmpeg`；桌面播放需要可用显示服务。Ubuntu 的系统包示例：

```bash
sudo apt-get update
sudo apt-get install -y git curl build-essential cmake python3-dev pipx \
  libegl1 libgl1 libopengl0 libglib2.0-0 libgomp1 libvulkan1 \
  libx11-6 libxext6 libxrender1 ffmpeg tzdata
```

随后在仓库根目录执行：

```bash
pipx install 'uv==0.12.9'
export PATH="$HOME/.local/bin:$PATH"
./tools/setup_native.sh
source .venv/bin/activate
python tools/download_data_native.py
python tools/urdf_to_mjcf.py
python tools/mjcf_to_usd.py
python tools/check_independence.py
python -m unittest discover -s tests -v
```

脚本要求 `uv --version` 为 0.12.9；若已有不同版本的 uv，请先确认 PATH 指向上述安装。不要额外混装 `usd-core`，USD 相关依赖以锁文件为准。激活 `.venv` 后，从仓库根目录运行 README 中去掉 `docker exec supermimic_isaaclab` 的 Python 命令。

若复用现有 conda/venv，必须核对固定 Isaac Lab 与全部二进制依赖；导入路径优先本仓库源码不代表二进制依赖自动兼容。不要将旧项目的 `IsaacLab/` 或 `data/` 做成跨目录软链接，独立性检查会拒绝这类资产依赖。

## 检查层次

1. `python tools/check_independence.py`：检查 Manager 环境、29 关节 FK、标准资产、USD 依赖闭包和本地源码路径。
2. `python -m unittest discover -s tests -v`：CPU 逻辑、真实 Managers 配合物理替身、导出等回归测试。部分测试仍需标准 G1 MJCF 和 Isaac Lab 依赖。
3. `python tools/test_env.py --num_envs 8`：启动真实 GPU 环境，检查状态与 FK。
4. 短训练、checkpoint 评估和播放：检查完整 CUDA 编译与学习流程；单元测试不能替代这一层。

## 常见问题

| 现象 | 检查方法 |
|---|---|
| `No module named isaaclab` 或缺少 Newton 类 | 检查 `IsaacLab/` 固定提交、激活的解释器和完整锁定依赖 |
| 找不到 `g1_29dof.xml` / `.usda` | 下载 URDF 后还需分别运行两个转换器；只下载 CSV 不够 |
| `torch.cuda.is_available()` 为 False | 检查驱动、GPU 可见性和容器 GPU 配置；`--headless` 仍需要 GPU |
| 首次运行长时间编译 | `torch.compile` 和 Warp 首次编译会创建缓存；保留 `.cache/` |
| 显存不足 | 同时减小 `num_envs`、`buffer_steps`、`wm_batch`、`pi_batch`；至少保留有效采样窗口 |
| 桌面窗口无法打开 | 确认图形会话/OpenGL；默认 Docker 启动器未配置显示转发 |
| 视频编码失败 | 检查 `ffmpeg` 与 EGL，录制轨迹和视频渲染是两个独立步骤 |

资产转换器使用 standalone USD importers，需在独立 Python 进程中运行；不要在已经启动的 Isaac Lab 仿真进程中进行转换。初次启动仿真还可能读取 Isaac Lab 的远端公共 ground 资产，网络受限时需提前准备对应缓存。
