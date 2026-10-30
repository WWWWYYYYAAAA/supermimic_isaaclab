# SuperMimic Isaac Lab

This repository is a refactoring of [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed.git), built around Isaac Lab's **manager-based environment, `ManagerBasedRLEnv`**. It is an **unofficial reproduction** of [*SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model*](https://arxiv.org/abs/2610.28480).

The project focuses on tracking LAFAN1 reference motions with the 29-DoF Unitree G1. It retains SuperMimic's learning loop: simulator collection → world-model fitting from replay → policy updates through imagined rollouts with BPTT → critic updates. Isaac Lab Managers organize the environment, with Newton / MuJoCo-Warp providing kit-less physics. The trainer does not use PPO.

Training, evaluation, interactive playback, and TorchScript export are included. See [PAPER_ALIGNMENT.md](PAPER_ALIGNMENT.md) for implementation differences and known limitations. Performance reported by the upstream repository is not presented as validation of this refactoring.

## Actual runtime environment

The following configuration, memory usage, and training throughput are **reported by the maintainer**:

| Item | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| CPU | AMD Ryzen 9 9950X |
| Parallel environments | 4096 |
| GPU memory usage | Approximately **22.4 GB** (21362 MiB, approximately 20.9 GiB) |
| Training throughput | Approximately **3 iterations/s** |

GB uses decimal units (1 GB = 10⁹ bytes), while GiB uses binary units (1 GiB = 1024 MiB). Memory usage and throughput depend on replay capacity, batch sizes, and other training settings. These figures describe the maintainer's run, not a general performance guarantee.

## Repository layout

| Path | Contents |
|---|---|
| `supermimic/tasks/tracking/` | Manager environment, MDP interfaces, and configuration |
| `supermimic/tasks/tracking/config/g1/flat_env_cfg.py` | Standard G1 model, PD control, randomization, rewards, and observations |
| `supermimic/tasks/tracking/config/g1/agents/supermimic_cfg.py` | Training hyperparameters and CLI |
| `supermimic/training/trainer.py` | World-model, BPTT policy, and critic updates |
| `supermimic/{motion,mdp,networks,buffer}.py` | Motion library, analytic MDP, networks, and replay buffer |
| `tools/` | Setup, asset conversion, playback, export, and validation tools |
| `tests/` | CPU numerical, configuration, Manager lifecycle, and export tests |
| `docker/` | Pinned runtime environment and container launcher |
| `IsaacLab/`, `data/` | Installed or downloaded locally; excluded from Git |
| `logs/`, `exports/`, `results/` | Runtime outputs; excluded from Git |

Code and assets are resolved within this repository without relying on sibling project directories. `supermimic/__init__.py` prioritizes the pinned source under this checkout's `IsaacLab/source/`.

## Environment setup

See the [environment guide](docs/ENVIRONMENT.md) for detailed installation steps, dependency versions, and troubleshooting. Docker is the recommended installation path. Dependencies are managed by `uv.lock` from the pinned Isaac Lab commit.

You need Linux, an NVIDIA GPU and driver compatible with the selected CUDA runtime, Git, Docker, and the NVIDIA Container Toolkit. Training, physics evaluation, and headless playback require CUDA. Policy export can run on a CPU. Interactive playback additionally requires a graphical desktop and OpenGL; video encoding requires `ffmpeg`.

```bash
git clone git@github.com:WWWWYYYYAAAA/supermimic_isaaclab.git
cd supermimic_isaaclab

# Pin Isaac Lab to 46d6cc3 and build the kit-less base and project images
./tools/setup.sh

# GPU index and container memory limit; adjust to your machine
./docker/run_container.sh 0 128g

# Download and verify standard G1/LAFAN1 assets, then convert MJCF/USD in Docker
./tools/download_data.sh

# Check CUDA, dependencies, and assets
# Run these docker exec commands from the repository root on the host
docker exec supermimic_isaaclab python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
docker exec supermimic_isaaclab python tools/check_independence.py
```

The image is `supermimic_isaaclab:latest`, the container is `supermimic_isaaclab`, and the repository is mounted at `/workspace/supermimic_isaaclab`. For a native environment, follow the [native installation instructions](docs/ENVIRONMENT.md), then omit `docker exec supermimic_isaaclab` from the Python commands below.

Motion data comes from [LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset): 40 G1 clips with 264,705 source frames at 30 FPS, resampled to 50 Hz when loaded. The downloader records the resolved dataset revision and file hashes in `data/source_manifest.json`.

```text
data/
├── lafan1_g1/*.csv
├── g1_description/
│   ├── g1_29dof_rev_1_0.urdf
│   ├── g1_29dof.xml
│   └── meshes/
└── g1_usd/g1_29dof/
    ├── g1_29dof.usda
    └── ...                 # USD payloads and dependent assets
```

The asset downloader covers the robot and motions. The first simulator launch may also fetch Isaac Lab's public ground-plane USD and textures from NVIDIA; prepare that cache before running without network access.

## Training

```bash
# Example with fewer environments and smaller batches; adjust to available VRAM
# Other algorithm settings retain their defaults
docker exec supermimic_isaaclab python supermimic/train.py \
  --num_envs 512 --buffer_steps 256 --pi_batch 512 --wm_batch 512 \
  --log_dir logs/lafan1

# Show all options
docker exec supermimic_isaaclab python supermimic/train.py --help
```

Defaults use 4096 environments, a replay capacity of 1000 steps per environment, and 4096 imagined starting states, which require more GPU memory than this example. The first run compiles `torch.compile` kernels. Changing batch sizes changes the training conditions, so convergence times from different configurations are not directly comparable.

Log directories receive an Asia/Shanghai (UTC+8) timestamp by default, for example `logs/lafan1_YYYYMMDD_HHMMSS_ffffff`. The actual directory is printed in the terminal. `--no-timestamp` uses the exact supplied path and requires it to be absent or empty. Use the **actual run directory** for evaluation, playback, and export.

| Output | Contents |
|---|---|
| `args.json` | Effective CLI arguments |
| `task_settings.json` | PD, randomization, reward, observation, and model configuration snapshot |
| `train_log.csv` | Training metrics |
| `ckpt_002000.pt`, etc. | Checkpoints saved every 2000 iterations by default |
| `last.pt` | Full model and optimizer states, updated every 200 iterations by default |

The training entry point currently rejects `--resume`: checkpoints do not save and restore the replay buffer, so valid training windows cannot be guaranteed after recovery. Start training in a new directory. Checkpoints remain usable for evaluation, playback, and policy export.

## Evaluation and recording

Set `RUN` to the actual directory printed during training. The checkpoint in these examples must already exist.

```bash
RUN=logs/lafan1_YYYYMMDD_HHMMSS_ffffff

docker exec supermimic_isaaclab python supermimic/evaluate.py \
  --run_dir "$RUN" --ckpt "$RUN/ckpt_002000.pt"

# Watch for new numbered checkpoints and write evaluation results to eval.csv
docker exec supermimic_isaaclab python supermimic/evaluate.py \
  --run_dir "$RUN" --watch

# Record trajectories for one checkpoint, then render a video
docker exec supermimic_isaaclab python supermimic/evaluate.py \
  --run_dir "$RUN" --ckpt "$RUN/ckpt_002000.pt" --record "$RUN/traj.npz"
docker exec -e MUJOCO_GL=egl supermimic_isaaclab python tools/render.py \
  "$RUN/traj.npz" "$RUN/tracking.mp4" --clips dance1_subject1 --seconds 10
```

Evaluation starts at the first frame of each clip, disables observation noise and physical parameter randomization, and uses the saved configuration's nominal parameters. For custom motions, pass `--motion_dir /path/to/motions` to training, evaluation, and playback. CSV files must use the standard 36-column, 30 FPS format: root position (3), root quaternion in XYZW order (4), and joint positions in the established 29-joint order.

## Interactive playback

Run from a native environment with CUDA and a graphical desktop:

```bash
python tools/play_tracking.py --ckpt "$RUN/ckpt_002000.pt"
python tools/play_tracking.py --run_dir "$RUN" --clips dance1_subject1 jumps1_subject1
```

Without `--ckpt`, playback selects the latest numbered checkpoint in `--run_dir`. Without `--clips`, it cycles through all clips in the motion directory in name order. The solid robot follows the policy; the translucent blue robot shows the reference motion. MuJoCo-Warp in the Manager environment computes physics, while the desktop MuJoCo viewer displays states and supplies mouse perturbations.

| Control | Action |
|---|---|
| Space / R | Pause or resume / replay the current clip |
| N / P | Next / previous clip |
| G / F | Toggle the reference robot / camera tracking |
| Double-click the solid robot | Select a body for perturbation |
| Ctrl + right mouse drag | Apply force |
| Ctrl + left mouse drag | Apply torque |

Use `--speed 0.5` to slow the display, `--paused` to start paused, or `--global-reference` to display the reference's original world XY position. Standard G1 disables physical self-collisions by default. Playback restores the saved checkpoint setting when available; `--self-collisions` or `--no-self-collisions` explicitly overrides it.

A CUDA physics check without a desktop can run in the container:

```bash
docker exec supermimic_isaaclab python tools/play_tracking.py \
  --ckpt "$RUN/ckpt_002000.pt" --headless --steps 300
```

## TorchScript policy export

```bash
docker exec supermimic_isaaclab python tools/export_policy.py \
  --ckpt "$RUN/ckpt_002000.pt"
```

By default, export creates the full policy, encoder, and decoder under `export/` beside the checkpoint, each with JSON metadata:

```text
export/policy_ckpt_002000.pt
export/policy_ckpt_002000_encoder.pt
export/policy_ckpt_002000_decoder.pt
export/*.pt.json
```

Use `--out exports/policy.pt` to select an output path, `--component policy|encoder|decoder|all` to select components, or `--check-inputs` to include runtime shape checks. Export does not launch a simulator or require CUDA, MuJoCo, or Isaac Lab; it requires a PyTorch version compatible with the checkpoint.

The default interface is shown below. Always follow the layout saved in the accompanying `.pt.json` file.

```python
import torch

policy = torch.jit.load("path/to/export/policy_ckpt_002000.pt", map_location="cpu").eval()
# obs_history: float32 [B, 10, 93], oldest frame first, built and scaled per task config
# motion_goal: float32 [B, 730], following MotionLib.goal's reference feature layout
with torch.no_grad():
    actions = policy(obs_history.flatten(start_dim=1), motion_goal)  # [B, 29]
```

The encoder maps a `[B, 730]` goal to a `[B, 32]` latent. Call the decoder as `decoder(obs_flat, latent)`, with `[B, 930]` observation history and the latent. All inputs must be `float32`. The full policy includes observation and goal normalization and produces normalized actions. An external controller must construct PD targets using the default joint positions, nominal offsets, and action scales in the metadata. Exports do not contain the motion library, simulator, or hardware communication.

## Configuration and validation

See [CONFIGURATION.md](CONFIGURATION.md) for detailed settings. Control runs at 50 Hz, with four 5 ms Newton substeps per control step. The default actor receives 10 frames of 93-dimensional proprioception and a goal containing 10 frames of 73-dimensional reference features. The critic receives state + goal + context; the world model receives state + action + context.

After installing dependencies and generating the standard assets, run:

```bash
docker exec supermimic_isaaclab python -m unittest discover -s tests -v
docker exec supermimic_isaaclab python -m unittest tools.test_motion_goal -v
docker exec supermimic_isaaclab python tools/audit_paper.py

# Check the actual GPU solver separately from CPU unit tests
docker exec supermimic_isaaclab python tools/test_env.py --num_envs 8
```

Validation performed during this refactoring:

- **59/59 CPU unit tests** and **3 motion-goal tests** passed. These cover configuration propagation, learning updates, evaluation, input contracts, and export, using real Isaac Lab Managers with a deterministic CPU physics substitute where applicable.
- The repository independence check passed, including local source paths and asset dependencies.
- A real GPU smoke test completed training and checkpoint saving with **2 environments and 2 iterations**, using a short LAFAN1 excerpt of **300 source frames** with **`torch.compile` disabled**. All three TorchScript components exported successfully; checkpoint evaluation, trajectory recording, and **20 steps of headless playback** also completed.

These checks do not establish full training convergence or reproduction of the paper's performance. A fresh Docker installation has not been validated end to end.

The current actor uses 93 values per frame and BatchNorm statistics; world-model and critic input semantics are identified by `model_input_version=2`. Playback or export compatibility with older weights depends on saved configuration and normalization keys, not just matching tensor dimensions.

## Attribution and licenses

This refactoring builds on [zita-ch/supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) and retains its [MIT License](LICENSE) and copyright notice. The paper, Isaac Lab, datasets, and robot assets retain their respective licenses.

- Paper: [SuperMimic, arXiv:2610.28480](https://arxiv.org/abs/2610.28480).
- Isaac Lab: [pinned commit `46d6cc3`](https://github.com/isaac-sim/IsaacLab/commit/46d6cc3d7b115906fad12efd372fbf69f4e3d3ed).
- LAFAN1 motions and G1 assets: [download source and dataset documentation](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset). LAFAN1 uses CC BY-NC-ND 4.0, and the Unitree G1 description uses BSD-3-Clause. Data and generated assets are not distributed with this repository.
