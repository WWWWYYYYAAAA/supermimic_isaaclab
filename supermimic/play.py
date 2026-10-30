"""Live G1 motion tracking with ManagerBasedRLEnv / MuJoCo-Warp physics.

The desktop MuJoCo viewer displays the policy/reference and forwards mouse forces.
Space: pause; R: restart; N/P: next/previous clip; G: ghost; F: follow camera.
Double-click the solid robot, then Ctrl + right-drag to pull or Ctrl + left-drag to rotate.
Self-collisions follow the saved task settings unless explicitly overridden.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import queue
import runpy
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.paths import REPO

os.environ.setdefault("MUJOCO_GL", "glfw")

ROOT = REPO


def build_parser(include_launcher=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        "--checkpoint",
        type=Path,
        help="Defaults to the latest numbered checkpoint in --run_dir",
    )
    parser.add_argument(
        "--run_dir", type=Path, default=ROOT / "logs/run"
    )
    parser.add_argument("--clips", nargs="+", help="Motion CSV stems to cycle through")
    parser.add_argument("--motion_dir", type=Path)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Display speed only; simulation dt is unchanged",
    )
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--global-reference", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Finite physics check without opening a desktop viewer",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after this many control steps; 0 means interactive",
    )
    parser.add_argument(
        "--self-collisions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override saved self-collisions; otherwise use the saved task or standard G1 default",
    )
    if include_launcher:
        from isaaclab.app import add_launcher_args

        add_launcher_args(parser)
    else:
        parser.add_argument("--device", default="cuda:0")
    return parser


def parse_args(argv=None, include_launcher=True):
    parser = build_parser(include_launcher)
    args = parser.parse_args(argv)
    if not math.isfinite(args.speed) or args.speed <= 0 or args.steps < 0:
        parser.error("speed must be positive and finite; steps must be nonnegative")
    if args.headless and (args.steps == 0 or args.paused):
        parser.error("--headless requires --steps > 0 and cannot be paused")
    return args


@dataclass
class Playback:
    checkpoint_path: Path
    checkpoint: dict
    env_cfg: object
    motion_dir: Path
    clips: list[str]
    files: list[str]
    checkpoint_self_collisions: bool | None


def prepare_playback(args) -> Playback:
    """Resolve inputs on CPU and restore the checkpoint's task settings."""
    import torch
    from supermimic.tasks.tracking.config.g1.flat_env_cfg import (
        G1FlatEnvCfg,
    )
    from supermimic.tasks.tracking.settings import apply_task_settings

    checkpoint_path = args.ckpt
    if checkpoint_path is None:
        checkpoints = sorted(args.run_dir.expanduser().glob("ckpt_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(
                f"No saved checkpoint in {args.run_dir}; provide --ckpt"
            )
        checkpoint_path = checkpoints[-1]
    checkpoint_path = checkpoint_path.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not {"pi", "args"} <= checkpoint.keys():
        raise ValueError(
            "Playback requires a checkpoint with pi weights and training args"
        )
    saved_args = checkpoint["args"]
    for name in ("goal_frames", "hist"):
        value = saved_args.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Checkpoint {name} must be a positive integer")
    cfg = G1FlatEnvCfg()
    settings = checkpoint.get("task_settings")
    settings_path = checkpoint_path.parent / "task_settings.json"
    if settings is None and settings_path.is_file():
        settings = json.loads(settings_path.read_text())
    checkpoint_collisions = None
    if settings is not None:
        apply_task_settings(cfg, settings)
        checkpoint_collisions = settings["self_collisions"]
        if cfg.policy_observation.history != saved_args["hist"]:
            raise ValueError(
                "Checkpoint history and saved observation settings disagree"
            )
        if cfg.commands.motion.goal_frames != saved_args["goal_frames"]:
            raise ValueError("Checkpoint goal size and saved task settings disagree")
    cfg.policy_observation.history = saved_args["hist"]
    cfg.commands.motion.goal_frames = saved_args["goal_frames"]
    if args.self_collisions is not None:
        cfg.self_collisions = args.self_collisions
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.prepare()
    motion_dir = (args.motion_dir or Path(cfg.motion_dir)).expanduser().resolve()
    available = {path.stem: path for path in sorted(motion_dir.glob("*.csv"))}
    clips = args.clips or list(available)
    if not clips or any(name not in available for name in clips):
        raise ValueError(
            f"Unknown or missing motion clips: {clips}; directory: {motion_dir}"
        )
    for path in (cfg.robot_model.mjcf, cfg.robot_model.usd):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return Playback(
        checkpoint_path,
        checkpoint,
        cfg,
        motion_dir,
        list(clips),
        [str(available[name]) for name in clips],
        checkpoint_collisions,
    )


def preview_model(mjcf_path: str):
    """Construct the local solid/ghost display scene, without rendering or stepping it."""
    import mujoco
    import numpy as np

    namespace = runpy.run_path(
        str(ROOT / "tools/render.py"), run_name="tracking_preview_scene"
    )
    namespace["two_robot_xml"].__globals__["MJCF"] = mjcf_path
    model = mujoco.MjModel.from_xml_string(namespace["two_robot_xml"]())
    if model.nq != 72:
        raise ValueError("Expected two G1 robots with 36 position coordinates each")
    model.vis.headlight.ambient[:] = 0.45
    model.vis.headlight.diffuse[:] = 0.55
    ghosts = np.array(
        [
            index
            for index in range(model.ngeom)
            if model.body(int(model.geom_bodyid[index])).name.startswith("ref_")
        ]
    )
    return model, ghosts


def set_preview_state(
    model,
    data,
    robot_pos,
    robot_quat,
    joints,
    ref_pos,
    ref_rotation,
    ref_joints,
    global_reference=False,
    *,
    robot_velocity=None,
):
    """Map policy/reference states to the desktop model; never integrate its physics."""
    import mujoco
    import numpy as np

    data.qpos[:3] = robot_pos
    data.qpos[3:7] = robot_quat  # wxyz
    data.qpos[7:36] = joints
    data.qpos[36:39] = ref_pos
    if not global_reference:
        data.qpos[36:38] = robot_pos[:2]
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.asarray(ref_rotation, dtype=np.float64).reshape(9))
    data.qpos[39:43] = quat
    data.qpos[43:72] = ref_joints
    mujoco.mj_kinematics(model, data)
    data.qvel[:] = 0
    if robot_velocity is not None:
        linear, angular, joint = robot_velocity
        data.qvel[:3] = linear
        # MuJoCo free-joint angular velocity is expressed in the root frame.
        rotation = data.xmat[model.body("pelvis").id].reshape(3, 3)
        data.qvel[3:6] = rotation.T @ angular
        data.qvel[6:35] = joint
    # Mouse spring initialization needs the mass matrix, and damping needs
    # body velocities. Forward computes these without stepping desktop physics.
    mujoco.mj_forward(model, data)


class MousePerturbation:
    """Forward desktop mouse wrenches to the live articulation in world space."""

    def __init__(self, model, data, robot, origin):
        self.model, self.data, self.robot = model, data, robot
        self.origin = origin.detach().cpu().numpy()
        self.composer = robot.permanent_wrench_composer
        self.active = False
        names = {name: index for index, name in enumerate(robot.body_names)}
        self.body_ids = {}
        root = model.body("pelvis").id
        for body in range(1, model.nbody):
            if int(model.body_rootid[body]) != root:
                continue  # The floor and blue reference never drive the robot.
            ancestor = body
            while ancestor and model.body(ancestor).name not in names:
                ancestor = int(model.body_parentid[ancestor])
            if not ancestor:
                raise ValueError(f"Cannot map viewer body {model.body(body).name} to the robot")
            self.body_ids[body] = names[model.body(ancestor).name]

    def clear(self, perturb=None):
        if self.active:
            self.composer.reset()
            self.active = False
        self.data.xfrc_applied[:] = 0
        if perturb is not None:
            perturb.active = perturb.active2 = 0

    def apply(self, perturb):
        """Called under the viewer lock before each live control step."""
        import mujoco
        import numpy as np
        import torch

        self.data.xfrc_applied[:] = 0
        selected = int(perturb.select)
        if not (perturb.active or perturb.active2) or selected not in self.body_ids:
            self.clear()
            return
        mujoco.mjv_applyPerturbForce(self.model, self.data, perturb)
        wrench = self.data.xfrc_applied[selected].copy()
        if not np.isfinite(wrench).all():
            self.clear(perturb)
            raise RuntimeError("Nonfinite mouse perturbation force")
        # MuJoCo's torque is about the displayed body's CoM. Supplying that
        # position also handles fixed links merged into a simulator ancestor,
        # and any difference between displayed and randomized simulator CoMs.
        position = self.data.xipos[selected] + self.origin
        tensor = lambda values: torch.tensor(
            values, dtype=torch.float32, device=self.robot.device
        ).reshape(1, 1, 3)
        self.active = True
        self.composer.set_forces_and_torques_index(
            forces=tensor(wrench[:3]),
            torques=tensor(wrench[3:]),
            positions=tensor(position),
            body_ids=[self.body_ids[selected]],
            is_global=True,
        )


def build_runtime(playback, device):
    """Build configured FK/MDP and load a deterministic policy, including shape checks."""
    import torch
    from supermimic import networks
    from supermimic.kinematics import G1Kinematics
    from supermimic.mdp import MDP
    from supermimic.motion import MotionLib
    from supermimic.robot_spec import default_joint_pos_list

    cfg = playback.env_cfg
    ck = playback.checkpoint
    saved = ck["args"]
    networks.AMP_DTYPE = torch.bfloat16 if saved.get("amp", "none") == "bf16" else None
    kin = G1Kinematics(device, mjcf_path=cfg.robot_model.mjcf)
    motion = MotionLib(playback.files, kin, device)
    mdp = MDP(
        kin,
        motion,
        torch.tensor(default_joint_pos_list(), device=device),
        cfg=cfg.analytic_reward,
        observation_cfg=cfg.policy_observation,
        xi_dim=cfg.randomization.context_dim,
    )
    policy = networks.Policy(
        mdp.obs_dim, 73 * saved["goal_frames"], hist=saved["hist"]
    ).to(device)
    try:
        policy.load_state_dict(ck["pi"])
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint policy shapes do not match the restored observation/goal settings"
        ) from exc
    policy.eval()
    return policy, motion, mdp


def play_loop(env, policy, motion, mdp, args, mjcf_path, *, viewer_factory=None):
    """Run one simulated environment; optional viewer resources are always closed."""
    import torch
    from supermimic.motion import CONTROL_DT
    from supermimic.rotations import mat_to_quat

    saved_history = args.history_length
    goal_frames = args.goal_frames
    ids = torch.zeros(1, dtype=torch.long, device=env.device)
    keys = queue.SimpleQueue()
    clip_index = steps = episode_steps = 0
    paused, follow, show_ghost = args.paused, True, True
    total_reward, hold_until, hist = 0.0, None, None
    model = display = ghosts = viewer = None
    mouse = None

    def reset():
        nonlocal hist, total_reward, episode_steps, hold_until
        if mouse is not None:
            with viewer.lock():
                mouse.clear(viewer.perturb)
        env.done_eval[:] = False
        env.failed_eval[:] = False
        env.reset(ids, motion.clip_start[clip_index : clip_index + 1])
        env.refresh()
        hist, total_reward, episode_steps, hold_until = None, 0.0, 0, None
        duration = (int(motion.clip_len[clip_index]) - 1) * CONTROL_DT
        print(f"Playing {motion.names[clip_index]} ({duration:.2f}s)", flush=True)

    def display_state():
        state = env.state()
        frame = int(env.f[0].item())
        set_preview_state(
            model,
            display,
            state[0][0].cpu().numpy(),
            mat_to_quat(state[1])[0].cpu().numpy(),
            state[4][0].cpu().numpy(),
            motion.root_p[frame].cpu().numpy(),
            motion.root_R[frame].cpu().numpy(),
            motion.q[frame].cpu().numpy(),
            args.global_reference,
            robot_velocity=tuple(state[index][0].cpu().numpy() for index in (2, 3, 5)),
        )

    with torch.no_grad():
        try:
            reset()
            if not args.headless:
                import mujoco

                if viewer_factory is None:
                    from mujoco import viewer as mj_viewer

                    viewer_factory = mj_viewer.launch_passive
                model, ghosts = preview_model(mjcf_path)
                display = mujoco.MjData(model)
                display_state()
                viewer = viewer_factory(model, display, key_callback=keys.put)
                with viewer.lock():
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                        3.2,
                        135,
                        -15,
                    )
                    viewer.cam.lookat[:] = [display.qpos[0], display.qpos[1], 0.8]
                viewer.sync()
                mouse = MousePerturbation(
                    model, display, env.scene["robot"], env.scene.env_origins[0]
                )
                print("MuJoCo tracking window opened.", flush=True)
            while (viewer is None or viewer.is_running()) and (
                not args.steps or steps < args.steps
            ):
                tick = time.perf_counter()
                while not keys.empty():
                    key = keys.get()
                    if key == 32:
                        paused = not paused
                    elif key == ord("R"):
                        reset()
                    elif key in (ord("N"), ord("P")):
                        clip_index = (
                            clip_index + (1 if key == ord("N") else -1)
                        ) % len(motion.names)
                        reset()
                    elif key == ord("G"):
                        show_ghost = not show_ghost
                    elif key == ord("F"):
                        follow = not follow
                if mouse is not None and (paused or hold_until is not None):
                    with viewer.lock():
                        mouse.clear(viewer.perturb)
                if not paused:
                    if hold_until is not None:
                        if args.headless or time.perf_counter() >= hold_until:
                            clip_index = (clip_index + 1) % len(motion.names)
                            reset()
                    else:
                        obs = mdp.obs(env.state(), env.a_prev, env.xi, env.f)
                        hist = (
                            obs[:, None].repeat(1, saved_history, 1)
                            if hist is None
                            else torch.cat((hist[:, 1:], obs[:, None]), dim=1)
                        )
                        action = policy(hist.flatten(-2), motion.goal(env.f, goal_frames))
                        if not torch.isfinite(action).all():
                            raise RuntimeError("Policy produced nonfinite actions")
                        if mouse is not None:
                            with viewer.lock():
                                mouse.apply(viewer.perturb)
                        transition = env.step(action)
                        total_reward += float(transition["r"][0])
                        episode_steps += 1
                        steps += 1
                        if not all(
                            bool(torch.isfinite(value).all()) for value in env.state()
                        ):
                            raise RuntimeError("Nonfinite simulator state")
                        if steps % 50 == 0:
                            reward = total_reward / (
                                episode_steps * mdp.max_step_reward
                            )
                            print(
                                f"{motion.names[clip_index]} t={episode_steps * CONTROL_DT:.1f}s mean_reward={reward:.3f}",
                                flush=True,
                            )
                        if bool(transition["done"][0]):
                            status = (
                                "FAILED" if bool(transition["term"][0]) else "COMPLETED"
                            )
                            print(
                                f"{status}: {motion.names[clip_index]}, t={episode_steps * CONTROL_DT:.2f}s",
                                flush=True,
                            )
                            hold_until = time.perf_counter() + 1.5
                            if mouse is not None:
                                with viewer.lock():
                                    mouse.clear(viewer.perturb)
                if viewer is not None:
                    with viewer.lock():
                        display_state()
                        model.geom_rgba[ghosts, 3] = 0.25 if show_ghost else 0.0
                        if follow:
                            viewer.cam.lookat[:] = [
                                display.qpos[0],
                                display.qpos[1],
                                0.8,
                            ]
                    viewer.sync()
                    time.sleep(
                        max(0.0, CONTROL_DT / args.speed - (time.perf_counter() - tick))
                    )
        finally:
            if viewer is not None:
                try:
                    if mouse is not None:
                        with viewer.lock():
                            mouse.clear(viewer.perturb)
                finally:
                    viewer.close()
    return steps


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv, include_launcher=not any(value in ("-h", "--help") for value in argv))
    playback = prepare_playback(args)
    import torch

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError(
            "Playback uses CUDA MuJoCo-Warp physics; an NVIDIA GPU is required"
        )
    from isaaclab.app import launch_simulation
    from supermimic.env import TrackingSim

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    policy, motion, mdp = build_runtime(playback, args.device)
    args.goal_frames = playback.checkpoint["args"]["goal_frames"]
    args.history_length = playback.checkpoint["args"]["hist"]
    iteration = playback.checkpoint.get("iteration", "unknown")
    playback.checkpoint = {}  # Release unused world-model/optimizer tensors loaded on CPU.
    print(f"Checkpoint iteration {iteration}: {playback.checkpoint_path}", flush=True)
    print(
        "Physics: ManagerBasedRLEnv / CUDA MuJoCo-Warp; 1 environment; 50 Hz. Solid=policy, blue=reference.",
        flush=True,
    )
    trained_collisions = (
        "unknown (legacy checkpoint)"
        if playback.checkpoint_self_collisions is None
        else playback.checkpoint_self_collisions
    )
    print(
        f"Self-collisions: runtime={playback.env_cfg.self_collisions}; checkpoint={trained_collisions}",
        flush=True,
    )
    print(
        "Space pause | R restart | N/P next/previous | G ghost | F camera follow\n"
        "Double-click solid robot to select | Ctrl+right-drag pull | Ctrl+left-drag rotate",
        flush=True,
    )
    cfg = playback.env_cfg
    with launch_simulation(cfg.sim, args):
        env = TrackingSim(
            cfg.sim,
            1,
            motion,
            mdp,
            eval_mode=True,
            env_cfg=cfg,
            goal_frames=args.goal_frames,
        )
        try:
            steps = play_loop(env, policy, motion, mdp, args, cfg.robot_model.mjcf)
        finally:
            env.close()
    print(f"Finished {steps} policy control steps.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        pass
