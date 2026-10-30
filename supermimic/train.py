"""Compatible training entry point: python supermimic/train.py [options]."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.config import train_parser


def main(argv=None, *, env_cfg=None):
    argv = sys.argv[1:] if argv is None else argv
    # Help and unsupported resume requests do not need simulator dependencies.
    preflight = train_parser(include_launcher=False)
    options, _ = preflight.parse_known_args(argv)
    if options.resume:
        preflight.error(
            "--resume is unavailable because replay-buffer restoration is not implemented; start a fresh run"
        )

    from isaaclab.app import launch_simulation
    from supermimic.env import TrackingSim
    from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from supermimic.training.trainer import Trainer

    env_cfg = G1FlatEnvCfg() if env_cfg is None else env_cfg
    parser = train_parser()
    parser.set_defaults(
        device=env_cfg.sim.device,
        motion_dir=env_cfg.motion_dir,
        log_dir=env_cfg.log_dir,
        num_envs=env_cfg.scene.num_envs,
        goal_frames=env_cfg.commands.motion.goal_frames,
        hist=env_cfg.policy_observation.history,
    )
    args = parser.parse_args(argv)
    args.device = args.device or env_cfg.sim.device
    env_cfg.sim.device = args.device
    env_cfg.policy_observation.history = args.hist
    env_cfg.commands.motion.goal_frames = args.goal_frames
    env_cfg.prepare()
    from supermimic.training.run_directory import create_run_directory

    args.log_dir = str(create_run_directory(args.log_dir, args.timestamp))
    print(f"Training output: {args.log_dir}", flush=True)
    trainer = Trainer(args, device=args.device, env_cfg=env_cfg)
    sim_cfg = env_cfg.sim
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(
            sim_cfg,
            args.num_envs,
            trainer.motion,
            trainer.mdp,
            gamma=args.gamma,
            goal_frames=args.goal_frames,
            env_cfg=env_cfg,
            max_episode_steps=round(
                env_cfg.episode_length_s / (sim_cfg.dt * env_cfg.decimation)
            ),
        )
        try:
            trainer.run(env)
        finally:
            env.close()


if __name__ == "__main__":
    main()
