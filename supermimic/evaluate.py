"""Compatible evaluation entry point; importing it does not launch a simulator."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.config import eval_parser


def main(argv=None, *, env_cfg=None):
    argv = sys.argv[1:] if argv is None else argv
    if any(value in ("-h", "--help") for value in argv):
        eval_parser(include_launcher=False).parse_args(argv)

    from isaaclab.app import launch_simulation
    from supermimic.env import TrackingSim
    from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from supermimic.evaluation import Evaluator

    env_cfg = G1FlatEnvCfg() if env_cfg is None else env_cfg
    parser = eval_parser()
    parser.set_defaults(device=env_cfg.sim.device, motion_dir=env_cfg.motion_dir)
    args = parser.parse_args(argv)
    from supermimic.tasks.tracking.settings import (
        load_task_settings,
        apply_task_settings,
    )

    saved = load_task_settings(
        Path(args.ckpt).parent if args.ckpt else args.run_dir, args.ckpt
    )
    if saved is not None:
        apply_task_settings(env_cfg, saved)
    args.device = args.device or env_cfg.sim.device
    env_cfg.sim.device = args.device
    env_cfg.prepare()
    evaluator = Evaluator(args.motion_dir, device=args.device, env_cfg=env_cfg)
    sim_cfg = env_cfg.sim
    with launch_simulation(sim_cfg, args):
        env = TrackingSim(
            sim_cfg,
            evaluator.C,
            evaluator.motion,
            evaluator.mdp,
            eval_mode=True,
            env_cfg=env_cfg,
            goal_frames=env_cfg.commands.motion.goal_frames,
        )
        try:
            evaluator.run(env, args)
        finally:
            env.close()


if __name__ == "__main__":
    main()
