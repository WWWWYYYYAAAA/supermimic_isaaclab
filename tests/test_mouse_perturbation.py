"""Exercise native MuJoCo mouse wrenches without a display or CUDA context."""

import contextlib
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch

from supermimic import play
from supermimic.env import TrackingSim
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from tests.support import make_motion, manager_env


class MousePerturbationTest(unittest.TestCase):
    def test_native_drag_maps_world_wrench_and_clears_on_release_or_ghost(self):
        model = mujoco.MjModel.from_xml_string("""
        <mujoco><option gravity="0 0 0"/><worldbody>
          <body name="pelvis" pos="0 0 1"><freejoint/>
            <geom type="sphere" size=".1" mass="2"/>
            <body name="hand" pos=".3 0 0"><geom type="sphere" size=".05" mass=".1"/></body>
          </body>
          <body name="ref_pelvis" pos="2 0 1"><freejoint/>
            <geom type="sphere" size=".1" mass="2"/>
          </body>
        </worldbody></mujoco>""")
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        composer = Mock()
        # The fixed hand is merged into the live pelvis; order includes another body.
        robot = NS(body_names=["unused", "pelvis"], device="cpu",
                   permanent_wrench_composer=composer)
        origin = torch.tensor([4., 5., 0.])
        bridge = play.MousePerturbation(model, data, robot, origin)
        pert = mujoco.MjvPerturb()
        pert.select = model.body("hand").id
        pert.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
        pert.localmass = 1
        pert.refselpos[:] = data.xpos[pert.select] + [0., .1, 0.]
        bridge.apply(pert)
        applied = composer.set_forces_and_torques_index.call_args.kwargs
        self.assertEqual(applied["body_ids"], [1])
        self.assertTrue(applied["is_global"])
        self.assertGreater(applied["forces"][0, 0, 1], 0)
        np.testing.assert_allclose(applied["forces"].numpy()[0, 0], data.xfrc_applied[pert.select, :3])
        np.testing.assert_allclose(applied["positions"].numpy()[0, 0], data.xipos[pert.select] + origin.numpy())
        # The actual MuJoCo-generated wrench moves a dynamic body.
        mujoco.mj_step(model, data)
        self.assertGreater(data.qvel[1], 0)
        pert.active = 0
        bridge.apply(pert)
        composer.reset.assert_called_once()
        self.assertFalse(np.any(data.xfrc_applied))
        pert.active = mujoco.mjtPertBit.mjPERT_ROTATE
        pert.refquat[:] = [np.cos(.1), 0., 0., np.sin(.1)]
        bridge.apply(pert)
        self.assertGreater(composer.set_forces_and_torques_index.call_args.kwargs["torques"].norm(), 0)
        pert.select = model.body("ref_pelvis").id
        bridge.apply(pert)
        self.assertEqual(composer.reset.call_count, 2)
        self.assertEqual(composer.set_forces_and_torques_index.call_count, 2)
        bridge.clear(pert)
        self.assertEqual(pert.active | pert.active2, 0)

    def test_play_loop_forwards_drag_before_step_and_clears_on_pause_reset_exit(self):
        torch.set_num_threads(2)
        _, motion, mdp = make_motion()
        adapter = TrackingSim.__new__(TrackingSim)
        adapter.env = manager_env(motion, mdp, eval_mode=True, n=1)
        robot = adapter.scene["robot"]
        robot.device, robot.body_names = "cpu", ["pelvis"]
        current = []
        composer = Mock()
        composer.set_forces_and_torques_index.side_effect = lambda **kw: current.append(kw)
        composer.reset.side_effect = current.clear
        robot.permanent_wrench_composer = composer
        steps_with_force = []
        original_step = adapter.step

        def step(action):
            steps_with_force.append(bool(current))
            return original_step(action)

        adapter.step = step
        args = play.parse_args(["--steps", "3"], False)
        args.history_length = args.goal_frames = 2
        viewer = NS(lock=contextlib.nullcontext, cam=NS(lookat=np.zeros(3)),
                    perturb=mujoco.MjvPerturb(), is_running=lambda: True, close=Mock())

        def factory(model, data, key_callback):
            count = 0

            def sync():
                nonlocal count
                count += 1
                if count in (1, 4):
                    viewer.perturb.select = model.body("pelvis").id
                    viewer.perturb.active = mujoco.mjtPertBit.mjPERT_TRANSLATE
                    viewer.perturb.localmass = 1
                    viewer.perturb.refselpos[:] = data.xpos[viewer.perturb.select] + [.1, 0., 0.]
                elif count == 2:
                    key_callback(32)  # Pause after first step.
                elif count == 3:
                    self.assertFalse(current)
                    key_callback(ord("R"))
                    key_callback(32)  # Restart and resume.

            viewer.sync = sync
            return viewer

        with patch.object(play.time, "sleep"):
            play.play_loop(adapter, lambda h, g: torch.zeros(1, 29), motion, mdp,
                           args, G1FlatEnvCfg().robot_model.mjcf, viewer_factory=factory)
        self.assertEqual(steps_with_force, [True, False, True])
        self.assertFalse(current)
        self.assertEqual(viewer.perturb.active | viewer.perturb.active2, 0)
        viewer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
