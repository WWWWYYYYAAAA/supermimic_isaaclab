"""Paper input groups and explicit exclusion of reference-derived observations."""

import unittest
from unittest.mock import Mock

import torch

from supermimic.mdp import STATE_DIM, flatten_state
from supermimic.networks import WorldModel
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.tasks.tracking.settings import task_settings, apply_task_settings
from tests.support import make_motion


class ModelInputContractTest(unittest.TestCase):
    def test_g1_disables_reference_orientation_in_all_network_groups(self):
        cfg = G1FlatEnvCfg()
        for name in ("policy_observation", "critic_observation", "world_model_observation"):
            self.assertIsNone(getattr(cfg, name).reference_orientation)
        self.assertEqual(cfg.policy_observation.dim, 93)
        saved = task_settings(cfg)
        for name in ("observations", "critic_observations", "world_model_observations"):
            self.assertIsNone(saved[name]["reference_orientation"])
        restored = apply_task_settings(G1FlatEnvCfg(), saved)
        self.assertEqual(task_settings(restored), saved)
        cfg.critic_observation.reference_orientation = True
        with self.assertRaisesRegex(ValueError, "must be None"):
            cfg.prepare()

    def test_privileged_inputs_are_full_state_and_only_paper_conditioning(self):
        torch.set_num_threads(2)
        _, motion, mdp = make_motion()
        frames = torch.tensor([2, 7])
        state = motion.ref_state(frames)
        goal = motion.goal(frames, 10)
        context = torch.randn(2, 33)
        action = torch.randn(2, 29)
        # Construction of actor/critic inputs must not query reference data or FK.
        mdp.motion = Mock(spec=[])
        mdp.kin = Mock(spec=[])
        self.assertEqual(mdp.obs(state, action, context, frames).shape, (2, 93))
        critic = mdp.critic_features(state, goal, context)
        self.assertEqual(critic.shape, (2, 834))
        torch.testing.assert_close(critic[:, :STATE_DIM], flatten_state(state))
        torch.testing.assert_close(critic[:, STATE_DIM:STATE_DIM + 730], goal)
        torch.testing.assert_close(critic[:, -33:], context)
        model = WorldModel(hidden=(16, 16))
        world = model.inputs(state, action, context)
        self.assertEqual(world.shape, (2, 133))
        torch.testing.assert_close(world[:, :STATE_DIM], critic[:, :STATE_DIM])
        torch.testing.assert_close(world[:, STATE_DIM:STATE_DIM + 29], action)
        torch.testing.assert_close(world[:, -33:], context)
        # Global root XY is retained, rather than silently reduced to root height.
        shifted = (state[0] + torch.tensor([1., 2., 0.]), *state[1:])
        difference = model.inputs(shifted, action, context) - world
        torch.testing.assert_close(difference[:, :3], torch.tensor([[1., 2., 0.]]).expand(2, -1))
        torch.testing.assert_close(difference[:, 3:], torch.zeros_like(difference[:, 3:]))


if __name__ == "__main__":
    unittest.main()
