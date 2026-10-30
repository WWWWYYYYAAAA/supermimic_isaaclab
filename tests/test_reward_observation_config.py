"""Task reward and sensor edits must change both learning and simulator heads."""

from dataclasses import replace
import unittest
import torch

from supermimic.mdp import MDP, OBS_DIM, OBS_NOISE, normalize_xi
from supermimic.motion import CONTROL_DT
from supermimic.rotations import mat_to_6d
from supermimic.tasks.tracking.observation_cfg import ObservationCfg, SensorTermCfg
from supermimic.tasks.tracking.reward_cfg import RewardCfg
from tests.support import make_motion


class RewardObservationConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin, cls.motion, cls.baseline = make_motion()
        cls.frames = torch.tensor([2, 7])
        cls.state = cls.motion.ref_state(cls.frames)
        cls.previous = torch.linspace(-0.2, 0.2, 58).reshape(2, 29)
        cls.xi = torch.zeros(2, 33)

    def make_mdp(self, **kwargs):
        return MDP(self.kin, self.motion, self.baseline.default_q, **kwargs)

    def test_default_sensor_layout_and_noise_match_paper(self):
        p, rotation, v, omega, q, qd = self.state
        expected = torch.cat(
            (
                q - self.baseline.default_q,
                qd,
                (rotation.transpose(-1, -2) @ omega.unsqueeze(-1)).squeeze(-1),
                -rotation[..., 2, :],
                self.previous,
            ),
            -1,
        )
        actual = self.baseline.obs(self.state, self.previous, self.xi, self.frames)
        self.assertEqual(actual.shape[-1], OBS_DIM)
        self.assertEqual(OBS_DIM, 93)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.manual_seed(41)
        expected_noisy = expected + (2 * torch.rand_like(expected) - 1) * OBS_NOISE
        expected_rng = torch.get_rng_state()
        torch.manual_seed(41)
        torch.testing.assert_close(
            self.baseline.add_noise(actual), expected_noisy, rtol=0, atol=0
        )
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)

    def test_default_observation_does_not_read_reference_orientation(self):
        # An invalid reference index must not matter for proprioceptive obs.
        expected = self.baseline.obs(self.state, self.previous, self.xi, self.frames)
        actual = self.baseline.obs(self.state, self.previous, self.xi, self.frames + 1000000)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_orientation_can_be_restored_from_saved_settings(self):
        cfg = ObservationCfg()
        cfg.reference_orientation = SensorTermCfg(noise=0.05)
        legacy = self.make_mdp(observation_cfg=cfg)
        obs = legacy.obs(self.state, self.previous, self.xi, self.frames)
        self.assertEqual(obs.shape[-1], 99)
        torch.testing.assert_close(obs[:, :93], self.baseline.obs(
            self.state, self.previous, self.xi, self.frames
        ))
        torch.testing.assert_close(obs[:, 93:], mat_to_6d(
            self.state[1].transpose(-1, -2) @ self.motion.root_R[self.frames]
        ))

    def test_sensor_disabling_scaling_and_noise_are_one_layout(self):
        cfg = ObservationCfg()
        cfg.reference_orientation = None
        cfg.previous_action.enabled = False
        cfg.joint_vel.scale = 0.1
        cfg.joint_vel.noise = 0.25
        cfg.base_ang_vel.scale = 0.25
        model = self.make_mdp(observation_cfg=cfg)
        raw = self.baseline.obs(self.state, self.previous, self.xi, self.frames)
        expected = raw[:, :64].clone()
        expected[:, 29:58] *= 0.1
        expected[:, 58:61] *= 0.25
        actual = model.obs(self.state, self.previous, self.xi, self.frames)
        self.assertEqual(model.obs_dim, 64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            model.obs_noise[29:58], torch.full((29,), 0.025), rtol=0, atol=0
        )
        torch.manual_seed(7)
        noisy = model.add_noise(actual)
        torch.manual_seed(7)
        expected = actual + (2 * torch.rand_like(actual) - 1) * model.obs_noise
        torch.testing.assert_close(noisy, expected, rtol=0, atol=0)
        cfg.noise_enabled = False
        rng = torch.get_rng_state().clone()
        self.assertIs(model.add_noise(actual), actual)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_reward_weight_and_upper_bound_follow_config_with_gradients(self):
        cfg = RewardCfg(
            w_body_pos=0,
            w_body_ori=0,
            w_body_linvel=0,
            w_body_angvel=0,
            w_root_pos=2,
            w_root_ori=0,
            w_action_rate=0,
            w_joint_limit=0,
            root_pos_std=0.2,
        )
        model = self.make_mdp(cfg=cfg)
        state = list(self.state)
        state[0] = (state[0] + torch.tensor([0.1, 0, 0])).detach().requires_grad_(True)
        action = torch.zeros(2, 29)
        rate = model.reward_rate(state, self.frames, action, action)
        reward = model.reward(state, self.frames, action, action)
        expected = 2 * torch.exp(
            -((state[0] - self.motion.root_p[self.frames]) ** 2).sum(-1) / 0.2**2
        )
        torch.testing.assert_close(rate, expected, rtol=0, atol=0)
        torch.testing.assert_close(reward, rate * CONTROL_DT, rtol=0, atol=0)
        self.assertEqual(model.max_step_reward, 0.04)
        actual_grad = torch.autograd.grad(reward.sum(), state[0], retain_graph=True)[0]
        expected_grad = torch.autograd.grad((expected * CONTROL_DT).sum(), state[0])[0]
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_soft_termination_uses_configured_temperatures(self):
        cfg = RewardCfg(
            soft_term_height_temperature=0.1, soft_term_tilt_temperature=0.2
        )
        model = self.make_mdp(cfg=cfg)
        state = list(self.state)
        state[0] = state[0] + torch.tensor([0.0, 0.0, 0.1])
        expected = 1 - (
            (1 - torch.sigmoid(torch.tensor((0.1 - 0.25) / 0.1)))
            * (1 - torch.sigmoid(torch.tensor(-0.8 / 0.2)))
        )
        torch.testing.assert_close(
            model.soft_termination(state, self.frames), expected.expand(2)
        )

    def test_invalid_configuration_fails_before_training(self):
        for cfg in (
            RewardCfg(w_root_pos=-1),
            RewardCfg(w_action_rate=1),
            RewardCfg(body_pos_std=0),
            RewardCfg(soft_limit_factor=1.1),
            RewardCfg(soft_term_height_temperature=0),
        ):
            with self.assertRaises(ValueError):
                self.make_mdp(cfg=cfg)
        cfg = ObservationCfg()
        for _, _, term in cfg.terms():
            if term is not None:
                term.enabled = False
        with self.assertRaisesRegex(ValueError, "at least one observation"):
            self.make_mdp(observation_cfg=cfg)
        cfg = ObservationCfg()
        cfg.joint_pos.noise = -0.01
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.make_mdp(observation_cfg=cfg)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            self.make_mdp(observation_cfg=replace(ObservationCfg(), history=0))

    def test_pd_context_normalization_preserves_old_layout(self):
        base = torch.zeros(2, 33)
        base[:, 0] = 0.75
        torch.testing.assert_close(
            normalize_xi(base), torch.zeros_like(base), rtol=0, atol=0
        )
        extended = torch.cat(
            (base, torch.full((2, 29), 1.25), torch.full((2, 29), 0.75)), -1
        )
        normalized = normalize_xi(extended)
        torch.testing.assert_close(
            normalized[:, :33], normalize_xi(base), rtol=0, atol=0
        )
        torch.testing.assert_close(
            normalized[:, 33:62], torch.full((2, 29), 0.5), rtol=0, atol=0
        )
        torch.testing.assert_close(
            normalized[:, 62:], torch.full((2, 29), -0.5), rtol=0, atol=0
        )
        self.assertEqual(
            self.make_mdp(xi_dim=91).critic_extra, self.baseline.critic_extra + 58
        )


if __name__ == "__main__":
    unittest.main()
