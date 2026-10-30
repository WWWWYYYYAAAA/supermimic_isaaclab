"""Physical parameter writes must agree with the recorded world-model context."""

import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

from supermimic.robot_spec import G1_JOINT_NAMES, motor_of, _kp, _kd, action_scale
from supermimic.tasks.tracking.control_cfg import PDControlCfg, joint_pd_parameters
from supermimic.tasks.tracking.randomization_cfg import RandomizationCfg
from supermimic.tasks.tracking.mdp import events


class PhysicalWrites:
    def __init__(self, n):
        self.com = torch.arange(n * 3, dtype=torch.float32).reshape(n, 1, 3) * 0.01
        self.data = NS(body_com_pose_b=NS(torch=self.com.clone()))
        self.kp = torch.zeros(n, 29)
        self.kd = torch.zeros(n, 29)
        self.gain_writes = 0

    def set_coms_index(self, *, coms, body_ids, env_ids=None):
        rows = slice(None) if env_ids is None else env_ids
        self.com[rows] = coms
        self.data.body_com_pose_b.torch[rows] = coms

    def write_joint_stiffness_to_sim_index(self, *, stiffness, env_ids=None):
        self.kp[slice(None) if env_ids is None else env_ids] = stiffness
        self.gain_writes += 1

    def write_joint_damping_to_sim_index(self, *, damping, env_ids=None):
        self.kd[slice(None) if env_ids is None else env_ids] = damping
        self.gain_writes += 1


def fixture(cfg=None, *, evaluation=False):
    cfg = copy.deepcopy(cfg or RandomizationCfg())
    cfg.validate()
    gains = joint_pd_parameters(PDControlCfg())
    command = NS(
        randomization_cfg=cfg,
        N=3,
        device="cpu",
        eval_mode=evaluation,
        torso_id=0,
        robot=PhysicalWrites(3),
        xi=torch.zeros(3, cfg.context_dim),
        nominal_kp=torch.tensor([gains[j][0] for j in G1_JOINT_NAMES]),
        nominal_kd=torch.tensor([gains[j][1] for j in G1_JOINT_NAMES]),
    )
    command._mu_view = torch.zeros(3, 1)
    return NS(reference=command, device="cpu"), command


def apply_friction(command):
    command._mu_view[:] = command.xi[:, :1]


class RandomizationConfigTest(unittest.TestCase):
    def test_reset_noise_ranges_reach_reference_state_writes(self):
        from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
        from tests.support import make_motion, manager_env

        _, motion, mdp = make_motion()
        cfg = G1FlatEnvCfg()
        cfg.randomization.reset.root_position = (
            (0.01, 0.01),
            (-0.02, -0.02),
            (0.03, 0.03),
        )
        cfg.randomization.reset.root_rotation = ((0.0, 0.0),) * 3
        cfg.randomization.reset.root_linear_velocity = ((0.0, 0.0),) * 3
        cfg.randomization.reset.root_angular_velocity = ((0.0, 0.0),) * 3
        cfg.randomization.reset.joint_position = (0.0, 0.0)
        env = manager_env(motion, mdp, env_cfg=cfg)
        unchanged = tuple(value[0].clone() for value in env.reference.state())
        frame = torch.tensor([7])
        env.reset_reference(torch.tensor([1]), frame)
        state, reference = env.reference.state(), motion.ref_state(frame)
        torch.testing.assert_close(
            state[0][1],
            reference[0][0] + torch.tensor([0.01, -0.02, 0.03]),
            rtol=0,
            atol=0,
        )
        for actual, expected in zip(state[1:], reference[1:]):
            torch.testing.assert_close(actual[1], expected[0], rtol=0, atol=0)
        for actual, expected in zip(state, unchanged):
            torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)

    def test_nominal_pd_defaults_and_regex_overrides(self):
        gains = joint_pd_parameters(PDControlCfg())
        for name in G1_JOINT_NAMES:
            _, _, arm = motor_of(name)
            self.assertEqual(gains[name], (_kp(arm), _kd(arm), action_scale(name)))
        config = PDControlCfg(
            stiffness_scale=2,
            damping_scale=0.5,
            stiffness_overrides={".*_knee_joint": 100.0},
            damping_overrides={"left_knee_joint": 4.0},
        )
        changed = joint_pd_parameters(config)
        self.assertEqual(changed["left_knee_joint"], (100.0, 4.0, 0.25 * 139 / 100))
        self.assertEqual(changed["right_knee_joint"][0], 100.0)
        self.assertEqual(changed["waist_yaw_joint"][0], gains["waist_yaw_joint"][0] * 2)
        self.assertEqual(
            changed["waist_yaw_joint"][1], gains["waist_yaw_joint"][1] * 0.5
        )
        self.assertEqual(changed["waist_yaw_joint"][2], gains["waist_yaw_joint"][2] / 2)

    def test_default_startup_random_draws_match_baseline_order(self):
        env, command = fixture()
        torch.manual_seed(117)
        mu = torch.empty(3).uniform_(0.3, 1.2)
        com = torch.stack(
            (
                torch.empty(3).uniform_(-0.025, 0.025),
                torch.empty(3).uniform_(-0.05, 0.05),
                torch.empty(3).uniform_(-0.05, 0.05),
            ),
            -1,
        )
        expected_rng = torch.get_rng_state()
        nominal_com = command.robot.com[:, 0].clone()
        torch.manual_seed(117)
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
        torch.testing.assert_close(command.xi[:, 0], mu, rtol=0, atol=0)
        torch.testing.assert_close(command.xi[:, 30:33], com, rtol=0, atol=0)
        torch.testing.assert_close(
            command.robot.com[:, 0], nominal_com + com, rtol=0, atol=0
        )
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertEqual(command.xi.shape[-1], 33)
        self.assertEqual(command.robot.gain_writes, 0)

    def test_reset_pd_factors_match_partial_physical_gain_writes(self):
        cfg = RandomizationCfg()
        cfg.pd_gains.enabled = True
        cfg.pd_gains.kp_scale = (0.8, 0.8)
        cfg.pd_gains.kd_scale = (1.2, 1.2)
        env, command = fixture(cfg)
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
            events.reset_context(command, torch.tensor([1]))
        self.assertEqual(command.xi.shape[-1], 91)
        torch.testing.assert_close(
            command.xi[[0, 2], 33:], torch.ones(2, 58), rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.xi[1, 33:62], torch.full((29,), 0.8), rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.xi[1, 62:], torch.full((29,), 1.2), rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.robot.kp[1], command.nominal_kp * 0.8, rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.robot.kd[1], command.nominal_kd * 1.2, rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.robot.kp[0], command.nominal_kp, rtol=0, atol=0
        )

    def test_startup_pd_and_context_are_held_across_episode_resets(self):
        cfg = RandomizationCfg()
        cfg.joint_offsets.enabled = False
        cfg.pd_gains.enabled = True
        cfg.pd_gains.mode = "startup"
        env, command = fixture(cfg)
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
            before = command.xi.clone()
            rng = torch.get_rng_state()
            events.reset_context(command, torch.tensor([0, 2]))
        torch.testing.assert_close(command.xi, before, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(
            command.robot.kp, command.nominal_kp * before[:, 33:62], rtol=0, atol=0
        )

    def test_reset_friction_com_and_offsets_affect_only_selected_environments(self):
        cfg = RandomizationCfg()
        cfg.friction.mode, cfg.friction.range = "reset", (0.6, 0.6)
        cfg.torso_com.mode = "reset"
        cfg.torso_com.ranges = ((0.01, 0.01), (0.02, 0.02), (0.03, 0.03))
        cfg.joint_offsets.range = (0.005, 0.005)
        env, command = fixture(cfg)
        original_com = command.robot.com.clone()
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
            events.reset_context(command, torch.tensor([1]))
            once = command.robot.com.clone()
            events.reset_context(command, torch.tensor([1]))
        torch.testing.assert_close(command.xi[[0, 2], 0], torch.ones(2), rtol=0, atol=0)
        torch.testing.assert_close(command.xi[1, 0], torch.tensor(0.6), rtol=0, atol=0)
        torch.testing.assert_close(
            command._mu_view[:, 0], command.xi[:, 0], rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.xi[1, 1:30], torch.full((29,), 0.005), rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.robot.com[[0, 2]], original_com[[0, 2]], rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.robot.com[1, 0],
            original_com[1, 0] + torch.tensor([0.01, 0.02, 0.03]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(command.robot.com, once, rtol=0, atol=0)

    def test_disabled_and_evaluation_modes_do_not_draw_random_physics(self):
        cfg = RandomizationCfg()
        cfg.pd_gains.enabled = True
        env, command = fixture(cfg, evaluation=True)
        rng = torch.get_rng_state()
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
            events.reset_context(command, torch.tensor([0, 1]))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(command.xi[:, 0], torch.ones(3), rtol=0, atol=0)
        torch.testing.assert_close(
            command.xi[:, 1:33], torch.zeros(3, 32), rtol=0, atol=0
        )
        torch.testing.assert_close(
            command.xi[:, 33:], torch.ones(3, 58), rtol=0, atol=0
        )
        cfg = RandomizationCfg()
        cfg.friction.enabled = cfg.torso_com.enabled = cfg.joint_offsets.enabled = False
        cfg.friction.nominal = 0.7
        env, command = fixture(cfg)
        rng = torch.get_rng_state()
        with (
            patch.object(events, "_setup_friction"),
            patch.object(events, "_apply_friction", apply_friction),
        ):
            events.startup_context(env, None)
            events.reset_context(command, torch.tensor([0, 1]))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(
            command.xi[:, 0], torch.full((3,), 0.7), rtol=0, atol=0
        )

    def test_configuration_errors_fail_before_simulation(self):
        with self.assertRaisesRegex(ValueError, "matches no G1 joints"):
            joint_pd_parameters(PDControlCfg(stiffness_overrides={"missing_joint": 5}))
        with self.assertRaisesRegex(ValueError, "positive"):
            joint_pd_parameters(PDControlCfg(stiffness_scale=0))
        cfg = RandomizationCfg()
        cfg.pd_gains.kp_scale = (-1.0, 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            cfg.validate()
        cfg = RandomizationCfg()
        cfg.friction.mode = "interval"
        with self.assertRaisesRegex(ValueError, "startup or reset"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
