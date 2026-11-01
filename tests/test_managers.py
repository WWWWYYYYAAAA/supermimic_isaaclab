import unittest
import torch
from supermimic.mdp import flatten_state
from tests.support import make_motion, manager_env


class ManagerLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kin, cls.motion, cls.mdp = make_motion()

    def test_native_environment_config_preserves_physics_timing(self):
        from supermimic.tasks.config import TrackingEnvCfg
        from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg

        cfg = TrackingEnvCfg()
        self.assertIsInstance(cfg.sim.physics, NewtonCfg)
        self.assertIsInstance(cfg.sim.physics.solver_cfg, MJWarpSolverCfg)
        self.assertEqual(cfg.sim.dt * cfg.decimation, 0.02)
        self.assertEqual(cfg.sim.physics.num_substeps, 4)
        self.assertEqual(cfg.sim.physics.solver_cfg.integrator, "implicitfast")
        self.assertEqual(cfg.scene.num_envs, 4096)

    def test_step_reward_and_action_match_analytic_heads(self):
        env = manager_env(self.motion, self.mdp, reset_noise=False)
        ref = env.reference
        env.reset_reference(torch.arange(2), torch.tensor([2, 8]))
        before = flatten_state(ref.state())
        rng = torch.get_rng_state().clone()
        action = torch.full((2, 29), 0.1)
        expected_target = ref.default_q + ref.xi[:, 1:30] + ref.alpha * action
        obs, reward, terminated, timeout, extras = env.step(action)
        out = extras["transition"]
        torch.testing.assert_close(
            env.scene["robot"].target, expected_target, rtol=0, atol=0
        )
        torch.testing.assert_close(out["s"], before, rtol=0, atol=0)
        torch.testing.assert_close(
            reward,
            self.mdp.reward(ref.state(), ref.f, action, out["a_prev"]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            obs["policy"], self.mdp.obs(ref.state(), action, ref.xi, ref.f)
        )
        torch.testing.assert_close(ref.f, torch.tensor([3, 9]))
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        self.assertFalse(bool((terminated | timeout).any()))
        self.assertEqual(env._sim_step_counter, 1)
        self.assertEqual(env.action_manager.active_terms, ["joints"])
        self.assertEqual(env.command_manager.active_terms, ["motion"])

    def test_timeout_keeps_terminal_state_and_bootstraps_return(self):
        env = manager_env(self.motion, self.mdp, reset_noise=False, max_episode_steps=1)
        env.reset_reference(torch.arange(2), torch.tensor([3, 70]))
        old_bins = env.reference.start_bin.clone()
        _, _, term, timeout, extras = env.step(torch.ones(2, 29) * 0.2)
        out = extras["transition"]
        self.assertTrue(bool(timeout.all()))
        self.assertFalse(bool(term.any()))
        self.assertTrue(bool((env.reference.ep_step == 0).all()))
        self.assertTrue(bool((out["ep_len"] == 1).all()))
        self.assertFalse(torch.equal(out["s1"], flatten_state(env.reference.state())))
        expected = 0.2 * out["r"] / (1 - env.gamma)
        torch.testing.assert_close(
            env.reference.bin_G[old_bins], expected, rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            out["r"],
            self.mdp.reward(
                __import__(
                    "supermimic.mdp", fromlist=["unflatten_state"]
                ).unflatten_state(out["s1"]),
                out["f"] + 1,
                out["a"],
                out["a_prev"],
            ),
        )

    def test_failure_and_clip_end_stay_distinct_in_evaluation(self):
        env = manager_env(self.motion, self.mdp, eval_mode=True)
        frames = torch.tensor([3, int(self.motion.tracking_end(env.reference.cfg.goal_frames)[1]) - 1])
        env.reset_reference(torch.arange(2), frames)
        env.scene["robot"].force_failure = True
        _, _, term, timeout, extras = env.step(torch.zeros(2, 29))
        out = extras["transition"]
        torch.testing.assert_close(term, torch.tensor([True, False]))
        torch.testing.assert_close(timeout, torch.tensor([False, True]))
        self.assertTrue(bool(env.reference.done_eval.all()))
        torch.testing.assert_close(env.reference.failed_eval, term)
        torch.testing.assert_close(env.reference.f, frames + 1)
        torch.testing.assert_close(out["s1"], flatten_state(env.reference.state()))
        self.assertNotIn("ep_len", out)
        self.assertTrue(bool((env.reference.xi[:, 0] == 1).all()))
        self.assertEqual(torch.count_nonzero(env.reference.xi[:, 1:]).item(), 0)


if __name__ == "__main__":
    unittest.main()
