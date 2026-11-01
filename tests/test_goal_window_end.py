"""Terminate on arrival at a full goal ending at the last reference frame."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.training.trainer import Trainer
from tests.support import make_motion, manager_env


class GoalWindowEndTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        _, self.motion, self.mdp = make_motion()

    def test_arriving_at_window_boundary_is_timeout_not_tracking_failure(self):
        for K in (1, 10):
            with self.subTest(K=K):
                cfg = G1FlatEnvCfg()
                cfg.commands.motion.goal_frames = K
                env = manager_env(self.motion, self.mdp, env_cfg=cfg, eval_mode=True, reset_noise=False)
                end = self.motion.tracking_end(K)
                env.reset_reference(torch.arange(2), end - torch.tensor([2, 1]))
                _, _, term, timeout, extras = env.step(torch.zeros(2,29))
                torch.testing.assert_close(term, torch.tensor([False, False]))
                torch.testing.assert_close(timeout, torch.tensor([False, True]))
                out = extras['transition']
                self.assertEqual(int(out['f'][1]+1+K-1), int(self.motion.clip_end[1]))
                self.assertFalse(out['term'].any())
                self.assertTrue(out['done'][1])
                # Match evaluator handling of finished batch members.
                env.reference.f = torch.minimum(env.reference.f, end-1)
                # The first environment becomes terminal exactly one step later.
                _, _, _, timeout, _ = env.step(torch.zeros(2,29))
                self.assertTrue(timeout[0])

    def test_random_resets_exclude_terminal_frames_and_empty_tail_bins(self):
        cfg = G1FlatEnvCfg(); cfg.commands.motion.goal_frames = 64
        env = manager_env(self.motion, self.mdp, env_cfg=cfg, reset_noise=False)
        f, bins = env.reference.sample_frames(4096)
        end = self.motion.tracking_end(64)
        self.assertTrue((f < end[self.motion.clip_id[f]]).all())
        self.assertTrue((f >= self.motion.clip_start[self.motion.clip_id[f]]).all())
        self.assertTrue((self.motion.bin_start[bins] < end[self.motion.bin_clip[bins]]).all())
        invalid = self.motion.bin_start >= end[self.motion.bin_clip]
        self.assertTrue(invalid.any())  # fixture really contains excluded bins
        self.assertFalse(invalid[bins].any())

    def test_imagination_requires_full_goal_at_bootstrap_state(self):
        t = Trainer.__new__(Trainer)
        t.motion, t.K, t.HZ, t.Hh = self.motion, 10, 4, 2
        limit = int(self.motion.tracking_end(t.K)[0])-t.HZ
        candidate = torch.tensor([limit+1, limit, limit-1, limit+2])
        t.buffer = SimpleNamespace(sample_starts=Mock(return_value=(
            {'f':candidate}, {'f':candidate[:,None].expand(-1,2)})))
        s, _ = t.sample_valid_starts(2)
        torch.testing.assert_close(s['f'],torch.tensor([limit,limit-1]))
        self.assertTrue((s['f']+t.HZ+t.K-1 <= self.motion.clip_end[0]).all())

    def test_short_clip_is_rejected_before_sampling(self):
        for K in (0, int(self.motion.clip_len.min())):
            with self.assertRaises(ValueError): self.motion.validate_goal_window(K)
