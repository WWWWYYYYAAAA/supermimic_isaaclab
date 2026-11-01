"""Pose-only anchor FK must preserve full-FK values and policy gradients."""
import unittest
from unittest.mock import patch
import torch
from supermimic.rotations import so3_exp, mat_to_6d
from tests.support import make_motion


class AnchorKinematicsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(123)
        self.kin, self.motion, self.mdp = make_motion()

    def random_state(self):
        S = list(self.motion.ref_state(torch.tensor([2, 7, 11])))
        S[0] = (S[0]+torch.randn(3,3)*.2).requires_grad_()
        S[1] = so3_exp(torch.randn(3,3)*.4).requires_grad_()
        S[4] = (S[4]+torch.randn(3,29)*.3).requires_grad_()
        return tuple(S)

    def test_pose_matches_every_tracked_body_for_nonzero_joints(self):
        S = self.random_state()
        bp, bR, _, _ = self.kin.forward(*S)
        for j, body in enumerate(self.kin.tracked_idx):
            p, R = self.kin.body_pose(S[0], S[1], S[4], body)
            torch.testing.assert_close(p, bp[:,j], rtol=0, atol=0)
            torch.testing.assert_close(R, bR[:,j], rtol=0, atol=0)
        self.assertEqual(len(self.kin.body_chains[self.mdp.anchor_body_index]), 3)

    def test_anchor_values_and_derivatives_match_full_fk_without_calling_it(self):
        S = self.random_state()
        f = torch.tensor([2,7,11])
        bp,bR,_,_ = self.kin.forward(*S)
        i = self.mdp.anchor_index
        Rt = bR[:,i].transpose(-1,-2)
        old = torch.cat(((Rt@(self.motion.body_p[f,i]-bp[:,i]).unsqueeze(-1)).squeeze(-1),
                         mat_to_6d(Rt@self.motion.body_R[f,i])), -1)
        weights = torch.randn_like(old)
        inputs = (S[0], S[1], S[4])
        old_grad = torch.autograd.grad((old*weights).sum(), inputs)
        with patch.object(self.kin, 'forward', side_effect=AssertionError('full FK must not be used for anchor')):
            new = self.mdp.anchor_observation(S,f)
        new_grad = torch.autograd.grad((new*weights).sum(), inputs)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        for actual, expected in zip(new_grad,old_grad):
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)

    def test_compiled_pose_matches_eager_values_and_gradients(self):
        S = self.random_state()
        body = self.mdp.anchor_body_index
        def pose(p,R,q): return self.kin.body_pose(p,R,q,body)
        compiled = torch.compile(pose,backend='aot_eager',fullgraph=True)
        eager = pose(S[0],S[1],S[4])
        actual = compiled(S[0],S[1],S[4])
        torch.testing.assert_close(actual,eager)
        weights = (torch.randn_like(eager[0]),torch.randn_like(eager[1]))
        grad = lambda result: torch.autograd.grad(sum((a*w).sum() for a,w in zip(result,weights)),(S[0],S[1],S[4]))
        torch.testing.assert_close(grad(actual),grad(eager))
