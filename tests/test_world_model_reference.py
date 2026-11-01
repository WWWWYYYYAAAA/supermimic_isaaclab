"""Residual integration, learned termination, gradients and terminal replay labels."""
from types import SimpleNamespace
import unittest
import torch
from supermimic.buffer import ReplayBuffer
from supermimic.mdp import flatten_state
from supermimic.networks import WorldModel
from supermimic.rotations import so3_exp
from supermimic.training.trainer import Trainer
from tests.support import make_motion

class WorldModelReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        _,self.motion,self.mdp=make_motion()
        self.f=torch.tensor([2,7]); self.S=self.motion.ref_state(self.f)

    def test_zero_output_extrapolates_and_nonzero_residuals_roundtrip(self):
        model=WorldModel(hidden=(16,16))
        S=list(self.S)
        S[1]=so3_exp(torch.tensor([[.2,-.3,.7],[-.1,.4,-.5]]))
        S[2]=torch.randn(2,3); S[3]=torch.randn(2,3); S[5]=torch.randn(2,29)
        zero=model.step(S,torch.zeros(2,29),torch.zeros(2,33))
        torch.testing.assert_close(zero[0],S[0]+.02*S[2])
        torch.testing.assert_close(zero[4],S[4]+.02*S[5])
        delta=torch.randn(2,70)*.15
        out=model.integrate(S,delta)
        torch.testing.assert_close(model.targets(S,out),delta*model.out_scale,atol=2e-6,rtol=1e-4)
        zero[1][:,1,0].sum().backward()
        self.assertGreater(model.net[-1].bias.grad[38:41].abs().sum().item(),0)

    def test_physical_loss_masks_episode_boundaries(self):
        learner=Trainer.__new__(Trainer)
        errors=torch.full((2,2,70),.5,requires_grad=True)
        learner.wm=SimpleNamespace(error_scale=torch.ones(70),state_error=lambda pred,actual:errors)
        mask=torch.tensor([[1.,0.],[1.,1.]])
        target=self.motion.ref_state(torch.tensor([[2,3],[7,8]]))
        loss,last=learner._wm_err(target,flatten_state(target),mask)
        torch.testing.assert_close(loss,torch.tensor(70*.25*.75))
        loss.backward()
        self.assertEqual(errors.grad[0,1].abs().sum().item(),0)

    def test_neural_head_bce_trains_head_and_dynamics_and_actor_state_gradients(self):
        model=WorldModel(hidden=(16,16))
        out=model.step(self.S,torch.zeros(2,29),torch.zeros(2,33))
        logits=model.term_head(self.mdp.termination_features(out,self.f+1))
        torch.nn.functional.binary_cross_entropy_with_logits(logits,torch.tensor([0.,1.])).backward()
        self.assertGreater(model.term_head.net[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(model.net[-1].weight.grad.abs().sum().item(),0)
        model.requires_grad_(False)
        p=self.S[0].clone().requires_grad_()
        prob=model.termination_probability(self.mdp,(p,*self.S[1:]),self.f)
        prob.sum().backward()
        self.assertGreater(p.grad.abs().sum().item(),0)
        analytic=WorldModel(hidden=(16,),termination_head='analytic')
        self.assertIsNone(analytic.term_head)
        torch.testing.assert_close(analytic.termination_probability(self.mdp,self.S,self.f),self.mdp.soft_termination(self.S,self.f))

    def test_replay_separates_timeouts_and_failures_and_masks_resets(self):
        replay=ReplayBuffer(3,1,'cpu')
        for done,term in ((True,False),(True,True),(False,False)):
            row={k:torch.zeros_like(v[0]) for k,v in replay.data.items()}
            row['done'][:]=done; row['term'][:]=term
            replay.add(row)
        w=replay.sample_windows(1,3)
        torch.testing.assert_close(w['done'],torch.tensor([[True,True,False]]))
        torch.testing.assert_close(w['term'],torch.tensor([[False,True,False]]))
        torch.testing.assert_close(w['mask'],torch.tensor([[1.,0.,0.]]))
