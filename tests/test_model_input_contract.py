"""New input semantics, frame conventions, default switches and legacy guards."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import torch
from supermimic.config import train_parser
from supermimic.mdp import invariant_features
from supermimic.motion import motion_goal_dim, checkpoint_goal_dim
from supermimic.networks import WorldModel, MODEL_INPUT_VERSION
from supermimic.training.trainer import Trainer
from supermimic.rotations import so3_exp
from supermimic.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from supermimic.tasks.tracking.settings import task_settings, apply_task_settings
from tests.support import make_motion, manager_env

class ModelInputContractTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.kin, self.motion, self.mdp = make_motion()
        self.f = torch.tensor([2,7])
        self.S = self.motion.ref_state(self.f)

    def test_defaults_and_configuration_roundtrip(self):
        args = train_parser(False).parse_args([])
        self.assertTrue(args.anchor_obs)
        self.assertEqual(args.termination_head, 'neural')
        off = train_parser(False).parse_args(['--no-anchor_obs','--termination_head','analytic'])
        self.assertFalse(off.anchor_obs)
        cfg = G1FlatEnvCfg()
        self.assertTrue(cfg.commands.motion.anchor_obs)
        saved = task_settings(cfg)
        self.assertEqual(task_settings(apply_task_settings(G1FlatEnvCfg(), saved)),saved)

    def test_critic_has_root_drift_and_world_model_is_heading_invariant(self):
        S, f, mdp = self.S, self.f, self.mdp
        g = mdp.goal(S,f,10)
        self.assertEqual(g.shape,(2,689))
        xi, a = torch.randn(2,33), torch.randn(2,29)
        critic = mdp.critic_features(S,g,xi,f)
        self.assertEqual(critic.shape,(2,797))
        torch.testing.assert_close(critic[:,:71], invariant_features(S))
        torch.testing.assert_close(critic[:,-4:],torch.zeros(2,4),atol=1e-6,rtol=0)
        # Rotate the entire physical state, and translate only world XY.
        H=so3_exp(torch.tensor([[0.,0.,.7],[0.,0.,-.4]]))
        mv=lambda x:(H@x.unsqueeze(-1)).squeeze(-1)
        shifted=(mv(S[0])+torch.tensor([1.,2.,0.]),H@S[1],mv(S[2]),mv(S[3]),S[4],S[5])
        wm=WorldModel(hidden=(16,))
        torch.testing.assert_close(wm.inputs(shifted,a,xi),wm.inputs(S,a,xi),atol=1e-6,rtol=1e-5)
        drift=mdp.critic_features(shifted,g,xi,f)[:,-4:]
        torch.testing.assert_close(drift[:,-1],torch.tensor([.7,-.4]))
        self.assertGreater(drift[:,:3].abs().sum().item(),1.)
        expanded=torch.randn(2,91)
        self.assertEqual(mdp.critic_features(S,g,expanded,f).shape,(2,855))
        self.assertEqual(WorldModel(hidden=(16,),xi_dim=91).inputs(S,a,expanded).shape,(2,191))

    def test_actor_bias_sign_and_term_major_history(self):
        xi=torch.zeros(2,33); xi[:,1:30]=.01
        obs=self.mdp.obs(self.S,torch.zeros(2,29),xi,self.f)
        torch.testing.assert_close(obs[:,:29],self.S[4]-self.mdp.default_q+.01)
        hist=torch.arange(2*3*93.).reshape(2,3,93)
        flat=self.mdp.flatten_history(hist)
        expected=torch.cat([hist[:,:,sl].flatten(1) for sl in
            (slice(0,29),slice(29,58),slice(58,61),slice(61,64),slice(64,93))],-1)
        torch.testing.assert_close(flat,expected)

    def test_imagined_anchor_tracks_predicted_torso_and_has_gradients(self):
        mdp=self.mdp; mdp.observation_cfg.noise_enabled=False
        learner=Trainer.__new__(Trainer)
        learner.motion,learner.mdp,learner.K,learner.anchor_obs=self.motion,mdp,2,True
        learner.pi=lambda obs,goal: torch.zeros(2,29)
        q=self.S[4].clone().requires_grad_()
        # Torso anchor must include waist-joint motion, not just pelvis orientation.
        predicted=(*self.S[:4],q,self.S[5])
        learner.wm=Mock(); learner.wm.step.return_value=predicted
        ctx=torch.zeros(2,33)
        g=mdp.goal(self.S,self.f,2)
        result=learner._imag_step(self.S,torch.zeros(2,2,93),torch.zeros(2,29),self.f,ctx,ctx,g)
        torch.testing.assert_close(result[-1],mdp.goal(predicted,self.f+1,2))
        result[-1][:,-9:].sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertGreater(q.grad.abs().sum().item(),0)
        learner.anchor_obs=False
        result=learner._imag_step(self.S,torch.zeros(2,2,93),torch.zeros(2,29),self.f,ctx,ctx,g)
        self.assertEqual(result[-1].shape,(2,136))
        self.assertFalse(result[-1].requires_grad)

    def test_command_updates_anchor_after_robot_rotation_and_reset(self):
        cfg=G1FlatEnvCfg(); cfg.commands.motion.goal_frames=2
        env=manager_env(self.motion,self.mdp,env_cfg=cfg,reset_noise=False,n=2)
        env.reset_reference(torch.arange(2),self.motion.clip_start)
        cmd=env.reference; before=cmd.command.clone()
        self.assertEqual(before.shape,(2,145))
        env.scene['robot'].data.root_link_pose_w.torch[:,3:7]=torch.tensor([0.,0.,.1,(1-.01)**.5])
        torch.testing.assert_close(cmd.command[:,:136],before[:,:136])
        self.assertFalse(torch.allclose(cmd.command[:,-9:],before[:,-9:]))
        env.reset_reference(torch.arange(2),self.motion.clip_start)
        torch.testing.assert_close(cmd.command,self.mdp.goal(cmd.state(),cmd.f,2))

    def test_version_guard_rejects_same_width_wrong_semantics(self):
        self.assertEqual(MODEL_INPUT_VERSION,7)
        self.assertEqual(motion_goal_dim(10),689)
        self.assertEqual(motion_goal_dim(10,False),680)
        def ck(v,w,anchor=True):
            return dict(args=dict(goal_frames=10,hist=10,anchor_obs=anchor),model_input_version=v,
                        pi={'enc.0.weight':torch.empty(1,w)})
        self.assertEqual(checkpoint_goal_dim(ck(7,689)),689)
        self.assertEqual(checkpoint_goal_dim(ck(7,680,False)),680)
        for v,w in ((3,730),(4,736),(5,790),(6,730)):
            self.assertEqual(checkpoint_goal_dim(ck(v,w)),w)
        for w in (680,730,736,790):
            with self.assertRaises(ValueError): checkpoint_goal_dim(ck(7,w))
