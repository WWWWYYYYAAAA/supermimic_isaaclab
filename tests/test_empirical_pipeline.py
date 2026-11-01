"""Empirical statistics, motion dataset semantics and exported v7 policies."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from supermimic.normalization import EmpiricalNormalization
from supermimic.networks import Policy, Critic, WorldModel
from supermimic.export_policy import export_all, load_export_inputs
from tests.support import make_motion

class EmpiricalPipelineTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_running_moments_weight_samples_and_forward_never_updates(self):
        norm=EmpiricalNormalization(3)
        a=torch.randn(7,3)*2+3; b=torch.randn(2,5,3)*.4-2
        norm.update(a); norm.update(b)
        all_x=torch.cat((a,b.flatten(0,1)))
        torch.testing.assert_close(norm.mean,all_x.mean(0))
        torch.testing.assert_close(norm.std,all_x.std(0,unbiased=False))
        before={k:v.clone() for k,v in norm.state_dict().items()}
        x=torch.randn(4,3,requires_grad=True)
        torch.testing.assert_close(norm(x),(x-all_x.mean(0))/(all_x.std(0,unbiased=False)+.01))
        norm(x).sum().backward(); self.assertTrue(torch.isfinite(x.grad).all())
        for k,v in before.items(): torch.testing.assert_close(norm.state_dict()[k],v)
        norm.eval(); norm.update(torch.randn(10,3))
        for k,v in before.items(): torch.testing.assert_close(norm.state_dict()[k],v)
        one=EmpiricalNormalization(3); one.update(torch.ones(1,3))
        self.assertTrue(torch.isfinite(one(torch.ones(1,3))).all())

    def test_all_current_network_inputs_use_empirical_normalization(self):
        models=(Policy(93,689),Critic(797),WorldModel(hidden=(16,)))
        for model in models:
            self.assertFalse(any(isinstance(m,torch.nn.BatchNorm1d) for m in model.modules()))
        for norm in (models[0].obs_norm,models[0].goal_norm,models[1].norm,
                     models[2].in_norm,models[2].term_head.norm):
            self.assertIsInstance(norm,EmpiricalNormalization)
            self.assertEqual(list(norm.parameters()),[])

    def test_dataset_goal_current_frame_default_pose_and_clip_end(self):
        _,m,mdp=make_motion()
        frames=torch.stack((m.clip_end[0]-1,m.clip_start[1]))
        g=m.goal(frames,3).reshape(2,3,68)
        for row,f in enumerate(frames.tolist()):
            for j in range(3):
                idx=min(f+j,int(m.clip_end[m.clip_id[f]]))
                torch.testing.assert_close(g[row,j,:29],m.q[idx]-mdp.default_q)
                torch.testing.assert_close(g[row,j,29:58],m.qd[idx])
                torch.testing.assert_close(g[row,j,58:61],-m.root_R[idx,2,:])
                torch.testing.assert_close(g[row,j,61:64],m.root_R[idx].T@m.root_w[idx])
                self.assertEqual(g[row,j,67].item(),m.root_p[idx,2].item())
        torch.testing.assert_close(g[0,1],g[0,2])
        self.assertGreater(g[0,2,29:58].abs().sum().item(),0)
        self.assertFalse(torch.equal(g[0,2,:29],m.q[m.clip_start[1]]-mdp.default_q))

    def test_new_torchscript_encoder_decoder_equal_training_policy_and_metadata(self):
        for anchor in (False,True):
            with self.subTest(anchor=anchor), tempfile.TemporaryDirectory() as tmp:
                d=136+9*anchor
                pi=Policy(93,d,hist=2)
                pi.obs_norm.update(torch.randn(17,186)*3+2)
                pi.goal_norm.update(torch.randn(13,d)*.5-.2)
                pi.eval()
                ck=dict(pi=pi.state_dict(),args=dict(hist=2,goal_frames=2,anchor_obs=anchor),model_input_version=7)
                source=Path(tmp)/'last.pt'; torch.save(ck,source)
                outputs=export_all(source,Path(tmp)/'policy.pt',task='g1',check_inputs=True)
                modules={k:torch.jit.load(str(v[0])) for k,v in outputs.items()}
                for batch in (1,3):
                    o,g=torch.randn(batch,186),torch.randn(batch,d)
                    with torch.no_grad():
                        expected=pi(o,g)
                        torch.testing.assert_close(modules['policy'](o,g),expected)
                        torch.testing.assert_close(modules['decoder'](o,modules['encoder'](g)),expected)
                meta=json.loads(outputs['policy'][1].read_text())
                self.assertEqual(meta['normalization_type'],'EmpiricalNormalization')
                self.assertEqual(meta['inputs']['motion_goal']['shape'],['batch',d])
                self.assertEqual(meta['inputs']['motion_goal']['future_frame_dim'],68)
                self.assertIn('term-major',meta['inputs']['obs_history']['flatten_order'])
                self.assertEqual(bool(meta['goal_layout_suffix']),anchor)
                bad=dict(ck,model_input_version=6); torch.save(bad,source)
                with self.assertRaises(ValueError): load_export_inputs(source,task='g1')
