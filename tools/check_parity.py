"""Compare v7 CPU contracts between the two maintained SuperMimic checkouts.

Usage: python tools/check_parity.py --baseline ../supermimic_isaaclab
Historical pre-v7 behavior is covered by legacy playback/export tests instead.
"""
from pathlib import Path
import argparse
import subprocess
import sys
import tempfile
import torch

PROBE = r'''
import sys
import torch
sys.path.insert(0, sys.argv[1])
from tests.support import make_motion
from supermimic.networks import WorldModel, Policy, MODEL_INPUT_VERSION
from supermimic.mdp import normalize_xi
assert MODEL_INPUT_VERSION == 7, 'Parity baseline must use the v7 contract'
torch.set_num_threads(2)
torch.manual_seed(42)
_, motion, mdp = make_motion()
f = torch.tensor([2,7]); S = motion.ref_state(f)
a = torch.randn(2,29); xi = torch.randn(2,33)*.01
obs = mdp.obs(S,a,xi,f)
hist = mdp.flatten_history(obs[:,None].expand(-1,2,-1))
goal = mdp.goal(S,f,2)
wm = WorldModel(hidden=(16,16))
pi = Policy(93,145,hist=2)
pi.obs_norm.update(hist); pi.goal_norm.update(goal)
wm.in_norm.update(wm.inputs(S,a,normalize_xi(xi)))
predicted = wm.step(S,a,normalize_xi(xi))
result = dict(obs=obs, history=hist, goal=goal,
              critic=mdp.critic_features(S,goal,normalize_xi(xi),f),
              dynamics=wm.inputs(S,a,normalize_xi(xi)),
              prediction=predicted, action=pi(hist,goal),
              termination=wm.termination_probability(mdp,predicted,f+1))
torch.save(result,sys.argv[2])
'''

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    args=parser.parse_args()
    current=Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        outputs=[]
        for index,root in enumerate((current,args.baseline.resolve())):
            path=Path(tmp)/f'{index}.pt'
            subprocess.run([sys.executable,'-c',PROBE,str(root),str(path)],cwd=tmp,check=True)
            outputs.append(torch.load(path,weights_only=True))
        for name,actual in outputs[0].items():
            torch.testing.assert_close(actual,outputs[1][name],rtol=0,atol=0)
            print(f'PASS: {name} (exact CPU equality)')

if __name__=='__main__': main()
