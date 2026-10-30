"""ManagerBasedRLEnv with SuperMimic's replay and evaluation lifecycle.

The inherited step executes the real Isaac Lab managers. Hooks capture terminal
states before autoreset and preserve evaluation's no-autoreset behavior.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from supermimic.mdp import flatten_state
from .tracking_env_cfg import TrackingEnvCfg


class TrackingEnv(ManagerBasedRLEnv):
    def __init__(
        self,
        cfg: TrackingEnvCfg,
        motion,
        mdp,
        gamma=0.99,
        eval_mode=False,
        max_episode_steps=500,
        reset_noise=True,
    ):
        self.motion, self.mdp, self.gamma = motion, mdp, gamma
        self.eval_mode = eval_mode
        self.episode_steps, self.reset_noise = max_episode_steps, reset_noise
        self._reset_frames = None
        self._pending = None
        self._next_state = None
        self._inside_step = False
        super().__init__(cfg)
        self.render_enabled = False
        self.reset_reference(torch.arange(self.num_envs, device=self.device))

    @property
    def reference(self):
        return self.command_manager.get_term("motion")

    def transition_state(self):
        """One snapshot shared by termination, reward and replay before reset."""
        if self._next_state is None:
            self._next_state = self.reference.state()
        return self._next_state

    def step(self, action):
        ref = self.reference
        self._pending = {
            "s": flatten_state(ref.state()),
            "a": action.to(self.device).clone(),
            "a_prev": ref.a_prev.clone(),
            "f": ref.f.clone(),
            "ep_step": ref.ep_step.clone(),
            "xi": ref.xi.clone(),
        }
        self._next_state = None
        self._inside_step = True
        try:
            result = super().step(action)
        finally:
            self._inside_step = False
        return result

    def finish_transition(self):
        """Commit a transition exactly once, before any reference reset."""
        if self._pending is None:
            return
        out, self._pending = self._pending, None
        ref = self.reference
        reward = self.reward_buf.clone()
        term = self.reset_terminated.clone()
        done = (term | self.reset_time_outs).clone()
        out.update(
            s1=flatten_state(self.transition_state()),
            r=reward,
            term=term,
            done=done,
            terms=self._reward_terms,
        )
        ref.ep_disc_ret += (self.gamma ** out["ep_step"].float()) * reward
        ref.ep_ret += reward
        ref.f = out["f"] + 1
        ref.ep_step = out["ep_step"] + 1
        ref.a_prev = out["a"].clone()
        if self.eval_mode:
            ref.failed_eval |= term & ~ref.done_eval
            ref.done_eval |= done
        else:
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            if len(ids):
                duration = ref.ep_step[ids].float()
                bootstrap = torch.where(
                    term[ids],
                    0.0,
                    (self.gamma**duration)
                    * (ref.ep_ret[ids] / duration)
                    / (1 - self.gamma),
                )
                returns = ref.ep_disc_ret[ids] + bootstrap
                bins = ref.start_bin[ids]
                ref.bin_G.index_put_((bins,), 0.8 * ref.bin_G[bins] + 0.2 * returns)
                out["ep_len"] = duration
                out["ep_ret_norm"] = ref.ep_ret[ids] / (
                    duration * self.mdp.max_step_reward
                )
        self.extras["transition"] = out

    def _reset_idx(self, env_ids):
        if self._inside_step:
            self.finish_transition()
            if self.eval_mode:
                # Evaluator masks finished clips; they remain in the batch.
                return
        super()._reset_idx(env_ids)

    def reset_reference(self, env_ids, frames=None):
        """Explicit reset used by legacy tools and deterministic evaluation."""
        if not len(env_ids):
            return
        self._reset_frames = frames
        try:
            self._reset_idx(env_ids)
        finally:
            self._reset_frames = None
        self._next_state = None
