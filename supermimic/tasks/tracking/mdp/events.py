"""Configured physical randomization, recorded in the world-model context."""

import torch
import warp as wp
from newton import ModelFlags


def _sample_vector(n, ranges, device):
    # Retain separate XYZ draws and their order for default RNG compatibility.
    return torch.stack(
        tuple(torch.empty(n, device=device).uniform_(*bounds) for bounds in ranges), -1
    )


def _set_com(command, env_ids=None):
    rows = slice(None) if env_ids is None else env_ids
    value = (command._default_com[rows] + command.xi[rows, 30:33]).unsqueeze(1)
    kwargs = {} if env_ids is None else {"env_ids": env_ids}
    command.robot.set_coms_index(coms=value, body_ids=[command.torso_id], **kwargs)


def _set_pd(command, env_ids=None):
    """Implicit gains are articulation-owned in the pinned Isaac Lab backend."""
    rows = slice(None) if env_ids is None else env_ids
    command.robot.write_joint_stiffness_to_sim_index(
        stiffness=command.nominal_kp * command.xi[rows, 33:62], env_ids=env_ids
    )
    command.robot.write_joint_damping_to_sim_index(
        damping=command.nominal_kd * command.xi[rows, 62:91], env_ids=env_ids
    )


def startup_context(env, env_ids):
    command = env.reference
    cfg = command.randomization_cfg
    dev = env.device
    _setup_friction(command)
    command._default_com = command.robot.data.body_com_pose_b.torch[
        :, command.torso_id, :3
    ].clone()
    command.xi[:, 0] = cfg.friction.nominal
    command.xi[:, 1:30] = cfg.joint_offsets.nominal
    command.xi[:, 30:33] = torch.tensor(cfg.torso_com.nominal, device=dev)
    if not command.eval_mode:
        if cfg.friction.enabled and cfg.friction.mode == "startup":
            command.xi[:, 0] = torch.empty(command.N, device=dev).uniform_(
                *cfg.friction.range
            )
        if cfg.torso_com.enabled and cfg.torso_com.mode == "startup":
            command.xi[:, 30:33] = _sample_vector(command.N, cfg.torso_com.ranges, dev)
            _set_com(command)
        elif any(cfg.torso_com.nominal):
            _set_com(command)
        if cfg.joint_offsets.enabled and cfg.joint_offsets.mode == "startup":
            command.xi[:, 1:30] = torch.empty(command.N, 29, device=dev).uniform_(
                *cfg.joint_offsets.range
            )
    elif any(cfg.torso_com.nominal):
        _set_com(command)
    _apply_friction(command)
    if cfg.pd_gains.enabled:
        command.xi[:, 33:] = 1.0
        if not command.eval_mode and cfg.pd_gains.mode == "startup":
            command.xi[:, 33:62] = torch.empty(command.N, 29, device=dev).uniform_(
                *cfg.pd_gains.kp_scale
            )
            command.xi[:, 62:91] = torch.empty(command.N, 29, device=dev).uniform_(
                *cfg.pd_gains.kd_scale
            )
        _set_pd(command)


def reset_context(command, env_ids):
    """Called after reference pose noise, at the original joint-offset draw site."""
    cfg = command.randomization_cfg
    if command.eval_mode:
        return
    n, dev = len(env_ids), command.device
    if cfg.joint_offsets.enabled and cfg.joint_offsets.mode == "reset":
        command.xi[env_ids, 1:30] = torch.empty(n, 29, device=dev).uniform_(
            *cfg.joint_offsets.range
        )
    if cfg.friction.enabled and cfg.friction.mode == "reset":
        command.xi[env_ids, 0] = torch.empty(n, device=dev).uniform_(
            *cfg.friction.range
        )
        _apply_friction(command)
    if cfg.torso_com.enabled and cfg.torso_com.mode == "reset":
        command.xi[env_ids, 30:33] = _sample_vector(n, cfg.torso_com.ranges, dev)
        _set_com(command, env_ids)
    if cfg.pd_gains.enabled and cfg.pd_gains.mode == "reset":
        command.xi[env_ids, 33:62] = torch.empty(n, 29, device=dev).uniform_(
            *cfg.pd_gains.kp_scale
        )
        command.xi[env_ids, 62:91] = torch.empty(n, 29, device=dev).uniform_(
            *cfg.pd_gains.kd_scale
        )
        _set_pd(command, env_ids)


def _setup_friction(self):
    mgr = self.sim.backend_manager
    model = mgr.get_model()
    self._mu_view = wp.to_torch(
        self.robot._root_view.get_attribute("shape_material_mu", model)[:, 0]
    )
    shape_body = wp.to_torch(model.shape_body)
    mu_all = wp.to_torch(model.shape_material_mu)
    cfg = self.randomization_cfg.friction
    # MuJoCo combines the larger of two material coefficients. Ground must not
    # raise the selected robot coefficient, including the pre-reset nominal one.
    ground = cfg.nominal
    if cfg.enabled and not self.eval_mode:
        ground = min(ground, cfg.range[0])
    mu_all[shape_body < 0] = ground


def _apply_friction(self):
    self._mu_view[:] = self.xi[:, :1]
    self.sim.backend_manager.add_model_change(ModelFlags.SHAPE_PROPERTIES)


def reset_reference(env, env_ids):
    env.reference.reset_reference(env_ids, env._reset_frames)
