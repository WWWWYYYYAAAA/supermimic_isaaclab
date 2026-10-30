"""Manager terms sharing analytic heads with the differentiable world model."""


def proprioception(env):
    ref = env.reference
    # Collector/imagination own noise/history. Do not consume extra random draws.
    return env.mdp.obs(ref.state(), ref.a_prev, ref.xi, ref.f)


def reward_rate(env):
    ref = env.reference
    rate, diagnostics = env.mdp.reward_rate(
        env.transition_state(),
        ref.f + 1,
        env.action_manager.action,
        ref.a_prev,
        terms=True,
    )
    env._reward_terms = diagnostics
    return rate


def tracking_failure(env):
    return env.mdp.termination(env.transition_state(), env.reference.f + 1)


def clip_or_time_limit(env):
    ref = env.reference
    clip_end = env.motion.clip_end[env.motion.clip_id[ref.f]]
    return (ref.f + 1 >= clip_end) | (ref.ep_step + 1 >= ref.max_episode_steps)
