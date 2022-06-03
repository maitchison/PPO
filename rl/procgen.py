"""
Helper to create wrapped mujoco environments
"""

import gym
import numpy as np

from . import wrappers
from . import config

def make(env_id:str, monitor_video=False, seed=None, args=None, determanistic_saving=True):
    """
    Construct environment of given name, including any required wrappers.
    @determanistic_saving: When true RND is saved with the environment, so restoring will always produce the same
        results. When false RNG is not persisted through saving, which can be helpful when generating return samples.
    """

    args = args or config.args

    env_name = f"procgen:procgen-{env_id}-v0"

    env = gym.make(env_name)

    env = env.unwrapped

    env = wrappers.LabelEnvWrapper(env, env_id)

    if seed is not None:
        np.random.seed(seed)
        env.seed(seed)

    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    env = F32Wrapper(env)

    env = wrappers.SaveEnvStateWrapper(env, determanistic=determanistic_saving)

    env = wrappers.EpisodeScoreWrapper(env)

    env = wrappers.MonitorWrapper(env, monitor_video=False)

    return env
