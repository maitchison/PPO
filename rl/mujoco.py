"""
Helper to create wrapped mujoco environments
"""

import gym
import numpy as np

from . import wrappers
from . import config

class F32Wrapper(gym.Wrapper):
    """
    Casts state down to float32.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.observation_space.dtype = np.float32

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs.astype(np.float32), rew, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs.astype(np.float32)


def make(env_id:str, monitor_video=False, seed=None, args=None, determanistic_saving=True):
    """
    Construct environment of given name, including any required wrappers.
    @determanistic_saving: When true RND is saved with the environment, so restoring will always produce the same
        results. When false RNG is not persisted through saving, which can be helpful when generating return samples.
    """

    # this global reference will not work on windows when we spawn instead of fork,
    # so make sure to pass args in as an argument.
    args = args or config.args

    env_name = f"{env_id}-v2"

    env = gym.make(env_name)

    env = env.unwrapped

    env = wrappers.LabelEnvWrapper(env, env_id)

    if seed is not None:
        np.random.seed(seed)
        env.seed(seed)

    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    env = F32Wrapper(env)

    # doesn't work...
    #env = wrappers.SaveEnvStateWrapper(env, determanistic=determanistic_saving)

    env = wrappers.EpisodeScoreWrapper(env)

    env = wrappers.MonitorWrapper(env, monitor_video=False)

    return env
