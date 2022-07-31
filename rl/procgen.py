"""
Helper to create wrapped procgen environments
"""

import gym
import numpy as np

from . import wrappers
from . import config

class ProcGenWrapper(gym.Wrapper):
    """
    Swaps channel order for procgen.
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        H, W, C = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (C, H, W), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process(obs), reward, done, info

    def reset(self):
        obs = self.env.reset()
        return self._process(obs)

    def _process(self, obs):
        return obs.transpose(2, 0, 1)


def make(env_id:str, monitor_video=False, seed=None, args=None, difficulty:str='hard', determanistic_saving=True):
    """
    Construct environment of given name, including any required wrappers.
    @determanistic_saving: When true RND is saved with the environment, so restoring will always produce the same
        results. When false RNG is not persisted through saving, which can be helpful when generating return samples.
    """

    args = args or config.args

    env_name = f"procgen:procgen-{env_id}-v0"

    if seed is not None:
        # setting seed here might help procgen init?
        np.random.seed(seed)

    # procgen defaults to using hard, so just use gym to create env.
    env_args = {'distribution_mode': difficulty}
    if seed is not None:
        env_args['rand_seed'] = seed
    env = gym.make(env_name, **env_args)

    env = wrappers.MonitorWrapper(env, monitor_video=monitor_video)

    env = ProcGenWrapper(env)

    env = wrappers.LabelEnvWrapper(env, env_id)

    # no timeout...
    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    env = wrappers.EpisodeScoreWrapper(env)


    env = wrappers.NullActionWrapper(env)


    return env
