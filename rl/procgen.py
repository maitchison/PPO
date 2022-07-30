"""
Helper to create wrapped procgen environments
"""

import gym
import numpy as np
from procgen import ProcgenGym3Env

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


def make(env_id:str, monitor_video=False, seed=None, args=None, determanistic_saving=True):
    """
    Construct environment of given name, including any required wrappers.
    @determanistic_saving: When true RND is saved with the environment, so restoring will always produce the same
        results. When false RNG is not persisted through saving, which can be helpful when generating return samples.
    """

    args = args or config.args

    env_name = f"procgen:procgen-{env_id}-v0"

    # procgen defaults to using hard, so just use gym to create env.
    #env = ProcgenGym3Env(env_id, distribution_mode="hard")
    env_args = {}
    if seed is not None:
        env_args['rand_seed'] = seed
    env = gym.make(env_name, **env_args)

    env = wrappers.MonitorWrapper(env, monitor_video=monitor_video)

    env = ProcGenWrapper(env)

    env = wrappers.LabelEnvWrapper(env, env_id)

    if seed is not None:
        np.random.seed(seed)

    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    # no state saving for procgen
    #env = wrappers.SaveEnvStateWrapper(env, determanistic=determanistic_saving)

    env = wrappers.EpisodeScoreWrapper(env)

    env = wrappers.TimeAwareWrapper(env, log=True)

    if args.reward_scale != 1.0 and not args.reward_normalization:
        env = wrappers.RewardScaleWrapper(env, args.reward_scale)

    # todo: include time aware... (and maybe action aware)
    # if args.embed_time:
    #     # must come after frame_stack
    #     env = wrappers.TimeAwareWrapper(env)

    env = wrappers.NullActionWrapper(env)

    return env
