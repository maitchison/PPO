"""
Helper to create wrapped mujoco environments
"""

import gym
import numpy as np
import cv2

from . import wrappers
from . import config

class ProcGenWrapper(gym.Wrapper):
    """
    Nothing to actaually to here.
    We used to transpose the channel order, but this is done with a wrapper now.

    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        H, W, C = self.env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (H, W, C), dtype=np.uint8)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["channels"] = ["Gray", "Gray", "Gray"]
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs


def make(env_id: str, monitor_video=False, seed=None, args=None, difficulty:str='hard', determanistic_saving=True):
    """
    Construct environment of given name, including any required wrappers.
    @determanistic_saving: When true RND is saved with the environment, so restoring will always produce the same
        results. When false RNG is not persisted through saving, which can be helpful when generating return samples.
    """

    args = args or config.args

    assert args.frame_skip == 1, "Frame skip should be 1 for procgen"

    env_name = f"procgen:procgen-{env_id}-v0"

    if seed is not None:
        # setting seed here might help procgen init?
        np.random.seed(seed)

    # procgen defaults to using hard, so just use gym to create env.
    env_args = {'distribution_mode': difficulty}
    if seed is not None:
        env_args['rand_seed'] = seed

    # stub
    # if seed is None:
    #     print("Generating procgen env with no seed.")
    # else:
    #     print(f"Generating procgen env with seed {seed}.")

    env = gym.make(env_name, **env_args)

    env = wrappers.LabelEnvWrapper(env, "env_id", env_id)

    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    env = ProcGenWrapper(env)

    env = wrappers.MonitorWrapper(env, monitor_video=monitor_video)

    env = wrappers.ColorTransformWrapper(env, args.color_mode)

    if args.embed_time:
        env = wrappers.TimeChannelWrapper(env)

    # reset of pipeline expects channels first
    env = wrappers.ChannelsFirstWrapper(env)

    env = wrappers.EpisodeScoreWrapper(env)

    if args.embed_action:
        # note: this is slightly different to how we do this for atari, where entire history is given...
        # also, make sure this is the correction action...
        # hmmm.. switch to a channel for this.. ?
        env = wrappers.ActionAwareWrapper(env)

    env = wrappers.NullActionWrapper(env)

    return env