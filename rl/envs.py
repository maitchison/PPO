"""
Library for creating vector environments
"""

from rl.config import args

import numpy as np

import gym

# should be in rl.envs
from rl import atari, mujoco, procgen
from rl import hybridVecEnv         # this is my vector env, it's a bit clunky, but it gets the job done.
from rl import wrappers

def make_env(env_type, env_id, **kwargs):
    """
    Makes a single environment (using classic method)
    """
    if env_type == "atari":
        make_fn = atari.make
    elif env_type == "mujoco":
        make_fn = mujoco.make
    elif env_type == "procgen":
        make_fn = procgen.make
    else:
        raise ValueError(f"Invalid environment type {env_type}")
    return make_fn(env_id, **kwargs)


def create_envs_envpool(N=None, monitor_video=False):
    """
    Creates environments using (faster) envpool. Not all features supported et
    """
    pass


def create_envs_classic(N=None, monitor_video=False):
    """
    Creates (vectorized) environments for runner
    This is the old version that creates processes to run individual envs, then joints them together into a vector
    env at the end. It's a bit slow.

    """
    N = N or args.agents
    base_seed = args.seed
    if base_seed is None or base_seed < 0:
        base_seed = np.random.randint(0, 9999)
    env_fns = [lambda i=i: make_env(args.env_type, env_id=args.get_env_name(i), args=args, seed=base_seed + (i * 997),
                                    monitor_video=monitor_video) for i in range(N)]

    if args.sync_envs:
        vec_env = gym.vector.SyncVectorEnv(env_fns)
    else:
        vec_env = hybridVecEnv.HybridAsyncVectorEnv(
            env_fns,
            copy=False,
            max_cpus=args.workers,
            verbose=True
        )

    # ema normalization is handled externally.
    if args.reward_normalization == "rms":
        vec_env = wrappers.VecNormalizeRewardWrapper(
            vec_env,
            gamma=args.reward_normalization_gamma,
            mode="rms",
            clip=args.reward_normalization_clipping,
        )

    if args.max_repeated_actions > 0 and args.env_type != "mujoco":
        vec_env = wrappers.VecRepeatedActionPenalty(vec_env, args.max_repeated_actions,
                                                         args.repeated_action_penalty)

    return vec_env