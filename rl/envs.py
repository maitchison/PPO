"""
Library for creating vector environments
"""

from rl.config import args

import numpy as np
import gym.wrappers

import gym

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

    # lazy load, as this might not be on cluster yet...
    import envpool

    # build id
    env_id = f"{args.env.type}:{args.env.name}"

    # build gym args
    gym_args = {}
    if args.env.type == "procgen":
        gym_args.update(
            difficulty = args.env.procgen_difficulty
        )
    if args.env.type == "atari":
        gym_args.update(
            repeat_action_probability=args.env.repeat_action_probability,
            full_action_space=args.env.full_action_space,
        )

    envs = envpool.make_gym(env_id, **gym_args)

    # todo: make sure no timelimit was applied

    if args.env.type == "atari":
        # todo: rom check
        pass

    # standard wrappers
    # label env_id
    # label seed
    # set seeds (do above?)

    # todo...
    # per step termination prob
    # save env state (might be hard?)

    if args.env.is_vision_env:

        # vision processing... it's quite a long list...
        # i'll try to integrate these two as much as I can.

        if args.env.type == "atari":
            # maybe we can change the config to get env pool to do this?
            assert args.env.color_mode == "bw", f"Envpool generates black and white frames, so {args.env.color_mode} format is not supported."
            assert args.env.res_x == args.env.res_y, "Atari preprocessing only supports square resolutions."

        # -------------------------
        # for atari...
        # -------------------------
        # noop start

        # [done] frame skip
        # timelimit
        # mr info
        # monitor video
        # ep score
        # reward clipping
        # atari processing (scale?)

        envs = gym.wrappers.AtariPreprocessing(
            envs,
            noop_max=args.env.noop_duration,
            frame_skip=args.env.frame_skip,
            screen_size=args.env.res_x,
            terminal_on_life_loss=args.env.atari_terminal_on_loss_of_life,
            grayscale_obs=False,
            scale_obs=False,
        )
        # zero obs
        # color transform
        # terminal on loss of life
        # differed reward
        # embed action
        # [done] framestack
        # embed time
        # embed state (remove?)
        # channels first (if needed)
        # null action wrapper (won't work anymore... remove, this will make desync impossiable though)

        # -------------------------
        # for procgen...
        # -------------------------
        # timelimit
        # procgen wrapper
        # monitor video
        # color transform
        # embed time
        # channels first
        # ep score
        # embed action
        # null action
    else:
        # this is really just for mujoco
        raise Exception("Mujoco not yet supported with envpool.")

    # other standard

    return envs



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
    env_fns = [lambda i=i: make_env(args.env.type, env_id=args.env.name, args=args, seed=base_seed + (i * 997),
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
    if args.env.reward_normalization == "rms":
        vec_env = wrappers.VecNormalizeRewardWrapper(
            vec_env,
            gamma=args.reward_normalization_gamma,
            mode="rms",
            clip=args.env.reward_normalization_clipping,
        )

    if args.env.max_repeated_actions > 0 and args.env.type != "mujoco":
        vec_env = wrappers.VecRepeatedActionPenalty(vec_env, args.env.max_repeated_actions,
                                                         args.env.repeated_action_penalty)

    return vec_env