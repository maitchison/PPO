import gym
import numpy as np
from collections import defaultdict

from . import wrappers
from .config import args

ENV_STATE = {}

# get list of game environments...
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def get_env_state(key):
    return ENV_STATE.get(key, None)

def make(non_determinism=None, monitor_video=False):
    """ Construct environment of given name, including any required wrappers."""

    env_type = None

    env_name = args.env_name

    for k,v in _game_envs.items():
        if env_name in v:
            env_type = k

    env = gym.make(env_name)

    if env_name == "MemorizeNoFrameskip-v4":
        env.set_number_of_actions_and_cards(args.memorize_actions, args.memorize_cards)

    # default non-determinism
    non_determinism = non_determinism or ("noop" if args.noop_start else "none")

    if env_type == "atari":

        assert "NoFrameskip" in env_name

        env = wrappers.TimeLimitWrapper(env, 60 * 60 * 30)

        non_determinism = non_determinism.lower()
        if non_determinism == "noop":
            env = wrappers.NoopResetWrapper(env, noop_max=30)
            env = wrappers.FrameSkipWrapper(env, min_skip=4, max_skip=4, reduce_op=np.max)
        elif non_determinism == "frame-skip":
            env = wrappers.NoopResetWrapper(env, noop_max=30)
            env = wrappers.FrameSkipWrapper(env, min_skip=2, max_skip=5, reduce_op=np.max)
        elif non_determinism == "none":
            env = wrappers.FrameSkipWrapper(env, min_skip=4, max_skip=4, reduce_op=np.max)
        else:
            raise Exception("Invalid non determinism type {}.".format(non_determinism))

        env = wrappers.MonitorWrapper(env, monitor_video=monitor_video)

        if args.input_crop:
            env = wrappers.FrameCropWrapper(env, None, None, 34, -16)

        # apply filter
        if args.filter == "none":
            pass
        elif args.filter == "hash":
            env = wrappers.HashWrapper(env, args.hash_size)
        elif args.filter == "hash_time":
            env = wrappers.HashWrapper(env, args.hash_size, use_time=True)
        else:
            raise Exception("Invalid observation filter {}.".format(args.filter))

        env = wrappers.AtariWrapper(env, width=args.res_x, height=args.res_y, grayscale=not args.color)

        if args.time_aware:
            env = wrappers.TimeAwareWrapper(env, max_time=60 * 60 * 30)

        if args.ed_type != "none":
            env = wrappers.EpisodicDiscounting(env, args.ed_type, args.ed_gamma)

        env = wrappers.FrameStack(env, n_stacks=args.frame_stack)

        env = wrappers.EpisodeScoreWrapper(env)

        env = wrappers.NullActionWrapper(env)

    else:
        raise Exception("Unsupported env_type {} for env {}".format(env_type, env_name))

    return env