import gym
import numpy as np
from collections import defaultdict

from . import wrappers, config, ale_roms

# get list of game environments...
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def make(non_determinism=None, monitor_video=False, seed=None, args=None):
    """ Construct environment of given name, including any required wrappers."""

    # this global reference wont work on windows when we spawn instead of fork,
    # so make sure to pass args in as an argument.
    args = args or config.args

    env_type = None

    env_name = args.env_name

    for k,v in _game_envs.items():
        if env_name in v:
            env_type = k

    env = gym.make(env_name, full_action_space=args.full_action_space)

    if seed is not None:
        np.random.seed(seed)
        env.seed(seed)

    if env_name == "MemorizeNoFrameskip-v4":
        env.set_number_of_actions_and_cards(args.memorize_actions, args.memorize_cards)

    # default non-determinism
    non_determinism = non_determinism or ("noop" if args.noop_start else "none")

    if args.timeout > 0:
        env = wrappers.TimeLimitWrapper(env, args.timeout)

    if env_type == "atari":

        assert "NoFrameskip" in env_name

        env = wrappers.SaveEnvStateWrapper(env)

        non_determinism = non_determinism.lower()
        if non_determinism == "noop":
            env = wrappers.NoopResetWrapper(env, noop_max=args.noop_duration)
            env = wrappers.FrameSkipWrapper(env, min_skip=args.frame_skip, max_skip=args.frame_skip, reduce_op=np.max)
        elif non_determinism == "frame-skip":
            env = wrappers.NoopResetWrapper(env, noop_max=args.noop_duration)
            env = wrappers.FrameSkipWrapper(env, min_skip=int(args.frame_skip*0.5), max_skip=int(args.frame_skip*1.5), reduce_op=np.max)
        elif non_determinism == "none":
            env = wrappers.FrameSkipWrapper(env, min_skip=args.frame_skip, max_skip=args.frame_skip, reduce_op=np.max)
        else:
            raise Exception("Invalid non determinism type {}.".format(non_determinism))

        env = wrappers.MonitorWrapper(env, monitor_video=monitor_video)
        env = wrappers.EpisodeScoreWrapper(env)

        if args.input_crop:
            env = wrappers.FrameCropWrapper(env, None, None, 34, -16)

        if args.reward_clipping == "off":
            pass
        elif args.reward_clipping == "sqrt":
            env = wrappers.SqrtRewardWrapper(env)
        else:
            try:
                clip = float(args.reward_clipping)
            except:
                raise ValueError("reward_clipping should be off, sqrt, or a float")
            env = wrappers.ClipRewardWrapper(env, clip)

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

        if args.ed_type != "none":
            env = wrappers.EpisodicDiscounting(env, args.ed_type, args.ed_gamma)

        if args.reward_scale != 1.0 and not args.reward_normalization:
            env = wrappers.RewardScaleWrapper(env, args.reward_scale)

        if args.terminal_on_loss_of_life:
            env = wrappers.EpisodicLifeEnv(env)

        if args.deferred_rewards != 0:
            env = wrappers.DeferredRewardWrapper(env, args.deferred_rewards)

        env = wrappers.FrameStack(env, n_stacks=args.frame_stack)

        if args.time_aware:
            # must come after frame_stack
            env = wrappers.TimeAwareWrapper(env)

        env = wrappers.NullActionWrapper(env)

    else:
        raise Exception("Unsupported env_type {} for env {}".format(env_type, env_name))

    return env
