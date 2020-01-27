import gym
import numpy as np
import random
from collections import defaultdict

from . import wrappers
from .config import args
from .models import NatureCNN_Net

import torch

from gym.envs.registration import register

register(
    id='MemorizeNoFrameskip-v4',
    entry_point='rl.atari:MemorizeGame',
)

ENV_STATE = {}

# get list of game environments...
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

def get_env_state(key):
    return ENV_STATE.get(key, None)


class RandomReward(wrappers.AuxReward):
    """
    Adds random auxilary rewards to environment.
    """

    def __init__(self, env, scale = 1.0, seed = None, decay_rewards=0.9):

        super().__init__(env, lambda prev_obs, action, obs : self.get_random_reward(prev_obs, action, obs))
        self.scale = scale

        self.initial_seed = seed or random.randint(0,1e9)

        self.random_network = NatureCNN_Net(env.observation_space.shape, hidden_units=16)

        self.small_rate = 1/10
        self.big_rate = 1/100
        self.decay_rewards = decay_rewards

        self.given_rewards = defaultdict(int)
        self.counter = 0

    def _mark_reward(self, reward_value, state):
        """ Returns value of given reward, and increments it's visited counter."""

        self.given_rewards[state] += 1
        reward_value = reward_value * (0.9**self.given_rewards[state])

        if self.counter % 100 == 1:
            print("reward {} in state {} with seed {}".format(reward_value, state, self.seed))

        self.counter += 1

        return reward_value

    def get_random_reward(self, prev_obs, action, obs):

        # run the final state through a randomly connected neural network then
        # output 16 bits -> 65536 states (i.e. random mapping to a discrete state space)
        # assign some percentage of these 'states' as high value states (1 reward)
        # assign another percentage of these 'states' as low value states (0.1 reward)

        # some prep is required to get the input ready for the model.
        obs_in = torch.tensor(obs[np.newaxis, :, :, :]).float()/255.0
        out = self.random_network(obs_in)
        out = torch.tanh(out)
        out = out.detach().cpu().numpy()
        out = out[0]

        out = [1 if x >= 0 else 0 for x in out]
        state = int("".join(str(i) for i in out), 2)

        if state in self.high_reward_states:
            reward = self._mark_reward(1.0, state)
            self._set_seed(self.seed + 1)
            return reward
        elif state in self.small_reward_states:
            reward = self._mark_reward(0.1, state)
            self._set_seed(self.seed + 1)
            return reward
        else:
            return 0

    def _set_seed(self, seed):
        """ Sets rewards to given seed. """

        st0 = np.random.get_state()

        np.random.seed(seed)

        state_space = 2**16

        small_rewards = int(state_space * self.small_rate)
        big_rewards = int(state_space * self.big_rate)

        self.high_reward_states = set(np.random.choice(state_space, big_rewards))
        self.small_reward_states = set(np.random.choice(state_space, small_rewards))
        self.seed = seed

        np.random.set_state(st0)


    def reset(self):
        self._set_seed(self.initial_seed)
        return self.env.reset()


def make(env_name, non_determinism=None):
    """ Construct environment of given name, including any required wrappers."""

    env_type = None

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

        env = wrappers.MonitorWrapper(env)

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

        if args.use_atn:
            env = wrappers.FoveaWrapper(env, width=args.res_x, height=args.res_y, global_frame_skip=args.atn_global_frame_skip)
        else:
            env = wrappers.AtariWrapper(env, width=args.res_x, height=args.res_y, grayscale=not args.color)
            env = wrappers.FrameStack(env)

        if args.random_rewards:
            print("Enabling random rewards")
            # this is done before normalization which isn't great... maybe normalize before and after?
            # also what scale is needed so that this matches the normalized rewards? Hard to say... but maybe 10?
            env = RandomReward(env, scale=10, seed=1) # each worker should really get it's own seed, and own parameters...

        if args.reward_normalization:
            env = wrappers.NormalizeRewardWrapper(env,
                                                  initial_state=get_env_state("returns_norm_state")
                                                  )


        if args.reward_clip:
            env = wrappers.ClipRewardWrapper(env, args.reward_clip)

    else:
        raise Exception("Unsupported env_type {} for env {}".format(env_type, env_name))

    return env