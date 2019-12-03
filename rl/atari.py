import gym
import numpy as np
import random
from gym.utils import seeding
from collections import defaultdict

from . import wrappers, utils
from .config import args

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

class MemorizeGame(gym.Env):
    """
    Note: some modification to the game.
    The number of classes matters, make this a no-op and 4 directions, as would be 2 bits per example.
    Make the game human playable
    Have the game work that when player presses a button it checks if it' correct then moves onto the new card
    With a 10 second time-out. Also make it that after 10 cards the episode ends.
    I'd need a 12 frame delay too (1/5th of a second), otherwise repeat action will trigger the next card.
    This means a fast agent would be able to get though around 1 card every 16 frames, which is super fast.
    Maybe add a very small penality while card is up to give faster players a higher score (but only slightly)
    This is mostly so the agent doesn't learn to delay punishment for incorrect cards. (or just don't have punishment?)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self):

        self.action_count = 6
        self.set_number_of_cards(100)

        self._height = 210
        self._width = 160

        self.counter = 0
        self.current_action = 0
        self.key = 0
        self.answer = 0

        self.action_space = gym.spaces.Discrete(self.action_count)

        self.observation_space = self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, 3),
            dtype=np.uint8,
        )

        self.seed()
        self.reset()

    def _key_to_action(self, key):
        """ Returns the action that goes with a specific key. """
        return key % self.action_count

    def set_number_of_cards(self, card_count):
        self.card_count = card_count
        self.solutions = [(key, self._key_to_action(key),) for key in range(self.card_count)]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """
        """
        self._take_action(action)
        reward = self._get_reward()

        episode_over = self.counter > 3600 # games last for 30 seconds
        self.counter += 1
        if self.counter % 60 == 0: # change every 1 second.
            self._randomize_state()

        obs = utils.generate_hash_image(self.key, (7, 7), (self._width, self._height, 3))

        return obs, reward, episode_over, {}

    @property
    def _n_actions(self):
        return self.action_count

    def get_action_meanings(self):
        return ["NOOP" for _ in range(self._n_actions)]

    def reset(self):
        self.counter = 0
        self.current_action = 0
        self._randomize_state()

    def _randomize_state(self):
        self.key, self.answer = random.choice(self.solutions)

    def _render(self, mode='human', close=False):
        pass

    def _take_action(self, action):
        self.current_action = action

    def _get_reward(self):
        """ Reward is given guessing correct answer. """
        if self.counter < 50: # don't give out reward during first 50 moves (i.e. before game starts).
            return 0
        # these rewards are roughly balanced so perfect place is close to 10, and random play is close to 0.
        reward = 1 if self.current_action == self.answer else (-1/5)
        return reward / (3600-50) * 10


def make(env_name, non_determinism="noop"):
    """ Construct environment of given name, including any required wrappers."""

    env_type = None

    for k,v in _game_envs.items():
        if env_name in v:
            env_type = k

    env = gym.make(env_name)

    if env_name == "MemorizeNoFrameskip-v4":
        env.set_number_of_cards(args.memorize_cards)

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

        env = wrappers.AtariWrapper(env, width=args.res_x, height=args.res_y, grayscale=not args.color)

        if args.use_rnd:
            # rnd requires a normalized copy of the state...
            env = wrappers.NormalizeObservationsWrapper(env, clip=5.0, shadow_mode=True,
                                                        initial_state=get_env_state("observation_norm_state")
                                                        )

        if args.reward_normalization:
            env = wrappers.NormalizeRewardWrapper(env,
                                                  initial_state=get_env_state("returns_norm_state")
                                                  )

        if args.reward_clip:
            env= wrappers.ClipRewardWrapper(env, args.reward_clip)

    else:
        raise Exception("Unsupported env_type {} for env {}".format(env_type, env_name))

    return env