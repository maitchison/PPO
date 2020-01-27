from gym.utils import seeding
import gym
import numpy as np

from . import utils

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

        super().__init__()

        self._card_count = None
        self._action_count = None

        self.set_number_of_actions_and_cards(2, 100)

        self._height = 210
        self._width = 160

        self.counter = 0
        self.current_action = 0
        self.key = 0
        self.answer = 0

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
        return key % self._action_count

    def set_number_of_actions_and_cards(self, action_count, card_count):
        self._action_count = action_count
        self._card_count = card_count
        self.solutions = [(key, self._key_to_action(key),) for key in range(self._card_count)]
        self.action_space = gym.spaces.Discrete(self._action_count)

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
        return self._action_count

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

        reward = 1 if self.current_action == self.answer else (-1 / (self._action_count - 1))

        return reward / (3600-50) * 10
