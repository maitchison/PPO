import gym
import math
import numpy as np
import cv2

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class RunningMeanStd(object):
    # from https://github.com/openai/baselines/blob/1b092434fc51efcb25d6650e287f07634ada1e08/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):

        if type(x) in [float, int]:
            batch_mean = x
            batch_var = 0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

class NoopResetWrapper(gym.Wrapper):
    """
    from
    https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class HashWrapper(gym.Wrapper):
    """
    Maps observation onto a random sequence of pixels.
    This is helpful for testing if the agent is simply memorizing the environment, as no generalization between
    states is possiable under this observation.

    Note: we assume channels is last, which means this really only works if applied after atari processing.
    """

    def __init__(self, env, hash_size, use_time=False):
        """
        Map observation to a hash of observation.
        """
        super().__init__(env)
        self.env = env
        self.use_time = use_time
        self.hash_size = hash_size
        self.counter = 0

    def step(self, action):

        original_obs, reward, done, info = self.env.step(action)

        if self.use_time:
            state_hash = self.counter
        else:
            state_hash = hash(original_obs.data.tobytes())

        w, h, c = original_obs.shape

        rng = np.random.RandomState(state_hash % (2**32)) # ok... this limits us to 32bits.. might be a better way to do this?

        # seed the random generator and create an random 42x42 observation.
        # note: I'm not sure how many bits the numpy random generate will use, it's posiable it's a lot less than
        # 1024. One option is then to break up the observation into parts. Another would be to just assume that the
        # number of reachable states is much much less than this, and that the chance of a collision (alaising) is
        # very low.
        new_obs = rng.randint(0, 1+1, (self.hash_size,self.hash_size), dtype=np.uint8) * 255
        new_obs = cv2.resize(new_obs, (h, w), interpolation=cv2.INTER_NEAREST)
        new_obs = new_obs[:, :, np.newaxis]

        new_obs = np.concatenate([new_obs]*c, axis=2)

        self.counter += 1

        return new_obs, reward, done, info

    def reset(self):
        self.counter = 0
        return self.env.reset()

class FrameSkipWrapper(gym.Wrapper):
    """
    from
    https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
    """
    def __init__(self, env, min_skip=4, max_skip=None, reduce_op=np.max):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        if max_skip is None:
            max_skip = min_skip
        assert env.observation_space.dtype == "uint8"
        assert min_skip >= 1
        assert max_skip >= min_skip
        # most recent raw observations
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._min_skip = min_skip
        self._max_skip = max_skip
        self._reduce_op = reduce_op

    def step(self, action):
        """Repeat action, sum reward, and mean over last observations."""
        total_reward = 0.0
        done = None
        info = None
        skip = np.random.randint(self._min_skip, self._max_skip+1)
        for i in range(skip):
            obs, reward, done, info = self.env.step(action)
            if i == skip - 2: self._obs_buffer[0] = obs
            if i == skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        reduce_frame = self._reduce_op(self._obs_buffer, axis=0)

        return reduce_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class NormalizeRewardWrapper(gym.Wrapper):

    def __init__(self, env, clip=5.0, initial_state=None):
        """
        Normalizes returns

        """
        super().__init__(env)

        self.env = env
        self.clip = clip
        self.epsilon = 1e-4
        self.current_return = 0
        self.ret_rms = RunningMeanStd(shape=())
        if initial_state is not None:
            self.restore_state(initial_state)

    def step(self, action):

        obs, reward, done, info = self.env.step(action)

        self.current_return = self.current_return * 0.99 + reward

        self.ret_rms.update(self.current_return)

        self.mean = self.ret_rms.mean
        self.std = math.sqrt(self.ret_rms.var)

        info["raw_reward"] = reward
        scaled_reward = reward / (self.std + self.epsilon)
        scaled_reward = np.clip(scaled_reward, -self.clip, +self.clip)

        info["returns_norm_state"] = self.save_state()

        return obs, scaled_reward, done, info

    def reset(self):
        self.current_return = 0
        return self.env.reset()

    def save_state(self):
        """
        Saves running statistics.
        """
        return tuple(float(x) for x in [self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count])

    def restore_state(self, state):
        """
        Restores running statistics.
        """
        self.ret_rms.mean, self.ret_rms.var, self.ret_rms.count = state


class ObservationMonitor(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["monitor_obs"] = obs.copy()
        return obs, reward, done, info

class AtariWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, n_stacks=4, grayscale=True, width=84, height=84, crop=False):
        """
        Stack and do other stuff...
        Input should be (210, 160, 3)
        Output is a stack of shape (nstacks, width, height)
        """

        super().__init__(env)

        self.env = env

        self.n_stacks = n_stacks
        self._width, self._height = width, height

        assert len(env.observation_space.shape) == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.shape[-1] == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.dtype == np.uint8, "Invalid dtype {}".format(env.observation_space.dtype)

        self.grayscale = grayscale
        self.crop = crop
        self.n_channels = self.n_stacks * (1 if self.grayscale else 3)
        self.stack = np.zeros((self.n_channels, self._width, self._height), dtype=np.uint8)

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.n_channels, self._width, self._height),
            dtype=np.uint8,
        )

    def _push_raw_obs(self, obs):

        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = obs[:, :, np.newaxis]

        width, height, channels = obs.shape

        if self.crop:
            obs = obs[34:-16, :, :]

        if (width, height) != (self._width, self._height):
            obs = cv2.resize(obs, (self._height, self._width), interpolation=cv2.INTER_AREA)

        if len(obs.shape) == 2:
            obs = obs[:, :, np.newaxis]

        self.stack = np.roll(self.stack, shift=-(1 if self.grayscale else 3), axis=0)

        if self.grayscale:
            self.stack[0:1, :, :] = obs[:, :, 0]
        else:
            obs = np.swapaxes(obs, 0, 2)
            obs = np.swapaxes(obs, 1, 2)
            self.stack[0:3, :, :] = obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_raw_obs(obs)
        return self.stack, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stacks):
            self._push_raw_obs(obs)
        return self.stack
