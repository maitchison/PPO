import gym
import math
import numpy as np
import cv2
import hashlib
from . import utils

class HashWrapper(gym.Wrapper):
    """
    Maps observation onto a random sequence of pixels.
    This is helpful for testing if the agent is simply memorizing the environment, as no generalization between
    states is possible under this observation.
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
            state_hash = int(hashlib.sha256(original_obs.data.tobytes()).hexdigest(), 16)

        # note: named tensor would help get this shape right...
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
    Performs frame skipping with max over last two frames.
    From https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
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


class ClipRewardWrapper(gym.Wrapper):

    def __init__(self, env, clip):
        super().__init__(env)
        self.env = env
        self.clip = clip

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = np.clip(reward, -self.clip, +self.clip)
        return obs, reward, done, info


class NormalizeObservationsWrapper(gym.Wrapper):
    """
    Normalizes observations.
    """
    def __init__(self, env, clip, shadow_mode=False, initial_state=None):
        super().__init__(env)

        self.env = env
        self.epsilon = 1e-4
        self.clip = clip
        self.obs_rms = utils.RunningMeanStd(shape=())
        self.shadow_mode = shadow_mode
        if initial_state is not None:
            self.obs_rms.restore_state(initial_state)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obs_rms.update(obs)
        self.mean = self.obs_rms.mean
        self.std = np.sqrt(self.obs_rms.var)

        info["observation_norm_state"] = self.obs_rms.save_state()

        if self.shadow_mode:
            return obs, reward, done, info
        else:
            scaled_obs = (obs - self.mean) / (self.std + self.epsilon)
            scaled_obs = np.clip(scaled_obs, -self.clip, +self.clip)
            scaled_obs = np.asarray(scaled_obs, dtype=np.float32)
            return scaled_obs, reward, done, info


class NormalizeRewardWrapper(gym.Wrapper):
    """
    Normalizes rewards such that returns are unit normal.
    """

    def __init__(self, env, initial_state=None):
        """
        Normalizes returns
        """
        super().__init__(env)

        self.env = env
        self.epsilon = 1e-4
        self.current_return = 0
        self.ret_rms = utils.RunningMeanStd(shape=())
        self.mean = 0.0
        self.std = 0.0
        if initial_state is not None:
            self.ret_rms.restore_state(initial_state)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_return = reward + self.current_return * 0.99 * (1-done)
        self.ret_rms.update(self.current_return)
        self.mean = self.ret_rms.mean
        self.std = math.sqrt(self.ret_rms.var)
        scaled_reward = reward / (self.std + self.epsilon)
        info["returns_norm_state"] = self.ret_rms.save_state()
        return obs, scaled_reward, done, info


class MonitorWrapper(gym.Wrapper):
    """
    Records a copy of the current observation and reward into info.
    This can be helpful to retain an unmodified copy of the input.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info["monitor_obs"] = obs.copy()
        info["raw_reward"] = reward
        return obs, reward, done, info

class FrameCropWrapper(gym.Wrapper):
    """
    Crops input frame.
    """

    def __init__(self, env: gym.Env, x1, x2, y1, y2):
        super().__init__(env)
        self.env = env
        self.cropping = (slice(y1, y2, 1), slice(x1, x2, 1))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = obs[self.cropping]
        return obs, reward, done, info

class TimeLimitWrapper(gym.Wrapper):
    """
    From https://github.com/openai/baselines/blob/master/baselines/common/wrappers.py
    """
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

class FoveaWrapper(gym.Wrapper):
    """
    Applies a fovea model to the Atari game. This involves stacking global grayscale low resolution frames with
    local high resolution color frames.
    """

    def __init__(self, env: gym.Env, global_stacks=4, local_stacks=4, width=42, height=42):
        """
        Stack and do other stuff...
        Input should be (210, 160, 3)
        Output is a stack of shape (nstacks, width, height)
        """

        super().__init__(env)

        self.env = env

        self.global_stacks = global_stacks
        self.local_stacks = local_stacks
        self._width, self._height = width, height

        assert global_stacks == local_stacks, "Gobal stacks not equal to local stacks not implemented yet."

        assert len(env.observation_space.shape) == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.shape[-1] == 3, "Invalid shape {}".format(env.observation_space.shape)
        assert env.observation_space.dtype == np.uint8, "Invalid dtype {}".format(env.observation_space.dtype)

        self.n_channels = global_stacks + local_stacks * 4
        self.stack = np.zeros((self.n_channels, self._width, self._height), dtype=np.uint8)

        self.local_x = 0
        self.local_y = 0
        self.blur_factor = 0

        # dx, dy
        self.action_map = [(0,0)]
        for step in [1, 4, 8]:
            self.action_map.append((-step, 0))
            self.action_map.append((+step, 0))
            self.action_map.append((0, -step))
            self.action_map.append((0, -step))
            self.action_map.append((-step, +step))
            self.action_map.append((+step, -step))
            self.action_map.append((-step, -step))
            self.action_map.append((+step, +step))

        self.channels = []
        for i in range(self.global_stacks):
            # right now these are all mixed up but what would be better is to have seperate stacks for each one
            # so I can have different levels of local and global stacks.
            self.channels.append("Gray-" + str(i))
            self.channels.append("ColorR-" + str(i))
            self.channels.append("ColorG-" + str(i))
            self.channels.append("ColorB-" + str(i))
            self.channels.append("Mask-" + str(i))

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.n_channels, self._width, self._height),
            dtype=np.uint8,
        )

    def _get_fovia_rect(self):
        x1 = np.clip(self.local_x, 0, 160 - self._width)
        y1 = np.clip(self.local_y, 0, 210 - self._height)
        x2 = x1 + self._width
        y2 = y1 + self._height
        return x1,y1,x2,y2

    def _push_raw_obs(self, obs):

        fr = self._get_fovia_rect()

        # generate fovea location frame
        mask = np.zeros((210, 160), dtype=np.uint8) + 64
        mask[fr[1]:fr[3], fr[0]:fr[2]] = 255
        mask = cv2.resize(mask, (self._height, self._width), interpolation=cv2.INTER_AREA)
        self._push(mask)

        # generate a local frame
        local_obs = obs[fr[1]:fr[3], fr[0]:fr[2]]
        assert local_obs.shape == (self._width, self._height, 3), "Invalid fovia rect {} - {}".format(str(fr), str(local_obs.shape))

        if self.blur_factor > 1:
            local_obs = cv2.blur(local_obs, (int(self.blur_factor), int(self.blur_factor)))

        self._push(local_obs)

        # generate the global frame
        global_obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        global_obs = cv2.resize(global_obs, (self._height, self._width), interpolation=cv2.INTER_AREA)
        self._push(global_obs)

    def _push(self, frame):

        if len(frame.shape) == 3:
            # push multiple frames one by one.
            for i in range(frame.shape[2]):
                self._push(frame[:,:,i])
        else:
            self.stack = np.roll(self.stack, shift=1, axis=0)
            self.stack[0:1, :, :] = frame

    def step(self, action):

        # get attention
        if type(action) is int:
            env_action = action
            movement_cost = 0
            self.blur_factor = 0
        else:
            env_action, fovia_action = tuple(action)
            dx, dy = self.action_map[fovia_action]
            movement_cost = abs(dx) + abs(dy)
            self.blur_factor = movement_cost
            self.local_x = np.clip(self.local_x + dx, 0, 160)
            self.local_y = np.clip(self.local_y + dy, 0, 210)

        obs, reward, done, info = self.env.step(env_action)

        info["attention_cost"] = movement_cost / 100

        self._push_raw_obs(obs)
        info["channels"] = self.channels[:]
        return self.stack, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(max(self.local_stacks, self.global_stacks)):
            self._push_raw_obs(obs)
        return self.stack


class AtariWrapper(gym.Wrapper):
    """
    Applies Atari frame warping, optional gray-scaling, and frame stacking as per nature paper.
    Note: unlike Nature the initial frame cropping is disabled by default.
    """

    def __init__(self, env: gym.Env, n_stacks=4, grayscale=True, width=84, height=84):
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
        self.n_channels = self.n_stacks * (1 if self.grayscale else 3)
        self.stack = np.zeros((self.n_channels, self._width, self._height), dtype=np.uint8)

        self.channels = []
        for i in range(self.n_stacks):
            if grayscale:
                self.channels.append("Gray-" + str(i))
            else:
                self.channels.append("ColorR-" + str(i))
                self.channels.append("ColorG-" + str(i))
                self.channels.append("ColorB-" + str(i))

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

        if (width, height) != (self._width, self._height):
            obs = cv2.resize(obs, (self._height, self._width), interpolation=cv2.INTER_AREA)

        if len(obs.shape) == 2:
            obs = obs[:, :, np.newaxis]

        self.stack = np.roll(self.stack, shift=(1 if self.grayscale else 3), axis=0)

        if self.grayscale:
            self.stack[0:1, :, :] = obs[:, :, 0]
        else:
            obs = np.swapaxes(obs, 0, 2)
            obs = np.swapaxes(obs, 1, 2)
            self.stack[0:3, :, :] = obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._push_raw_obs(obs)
        info["channels"] = self.channels[:]
        return self.stack, reward, done, info

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.n_stacks):
            self._push_raw_obs(obs)
        return self.stack

class NoopResetWrapper(gym.Wrapper):
    """
    Applies a random number of no-op actions before agent can start playing.
    From https://github.com/openai/baselines/blob/7c520852d9cf4eaaad326a3d548efc915dc60c10/baselines/common/atari_wrappers.py
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
