import os
import sys
import argparse

# using more than 2 threads locks up all the cpus and does not seem to improve performance.
# the gain from 2 CPUs to 1 is very minor too. (~10%)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def build_parser():
    parser = argparse.ArgumentParser(description="Trainer for PPO2")

    parser.add_argument("environment")

    parser.add_argument("--experiment_name", type=str, help="Name of the experiment.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")

    parser.add_argument("--agents", type=int, default=8)

    parser.add_argument("--filter", type=str, default="none", help="Add filter to agent observation ['none', 'hash']")
    parser.add_argument("--hash_size", type=int, default=42, help="Adjusts the hash tempalte generator size.")
    parser.add_argument("--restore", type=str2bool, default=False, help="Restores previous model if it exists. If set to false and new run will be started.")

    parser.add_argument("--crop_input", type=str2bool, default=False, help="enables atari input cropping.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="learning rate for adam optimizer")
    parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of environment steps per training step.")
    parser.add_argument("--epochs", type=int, default=200, help="Each epoch represents 1 million environment interactions.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
    parser.add_argument("--batch_epochs", type=int, default=4, help="Number of training epochs per training batch.")
    parser.add_argument("--evaluate_diversity", type=str2bool, default=False, help="Evalutes the diversity of rollouts during training.")
    parser.add_argument("--reward_clip", type=float, default=5.0)
    parser.add_argument("--mini_batch_size", type=int, default=1024)
    parser.add_argument("--sync", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--resolution", type=str, default="standard", help="['full', 'standard', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--ent_bonus", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--export_video", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--output_folder", type=str)
    parser.add_argument("--sticky_actions", type=str2bool, default=False)
    parser.add_argument("--model", type=str, default="cnn", help="['cnn', 'improved_cnn']")

    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    # must be done before import numpy...
    threads = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = threads
    os.environ["OPENBLAS_NUM_THREADS"] = threads
    os.environ["MKL_NUM_THREADS"] = threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = threads
    os.environ["NUMEXPR_NUM_THREADS"] = threads

import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import time
import itertools
import csv
import json
import uuid
import argparse
import math
import shutil
from collections import deque, defaultdict
import multiprocessing
import pickle
import socket

ENV_NORM_STATE = None

if __name__ == "__main__":
    torch.set_num_threads(int(threads))

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

GUID = uuid.uuid4().hex
LOCK_KEY = uuid.uuid4().hex
LOG_FOLDER = "/run/experiment [{}]".format(GUID[-8:])
NATS_TO_BITS = 1.0/math.log(2)
RES_X = 84
RES_Y = 84
USE_COLOR = False
OUTPUT_FOLDER = "runs"
DTYPE = torch.float
PROFILE_INFO = False
PRINT_EVERY = 10
SAVE_LOG_EVERY = 50
VERBOSE = True
HOSTNAME = socket.gethostname()

def get_auto_device():
    """ Returns the best device, CPU if no CUDA, otherwise GPU with most free memory. """
    if not torch.cuda.is_available():
        return "cpu"

    if torch.cuda.device_count() == 1:
        return "cuda"
    else:
        # use the device with the most free memory.
        try:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            return "cuda:"+str(np.argmax(memory_available))
        except:
            print("Warning: Failed to auto detect best GPU.")
            return "cuda"

def get_checkpoint_path(step, postfix):
    return os.path.join(LOG_FOLDER, "checkpoint-{}-{}".format(zero_format_number(step), postfix))

def show_cuda_info():
    print("Using device: {}".format(bcolors.BOLD+DEVICE+bcolors.ENDC))

def mse(a,b):
    """ returns MSE of a and b. """
    return (np.square(a - b, dtype=np.float32)).mean(dtype=np.float32)

def dtw(obs1, obs2):
    """ Calculates the distances between two observation sequences using dynamic time warping.
        obs1, obs2
            np array [N, C, W, H], where N is number of frames (they don't need to mathc), and C is channels which
                                   should be 1.

        ref: https://en.wikipedia.org/wiki/Dynamic_time_warping
    """

    n = obs1.shape[0]
    m = obs2.shape[0]

    DTW = np.zeros((n+1,m+1), dtype=np.float32) + float("inf")
    DTW[0,0] = 0

    obs1 = np.float32(obs1)
    obs2 = np.float32(obs2)

    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = mse(obs1[i-1], obs2[j-1])
            DTW[i,j] = cost + min(
                DTW[i - 1, j],
                DTW[i, j - 1],
                DTW[i - 1, j - 1]
            )

    return DTW[n, m]


"""
------------------------------------------------------------------------------------------------------------------------
    Wrappers
------------------------------------------------------------------------------------------------------------------------
"""


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

    def __init__(self, env, use_time=False):
        """
        Map observation to a hash of observation.
        """
        super().__init__(env)
        self.env = env
        self.use_time = use_time
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
        new_obs = rng.randint(0, 1+1, (args.hash_size,args.hash_size), dtype=np.uint8) * 255
        new_obs = cv2.resize(new_obs, (h, w), interpolation=cv2.INTER_NEAREST)
        new_obs = new_obs[:, :, np.newaxis]

        new_obs = np.concatenate([new_obs]*c, axis=2)

        self.counter += 1

        return new_obs, reward, done, info

    def reset(self):
        self.counter = 0
        return self.env.reset()


class BlindWrapper(gym.Wrapper):
    """
    Replaces observation with zeros. This tests if an agent can solve the task by memorizing a sequence of actions without
    considering the observation at all.
    """
    pass

class TickerWrapper(gym.Wrapper):
    """
    Replace observation with an indication of the current time. This tests how well an agent performs if it's only
    strategy is to memorize a sequence of key-presses.
    """
    pass

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


class DiscretizeActionWrapper(gym.Wrapper):

    def __init__(self, env, bins=10):
        """
        Convert continuous action space into discrete.
        """
        super().__init__(env)
        self.env = env

        assert isinstance(env.action_space, gym.spaces.Box)
        assert len(env.action_space.shape) == 1

        dims = env.action_space[0]

        self.action_map = []

        spans = [np.linspace(env.action_space.low[d], env.action_space.high[d], bins) for d in range(dims)]

        self.action_map = list(itertools.product(*spans))

        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def step(self, action):
        return self.env.step(self.action_map[action])


"""
------------------------------------------------------------------------------------------------------------------------
    Utilities
------------------------------------------------------------------------------------------------------------------------
"""


def make_environment(env_name, non_determinism="noop"):
    """ Construct environment of given name, including any required """

    env_type = None

    for k,v in _game_envs.items():
        if env_name in v:
            env_type = k

    env = gym.make(env_name)
    if env_type == "atari":
        assert "NoFrameskip" in env_name

        non_determinism = non_determinism.lower()
        if non_determinism == "noop":
            env = NoopResetWrapper(env, noop_max=30)
            env = FrameSkipWrapper(env, min_skip=4, max_skip=4, reduce_op=np.max)
        elif non_determinism == "frame-skip":
            env = NoopResetWrapper(env, noop_max=30)
            env = FrameSkipWrapper(env, min_skip=2, max_skip=5, reduce_op=np.max)
        elif non_determinism == "none":
            env = FrameSkipWrapper(env, min_skip=4, max_skip=4, reduce_op=np.max)
        else:
            raise Exception("Invalid non determinism type {}.".format(non_determinism))

        env = ObservationMonitor(env)

        # apply filter
        if args.filter == "none":
            pass
        elif args.filter == "hash":
            env = HashWrapper(env)
        elif args.filter == "hash_time":
            env = HashWrapper(env, use_time=True)
        else:
            raise Exception("Invalid observation filter {}.".format(args.filter))

        env = AtariWrapper(env, width=RES_X, height=RES_Y, grayscale=not USE_COLOR, crop=args.crop_input)

        env = NormalizeRewardWrapper(env, clip=args.reward_clip, initial_state=ENV_NORM_STATE)


    elif env_type == "classic_control":
        #env = NormalizeObservationWrapper(env)
        pass
    else:
        raise Exception("Unsupported env_type {} for env {}".format(env_type, env_name))

    if isinstance(env.action_space, gym.spaces.Box):
        env = DiscretizeActionWrapper(env)

    return env


def prod(X):
    y = 1
    for x in X:
        y *= x
    return y


def trace(s):
    print(s)


def sample_action_from_logp(logp):
    """ Returns integer [0..len(probs)-1] based on log probabilities. """

    # this would probably work
    # u = tf.random_uniform(tf.shape(self.logits), dtype=self.logits.dtype)
    # return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    # todo make this sample directly without exponentiation

    p = np.asarray(np.exp(logp), dtype=np.float64)

    # this shouldn't happen, but sometimes does
    if any(np.isnan(p)):
        raise Exception("Found nans in probabilities", p)

    p /= p.sum()  # probs are sometimes off by a little due to precision error
    return np.random.choice(range(len(p)), p=p)


class PolicyModel(nn.Module):

    def forward(self, x):
        raise NotImplemented()

    def policy(self, x):
        policy, value = self.forward(x)
        return policy

    def value(self, x):
        policy, value = self.forward(x)
        return value

    def set_device_and_dtype(self, device, dtype):

        self.to(device)

        if dtype == torch.half:
            self.half()
        elif dtype == torch.float:
            self.float()
        elif dtype == torch.double:
            self.double()
        else:
            raise Exception("Invalid dtype {} for model.".format(dtype))

        self.device, self.dtype = device, dtype


def get_CNN_output_size(input_size, kernel_sizes, strides, max_pool=False):
    """ Calculates CNN output size, if max_pool is true uses max_pool instead of stride."""
    size = input_size
    for kernel_size, stride in zip(kernel_sizes, strides):

        if max_pool:
            size = (size - (kernel_size - 1) - 1) // 1 + 1
            size = size // stride
        else:
            size = (size - (kernel_size - 1) - 1) // stride + 1
    return size


def add_xy(x):

    n, c, w, h = x.shape
    # from https://gist.github.com/leVirve/0377a8fbac455bfd44e374e5cf8b1260
    xx_channel = torch.arange(w).repeat(1, h, 1)
    yy_channel = torch.arange(h).repeat(1, w, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (w - 1)
    yy_channel = yy_channel.float() / (h - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(n, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(n, 1, 1, 1).transpose(2, 3)

    return torch.cat([
        x,
        xx_channel.type_as(x),
        yy_channel.type_as(x)], dim=1)


class ImprovedCNNModel(PolicyModel):
    """ An improved CNN model that uses 3x3 filters and smaller strides.
    """

    name = "Improved_CNN"

    def __init__(self, input_dims, actions, include_xy=True):

        super(ImprovedCNNModel, self).__init__()



        self.actions = actions
        c, w, h = input_dims

        self.include_xy = include_xy

        if self.include_xy:
            c = c + 2

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        w = get_CNN_output_size(w, [3, 3, 3, 3], [2, 2, 2, 1], max_pool=True)
        h = get_CNN_output_size(h, [3, 3, 3, 3], [2, 2, 2, 1], max_pool=True)

        self.out_shape = (64, w, h)

        self.d = prod(self.out_shape)
        self.fc = nn.Linear(self.d, 512)
        self.fc_policy = nn.Linear(512, actions)
        self.fc_value = nn.Linear(512, 1)

        self.set_device_and_dtype(DEVICE, DTYPE)

    def forward(self, x):
        """ forwards input through model, returns policy and value estimate. """

        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]

        assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
        assert len(x.shape) == 4, "input should be (N,C,W,H)"

        n,c,w,h = x.shape

        x = prep_for_model(x) * (1.0 / 255.0)

        # give filters access to x,y location
        if self.include_xy:
            x = add_xy(x)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv4(x))

        if x.shape[1:] != self.out_shape:
            raise Exception("Invalid output dims. Expected {} found {}.".format(x.shape, self.out_shape))

        x = F.relu(self.fc(x.view(n, self.d)))

        policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)

        return policy, value


class CNNModel(PolicyModel):
    """ Nature paper inspired CNN
    """

    name = "CNN"

    def __init__(self, input_dims, actions):

        super(CNNModel, self).__init__()

        self.actions = actions
        c, w, h = input_dims
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        w = get_CNN_output_size(w, [8, 4, 3], [4, 2, 1])
        h = get_CNN_output_size(h, [8, 4, 3], [4, 2, 1])

        self.out_shape = (64, w, h)

        self.d = prod(self.out_shape)
        self.fc = nn.Linear(self.d, 512)
        self.fc_policy = nn.Linear(512, actions)
        self.fc_value = nn.Linear(512, 1)

        self.set_device_and_dtype(DEVICE, DTYPE)

    def forward(self, x):
        """ forwards input through model, returns policy and value estimate. """

        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]

        assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
        assert len(x.shape) == 4, "input should be (N,C,W,H)"

        n,c,w,h = x.shape

        x = prep_for_model(x) * (1.0/255.0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        assert x.shape[1:] == self.out_shape, "Invalid output dims {} expecting {}".format(x.shape[1:], self.out_shape)

        x = F.relu(self.fc(x.view(n, self.d)))

        policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)

        return policy, value

def entropy(p):
    return -torch.sum(p * p.log2())

def log_entropy(logp):
    """entropy of logits, where logits are in nats."""
    return -(logp.exp() * logp).sum() * (NATS_TO_BITS)


def smooth(X, alpha=0.98):
    y = X[0]
    results = []
    for x in X:
        y = (1 - alpha) * x + (alpha) * y
        results.append(y)
    return results


def safe_mean(X):
    return np.mean(X) if len(X) > 0 else None

def safe_round(x, digits):
    return round(x, digits) if x is not None else x

def inspect(x):
    if isinstance(x, int):
        print("Python interger")
    elif isinstance(x, float):
        print("Python float")
    elif isinstance(x, np.ndarray):
        print("Numpy", x.shape, x.dtype)
    elif isinstance(x, torch.Tensor):
        print("{:<10}{:<25}{:<18}{:<14}".format("torch", str(x.shape), str(x.dtype), str(x.device)))
    else:
        print(type(x))

def nice_display(X, title):
    print("{:<20}{}".format(title, [round(float(x),2) for x in X[:5]]))

def prep_for_model(x):
    return torch.from_numpy(x).to(DEVICE, non_blocking=True).to(dtype=DTYPE)

def train_minibatch(model, optimizer, epsilon, vf_coef, ent_bonus, max_grad_norm,
                    prev_states, actions, returns, policy_logprobs, advantages, values):

    # todo:
    # sample from logps

    policy_logprobs = prep_for_model(policy_logprobs)
    advantages = prep_for_model(advantages)
    returns = prep_for_model(returns)
    old_pred_values = prep_for_model(values)

    mini_batch_size = len(prev_states)

    logps, value_prediction = model.forward(prev_states)

    ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])

    loss_clip = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages))

    # this one is taken from PPO2 baseline, reduces variance but not sure why? does it stop the values from moving
    # too much?
    value_prediction_clipped = old_pred_values + torch.clamp(value_prediction - old_pred_values, -epsilon, +epsilon)

    vf_losses1 = (value_prediction - returns).pow(2)
    vf_losses2 = (value_prediction_clipped - returns).pow(2)
    loss_value = - vf_coef * 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    loss_entropy = ent_bonus * log_entropy(logps) / mini_batch_size

    loss = -(loss_clip + loss_value + loss_entropy)  # gradient ascent.

    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    else:
        grad_norm = None

    optimizer.step()

    return (float(x) for x in [-loss, loss_clip, loss_value, loss_entropy, grad_norm])

class HybridAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """ Async vector env, that limits the number of worker threads spawned """

    def __init__(self, env_fns, max_cpus=8, verbose=False, **kwargs):
        if len(env_fns) <= max_cpus:
            # this is just a standard vec env
            super(HybridAsyncVectorEnv, self).__init__(env_fns, **kwargs)
            self.is_batched = False
        else:
            # create sequential envs for each worker
            assert len(env_fns) % max_cpus == 0, "Number of environments ({}) must be a multiple of the CPU count ({}).".format(len(env_fns), max_cpus)
            self.n_sequential = len(env_fns) // max_cpus
            self.n_parallel = max_cpus
            vec_functions = []
            for i in range(self.n_parallel):
                vec_functions.append(lambda : gym.vector.SyncVectorEnv(env_fns[i*self.n_sequential:(i+1)*self.n_sequential], **kwargs))

            if verbose:
                print("Creating {} cpu workers with {} environments each.".format(self.n_parallel, self.n_sequential))
            super(HybridAsyncVectorEnv, self).__init__(vec_functions, **kwargs)

            self.is_batched = True

    def reset(self):
        if self.is_batched:
            obs = super(HybridAsyncVectorEnv, self).reset()
            return np.reshape(obs, [-1, *obs.shape[2:]])
        else:
            return super(HybridAsyncVectorEnv, self).reset()

    def step(self, actions):
        if self.is_batched:

            # put actions into 2d python array.
            actions = np.reshape(actions, [self.n_parallel, self.n_sequential])
            actions = [list(actions[i]) for i in range(len(actions))]

            observations_list, rewards, dones, infos = super(HybridAsyncVectorEnv, self).step(actions)

            return (
                np.reshape(observations_list, [-1, *observations_list.shape[2:]]),
                np.reshape(rewards, [-1]),
                np.reshape(dones, [-1]),
                np.reshape(infos, [-1])
            )
        else:
            return super(HybridAsyncVectorEnv, self).step(actions)

def run_agents_vec(n_steps, model, vec_envs, states, episode_score, episode_len, score_history, len_history,
               state_shape, state_dtype, policy_shape):
    """
    Runs agents given number of steps, using a single thread, but batching the updates
    :param envs:
    :return:
        N is number of steps per run
        A is number of agents

        batch_prev_state [N, A, (obs dims)]
        ...

    """

    N = n_steps
    A = args.agents

    batch_prev_state = np.zeros([N, A, *state_shape], dtype=state_dtype)
    batch_action = np.zeros([N, A], dtype=np.int32)
    batch_reward = np.zeros([N, A], dtype=np.float32)
    batch_logpolicy = np.zeros([N, A, *policy_shape], dtype=np.float32)
    batch_terminal = np.zeros([N, A], dtype=np.bool)
    batch_value = np.zeros([N, A], dtype=np.float32)

    for t in range(N):

        logprobs, value = model.forward(states)

        logprobs = logprobs.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

        actions = np.asarray([sample_action_from_logp(prob) for prob in logprobs], dtype=np.int32)
        prev_states = states.copy()

        states, rewards, dones, infos = vec_envs.step(actions)

        raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(rewards, infos)], dtype=np.float32)

        # save a copy of the normalization statistics.
        norm_state = infos[0].get("returns_norm_state", None)
        if norm_state is not None:
            global ENV_NORM_STATE
            ENV_NORM_STATE = norm_state

        episode_score += raw_rewards
        episode_len += 1

        for i, done in enumerate(dones):
            if done:
                # reset is handled automatically by vectorized environments
                # so just need to keep track of book-keeping
                score_history.append(episode_score[i])
                len_history.append(episode_len[i])
                episode_score[i] = 0
                episode_len[i] = 0

        batch_prev_state[t] = prev_states
        batch_action[t] = actions
        batch_reward[t] = rewards
        batch_logpolicy[t] = logprobs
        batch_terminal[t] = dones
        batch_value[t] = value

    return (batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal, batch_value)


def with_default(x, default):
    return x if x is not None else default


def compose_frame(state_frame, rendered_frame):
    """ Puts together a composite frame containing rendered frame and state. """

    # note: untested on non-stacked states.

    # assume state is C, W, H
    # assume rendered frame is  is W, H, C
    assert state_frame.shape[0] < max(state_frame.shape), "Channels should be first on state {}".format(state_frame.shape)
    assert rendered_frame.shape[2] < max(state_frame.shape), "Channels should be last on rendered {}".format(
        rendered_frame.shape)

    # ---------------------------------------
    # preprocess frames

    # state was CWH but needs to be WHC
    state_frame = np.swapaxes(state_frame, 0, 2)
    state_frame = np.swapaxes(state_frame, 0, 1)
    # rendered frame is BGR but should be RGB
    rendered_frame = rendered_frame[...,::-1] # get colors around the right way...

    assert rendered_frame.dtype == np.uint8
    assert state_frame.dtype == np.uint8
    assert len(state_frame.shape) == 3
    assert len(rendered_frame.shape) == 3
    assert rendered_frame.shape[2] == 3, "Invalid rendered shape " + str(rendered_frame.shape)

    s_h, s_w, s_c = state_frame.shape
    r_h, r_w, r_c = rendered_frame.shape

    is_stacked = s_c % 4 == 0
    is_color = s_c % 3 == 0

    full_width = r_w + s_w * (2 if is_stacked else 1)
    full_height = max(r_h, s_h * (2 if is_stacked else 1))

    frame = np.zeros([full_height, full_width, 3], dtype=np.uint8)
    frame[:, :, :] += 30 # dark gray background.

    # place the rendered frame
    ofs_y = (full_height - r_h) // 2
    frame[ofs_y:ofs_y+r_h, 0:r_w] = rendered_frame

    # place state frames
    y_pad = (full_height - (s_h * 2)) // 2
    if is_stacked:
        i = 0
        for x in range(2):
            for y in range(2):
                dx = x * s_w + r_w
                dy = y * s_h + y_pad
                factor = 1 if x==0 and y==0 else 2 # darken all but first state for clarity
                if is_color:
                    frame[dy:dy+s_h, dx:dx+s_w, :] = state_frame[:, :, i*3:(i+1)*3] // factor
                else:
                    for c in range(3):
                        frame[dy:dy+s_h, dx:dx+s_w, c] = state_frame[:, :, i] // factor
                i += 1
    else:
        dx = r_w
        dy = y_pad
        if is_color:
            frame[dy:dy+s_h, dx:dx+s_w, :] = state_frame[:, :, :]
        else:
            for c in range(3):
                frame[dy:dy+s_h, dx:dx+s_w, c] = state_frame[:, :, :]

    return frame


def generate_rollouts(num_rollouts, model, env_name, resolution=0.5, max_length=2000, deterministic=False):
    """ Generates roll out with given model and environment name.
        returns observations.
            num_rollouts: Number of rollouts to generate
            model: The model to use
            env_name: Name of the environment
            resolution: Resolution of returned frames
            max_length: Maximum number of environment interactions before rollouts are automatically terminated.
            deterministic: Force a deterministic environment (but not policy)
        :returns
            observations as a list np arrays of dims [c,w,h] in uint8 format.
    """

    env_fns = [lambda : make_environment(env_name, non_determinism="none" if deterministic else "noop") for _ in range(num_rollouts)]
    env = HybridAsyncVectorEnv(env_fns)

    _ = env.reset()
    state, reward, done, info = env.step([0]*num_rollouts)
    rendered_frame = info[0].get("monitor_obs", state)
    w,h,c = rendered_frame.shape
    state = env.reset()

    frames = [[] for _ in range(num_rollouts)]

    is_running = [True] * num_rollouts

    counter = 0

    while any(is_running) and counter < max_length:

        logprobs = model.policy(state).detach().cpu().numpy()
        actions = np.asarray([sample_action_from_logp(prob) for prob in logprobs], dtype=np.int32)

        state, reward, done, info = env.step(actions)

        # append only frames for runs that are still running.
        for i in range(num_rollouts):
            if done[i]:
                is_running[i] = False
            if is_running[i]:
                rendered_frame = info[i].get("monitor_obs", state)
                rendered_frame = rendered_frame.mean(axis=2, dtype=np.float32).astype(np.uint8)  # get a black and white frame.
                if resolution != 1.0:
                    rendered_frame = cv2.resize(rendered_frame, (int(h * resolution), int(w * resolution)),
                                                interpolation=cv2.INTER_AREA)
                frames[i].append(rendered_frame)

        counter += 1

    env.close()

    return [np.asarray(frame_sequence) for frame_sequence in frames]


def evaluate_diversity(step, model, env_name, num_rollouts=8, save_rollouts=True, resolution=0.5):
    """ Generates multiple rollouts of agent, and evaluates the diversity of the rollouts.

    """

    results = []

    # we generate rollouts with the additional determanism turned on. This just removes the no-op starts
    # and gives us a better idea of how similar the runs are.
    rollouts = generate_rollouts(num_rollouts, model, env_name, resolution=resolution, deterministic=True)

    # get all distances between rollouts.
    for i in range(num_rollouts):
        for j in range(i+1, num_rollouts):
            a = rollouts[i][::5] # do comparision at around 3 fps.
            b = rollouts[j][::5]
            difference = dtw(a, b)

            results.append(difference)

    # save the rollouts for later.
    if save_rollouts:
        rollouts_name = get_checkpoint_path(step,"rollouts.dat")
        with open(rollouts_name, 'wb') as f:
            package = {"step":step, "rollouts": rollouts, "distances": results}
            pickle.dump(package, f)

    return results


def export_movie(filename, model, env_name):
    """ Exports a movie of agent playing game.
        which_frames: model, real, or both
    """

    scale = 2

    env = make_environment(env_name)
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("monitor_obs", state)

    # work out our height
    first_frame = compose_frame(state, rendered_frame)
    height, width, channels = first_frame.shape
    width = (width * scale) // 4 * 4 # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder
    video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    state = env.reset()

    # play the game...
    while not done:
        action = sample_action_from_logp(model.policy(state[np.newaxis])[0].detach().cpu().numpy())
        state, reward, done, info = env.step(action)
        rendered_frame = info.get("monitor_obs", state)

        frame = compose_frame(state, rendered_frame)
        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)


        video_out.write(frame)

    video_out.release()


def sig_fig(x, sf=6):
    """ returns x to 6 significant figures if x is a float and small, otherwise returns the input unchanged."""
    if type(x) is float or type(x) is np.float:
        digits = int(math.log10(abs(x)+0.00000000001))
        rounding = sf - digits
        if rounding < 0:
            rounding = 0
        return round(x, rounding)
    else:
        return x

def save_training_log(filename, training_log):
    with open(filename, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["Loss", "Loss_Clip", "Loss_Value", "Loss_Entropy",
                             "Ep_Score (100)", "Ep_Len (100)",
                             "Ep_Score (10)", "Ep_Len (10)",
                             "Elapsed", "Iteration", "Step", "FPS", "Gradient_Norm", "History"])

        for row in training_log:
            # convert values lower precision
            row = [sig_fig(x,sf=4) for x in row]
            csv_writer.writerow(row)

def save_profile_log(filename, timing_log):
    with open(filename, "w") as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["Step", "Rollout_Time", "Train_Time", "Step_Time", "Batch_Size", "FPS", "CUDA_Memory"])
        for row in timing_log:
            csv_writer.writerow(row)

def print_profile_info(timing_log, title="Performance results:"):

    timing_log = np.asarray(timing_log)

    rollout_time = timing_log[:, 1].mean()
    train_time = timing_log[:, 2].mean()
    step_time = timing_log[:, 3].mean()
    fps = timing_log[:, 5].mean()
    fps_std_error = timing_log[:, 5].std(ddof=1) / math.sqrt(len(timing_log))

    print(title+": {:.2f}ms / {:.2f}ms / {:.2f}ms  [{:.0f} FPS +- {:.1f}]".format(
        step_time, rollout_time, train_time, fps, fps_std_error))


def zero_format_number(x):
    return "{:03.0f}M".format(round(x/1e6))

def save_checkpoint(filename, step, model, optimizer, norm_state, logs):
    torch.save({
        'step': step ,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logs': logs,
        'norm_state': norm_state
    }, filename)

def get_checkpoints(path):
    """ Returns list of (epoch, filename) for each checkpoint in current LOG_FOLDER. """
    results = []
    if not os.path.exists(path):
        return []
    for f in os.listdir(path):
        if f.startswith("checkpoint") and f.endswith(".pt"):
            epoch = int(f[11:14])
            results.append((epoch, f))
    results.sort(reverse=True)
    return results

def load_checkpoint(model, optimizer, checkpoint_path):
    """ Restores model from checkpoint. Returns current env_step"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step = checkpoint['step']
    logs = checkpoint['logs']
    norm_state = checkpoint['norm_state']
    return step, logs, norm_state

def lock_job():

    # make sure there isn't another lock
    previous_lock = get_lock()
    if previous_lock is not None and previous_lock["key"] != LOCK_KEY:
        raise Exception("Could not get lock for job, another worker has a lock open.")

    lock = {
        'host': str(HOSTNAME),
        'time': str(time.time()),
        'status': "started",
        'key': str(LOCK_KEY)
    }

    lock_path = os.path.join(LOG_FOLDER, "lock.txt")
    with open(lock_path,"w") as f:
        json.dump(lock, f)

def release_lock():

    assert have_lock(), "Worker does not have lock."

    lock_path = os.path.join(LOG_FOLDER, "lock.txt")
    os.remove(lock_path)

def get_lock():
    """ Gets lock information for this job. """
    lock_path = os.path.join(LOG_FOLDER, "lock.txt")
    if os.path.exists(lock_path):
        return json.load(open(lock_path, "r"))
    else:
        return None

def have_lock():
    """ Returns if we currently have the lock."""
    lock = get_lock()
    return lock is not None and lock["key"] == LOCK_KEY


def train(env_name, model: nn.Module, n_iterations=10*1000, **kwargs):
    """
    Default parameters from stable baselines
    
    https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
    
    gamma             0.99
    n_steps            128
    ent_coef          0.01
    learning_rate   2.5e-4
    vf_coef            0.5
    max_grad_norm      0.5 (not used...)
    lam               0.95
    nminibatches         4
    noptepoch            4
    cliprange          0.1 
    
    atari usually requires ~10M steps 
    
    """

    lock_job()

    # get shapes and dtypes
    _env = make_environment(env_name)
    obs = _env.reset()
    state_shape = obs.shape
    state_dtype = obs.dtype
    policy_shape = model.policy(obs[np.newaxis])[0].shape
    _env.close()

    n_steps = kwargs.get("n_steps",128)  # steps per update (128)
    gamma = kwargs.get("gamma", 0.99)   # discount (0.99)
    lam = kwargs.get("lambda", 0.95)     # GAE parameter (0.95)
    mini_batch_size = kwargs.get("mini_batch_size", 256)
    epsilon = kwargs.get("epsilon", 0.1)
    vf_coef = kwargs.get("vf_coef", 0.5)  # how much loss to take from value function
    agents = kwargs.get("agents", 16)    # (8)
    batch_epochs = kwargs.get("batch_epochs", 4)
    ent_bonus = kwargs.get("ent_bonus", 0.01)
    learning_rate = kwargs.get("learning_rate", 2.5e-4)
    max_grad_norm = kwargs.get("max_grad_norm", 0.5)
    sync_envs = kwargs.get("sync", False)

    # epsilon = 1e-5 is required for stability.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-5)

    # save parameters
    params = {
        "n_steps": n_steps,
        "gamma": gamma,
        "lambda": lam,
        "mini_batch_size": mini_batch_size,
        "epsilon": epsilon,
        "vf_coef": vf_coef,
        "agents": agents,
        "model": model.name,
        "batch_epochs": batch_epochs,
        "ent_bonus": ent_bonus,
        "reward_clip": args.reward_clip,
        "hash_size": args.hash_size,
        "learning_rate": learning_rate,
        "filter": args.filter,
        "max_grad_norm": max_grad_norm,
        "n_iterations": n_iterations,
        "guid": GUID[-8:],
        "hostname": HOSTNAME
    }

    params.update(kwargs)

    batch_size = (n_steps * agents)

    # detect a previous experiment
    checkpoints = get_checkpoints(LOG_FOLDER)
    if len(checkpoints) > 0:
        global ENV_NORM_STATE
        print(bcolors.OKGREEN+"Previous checkpoint detected."+bcolors.ENDC)
        checkpoint_path = os.path.join(LOG_FOLDER, checkpoints[0][1])
        restored_step, logs, ENV_NORM_STATE = load_checkpoint(model, optimizer, checkpoint_path)
        training_log, timing_log, score_history, len_history = logs
        print("  (resumed from step {:.0f}M)".format(restored_step/1000/1000))
        start_iteration = (restored_step // batch_size) + 1
        did_restore = True
    else:
        start_iteration = 0
        did_restore = False

        training_log = []
        timing_log = []
        score_history = []
        len_history = []

    # make a copy of params
    with open(os.path.join(LOG_FOLDER, "params.txt"),"w") as f:
        f.write(json.dumps(params, indent=4))

    # make a copy of training script for reference
    try:
        shutil.copyfile("./ppo.py", os.path.join(LOG_FOLDER, "ppo.py"))
    except Exception as e:
        print("Failed to copy training file to log folder.", e)


    env_fns = [lambda : make_environment(env_name) for _ in range(agents)]
    vec_env = HybridAsyncVectorEnv(env_fns, max_cpus=args.workers, verbose=True) if not sync_envs else gym.vector.SyncVectorEnv(env_fns)

    print("Generated {} agents ({}) using {} ({}) model.".format(agents, "async" if not sync_envs else "sync", model.name, model.dtype))

    if start_iteration == 0 and (args.limit_epochs is None):
        print("Training for {:.1f}M steps".format(n_iterations/1000, n_iterations*batch_size/1000/1000))
    else:
        print("Training block from "+
              bcolors.WARNING+str(round(start_iteration * batch_size / 1000 / 1000))+"M"+bcolors.ENDC+" to ("+
              bcolors.WARNING+str(round(n_iterations * batch_size / 1000 / 1000))+"M"+bcolors.ENDC+
              " / "+str(round(args.epochs))+"M) steps")

    print()
    print("-" * 120)

    # initialize agent
    states = vec_env.reset()

    episode_score = np.zeros([agents], dtype = np.float32)
    episode_len = np.zeros([agents], dtype = np.int32)

    initial_start_time = time.time()

    fps_history = deque(maxlen=10 if not PROFILE_INFO else None)

    checkpoint_every = int(5e6)

    # add a few checkpoints early on

    checkpoints = [x // batch_size for x in range(0, n_iterations*batch_size+1, checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]] #add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    for iteration in range(start_iteration, n_iterations+1):

        env_step = iteration * batch_size

        # the idea here is that all our batch arrays are of dims
        # N, A, ...,
        # Where n is the number of steps, and A is the number of agents.
        # this means we can process each step as a vector

        start_time = time.time()

        # collect experience
        batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal, batch_value = run_agents_vec(
            n_steps, model, vec_env, states, episode_score, episode_len, score_history, len_history,
            state_shape, state_dtype, policy_shape)

        # estimate advantages

        # we calculate the advantages by going backwards..
        # estimated return is the estimated return being in state i
        # this is largely based off https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py
        batch_advantage = np.zeros([n_steps, agents], dtype=np.float32)
        value_next_i = model.value(states).detach().cpu().numpy()
        terminal_next_i = np.asarray([False] * agents)
        prev_adv = np.zeros([agents], dtype=np.float32)

        for i in reversed(range(n_steps)):
            delta = batch_reward[i] + gamma * value_next_i * (1.0-terminal_next_i) - batch_value[i]

            batch_advantage[i] = prev_adv = delta + gamma * lam * (1.0-terminal_next_i) * prev_adv

            value_next_i = batch_value[i]
            terminal_next_i = batch_terminal[i]

        batch_returns = batch_advantage + batch_value

        rollout_time = (time.time() - start_time) / batch_size

        start_train_time = time.time()

        # normalize batch advantages
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_loss = 0
        total_grad_norm = 0

        batch_arrays = [
            np.asarray(batch_prev_state.reshape([batch_size, *state_shape])),
            np.asarray(batch_action.reshape(batch_size)),
            np.asarray(batch_returns.reshape(batch_size)),
            np.asarray(batch_logpolicy.reshape([batch_size, *policy_shape])),
            np.asarray(batch_advantage.reshape(batch_size)),
            np.asarray(batch_value.reshape(batch_size))
        ]

        for i in range(batch_epochs):

            ordering = list(range(batch_size))
            np.random.shuffle(ordering)

            n_batches = math.ceil(batch_size / mini_batch_size)

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * mini_batch_size
                batch_end = (j + 1) * mini_batch_size
                sample = ordering[batch_start:batch_end]

                slices = (x[sample] for x in batch_arrays)

                loss, loss_clip, loss_value, loss_entropy, grad_norm = train_minibatch(
                    model, optimizer, epsilon, vf_coef, ent_bonus, max_grad_norm, *slices)

                total_loss_clip += loss_clip / (batch_epochs*n_batches)
                total_loss_value += loss_value / (batch_epochs*n_batches)
                total_loss_entropy += loss_entropy / (batch_epochs*n_batches)
                total_loss += loss / (batch_epochs*n_batches)
                total_grad_norm += grad_norm / (batch_epochs * n_batches)

        train_time = (time.time() - start_train_time) / batch_size

        step_time = (time.time() - start_time) / batch_size

        fps = 1.0 / (step_time)

        if PROFILE_INFO:

            if "cuda" in DEVICE:
                cuda_memory = torch.cuda.max_memory_allocated()
            else:
                cuda_memory = 0

            timing_log.append((iteration, rollout_time * 1000, train_time * 1000, step_time * 1000, batch_size, fps, cuda_memory/1024/1024))

            # print early timing information from second iteration.
            if iteration == 1:
                print_profile_info(timing_log, "Early timing results")

        fps_history.append(fps)

        history_string = "({:.1f} - {:.1f}) +- {:.2f}".format(
            min(score_history[-100:]), max(score_history[-100:]), np.std(score_history[-100:])
        ) if len(score_history) > 0 else ""

        training_log.append(
            (total_loss,
             total_loss_clip,
             total_loss_value,
             total_loss_entropy,
             safe_round(safe_mean(score_history[-100:]), 2),
             safe_round(safe_mean(len_history[-100:]),2),
             safe_round(safe_mean(score_history[-10:]), 2),
             safe_round(safe_mean(len_history[-10:]),2),
             time.time()-initial_start_time,
             iteration,
             env_step,
             int(np.mean(fps_history)),
             total_grad_norm,
             history_string
             )
        )

        if True:
            # save current step information.
            details = {}
            details["max_epochs"] = args.epochs
            details["completed_epochs"] = env_step / 1e6
            details["score"] = np.percentile(score_history, 95) if len(score_history) > 0 else None
            details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
            details["fps"] = int(np.mean(fps_history))
            frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
            details["eta"] = frames_remaining / details["fps"]
            details["host"] = HOSTNAME
            details["last_modified"] = time.time()
            with open(os.path.join(LOG_FOLDER, "progress.txt"),"w") as f:
                json.dump(details, f)

        if PRINT_EVERY:
            if iteration % (PRINT_EVERY * 10) == 0:
                print("{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>8}".format("iter", "step", "loss", "l_clip", "l_value",
                                                                      "l_ent", "ep_score", "ep_len", "elapsed", "fps"))
                print("-"*120)
            if iteration % PRINT_EVERY == 0 or iteration == n_iterations:
                print("{:>8}{:>8}{:>10.3f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.2f}{:>10.0f}{:>10}{:>8.0f} {:<10}".format(
                    str(iteration),
                    "{:.2f}M".format(iteration * n_steps * agents / 1000 / 1000),
                    training_log[-1][0],
                    training_log[-1][1],
                    training_log[-1][2],
                    training_log[-1][3],
                    with_default(training_log[-1][4], 0),
                    with_default(training_log[-1][5], 0),
                    "{:.0f} min".format(training_log[-1][8]/60),
                    training_log[-1][11],
                    with_default(training_log[-1][13], 0)
                ))

        # make sure we don't save the checkpoint we just restored from.
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            print()
            print(bcolors.OKGREEN+"Checkpoint: {}".format(LOG_FOLDER)+bcolors.ENDC)

            if args.save_checkpoints:
                checkpoint_name = get_checkpoint_path(env_step, "params.pt")
                logs = (training_log, timing_log, score_history, len_history)
                save_checkpoint(checkpoint_name, env_step, model, optimizer, ENV_NORM_STATE, logs)
                print("  -checkpoint saved")

            if args.export_video:
                video_name  = get_checkpoint_path(env_step, env_name+".mp4")
                export_movie(video_name, model, env_name)
                print("  -video exported")

            if args.evaluate_diversity:
                diversity = evaluate_diversity(env_step, model, env_name, save_rollouts=True)
                print("  -diversity of rollouts - mean={:.1f}k ({:.1f}k-{:.1f}k)".format(np.mean(diversity) / 1000,
                                                                                          np.min(diversity) / 1000,
                                                                                          np.max(diversity) / 1000))

            print()

        if iteration in [5, 10, 20, 30, 40] or iteration % SAVE_LOG_EVERY == 0 or iteration == n_iterations:

            save_training_log(os.path.join(LOG_FOLDER, "training_log.csv"), training_log)

            clean_training_log = training_log[10:] if len(training_log) >= 10 else training_log  # first sample is usually extreme.

            xs = [x[10] for x in clean_training_log]
            plt.figure(figsize=(8, 8))
            plt.grid()

            labels = ["loss", "loss_clip", "loss_value", "loss_entropy"]
            ys = [[x[i] for x in clean_training_log] for i in range(4)]
            colors = ["red", "green", "blue", "black"]

            for label, y, c in zip(labels, ys, colors):
                plt.plot(xs, y, alpha=0.2, c=c)
                plt.plot(xs, smooth(y), label=label, c=c)

            plt.legend()
            plt.ylabel("Loss")
            plt.xlabel("Env Step")
            plt.savefig(os.path.join(LOG_FOLDER, "losses.png"))
            plt.close()

            xs = []
            rewards = []
            lengths = []
            rewards10 = []
            lengths10 = []
            for i, x in enumerate(clean_training_log):
                if x[4] is None:
                    continue
                xs.append(x[10])
                rewards.append(x[4])
                lengths.append(x[5])
                rewards10.append(x[6])
                lengths10.append(x[7])

            if len(rewards) > 10:
                plt.figure(figsize=(8, 8))
                plt.grid()
                plt.plot(xs, rewards10, alpha=0.2)
                plt.plot(xs, rewards)
                plt.ylabel("Reward")
                plt.xlabel("Env Step")
                plt.savefig(os.path.join(LOG_FOLDER, "ep_reward.png"))
                plt.close()

                plt.figure(figsize=(8, 8))
                plt.grid()
                plt.plot(xs, lengths10, alpha=0.2)
                plt.plot(xs, lengths)
                plt.ylabel("Episode Length")
                plt.xlabel("Env Step")
                plt.savefig(os.path.join(LOG_FOLDER, "ep_length.png"))
                plt.close()

    if PROFILE_INFO:
        print_profile_info(timing_log, "Final timing results")
        save_profile_log(os.path.join(LOG_FOLDER, "timing_info.csv"), timing_log)

    # save a final score.
    if args.limit_epochs is None:
        # only write final score once we finish the last epoch.
        with open(os.path.join(LOG_FOLDER, "final_score.txt"), "w") as f:
            f.write(str(np.percentile(score_history,95)))

    release_lock()

    return training_log


def get_previous_experiment_guid(experiment_path, run_name):
    """ Looks for previous experiment with same run_name. Returns the guid if found. """
    if not os.path.exists(experiment_path):
        return None
    for f in os.listdir(experiment_path):
        if f[:-19] == run_name:
            guid = f[-17:-1]
            return guid
    return None


def run_experiment(experiment_name, run_name, env_name, model, n_iterations = 10000, **kwargs):

    global LOG_FOLDER
    global GUID

    if args.restore:
        # look for a previous experiment and use it if we find one...
        guid = get_previous_experiment_guid(os.path.join(OUTPUT_FOLDER, experiment_name), run_name)
        if guid is None:
            print(bcolors.FAIL+"Could not restore experiment {}:{}. Previous run not found.".format(experiment_name, run_name)+bcolors.ENDC)
        else:
            GUID = guid

    LOG_FOLDER = "{}/{}/{} [{}]".format(OUTPUT_FOLDER, experiment_name, run_name, GUID[-16:])

    print("Logging to folder", LOG_FOLDER)
    os.makedirs(LOG_FOLDER, exist_ok=True)

    env = make_environment(env_name)
    n_actions = env.action_space.n
    obs_space = env.observation_space.shape

    print("Playing {} with {} obs_space and {} actions.".format(env_name, obs_space, n_actions))

    train(env_name, model(obs_space, n_actions), n_iterations, **kwargs)


def set_default(dict, key, value):
    if key not in dict or dict[key] is None:
        dict[key] = value

def get_environment_name(environment, sticky_actions=False):
    environment = environment.capitalize()
    return "{}NoFrameskip-v{}".format(environment, "0" if sticky_actions else "4")

def set_num_frames(frames):
    set_default(exp_args, "n_iterations", round(int(frames) / (args.agents * args.n_steps)))

if __name__ == "__main__":

    exp_args = vars(args)

    if args.device.lower() == "auto":
        DEVICE = get_auto_device()
    else:
        DEVICE = args.device

    if args.workers < 0:
        args.workers = multiprocessing.cpu_count()
        while args.agents % args.workers != 0:
            # make sure workers divides number of jobs.
            args.workers -= 1

    args.resolution = args.resolution.lower()

    if args.resolution == "full":
        RES_X, RES_Y = 210, 160
    elif args.resolution == "high":
        RES_X, RES_Y = 128, 128
    elif args.resolution == "half-full":
        RES_X, RES_Y = 105, 80
    elif args.resolution == "standard":
        RES_X, RES_Y = 84, 84
    elif args.resolution == "half":
        RES_X, RES_Y = 42, 42
    else:
        raise Exception("Invalid resolution "+args.resolution)

    # get model
    if args.model.lower() == "cnn":
        args.model = CNNModel
    elif args.model.lower() == "improved_cnn":
        args.model = ImprovedCNNModel
    else:
        raise Exception("Invalid model name '{}', please use [cnn, improved_cnn]".format(args.model))

    # special environments
    if args.environment == "benchmark":
        set_default(exp_args, "n_iterations", 10)
        set_default(exp_args, "env_name", "AlienNoFrameskip-v4")
        set_default(exp_args, "model", CNNModel)
        PROFILE_INFO = True
        PRINT_EVERY = 0
    else:
        args.env_name = get_environment_name(args.environment, args.sticky_actions)
        if args.limit_epochs is not None:
            if args.limit_epochs >= args.epochs:
                # turn limit epochs if it is too high.
                args.limit_epochs = None
            else:
                set_num_frames(args.limit_epochs * 1e6)
        else:
            set_num_frames(args.epochs * 1e6)

    if args.output_folder is not None:
        print("Outputting to folder", args.output_folder)
        assert os.path.isdir(args.output_folder), "Can not find path "+args.output_folder
        OUTPUT_FOLDER = args.output_folder

    USE_COLOR = args.color

    set_default(exp_args, "experiment_name", exp_args["env_name"])

    show_cuda_info()

    run_experiment(**exp_args)

