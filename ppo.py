import os
import sys
import argparse
# using more than 2 threads locks up all the cpus and does not seem to improve performance.
# the gain from 2 CPUs to 1 is very minor too. (~10%)


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

    parser.add_argument("experiment")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=128)
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--sync", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--resolution", type=str, default="standard", help="['full', 'standard', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--export_video", type=str2bool, default=False)
    parser.add_argument("--run_name", type=str, default="experiments")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--output_folder", type=str)

    return parser

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

torch.set_num_threads(int(threads))

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    env_type = env.entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

GUID = uuid.uuid4().hex
LOG_FOLDER = "/run/experiment [{}]".format(GUID[-8:])
NATS_TO_BITS = 1.0/math.log(2)
RES_X = 84
RES_Y = 84
USE_COLOR = False
OUTPUT_FOLDER = "runs"
DTYPE = torch.float
PROFILE_INFO = False
PRINT_EVERY = 10
VERBOSE = True

def get_auto_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"

def show_cuda_info():

    global DEVICE
    print("Device:", DEVICE)
    if DEVICE == "cuda":
        device_id = torch.cuda.current_device()
        print(torch.cuda.get_device_name(device_id))

"""
------------------------------------------------------------------------------------------------------------------------
    Wrappers
------------------------------------------------------------------------------------------------------------------------
"""


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

class NormalizeObservationWrapper(gym.Wrapper):
    def __init__(self, env, clip=5.0):
        """
        Normalize and clip observations.
        """
        super().__init__(env)
        self.clip = clip
        self._n = 1000
        self._update_every = 100
        self.epsilon = 0.00001
        self.counter = 0

        self.observation_space = gym.spaces.Box(
            low = -clip,
            high = clip,
            shape = env.observation_space.shape,
            dtype=np.float32
        )

        self.history = deque(maxlen=self._n)
        self.env = env

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.history.append(obs)

        #note: this will be slow for large observation spaces, would be better to do a running average.
        if (self.counter % self._update_every) == 0:
            self.means = np.mean(np.asarray(self.history), axis=0)
            self.stds = np.std(np.asarray(self.history), axis=0)

        obs = np.asarray(obs, dtype=np.float32)
        obs = np.clip((obs - self.means) / (self.stds + self.epsilon), -self.clip, +self.clip)

        self.counter += 1

        return obs, reward, done, info


class NormalizeRewardWrapper(gym.Wrapper):

    def __init__(self, env, clip=2.0):
        """
        Normalizes returns
        """
        super().__init__(env)

        self.env = env
        self._n = 10000
        self._update_every = 100
        self.history = deque(maxlen=self._n)
        self.clip = clip
        self.epsilon = 0.00001
        self.counter = 0
        self.current_return = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.current_return = self.current_return * 0.99 + reward

        self.history.append(self.current_return)

        # note: would be better to switch to a running mean / std
        if (self.counter < 100) or (self.counter % (self._update_every)) == 0:
            self.mean = np.mean(self.history, axis=0)
            self.std = np.std(self.history, axis=0)

        last_raw_reward = reward
        reward = np.clip(reward / (self.std + self.epsilon), -self.clip, +self.clip)

        info["raw_reward"] = last_raw_reward

        self.counter += 1

        return obs, reward, done, info

    def reset(self):
        self.current_return = 0
        return self.env.reset()


class AtariWrapper(gym.Wrapper):

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
        info["raw_obs"] = obs
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


def make_environment(env_name):
    """ Construct environment of given name, including any required """

    env_type = None

    for k,v in _game_envs.items():
        if env_name in v:
            env_type = k

    env = gym.make(env_name)
    if env_type == "atari":
        assert "NoFrameskip" in env_name
        env = NoopResetWrapper(env, noop_max=30)
        env = FrameSkipWrapper(env, min_skip=4, max_skip=4, reduce_op=np.max)
        env = AtariWrapper(env, width=RES_X, height=RES_Y, grayscale=not USE_COLOR)
        env = NormalizeRewardWrapper(env)
    elif env_type == "classic_control":
        env = NormalizeObservationWrapper(env)
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


class MLPModel(nn.Module):
    """ A very simple Multi Layer Perceptron """

    def __init__(self, input_dims, actions):
        super(MLPModel, self).__init__()
        self.actions = actions
        self.d = prod(input_dims)
        self.fc1 = nn.Linear(self.d, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_policy = nn.Linear(64, actions)
        self.fc_value = nn.Linear(64, 1)
        self.to(DEVICE)

    def forward(self, x):
        x = torch.from_numpy(x).to(DEVICE).float() / 255.0
        x = x.reshape(-1, self.d)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x

    def policy(self, x):
        """ Returns logprobs"""
        x = self.forward(x)
        x = self.fc_policy(x)
        x = F.log_softmax(x, dim=1)
        return x

    def value(self, x):
        x = self.forward(x)
        x = self.fc_value(x).squeeze(dim=1)
        return x


class CNNModel(nn.Module):
    """ Nature paper inspired CNN """

    def __init__(self, input_dims, actions):
        super(CNNModel, self).__init__()
        self.actions = actions
        c, w, h = input_dims
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        w = (w - (8-1) - 1) // 4 + 1
        w = (w - (4-1) - 1) // 2 + 1
        w = (w - (3-1) - 1) // 1 + 1

        h = (h - (8-1) - 1) // 4 + 1
        h = (h - (4-1) - 1) // 2 + 1
        h = (h - (3-1) - 1) // 1 + 1

        self.out_shape = (64, w, h)

        self.d = prod(self.out_shape)
        self.fc = nn.Linear(self.d, 512)
        self.fc_policy = nn.Linear(512, actions)
        self.fc_value = nn.Linear(512, 1)
        self.to(DEVICE)

        if DTYPE == torch.half:
            self.half()
        elif DTYPE == torch.float:
            self.float()
        elif DTYPE == torch.double:
            self.double()
        else:
            raise Exception("Invalid dtype {} for model.".format(DTYPE))

    def forward(self, x):

        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]

        assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
        assert len(x.shape) == 4, "input should be (N,C,W,H)"

        n = x.shape[0]

        x = prep_for_model(x) * (1.0/255.0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.fc(x.view(n, self.d)))
        return x

    def policy(self, x):
        x = self.forward(x)
        x = self.fc_policy(x)
        x = F.log_softmax(x, dim=1)
        return x

    def value(self, x):
        x = self.forward(x)
        x = self.fc_value(x).squeeze(dim=1)
        return x


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

    logps = model.policy(prev_states)

    ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])

    loss_clip = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages))

    # people do this in different ways, I'm going to go for huber loss.
    #td = model.value(prev_states) - returns
    #loss_value = - vf_coef * torch.mean(td * td)
    #loss_value = - vf_coef * torch.mean(torch.min(td * td, torch.abs(td)))

    # this one is taken from PPO2 baseline, reduces variance but not sure why? does it stop the values from moving
    # too much?
    value_prediction = model.value(prev_states)
    value_prediction_clipped = old_pred_values + torch.clamp(value_prediction - old_pred_values, -epsilon, +epsilon)

    vf_losses1 = (value_prediction - returns).pow(2)
    vf_losses2 = (value_prediction_clipped - returns).pow(2)
    loss_value = - vf_coef * 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    loss_entropy = ent_bonus * log_entropy(logps) / mini_batch_size

    loss = -(loss_clip + loss_value + loss_entropy)  # gradient ascent.

    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None:
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    optimizer.step()

    return (float(x) for x in [-loss, loss_clip, loss_value, loss_entropy])

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
    A = vec_envs.num_envs

    batch_prev_state = np.zeros([N, A, *state_shape], dtype=state_dtype)
    batch_action = np.zeros([N, A], dtype=np.int32)
    batch_reward = np.zeros([N, A], dtype=np.float32)
    batch_logpolicy = np.zeros([N, A, *policy_shape], dtype=np.float32)
    batch_terminal = np.zeros([N, A], dtype=np.bool)

    for t in range(N):

        logprobs = model.policy(states).detach().cpu().numpy()
        actions = np.asarray([sample_action_from_logp(prob) for prob in logprobs], dtype=np.int32)
        prev_states = states.copy()

        states, rewards, dones, infos = vec_envs.step(actions)

        raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(rewards, infos)], dtype=np.float32)

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

    return (batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal)


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


def export_movie(model, env_name, filename):
    """ Exports a movie of agent playing game.
        which_frames: model, real, or both
    """

    filename += ".mp4"
    scale = 2

    env = make_environment(env_name)
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("raw_obs", state)

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
        rendered_frame = info.get("raw_obs", state)

        frame = compose_frame(state, rendered_frame)
        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format((width, height, 3), frame.shape)


        video_out.write(frame)

    video_out.release()


def save_training_log(filename, training_log):
    with open(filename, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["Loss", "Loss_Clip", "Loss_Value", "Loss_Entropy",
                             "Ep_Score (100)", "Ep_Len (100)",
                             "Ep_Score (10)", "Ep_Len (10)",
                             "Elapsed", "Iteration", "Step", "FPS", "History"])

        for row in training_log:
            csv_writer.writerow(row)

def zero_format_number(x):
    if x < 1e3:
        return "{:03.0f} ".format(x)
    elif x < 1e6:
        return "{:03.0f}K".format(x//1e3)
    elif x < 1e9:
        return "{:03.0f}M".format(x//1e6)
    else:
        return "{:03.0f}B".format(x//1e9)

def mma(X):
    """ returns string containing min, max, average, etc... """
    return "{2:.2f} [{0:.2f}-{1:.2f}] (std={3:.2f})".format(np.min(X), np.max(X), np.mean(X), np.std(X))


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
    n_batches = kwargs.get("n_batches", 4)
    epsilon = kwargs.get("epsilon", 0.1)
    vf_coef = kwargs.get("vf_coef", 0.5)  # how much loss to take from value function
    agents = kwargs.get("agents", 16)    # (8)
    epochs = kwargs.get("epochs", 4)
    ent_bonus = kwargs.get("ent_bonus", 0.01)
    alpha = kwargs.get("lr", 2.5e-4)
    max_grad_norm = kwargs.get("max_grad_norm", 0.5)
    sync_envs = kwargs.get("sync", False)

    # save parameters
    params = {
        "n_steps": n_steps,
        "gamma": gamma,
        "lambda": lam,
        "n_batches": n_batches,
        "epsilon": epsilon,
        "vf_coef": vf_coef,
        "agents": agents,
        "epochs": epochs,
        "ent_bouns": ent_bonus,
        "lr": alpha,
        "max_grad_norm": max_grad_norm
    }

    params.update(kwargs)

    # make a copy of params
    with open(os.path.join(LOG_FOLDER, "params.txt"),"w") as f:
        f.write(json.dumps(params, indent=4))

    # make a copy of training script for reference
    try:
        shutil.copyfile("./ppo.py", os.path.join(LOG_FOLDER, "ppo.py"))
    except Exception as e:
        print("Failed to copy training file to log folder.", e)

    batch_size = (n_steps * agents)
    mini_batch_size = batch_size // n_batches

    # epsilon = 1e-5 is required for stability.
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha, eps=1e-5)

    env_fns = [lambda : make_environment(env_name) for _ in range(agents)]
    vec_env = gym.vector.AsyncVectorEnv(env_fns) if not sync_envs  else gym.vector.SyncVectorEnv(env_fns)

    print("Generated {} agents [{}]".format(agents, "async" if not sync_envs else "sync"))

    print("Training for {:.1f}k iterations {:.2f}M steps".format(n_iterations/1000, n_iterations*batch_size/1000/1000))

    # initialize agent
    states = vec_env.reset()

    episode_score = np.zeros([agents], dtype = np.float32)
    episode_len = np.zeros([agents], dtype = np.int32)

    training_log = []
    timing_log = []

    score_history = []
    len_history = []


    initial_start_time = time.time()

    fps_history = deque(maxlen=10)

    for step in range(n_iterations+1):

        # the idea here is that all our batch arrays are of dims
        # N, A, ...,
        # Where n is the numer of steps, and A is the number of agents.
        # this means we can process each step as a vector

        start_time = time.time()

        # collect experience
        batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal = run_agents_vec(
            n_steps, model, vec_env, states, episode_score, episode_len, score_history, len_history,
            state_shape, state_dtype, policy_shape)

        # estimate values
        # note: we need to manipulate the dims into a batch first.
        batch_value = model.value(batch_prev_state.reshape([batch_size, *state_shape])).detach().cpu().numpy()
        batch_value = batch_value.reshape([n_steps, agents])
        batch_advantage = np.zeros([n_steps, agents], dtype=np.float32)

        # we calculate the advantages by going backwards..
        # estimated return is the estimated return being in state i
        # this is largely based off https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py
        value_next_i = model.value(states).detach().cpu().numpy()
        terminal_next_i = np.asarray([False] * agents)
        prev_adv = np.zeros([agents], dtype=np.float32)

        
        for i in reversed(range(n_steps)):
            delta = batch_reward[i] + gamma * value_next_i * (1.0-terminal_next_i) - batch_value[i]

            batch_advantage[i] = prev_adv = delta + gamma * lam * (1.0-terminal_next_i) * prev_adv

            value_next_i = batch_value[i]
            terminal_next_i = batch_terminal[i]

        batch_returns = batch_advantage + batch_value

        rollout_time = (time.time() - start_time)

        start_train_time = time.time()

        movie_every = 5*1000*1000 // batch_size # 1 movie every 5m environment steps

        # normalize batch advantages
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_loss = 0

        batch_arrays = [
            np.asarray(batch_prev_state.reshape([batch_size, *state_shape])),
            np.asarray(batch_action.reshape(batch_size)),
            np.asarray(batch_returns.reshape(batch_size)),
            np.asarray(batch_logpolicy.reshape([batch_size, *policy_shape])),
            np.asarray(batch_advantage.reshape(batch_size)),
            np.asarray(batch_value.reshape(batch_size))
        ]

        for i in range(epochs):

            ordering = list(range(batch_size))
            np.random.shuffle(ordering)

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * mini_batch_size
                batch_end = (j + 1) * mini_batch_size
                sample = ordering[batch_start:batch_end]

                slices = (x[sample] for x in batch_arrays)

                loss, loss_clip, loss_value, loss_entropy = train_minibatch(
                    model, optimizer, epsilon, vf_coef, ent_bonus, max_grad_norm, *slices)

                total_loss_clip += loss_clip / (epochs*n_batches)
                total_loss_value += loss_value / (epochs*n_batches)
                total_loss_entropy += loss_entropy / (epochs*n_batches)
                total_loss += loss / (epochs*n_batches)

        train_time = (time.time() - start_train_time)

        step_time = (time.time() - start_time)

        fps = 1.0 / (step_time / batch_size)

        if PROFILE_INFO:

            if "cuda" in DEVICE:
                cuda_memory = torch.cuda.max_memory_allocated()
            else:
                cuda_memory = 0

            timing_log.append((step, rollout_time, train_time, step_time, batch_size, fps, cuda_memory/1024/1024))

        fps_history.append(fps)

        history_string = "{}".format(
            [round(float(x), 2) for x in score_history[-5:]]
        )

        training_log.append(
            (float(total_loss),
             float(total_loss_clip),
             float(total_loss_value),
             float(total_loss_entropy),
             safe_round(safe_mean(score_history[-100:]), 2),
             safe_round(safe_mean(len_history[-100:]),2),
             safe_round(safe_mean(score_history[-10:]), 2),
             safe_round(safe_mean(len_history[-10:]),2),
             time.time()-initial_start_time,
             step,
             step * batch_size,
             int(np.mean(fps_history)),
             history_string
             )
        )

        if PRINT_EVERY and step % (PRINT_EVERY * 10) == 0:
            print("{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>8}".format("iter", "step", "loss", "l_clip", "l_value",
                                                                      "l_ent", "ep_score", "ep_len", "elapsed", "fps"))
            print("-"*120)
        if PRINT_EVERY and step % PRINT_EVERY == 0 or step == n_iterations:
            print("{:>8}{:>8}{:>10.3f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.2f}{:>10.0f}{:>10}{:>8.0f} {:<10}".format(
                str(step),
                "{:.2f}M".format(step * n_steps * agents / 1000 / 1000),
                training_log[-1][0],
                training_log[-1][1],
                training_log[-1][2],
                training_log[-1][3],
                with_default(training_log[-1][4], 0),
                with_default(training_log[-1][5], 0),
                "{:.0f} min".format(training_log[-1][8]/60),
                training_log[-1][11],
                with_default(training_log[-1][12], 0)
            ))

        if args.export_video and (step in [math.ceil(x / batch_size) for x in [0, 100*1000, 1000*1000]] or step % movie_every == 0):
            export_movie(model, env_name, "{}_{}".format(os.path.join(LOG_FOLDER, env_name), zero_format_number(step*batch_size)))

        if step in [10, 20, 30, 40] or step % 50 == 0:

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

        step_times = np.asarray([step_time for step, rollout_time, train_time, step_time, batch_size, FPS in timing_log]) / batch_size * 1000
        rollout_times = np.asarray([rollout_time for step, rollout_time, train_time, step_time, batch_size, FPS in timing_log]) / batch_size * 1000
        train_times = np.asarray([train_time for step, rollout_time, train_time, step_time, batch_size, FPS in timing_log]) / batch_size * 1000

        print("Average timings: {:.2f}ms /{:.2f}ms /{:.2f}ms".format(*(np.mean(x) for x in [step_times, rollout_times, train_times])))

        with open(os.path.join(LOG_FOLDER, "timing_info.csv"), "w") as f:
            csv_writer = csv.writer(f, delimiter=',')
            csv_writer.writerow(["Step", "Rollout", "Train", "Step", "Batch_Size", "FPS", "CUDA_Memory"])
            for row in timing_log:
                csv_writer.writerow(row)

    return training_log


def run_experiment(run_name, experiment_name, env_name, Model, n_iterations = 10000, **kwargs):

    global LOG_FOLDER
    LOG_FOLDER = "{}/{}/{} [{}]".format(OUTPUT_FOLDER, run_name, experiment_name, GUID[-16:])

    print("Logging to folder", LOG_FOLDER)
    os.makedirs(LOG_FOLDER, exist_ok=True)

    env = make_environment(env_name)
    n_actions = env.action_space.n
    obs_space = env.observation_space.shape

    print("Playing {} with {} obs_space and {} actions.".format(env_name, obs_space, n_actions))

    model = Model(obs_space, n_actions)
    train(env_name, model, n_iterations, **kwargs)


def set_default(dict, key, value):
    if key not in dict:
        dict[key] = value


def set_num_frames(frames):
    set_default(exp_args, "n_iterations", int(frames) // (args.agents * args.n_steps))

if __name__ == "__main__":


    exp_args = vars(args)

    if args.device.lower() == "auto":
        DEVICE = get_auto_device()
    else:
        DEVICE = args.device

    show_cuda_info()

    experiment = args.experiment.lower()

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

    if args.output_folder is not None:
        print("outputting to folder", args.output_folder)
        assert os.path.isdir(args.output_folder), "Can not find path "+args.output_folder
        OUTPUT_FOLDER = args.output_folder

    USE_COLOR = args.color

    # debugging experiments

    if experiment == "pong_small":
        set_default(exp_args, "n_iterations", 10)
        set_default(exp_args, "env_name", "PongNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "pong_mlp":
        set_default(exp_args, "n_iterations", 10)
        set_default(exp_args, "env_name", "PongNoFrameskip-v4")
        set_default(exp_args, "Model", MLPModel)
    elif experiment == "benchmark":
        set_default(exp_args, "n_iterations", 10)
        set_default(exp_args, "env_name", "AlienNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
        PROFILE_INFO = True
        PRINT_EVERY = 0

    # atari games

    elif experiment == "pong":
        set_num_frames(2e8)
        set_default(exp_args, "env_name", "PongNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "seaquest":
        set_num_frames(2e8)
        set_default(exp_args, "env_name", "SeaquestNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "alien":
        set_num_frames(2e8)
        set_default(exp_args, "env_name", "AlienNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "alien_50":
        set_num_frames(5e7)
        set_default(exp_args, "env_name", "AlienNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "breakout":
        set_num_frames(2e8)
        set_default(exp_args, "env_name", "BreakoutNoFrameskip-v4")
        set_default(exp_args, "Model", CNNModel)
    elif experiment == "cartpole":
        set_default(exp_args, "n_iterations", 2*1000)
        set_default(exp_args, "vf_coef", 0.0001)
        set_default(exp_args, "env_name", "CartPole-v0")
        set_default(exp_args, "Model", MLPModel)
    else:
        raise Exception("Invalid experiment {}.".format(experiment))

    set_default(exp_args, "run_name", exp_args["env_name"])

    run_experiment(**exp_args)

