# limit to 4 threads...
import os
os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

"""
Runs a single evaluation on given file
"""

import argparse

import json
import pickle
import time
import sys
import socket
import gzip

from typing import Union, List, Dict

import cv2
import numpy as np
import torch
import os
import traceback
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.colors
import numpy.random
from gym.vector import sync_vector_env
from rl import atari, config, utils, models, hybridVecEnv, rollout
from rl.config import args
import lz4.frame as lib
import os

DEVICE = "cuda:0"
REWARD_SCALE = float()
CURRENT_HORIZON = int()
PARALLEL_ENVS = 20 # number of environments to run in parallel
TEMP_LOCATION = os.path.expanduser("~/.cache/")

GENERATE_EVAL = False
GENERATE_MOVIES = True

WORKERS = 10

class CompressedStack():
    """
    A stack of nd arrays that are stored with compression

    gzip-0 = 0.9ms
    gzip-2 = 0.7ms
    gzip-5 = 2.3ms
    gzip-9 = 6.0ms

    zlib-0 = 0.3ms # uncompressed?
    zlib-1 = 0.8ms
    zlib-5 = 1.5ms

    lz4

    """


    def __init__(self):
        self.buffer = []
        self._uncompressed_size = 0
        self._compressed_size = 0
        self._compression_time = None

    def append(self, x:np.ndarray):
        start_time = time.time()
        compressed_data = lib.compress(x.tobytes())
        self.buffer.append((x.dtype, x.shape, compressed_data))
        self._uncompressed_size += x.nbytes
        self._compressed_size += len(compressed_data)
        self._compression_time = self._compression_time or (time.time()-start_time)
        self._compression_time = 0.999 * self._compression_time + 0.001 * (time.time()-start_time)

    @property
    def ratio(self):
        return self._compressed_size / self._uncompressed_size

    def get(self, index):
        dtype, shape, data = self.buffer[index]
        return np.frombuffer(lib.decompress(data), dtype=dtype).reshape(shape)


def backup_open_checkpoint(checkpoint_path: str, **tf_args):
    # todo: remove this and use rollout._open_checkpoint
    try:
        with gzip.open(os.path.join(checkpoint_path, ".gz"), 'rb') as f:
            return torch.load(f, **tf_args)
    except:
        pass

    try:
        # unfortunately some checkpoints were saved without the .gz so just try and fail to load them...
        with gzip.open(checkpoint_path, 'rb') as f:
            return torch.load(f, **tf_args)
    except:
        pass

    try:
        # unfortunately some checkpoints were saved without the .gz so just try and fail to load them...
        with open(checkpoint_path, 'rb') as f:
            return torch.load(f, **tf_args)
    except:
        pass

    raise Exception(f"Could not open checkpoint {checkpoint_path}")


def needs_rediscount():
    return args.tvf_gamma != args.gamma and args.use_tvf


def load_args(checkpoint_path):
    """
    Load config arguments from a checkpoint_path
    """
    args_path = os.path.join(os.path.split(checkpoint_path)[0], "params.txt")
    with open(args_path, 'r') as f:
        checkpoint_args = json.load(f)
        for k, v in checkpoint_args.items():
            vars(args)[k] = v
        args.log_folder = ''
        args.terminal_on_loss_of_life = False # always off for evaluation...

# load a model and evaluate performance
def load_checkpoint(checkpoint_path, device=None):
    """ Restores model from checkpoint. Returns current env_step"""

    global REWARD_SCALE
    global CURRENT_HORIZON

    load_args(checkpoint_path)

    args.env_name = utils.get_environment_name(args.environment, args.sticky_actions)
    args.res_x, args.res_y = (84, 84)

    args.experiment_name = Path(os.path.join(os.path.join(os.getcwd(), checkpoint_path))).parts[-3]

    # fix up frameskip, this happens for older versions of the code.
    if args.frame_skip == 0:
        args.frame_skip = 4

    env = atari.make(monitor_video=True)

    model = make_model(env)

    # some older versions might not have the open_checkpoint function... :(
    try:
        oc = rollout._open_checkpoint
    except:
        oc = backup_open_checkpoint
    checkpoint = oc(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    env_state = checkpoint["env_state"]
    CURRENT_HORIZON = checkpoint.get("current_horizon", 0)

    if "VecNormalizeRewardWrapper" in env_state:
        # the new way
        REWARD_SCALE = env_state['VecNormalizeRewardWrapper']['ret_rms'][1] ** 0.5
    else:
        # the old way
        atari.ENV_STATE = env_state
        REWARD_SCALE = env_state['returns_norm_state'][1] ** 0.5

    return model

def make_model(env):

    import inspect
    allowed_args = set(inspect.signature(models.TVFModel.__init__).parameters.keys())

    additional_args = {}

    additional_args['use_rnd'] = args.use_rnd
    additional_args['use_rnn'] = False

    try:
        additional_args['tvf_horizon_transform'] = rollout.horizon_scale_function
    except:
        # this was the old method
        additional_args['tvf_horizon_transform'] = lambda x: x / args.tvf_max_horizon

    try:
        network = args.network
    except:
        network = "nature"

    try:
        additional_args['tvf_time_transform'] = rollout.time_scale_function
    except:
        pass

    try:
        additional_args['tvf_hidden_units'] = args.tvf_hidden_units
    except:
        pass

    try:
        additional_args['tvf_average_reward'] = args.tvf_average_reward
    except:
        pass

    try:
        additional_args['tvf_value_scale_fn'] = args.tvf_value_scale_fn
        additional_args['tvf_value_scale_norm'] = args.tvf_value_scale_norm
    except:
        pass

    try:
        additional_args['hidden_units'] = args.hidden_units
    except:
        pass

    additional_args['tvf_activation'] = args.tvf_activation
    additional_args['tvf_max_horizon'] = args.tvf_max_horizon


    try:
        additional_args['architecture'] = args.architecture
    except:
        pass

    try:
        additional_args['architecture'] = args.architecture
    except:
        pass

    try:
        additional_args['tvf_n_value_heads'] = args.tvf_n_value_heads
    except:
        pass

    additional_args['head'] = network
    additional_args['network'] = network

    return models.TVFModel(
            input_dims=env.observation_space.shape,
            actions=env.action_space.n,
            device=DEVICE,
            dtype=torch.float32,
            **{k:v for k,v in additional_args.items() if k in allowed_args},
        )

class VT():
    """
    Coppied from rollout, should use rollout.ValueTransform, but older scripts won't have it...
    """

    TRANSFORM_EPSILON = 1e-2

    @staticmethod
    def H(x):
        if args.value_transform == "identity":
            return x
        elif args.value_transform == "sqrt":
            # formula was designed to handle large returns, but mine are scaled, so scale up by 1000
            # as ext_value is ~1.2, but returns in games are usually ~1000
            return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + VT.TRANSFORM_EPSILON * x

    @staticmethod
    def H_inv(x):
        if args.value_transform == "identity":
            return x
        elif args.value_transform == "sqrt":
            # from https://openreview.net/pdf?id=Sye57xStvB (but with corrected square...)
            return np.sign(x) * (((np.sqrt(
                1 + (4 * VT.TRANSFORM_EPSILON) * (np.abs(x) + 1 + VT.TRANSFORM_EPSILON)) - 1) / (
                                              2 * VT.TRANSFORM_EPSILON)) ** 2 - 1)


def discount_rewards(rewards, gamma):
    """
    Returns discounted sum of rewards
    """
    reward_sum = 0
    for k, reward in enumerate(rewards):
        reward_sum += reward * (gamma**k)
    return reward_sum

def rediscount_TVF(values, new_gamma):
    """
    Uses truncated value function to estimate value for given states.
    Rewards will be undiscounted, then rediscounted to the correct gamma.

    values: np array of shape [H]
    returns: rediscounted value estimates of shale [H]
    """
    H = len(values)
    rediscounted_values = np.zeros_like(values)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1

    for h in range(H):
        reward = (values[h] - prev) / old_discount
        prev = values[h]
        discounted_reward_sum += reward * discount
        old_discount *= args.tvf_gamma
        discount *= new_gamma
        rediscounted_values[h] = discounted_reward_sum
    return rediscounted_values


def rediscount_TVF_minimize_error(value_mu, value_std, new_gamma):
    """
    Uses truncated value function to get the 'best' horizon in terms of estimated error.
    Error is estimated as MSE error against cummulative rewards to an infinite horizon with the new_gamma

    values: np array of shape [K]
    returns: value estimate (float), horizon used
    """
    assert new_gamma < 1, "Finite sum requires gamma < 1"

    # first we rediscount the returns, and get error estimates
    K = len(value_mu)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1
    new_values = np.zeros_like(value_mu)
    bias_error = np.zeros_like(value_mu)
    var_error = np.zeros_like(value_mu)
    for k in range(K):
        reward = (value_mu[k] - prev) / old_discount
        prev = value_mu[k]
        discounted_reward_sum += reward * discount
        old_discount *= args.tvf_gamma
        discount *= new_gamma
        new_values[k] = discounted_reward_sum

        bias_error[k] = new_gamma ** k
        var_error[k] = value_std[k] ** 2

    return_estimate = new_values[-1] # not sure if this is the best way to get a return estimate...
    new_error = (return_estimate*bias_error)**2 + var_error
    best_k = np.argmin(new_error)
    return new_values[best_k]

def rediscount_TVF_dcyc(value_mu, value_std, new_gamma, alpha=10):
    """
    Uses truncated value function to get a "don't count your chickens" estimate.
    This works by discounting rewards until the uncertanty associated with them is retired.

    values: np array of shape [K]
    returns: value estimates of shape [k], and best estimate
    """

    # the idea here is to discount rewards based on the minimum future uncertanty

    # first we rediscount the returns, and get error estimates
    K = len(value_mu)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1
    new_values = np.zeros_like(value_mu)
    for k in range(K):
        reward = (value_mu[k] - prev) / old_discount
        future_risk = min(value_std[k:])
        # note value_mu k probably isn't right here..
        rho = (future_risk**2) / (0.0001+value_mu[k]**2)
        discount_factor = np.exp(-0.5*alpha*rho)
        prev = value_mu[k]
        discounted_reward_sum += reward * discount * discount_factor
        old_discount *= args.tvf_gamma
        discount *= new_gamma
        new_values[k] = discounted_reward_sum

    return new_values[-1]

def evaluate_model(model, filename, samples=16, max_frames = 30*60*15, temperature=1.0):

    # we play the games one at a time so as to not take up too much memory
    # this is required as the need to know the future rewards at the same time as the states.
    # running twice using fixed seeds would work too, but would require simulating twice

    episode_scores = []
    episode_lengths = []

    print(f"Evaluating {filename}:", end='', flush=True)

    remaining_samples = samples

    return_estimates = {}

    counter = 0
    noops_used = []

    while remaining_samples > 0:

        batch_samples = min(PARALLEL_ENVS, remaining_samples) # 1 at a time is slower, but better on the memory for long runs...
        buffers = generate_rollouts(
            model,
            max_frames,
            num_rollouts=batch_samples,
            temperature=temperature,
            include_horizons=False,
            seed_base=eval_args.seed+(counter*17)
        )

        for i, buffer in enumerate(buffers):
            # get game score and length
            raw_rewards = buffer["raw_rewards"]
            episode_score = sum(raw_rewards)
            episode_length = len(raw_rewards)
            episode_scores.append(episode_score)
            episode_lengths.append(episode_length)
            noops_used.append(buffer["noops"])
            print(".", end='')

        remaining_samples -= batch_samples
        counter += 1

    def print_it(label, x):
        print(f"{label:<20} {np.mean(x):.2f} +- {np.std(x)/(len(x)**0.5):.2f} [{np.min(x):.1f} to {np.max(x):.1f}]")

    print()
    print()
    print_it("Ep Score:", episode_scores)
    print_it("Ep Length:", episode_lengths)

    print()

    data = {
        'episode_lengths': episode_lengths,
        'episode_scores': episode_scores,
        'return_estimates': return_estimates,
        'noops': noops_used,
        'hostname': socket.gethostname()
    }

    with open(filename+".dat", "wb") as f:
        pickle.dump(data, f)

def generate_rollout(model, max_frames = 30*60*15, include_video=False, temperature=1.0, zero_time=False):
    return generate_rollouts(
        model, max_frames, include_video, num_rollouts=1, temperature=temperature, zero_time=zero_time
    )[0]

def generate_fake_rollout(num_frames = 30*60):
    """
    Generate a fake rollout for testing
    """
    return {
        'values': np.zeros([num_frames, args.tvf_n_horizons], dtype=np.float32),
        'time': np.zeros([num_frames], dtype=np.float32),
        'model_values': np.zeros([num_frames], dtype=np.float32),
        'rewards': np.zeros([num_frames], dtype=np.float32),
        'raw_rewards': np.zeros([num_frames], dtype=np.float32),
        'frames': np.zeros([num_frames, 210, 334, 3], dtype=np.uint8)
    }


def generate_rollouts(
        model,
        max_frames = 30*60*15,
        include_video=False,
        num_rollouts=1,
        temperature=1.0,
        include_horizons=True,
        zero_time=False,
        seed_base=None,
    ):
    """
    Generates rollouts
    """

    seed_base = seed_base or eval_args.seed

    env_fns = [lambda i=i: atari.make(monitor_video=include_video, seed=(i*997)+seed_base) for i in range(num_rollouts)]
    if num_rollouts > 16:
        env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=WORKERS)
    else:
        env = sync_vector_env.SyncVectorEnv(env_fns)

    states = env.reset()
    infos = None

    is_running = [True] * num_rollouts

    frame_count = 0

    buffers:List[Dict[str, Union[bool, list, np.ndarray, CompressedStack]]] = []

    for i in range(num_rollouts):
        buffers.append({
            'values': [],   # values for each horizon of dims [K]
            'tvf_discounted_values': [],  # values for each horizon discounted with TVF_gamma instead of gamma, [K]
            'times': [],  # normalized time step
            'raw_values': [],  # values for each horizon of dims [K] (unscaled)
            'errors': [],   # std error estimates for each horizon of dims [K]
            'model_values': [], # models predicted value (float)
            'rewards': [],   # normalized reward (which value predicts), might be clipped, episodic discounted etc
            'raw_rewards': [], # raw unscaled reward from the atari environment
            'noops': 0,
        })

        if include_video:
            buffers[-1]['frames'] = CompressedStack()  # video frames

    if include_horizons and args.use_tvf:
        horizons = np.repeat(np.arange(int(args.tvf_max_horizon*1.05))[None, :], repeats=num_rollouts, axis=0)
    else:
        horizons = np.repeat(np.arange(args.tvf_max_horizon, args.tvf_max_horizon+1)[None, :], repeats=num_rollouts, axis=0)

    times = np.zeros([num_rollouts])

    # get value transform (will not be in some of the older scripts)
    try:
        ValueTransform = rollout.ValueTransform
    except:
        ValueTransform = VT

    # set seeds
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)

    while any(is_running) and frame_count < max_frames:

        kwargs = {}

        # not sure how best to do this
        try:
            _ = rollout.package_aux_features
            is_old_code = False
        except:
            is_old_code = True

        if is_old_code:
            # old method
            kwargs['horizons'] = horizons
        else:
            # new method
            kwargs['aux_features'] = rollout.package_aux_features(horizons, times)

        with torch.no_grad():
            model_out = model.forward(
                states,
                **kwargs,
                **({'policy_temperature':temperature} if temperature is not None else {})
            )

        if temperature == 0:
            probs = model_out["argmax_policy"].detach().cpu().numpy()
            action = np.asarray([np.argmax(prob) for prob in probs], dtype=np.int32)
        else:
            log_probs = model_out["log_policy"].detach().cpu().numpy()
            if np.isnan(log_probs).any():
                raise Exception(f"NaN found in policy ({args.experiment_name}, {args.run_name}).")
            action = np.asarray([utils.sample_action_from_logp(prob) for prob in log_probs], dtype=np.int32)

        prev_states = states.copy()
        prev_times = times.copy()
        if infos is not None:
            prev_infos = infos.copy()
        else:
            prev_infos = None

        states, rewards, dones, infos = env.step(action)

        # this happens on the first frame, prev_infos is set to None as reset does not generate an info
        # so we use the next info to get the rendered frame... what a pain...
        if prev_infos is None:
            prev_infos = infos.copy()

        # go though each agent...
        for i in range(len(states)):

            if not is_running[i]:
                continue

            # make sure to include last state
            if dones[i]:
                is_running[i] = False

            # check for infos
            if "noop_start" in infos[i]:
                buffers[i]["noops"] = infos[i]["noop_start"]

            model_value = model_out["ext_value"][i].detach().cpu().numpy()
            raw_reward = infos[i].get("raw_reward", rewards[i])
            times[i] = infos[i]["time_frac"] if not zero_time else 0.0

            if 'frames' in buffers[i]:
                agent_layers = prev_states[i]
                channels = prev_infos[i].get("channels", None)
                rendered_frame = prev_infos[i].get("monitor_obs", prev_states[i])
                frame = utils.compose_frame(agent_layers, rendered_frame, channels)
                buffers[i]['frames'].append(frame)

            if horizons is not None:
                values = model_out["tvf_value"][i, :].detach().cpu().numpy()
                values = ValueTransform.H_inv(values)

                if needs_rediscount():
                    buffers[i]['tvf_discounted_values'].append(values)
                    values = rediscount_TVF(values, args.gamma)

                buffers[i]['values'].append(values)
                if "tvf_raw_value" in model_out:
                    raw_values = model_out["tvf_raw_value"][i, :].detach().cpu().numpy()
                    buffers[i]['raw_values'].append(raw_values)

            buffers[i]['model_values'].append(model_value)
            buffers[i]['rewards'].append(rewards[i])
            buffers[i]['times'].append(prev_times[i])
            buffers[i]['raw_rewards'].append(raw_reward)

        frame_count += 1

    # turn lists into np arrays
    for buffer in buffers:
        keys = list(buffer.keys())
        for key in keys:
            if key == "frames":
                continue
            if type(buffer[key]) in [int, float, str]:
                continue
            if len(buffer[key]) == 0:
                del buffer[key]
            else:
                buffer[key] = np.asarray(buffer[key])

    return buffers


class QuickPlot():
    """
    Class to handle fast plotting.
    Background is rendered by py plot, and plots are overlayed using custom drawing
    Supports only basic functions.
    Old plt.draw was ~40ms, this one is ?
    """
    def __init__(self, y_min=0, y_max=1000, log_scale=False, invert_score=False):
        self._y_min = y_min
        self._y_max = y_max
        self._background:np.ndarray
        self._transform: matplotlib.transforms.Transform
        self.log_scale = log_scale
        self._generate_background()
        self.invert_score = invert_score
        self.buffer = self._background.copy()

    def _generate_background(self):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(7, 4), dpi=100)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot([1], [0], label="True", c="lightcoral")
        plt.plot([1], [0], label="Pred", c="greenyellow")

        plt.ylim(self._y_min, self._y_max)
        plt.grid(alpha=0.2)
        if self.log_scale:
            plt.xlabel("log_2(1+h)")
            plt.xlim(1, np.log2(args.tvf_max_horizon + 1))
        else:
            plt.xlabel("h")
            plt.xlim(0, args.tvf_max_horizon)
        plt.ylabel("Score")
        plt.legend(loc="upper left")
        #plt.legend(loc="lower right")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_height, plot_width, _ = data.shape
        self._background = data[:, :, ::-1]
        self._transform = plt.gca().transData
        plt.close(fig)

    def clear(self):
        self.buffer = self._background.copy()

    def plot_pixel(self, x:int, y:int, c):
        # plots pixel at given co-ords
        h, w, channels = self.buffer.shape
        if y < 0:
            return
        if y >= h:
            return
        if x < 0:
            return
        if x >= w:
            return
        self.buffer[-y, x] = c

    def h_line(self, x1:int, x2:int, y:int, c):
        h, w, channels = self.buffer.shape
        if y < 0 or y >= h:
            return
        if x1 < 0:
            x1 = 0
        if x2 >= w:
            x2 = w
        if x2 < x1:
            return
        x1, x2 = min(x1,x2), max(x1, x2)
        self.buffer[-y, x1:x2+1] = c

    def v_line(self, x:int, y1:int, y2:int, c):
        h, w, channels = self.buffer.shape
        if x < 0 or x >= w:
            return
        y1, y2 = min(y1, y2), max(y1, y2)
        if y1 < 0:
            y1 = 0
        if y2 >= h:
            y2 = h
        self.buffer[-y2:-y1, x] = c

    def plot(self, xs, ys, color):
        """
        We assume xs are sorted.
        """

        if self.invert_score:
            ys = -np.asarray(ys)

        if self.log_scale:
            xs = np.log10(10+np.asarray(xs))

        c = mpl.colors.to_rgba(color)[:3][::-1] # swap from RGB to BGA
        c = (np.asarray(c, dtype=np.float32) * 255).astype(dtype=np.uint8)

        zipped_data = list(zip(xs, ys))
        transformed_data = self._transform.transform(zipped_data)
        tx, ty = transformed_data[:, 0], transformed_data[:, 1]

        old_x = int(tx[0])
        old_y = int(ty[0])

        # i'm doing a weird zig-zag thing here which, in this setting, makes more sense than straight line
        # interpolation.
        for x, y in zip(tx, ty):
            x = int(x)
            y = int(y)
            if x == old_x and y == old_y:
                # don't plot pixels twice
                continue
            new_x = x
            new_y = y
            self.h_line(old_x, new_x, old_y, c)
            self.v_line(new_x, old_y, new_y, c)
            old_x = new_x
            old_y = new_y

def export_movie(
        model,
        filename_base,
        max_frames = 30*60*15,
        include_score_in_filename=False,
        temperature=1.0,
        zero_time=False,
):
    """
    Modified version of export movie that supports display of truncated value functions
    In order to show the true discounted returns we write all observations to a buffer, which may take
    a lot of memory.
    """

    scale = 4

    env = atari.make(monitor_video=True, seed=1)
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("monitor_obs", state)

    # work out our height
    first_frame = utils.compose_frame(state, rendered_frame)
    height, width, channels = first_frame.shape
    width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    print(f"Video {filename_base} ", end='', flush=True)

    buffer = generate_rollout(model, max_frames, include_video=True, temperature=temperature, zero_time=zero_time)
    rewards = buffer["rewards"]
    raw_rewards = buffer["raw_rewards"]
    print(f"ratio:{buffer['frames'].ratio:.2f} ", end='', flush=True)
    print(f"score:{int(sum(raw_rewards)):,} ", end='', flush=True)

    # work out discounting values to make plotting a little faster
    discount_weights = args.gamma ** np.arange(0, args.tvf_max_horizon+1)

    #  work out how big our graph will be (with a coarse estimate)
    max_true_return = 0.0
    min_true_return = float('inf')
    step_size = max(len(rewards) // 100, 1)
    for t in range(0, len(rewards), step_size):
        true_rewards = rewards[t:t + args.tvf_max_horizon]
        true_returns = true_rewards * discount_weights[:len(true_rewards)]
        final_return = np.sum(true_returns)
        max_true_return = max(max_true_return, final_return)
        min_true_return = min(min_true_return, final_return)

    max_value_estimate = np.max(buffer["values"]) * REWARD_SCALE
    min_value_estimate = np.min(buffer["values"]) * REWARD_SCALE
    y_max = max(max_true_return, max_value_estimate)
    y_min = min(min_true_return, min_value_estimate)

    # draw background plot
    #inv_score = args.environment == "Skiing" # hack to make skiing plot correctly.
    inv_score = False # not needed any more
    log_fig = QuickPlot(y_min, y_max, log_scale=True, invert_score=inv_score)
    linear_fig = QuickPlot(y_min, y_max, log_scale=False, invert_score=inv_score)
    plot_height, plot_width = log_fig.buffer.shape[:2]

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    final_score = sum(raw_rewards)
    postfix = f" [{int(final_score):,}].mp4" if include_score_in_filename else ".mp4"
    video_filename = filename_base + postfix
    parts = os.path.split(video_filename)
    temp_filename = os.path.join(TEMP_LOCATION, "_"+parts[1])

    video_out = cv2.VideoWriter(temp_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width+plot_width, height), isColor=True)

    max_fps = float('-inf')
    min_fps = float('inf')

    marker_time = time.time()
    very_start_time = time.time()
    fps_list = []

    # run through video and generate data
    for t in range(len(rewards)):

        start_frame_time = time.time()

        # get frames
        frame = buffer["frames"].get(t)

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
            (width, height, 3), frame.shape)

        # extend for plot
        frame = np.pad(frame, ((0, 0), (0, plot_width), (0, 0)))

        # calculate actual truncated values using real future rewards
        true_rewards = rewards[t:t+args.tvf_max_horizon]
        true_returns = true_rewards * discount_weights[:len(true_rewards)]
        true_returns = np.cumsum(true_returns)

        # plotting...
        log_fig.clear()
        linear_fig.clear()

        if len(true_returns) < args.tvf_max_horizon+1:
            padded_true_returns = np.zeros([args.tvf_max_horizon+1], dtype=np.float32) + true_returns[-1]
            padded_true_returns[:len(true_returns)] = true_returns
            true_returns = padded_true_returns
        xs = list(range(len(true_returns)))
        ys = true_returns
        log_fig.plot(xs, ys, 'lightcoral')
        linear_fig.plot(xs, ys, 'lightcoral')

        # show current horizon
        log_fig.plot([CURRENT_HORIZON], [0], 'white')
        linear_fig.plot([CURRENT_HORIZON], [0], 'white')

        # plot predicted values
        if args.use_tvf:
            xs = list(range(len(buffer["values"][t])))
            ys = buffer["values"][t] * REWARD_SCALE  # model learned scaled rewards
            log_fig.plot(xs, ys, 'greenyellow')
            linear_fig.plot(xs, ys, 'greenyellow')
        else:
            # white dot representing value estimate
            xs = [args.tvf_max_horizon-10, args.tvf_max_horizon]
            y = buffer["model_values"][t] * REWARD_SCALE
            ys = [y, y]
            log_fig.plot(xs, ys, 'white')
            linear_fig.plot(xs, ys, 'white')

        if needs_rediscount():
            # plot originally predicted values (without rediscounting)
            xs = list(range(len(buffer["tvf_discounted_values"][t])))
            ys = buffer["tvf_discounted_values"][t] * REWARD_SCALE  # model learned scaled rewards
            log_fig.plot(xs, ys, 'green')
            linear_fig.plot(xs, ys, 'green')

        frame[:plot_height, -plot_width:] = log_fig.buffer
        frame[plot_height:plot_height*2, -plot_width:] = linear_fig.buffer

        video_out.write(frame)

        frame_time = time.time() - start_frame_time

        fps = 1 / frame_time
        fps_list.append(fps)
        max_fps = max(fps, max_fps)
        min_fps = min(fps, min_fps)

        # every 60 second print time
        if (time.time() - marker_time) > 60:
            print(f"<{t}/{len(rewards)} - {np.mean(fps_list[-100:]):.1f} FPS> ", end='', flush=True)
            marker_time = time.time()

    video_out.release()

    end_time = time.time()
    print(f"completed at {len(rewards) / (end_time - very_start_time):.1f} FPS")

    # rename temp file...
    try:
        os.rename(temp_filename, video_filename)
    except Exception as e:
        print(f"Warning: failed to rename {temp_filename} to {video_filename}: {e}")

    return video_filename


if __name__ == "__main__":

    # usage
    # python run_evaluation.py video ./bundle_0 temperatures=[-0.01 , -0.1, -0.5, -1] --max_epochs=200

    parser = argparse.ArgumentParser(description="Evaluation script for PPO/PPG/TVF")
    parser.add_argument("mode", help="[video|eval]")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--temperature", type=float, help="Temperature to use during evaluation (float).")
    parser.add_argument("--samples", type=int, default=64, help="Number of samples to use during evaluation.")
    parser.add_argument("--seed", type=int, default=1, help="Random Seed")
    parser.add_argument("--device", type=str, default=None, help="device to train on")

    eval_args = parser.parse_args()

    if eval_args.device is not None:
        DEVICE = eval_args.device

    if eval_args.mode == "video":
        GENERATE_EVAL = False
        GENERATE_MOVIES = True
    else:
        GENERATE_EVAL = True
        GENERATE_MOVIES = False

    temperature = eval_args.temperature if 'temperature' in vars(eval_args) else None
    samples = eval_args.samples if 'samples' in vars(eval_args) else None

    try:
        model = load_checkpoint(eval_args.checkpoint, device=DEVICE)
        if eval_args.mode == "video":
            video_filename = export_movie(
                model,
                os.path.splitext(eval_args.output_file)[0],
                include_score_in_filename=True,
                temperature=temperature,
            )
        elif eval_args.mode == "video_nt":
            video_filename = export_movie(
                model,
                os.path.splitext(eval_args.output_file)[0],
                include_score_in_filename=True,
                temperature=temperature,
                zero_time=True,
            )
        elif eval_args.mode == "eval":
            evaluate_model(
                model,
                os.path.splitext(eval_args.output_file)[0],
                samples=samples,
                temperature=temperature
            )
        else:
            raise Exception(f"Invalid mode {args.mode}")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(-1)

    sys.exit(0)
