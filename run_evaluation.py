# limit to 4 threads...
import os

# IPS: (for samples=1, mv_samples=100
# on my PC with cuda:0

# start: (false/false/false)
# 3.8 / 17.6 / 13.7

# step is the slowest part...
# why is processing so slow?

# no mask on forward, but with efficent forward gives
# 2.5 all the way through

# null action gives step of ...
# 16 - > 2.5

os.environ["MKL_NUM_THREADS"] = "4"
import torch
torch.set_num_threads(4)

"""
Runs a single evaluation on given file
"""

import argparse
import hashlib
import shutil

import json
import pickle
import time as clock
import sys
import socket
import gzip
import math
from tqdm import tqdm

from typing import Union, List, Dict

import cv2
import numpy as np
import torch

import traceback
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms
import matplotlib.colors
import numpy.random
from gym.vector import sync_vector_env
from rl import atari, procgen, config, utils, hybridVecEnv, rollout
from rl.config import args
import lz4.frame as lib
import os

DEVICE = "cpu"
REWARD_SCALE = float()
CURRENT_HORIZON = int()
PARALLEL_ENVS = 100 # (maximum) number of environments to run in parallel
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
        start_time = clock.time()
        compressed_data = lib.compress(x.tobytes())
        self.buffer.append((x.dtype, x.shape, compressed_data))
        self._uncompressed_size += x.nbytes
        self._compressed_size += len(compressed_data)
        self._compression_time = self._compression_time or (clock.time()-start_time)
        self._compression_time = 0.999 * self._compression_time + 0.001 * (clock.time()-start_time)

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
        args.device = eval_args.device

# load a model and evaluate performance
def load_checkpoint(checkpoint_path, device=None):
    """ Restores model from checkpoint. Returns current env_step"""

    global REWARD_SCALE
    global CURRENT_HORIZON

    load_args(checkpoint_path)

    args.experiment_name = Path(os.path.join(os.path.join(os.getcwd(), checkpoint_path))).parts[-3]

    # fix up frameskip, this happens for older versions of the code.
    if args.frame_skip == 0:
        args.frame_skip = 4

    # fix up horizon when in dna mode (used for plotting only)
    if not args.use_tvf:
        if args.gamma == 1.0:
            args.tvf_max_horizon = args.timeout // 4
        else:
            args.tvf_max_horizon = round(3/(1-args.gamma))

    import train
    model = train.make_model(args)

    # some older versions might not have the open_checkpoint function... :(
    try:
        oc = rollout._open_checkpoint
    except:
        oc = backup_open_checkpoint
    checkpoint = oc(checkpoint_path, map_location=device)

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        print(f"Warning, failed to load model, trying non-strict version: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    step = checkpoint['step']
    env_state = checkpoint["env_state"]
    CURRENT_HORIZON = checkpoint.get("current_horizon", 0)

    if "obs_rms" in checkpoint:
        model.obs_rms = checkpoint["obs_rms"]

    if "reward_scale" in checkpoint:
        REWARD_SCALE = checkpoint["reward_scale"]

    return model


# remove once old runs are done.
def __get_n_actions(space):

    import gym
    if type(space) == gym.spaces.Discrete:
        return space.n
    elif type(space) == gym.spaces.Box:
        assert len(space.shape) == 1
        return space.shape[0]
    else:
        raise ValueError(f"Action space of type {type(space)} not implemented yet.")

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

    counter = 0
    noops_used = []

    eval_data = []

    while remaining_samples > 0:

        batch_samples = min(PARALLEL_ENVS, remaining_samples)
        buffers = generate_rollouts(
            model,
            max_frames,
            num_rollouts=batch_samples,
            temperature=temperature,
            include_horizons=eval_args.eval_horizons,
            mv_return_samples=eval_args.mv_return_samples,
            mv_samples=eval_args.mv_samples,
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
            buffer['reward_scale'] = REWARD_SCALE
            eval_data.append(buffer)
            print(".", end='')

        remaining_samples -= batch_samples
        counter += 1

    with gzip.open(f"{filename}.eval.gz", 'wb') as f:
        pickle.dump(eval_data, f)

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
        'noops': noops_used,
        'hostname': socket.gethostname()
    }

    with open(filename+".summary.dat", "wb") as f:
        pickle.dump(data, f)


def generate_rollout(model, **kwargs):
    return generate_rollouts(model, num_rollouts=1, **kwargs)[0]


def generate_fake_rollout(num_frames = 30*60):
    """
    Generate a fake rollout for testing
    """
    return {
        'values': np.zeros([num_frames, args.tvf_value_heads], dtype=np.float32),
        'time': np.zeros([num_frames], dtype=np.float32),
        'model_values': np.zeros([num_frames], dtype=np.float32),
        'rewards': np.zeros([num_frames], dtype=np.float32),
        'raw_rewards': np.zeros([num_frames], dtype=np.float32),
        'frames': np.zeros([num_frames, 210, 334, 3], dtype=np.uint8)
    }

def make_envs(
        include_video:bool=False,
        seed_base:int=0,
        num_envs:int=1,
        force_hybrid_async:bool=False,
        determanistic_saving=True
):
    # create environment(s) if not already given
    env_fns = [lambda i=i: rollout.make_env(
        args.env_type,
        env_id=args.get_env_name(),
        monitor_video=include_video,
        seed=(i * 997) + seed_base,
        determanistic_saving=determanistic_saving,
    ) for i in
               range(num_envs)]
    if num_envs > 1 or force_hybrid_async:
        envs = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=WORKERS, copy=False)
    else:
        envs = sync_vector_env.SyncVectorEnv(env_fns)
    return envs

@torch.no_grad()
def generate_rollouts(
        model,
        max_frames = 30*60*15,
        include_video=False,
        num_rollouts=1,
        temperature=1.0,
        include_horizons:Union[bool, str] = True,
        zero_time = False,
        mv_return_samples:int=0,
        mv_samples:int=100,
        rewards_only:bool=False,
        venvs=None,
        env_states:np.ndarray=None,
        env_times:np.ndarray=None,
        seed_base=None,
        print_depth=0,
        verbose=True,
    ):
    """
    Generates rollouts

    @param venvs: If given the environment to use to generate the rollouts
    @param env_state: If given then initial state of the environments, if not given the environment will be reset.
    @param rewards_only: if true returns only the rewards gained.
    """

    start_time = clock.time()

    step_timer = utils.Timer('step', 1000)
    forward_timer = utils.Timer('forward', 1000)
    process_timer = utils.Timer('process', 1000)
    hash_timer = utils.Timer('hash', 1000)
    total_timer = utils.Timer('total', 1000)

    model.test_mode = True
    seed_base = seed_base or eval_args.seed

    if venvs is None:
        venvs = make_envs(include_video=include_video, seed_base=seed_base, num_envs=num_rollouts)

    if env_states is not None:
        states = env_states
    else:
        states = venvs.reset()

    infos = None
    is_running = [True] * num_rollouts
    t = 0
    buffers:List[Dict[str, Union[bool, list, np.ndarray, CompressedStack]]] = []

    for i in range(num_rollouts):
        if rewards_only:
            buffers.append({'prev_state_hash':[], 'is_running':[], 'rewards':[], 'raw_rewards':[]})
        else:
            buffers.append({
                'values': [],   # (scaled) values for each horizon of dims [K]
                'tvf_discounted_values': [],  # values for each horizon discounted with TVF_gamma instead of gamma, [K]
                'times': [],  # normalized time step
                'raw_values': [],  # values for each horizon of dims [K]
                'std': [],   # estimated std of return for each horizon of dims [K]
                'sqrt_m2': [],  # estimated sqrt of second moment of return for each horizon of dims [K]
                'model_values': [], # (policy) models predicted value (float)
                'rewards': [],   # normalized reward (which value predicts), might be clipped, episodic discounted etc
                'raw_rewards': [], # raw unscaled reward from the atari environment
                'mv_return_sample': [], # return samples for each horizon, specifically the discounted sum of (unscaled) rewards
                'prev_state_hash': [],  # hash for each prev_state , used to verify that runs are identical
                'is_running': [], # bool, True if agent is still alive, false otherwise.
                # 'uac_value': [],
                'actions': [],
                'probs': [],
                'noops': 0,
            })
            if include_video:
                buffers[-1]['frames'] = CompressedStack()  # video frames

    if not args.use_tvf:
        include_horizons = False

    if include_horizons is True:
        include_horizons = "full"
    elif include_horizons is False:
        include_horizons = "last"

    # special case for fixed head mode
    # if include_horizons:
    #     include_horizons = "standard"

    if include_horizons == "full":
        horizons = np.repeat(np.arange(int(args.tvf_max_horizon*1.05))[None, :], repeats=num_rollouts, axis=0)
    elif include_horizons == "debug":
        horizons = np.repeat(np.asarray([1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000])[None, :], repeats=num_rollouts, axis=0)
    elif include_horizons == "last":
        horizons = np.repeat(np.arange(args.tvf_max_horizon, args.tvf_max_horizon+1)[None, :], repeats=num_rollouts, axis=0)
    elif include_horizons == "standard":
        horizons = model.tvf_fixed_head_horizons
    else:
        raise ValueError(f"invalid horizons mode {include_horizons}")

    if env_times is None:
        times = np.zeros([num_rollouts])
    else:
        times = env_times

    # generate multiverse envs
    if mv_return_samples > 0:
        print("") # start on new line
        multi_envs = make_envs(
            seed_base=seed_base,
            num_envs=mv_return_samples,
            force_hybrid_async=True,
            determanistic_saving=False, # means that when we restore, RNG will be different.
        )
        multi_envs.reset()

        # do a first pass to get episode lengths, then use these to generate checkpoint locations.
        prerun_buffer = generate_rollouts(
            model,
            max_frames=max_frames,
            num_rollouts=num_rollouts,
            include_video=False,
            temperature=temperature,
            include_horizons=False,
            zero_time=zero_time,
            rewards_only=True,
            mv_return_samples=0,
            mv_samples=0,
            seed_base=seed_base,
        )

        ep_lengths = [sum(prerun_buffer[i]['is_running']) for i in range(num_rollouts)]

        # generate samples
        # ideally num_rollouts divides mv_samples, if not we generate more and sample down.
        mv_sample = np.zeros([num_rollouts, max(ep_lengths)+1], dtype=bool) # +1 because we process the last frame
        samples_per_run = int(math.ceil(mv_samples / num_rollouts))
        sample_list = []
        for i in range(num_rollouts):
            run_length = sum(prerun_buffer[i]['is_running'])
            run_samples = np.random.choice(run_length, samples_per_run, replace=False)
            sample_list.extend((i, j) for j in run_samples)
        np.random.shuffle(sample_list)
        for i, j in sample_list[:mv_samples]:
            mv_sample[i, j] = True
        print(f"Generating {np.sum(mv_sample)} samples...")

    else:
        multi_envs = None
        prerun_buffer = None
        mv_sample = None

    mv_samples_generated = 0

    # set seeds
    torch.manual_seed(seed_base)
    np.random.seed(seed_base)

    while any(is_running) and t < max_frames:

        total_timer.start()

        kwargs = {}

        if eval_args.verbose and t % 100 == 0:
            print(f" - step: {t:05d}/{max_frames} with {np.mean(is_running)*100:<3.0f}% running: ", end='')

        forward_timer.start()

        if rewards_only:
            kwargs['output'] = "policy"

        with torch.no_grad():
            model_out = model.forward(
                states,
                **kwargs,
                **({'policy_temperature': temperature} if temperature is not None else {})
            )

        log_probs = model_out["log_policy"].detach().cpu().numpy()
        forward_timer.stop()

        if eval_args.verbose and t % 100 == 0:
            print(f"forward: {forward_timer.time*1000:<10.1f}", end='')

        if temperature == 0:
            probs = model_out["argmax_policy"].detach().cpu().numpy()
            actions = np.asarray([np.argmax(prob) for prob in probs], dtype=np.int32)
        else:

            probs = np.exp(log_probs)
            if np.isnan(log_probs).any():
                raise Exception(f"NaN found in policy ({args.experiment_name}, {args.run_name}).")
            actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_probs], dtype=np.int32)

        prev_states = states.copy()
        prev_times = times.copy()
        if infos is not None:
            prev_infos = infos.copy()
        else:
            prev_infos = None

        # this can speed things up a lot by ignoring completed envs
        actions = np.asarray([(a if running else -1) for a, running in zip(actions, is_running)], dtype=np.int32)

        step_timer.start()

        states, rewards, dones, infos = venvs.step(actions)

        step_timer.stop()
        if eval_args.verbose and t % 100 == 0:
            print(f"step: {step_timer.time * 1000:<10.1f}", end='')

        # this happens on the first frame, prev_infos is set to None as reset does not generate an info
        # so we use the next info to get the rendered frame... what a pain...
        if prev_infos is None:
            prev_infos = infos.copy()

        process_timer.start()

        def append_buffer(key, value):
            if key in buffers[i]:
                if type(value) is np.ndarray:
                    value = value.copy()
                buffers[i][key].append(value)
            else:
                pass

        def set_buffer(key, value):
            if key in buffers[i]:
                buffers[i][key] = value
            else:
                pass

        # go though each agent...
        for i in range(len(states)):

            if not is_running[i]:
                continue

            if dones[i]:
                is_running[i] = False

            # 1. first do the simple stuff... rewards etc
            append_buffer('is_running', is_running[i])
            raw_reward = infos[i].get("raw_reward", rewards[i])
            try:
                if args.noisy_zero >= 0:
                    rewards *= 0
                    raw_reward *= 0
            except:
                pass
            append_buffer('rewards', rewards[i])
            append_buffer('raw_rewards', raw_reward)

            # 2. save a hash of the states for verification
            # first channel is enough.
            hash_timer.start()
            state_hash = int(hashlib.sha256(prev_states[i][0].tobytes()).hexdigest(), 16) % (2 ** 16)
            append_buffer('prev_state_hash', state_hash)
            hash_timer.stop()

            # if we did a pre-run make sure we are identical on our second pass.
            if prerun_buffer is not None:
                assert state_hash == prerun_buffer[i]['prev_state_hash'][
                    t], 'Second pass did not match. Check determanism'

            # next do the slower stuff, but only if needed.
            if rewards_only:
                continue

            # handle multiverse
            # round robin
            # the idea is to generate a sample from env_0, wait multiverse_period steps, then generate the next
            # sample on env_1, etc.
            if (multi_envs is not None) and (mv_sample[i, t]):

                # save randomness before potential multiverse
                rng_state = (np.random.get_state(), torch.get_rng_state())

                # prep multiverse envs...
                if type(venvs) is sync_vector_env.SyncVectorEnv:
                    root_env_state = utils.save_env_state(venvs.envs[i])
                elif type(venvs) is hybridVecEnv.HybridAsyncVectorEnv:
                    save_state = utils.save_env_state(venvs)
                    root_env_state = save_state["HybridAsyncVectorEnv"][f"vec_{i:03d}"]
                else:
                    raise ValueError(f"Invalid venv type {type(venvs)}")

                buffer = {}
                for j in range(multi_envs.n_sequential):
                    buffer[f"vec_{j:03d}"] = root_env_state
                for j in range(multi_envs.n_parallel):
                    pipe = multi_envs.parent_pipes[j]
                    pipe.send(('load', buffer))
                    error, ok = pipe.recv()
                    assert ok, "Failed env load with error:" + str(error)

                initial_states = np.asarray([states[i] for _ in range(mv_return_samples)])
                initial_times = np.asarray([times[i] for _ in range(mv_return_samples)])
                print(f"    - [t:{t:<5} i:{i:<4}]:", end='')
                mv_start_time = clock.time()
                multiverse_buffer = generate_rollouts(
                    model,
                    max_frames=max_frames-t,
                    num_rollouts=mv_return_samples,
                    include_video=False,
                    temperature=temperature,
                    include_horizons=False,
                    zero_time=zero_time,
                    rewards_only=True,
                    mv_return_samples=0,
                    mv_samples=0,
                    venvs=multi_envs,
                    env_states=initial_states,
                    env_times=initial_times,
                    seed_base=i*59+t*1013+6673,
                    verbose=False,
                )
                # calculate the returns at each horizon
                all_rewards = np.zeros([mv_return_samples, max_frames], dtype=np.float32)
                for j in range(mv_return_samples):
                    sample_rewards = multiverse_buffer[j]['raw_rewards']
                    all_rewards[j, :len(sample_rewards)] = sample_rewards
                discounts = np.asarray([args.gamma ** t for t in range(max_frames)], dtype=np.float32)
                discounted_rewards = all_rewards * discounts[None, :]
                discounted_returns = np.cumsum(discounted_rewards, axis=1)
                append_buffer('mv_return_sample', discounted_returns)

                mv_samples_generated += 1

                # show progress
                time_taken = clock.time() - mv_start_time
                ep_lengths = [sum(multiverse_buffer[i]['is_running']) for i in range(mv_return_samples)]
                fps = max(ep_lengths) * len(ep_lengths) / time_taken

                eta_hr = ((mv_samples - mv_samples_generated) * time_taken) / 60 / 60

                print(f" {mv_samples_generated:03d}/{mv_samples}", end=' ')
                print(f"length {int(np.mean(ep_lengths))} ({min(ep_lengths)}-{max(ep_lengths)})", end=', ')
                print(f"ips: {fps:.0f}", end=', ')
                print(f"took {time_taken/60:.1f}m eta: {eta_hr:.1f}h")

                np.random.set_state(rng_state[0])
                torch.set_rng_state(rng_state[1])
            else:
                append_buffer('mv_return_sample', None)

            # check for infos
            if "noop_start" in infos[i]:
                set_buffer("noops", infos[i]["noop_start"])

            model_value = model_out["ext_value"][i].detach().cpu().numpy()

            # old versions accidentally used time_frac... if time is there we use that instead
            time = 0.0
            if not zero_time:
                time = infos[i].get("time", infos[i]["time_frac"])
            times[i] = time

            append_buffer('actions', actions[i])
            append_buffer('probs', probs[i])

            if 'frames' in buffers[i]:
                agent_layers = prev_states[i]
                channels = prev_infos[i].get("channels", None)
                rendered_frame = prev_infos[i].get("monitor_obs", prev_states[i])
                frame = utils.compose_frame(agent_layers, rendered_frame, channels)
                append_buffer('frames', frame)

            if horizons is not None and args.use_tvf:
                values = model_out["tvf_value"][i, :, 0].detach().cpu().numpy()
                if needs_rediscount():
                    append_buffer('tvf_discounted_values', values)
                    values = rediscount_TVF(values, args.gamma)

                append_buffer('values', values)
                if "tvf_raw_value" in model_out:
                    raw_values = model_out["tvf_raw_value"][i, :].detach().cpu().numpy()
                    append_buffer('raw_values', raw_values)

            append_buffer('model_values', model_value)
            append_buffer('times', prev_times[i])


        process_timer.stop()
        total_timer.stop()

        if eval_args.verbose and t % 100 == 0:
            print(f"processing: {process_timer.time * 1000:<10.1f}", end='')
            print(f"hash: {sum(is_running) * hash_timer.time * 1000:<10.1f}", end='')
            print(f"total: {total_timer.time * 1000:<10.1f}", end='')
            print()

        t += 1

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
                if key in ["mv_return_sample"]:
                    buffer[key] = np.asarray(buffer[key], dtype=object)
                else:
                    buffer[key] = np.asarray(buffer[key])

    # output debugging
    if verbose:
        time_taken = clock.time() - start_time
        fps = t / time_taken
        ep_lengths = [sum(buffers[i]['is_running']) for i in range(num_rollouts)]
        print(" " * print_depth + f" length {int(np.mean(ep_lengths))} ({min(ep_lengths)}-{max(ep_lengths)}) ips: {fps*num_rollouts:.0f}")

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
            plt.xlabel("$\log_{10}(10+h)$")
            plt.xlim(1, np.log10(args.tvf_max_horizon + 10))
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
        """
        x,y in pixel space
        """
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

    def line(self, x1:int, y1:int, x2:int, y2:int, c):
        """
        x,y in pixel space
        """
        h, w, channels = self.buffer.shape
        if x1 < 0:
            x1 = 0
        if x2 >= w:
            x2 = w
        if y1 < 0:
            y1 = 0
        if y2 > h:
            y2 = h
        y = y1
        for x in range(x1, x2+1):
            self.buffer[-int(y), x] = c
            y += (y2-y1) / (x2+1-x1)

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
            #self.h_line(old_x, new_x, old_y, c)
            #self.v_line(new_x, old_y, new_y, c)
            self.line(old_x, old_y, new_x, new_y, c)
            old_x = new_x
            old_y = new_y

    def plot_between(self, xs, y_low, y_high, color, edges_only: bool=False):
        """
        We assume xs are sorted.
        """

        if self.log_scale:
            xs = np.log10(10+np.asarray(xs))

        c = mpl.colors.to_rgba(color)[:3][::-1] # swap from RGB to BGA
        c = (np.asarray(c, dtype=np.float32) * 255).astype(dtype=np.uint8)

        zipped_data = list(zip(xs, y_low))
        transformed_data = self._transform.transform(zipped_data)
        tx, ty1 = transformed_data[:, 0], transformed_data[:, 1]

        zipped_data = list(zip(xs, y_high))
        transformed_data = self._transform.transform(zipped_data)
        tx, ty2 = transformed_data[:, 0], transformed_data[:, 1]

        prev_x = tx[0]-1
        current_x = int(prev_x)
        prev_y1 = ty1[0]
        prev_y2 = ty2[0]

        def lerp(x, x1, x2, y1, y2):
            factor = (x - x1) / (x2-x1)
            return y2 * factor + y1 * (1-factor)

        for new_x, new_y1, new_y2 in zip(tx, ty1, ty2):
            while current_x < int(new_x):
                current_x += 1
                x = int(current_x)
                y_top = int(lerp(current_x, prev_x, new_x, prev_y1, new_y1))
                y_bottom = int(lerp(current_x, prev_x, new_x, prev_y2, new_y2))
                if edges_only:
                    self.plot_pixel(x, y_top, c)
                    self.plot_pixel(x, y_bottom, c)
                else:
                    self.v_line(x, y_top, y_bottom, c)

            prev_x = new_x
            prev_y1 = new_y1
            prev_y2 = new_y2


def export_movie(
        model,
        filename_base,
        max_frames:int = 30*60*15,
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

    env = rollout.make_env(args.env_type, env_id=args.get_env_name(), monitor_video=True, seed=1)
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("monitor_obs", state)

    # work out our height
    first_frame = utils.compose_frame(state, rendered_frame)
    height, width, channels = first_frame.shape
    width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    print(f"Video {filename_base} ", end='', flush=True)

    buffer = generate_rollout(
        model,
        max_frames=max_frames,
        include_video=True,
        temperature=temperature,
        mv_return_samples=eval_args.mv_return_samples,
        mv_samples=eval_args.mv_samples,
        zero_time=zero_time
    )
    rewards = buffer["rewards"]
    raw_rewards = buffer["raw_rewards"]
    print(f"ratio:{buffer['frames'].ratio:.2f} ", end='', flush=True)
    print(f"score:{int(sum(raw_rewards)):,} ", end='', flush=True)

    # work out discounting values to make plotting a little faster
    discount_weights = args.gamma ** np.arange(0, args.tvf_max_horizon+1)
    tvf_discount_weights = args.tvf_gamma ** np.arange(0, args.tvf_max_horizon + 1)

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

    max_return_sample = float('-inf')
    if "mv_return_sample" in buffer:
        for sample in buffer["mv_return_sample"]:
            if sample is not None:
                max_return_sample = max(max_return_sample, sample.max())

    key = "values" if args.use_tvf else "model_values"
    max_value_estimate = np.max(buffer[key]) / REWARD_SCALE
    min_value_estimate = np.min(buffer[key]) / REWARD_SCALE
    y_max = max(max_true_return, max_value_estimate, max_return_sample)
    y_min = min(min_true_return, min_value_estimate)

    # draw background plot
    inv_score = False # not needed any more
    fig = QuickPlot(y_min, y_max, log_scale=True, invert_score=inv_score)
    plot_height, plot_width = fig.buffer.shape[:2]

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    final_score = sum(raw_rewards)
    postfix = f" [{int(final_score):,}].mp4" if include_score_in_filename else ".mp4"
    video_filename = filename_base + postfix
    parts = os.path.split(video_filename)
    temp_filename = os.path.join(TEMP_LOCATION, "_"+parts[1])

    video_out = cv2.VideoWriter(temp_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width+plot_width, height), isColor=True)

    max_fps = float('-inf')
    min_fps = float('inf')

    marker_time = clock.time()
    very_start_time = clock.time()
    fps_list = []

    # run through video and generate data
    for t in tqdm(range(len(rewards))):

        start_frame_time = clock.time()

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
        true_tvf_discounted_returns = true_rewards * tvf_discount_weights[:len(true_rewards)]
        true_tvf_discounted_returns = np.cumsum(true_tvf_discounted_returns)

        # calculate return distribution when multiverse is enabled
        return_samples = buffer.get("mv_return_sample", None)
        if return_samples is not None:
            # not all timesteps have a return sample
            return_sample = return_samples[t]
        else:
            return_sample = None

        # plotting...
        fig.clear()


        if return_sample is not None:
            # plot return sample

            mean = np.mean(return_sample, axis=0)

            xs = list(range(len(mean)))
            low = np.quantile(return_sample, 0.5 - 0.9545/2, axis=0)
            high = np.quantile(return_sample, 0.5 + 0.9545/2, axis=0)
            fig.plot_between(xs, low, high, (0.10, 0.10, 0.20))
            low = np.quantile(return_sample, 0.50 - 0.341, axis=0)
            high = np.quantile(return_sample, 0.50 + 0.341, axis=0)
            fig.plot_between(xs, low, high, (0.25, 0.25, 0.35))
            low = np.min(return_sample, axis=0)
            high = np.max(return_sample, axis=0)
            fig.plot_between(xs, low, high, (0.45, 0.25, 0.35), edges_only=True)
            fig.plot(xs, mean, (0.45,0.45,0.55))
        else:
            # otherwise plot this episodes return
            if len(true_returns) < args.tvf_max_horizon+1:
                padded_true_returns = np.zeros([args.tvf_max_horizon+1], dtype=np.float32) + true_returns[-1]
                padded_true_returns[:len(true_returns)] = true_returns
                true_returns = padded_true_returns
            xs = list(range(len(true_returns)))
            ys = true_returns
            fig.plot(xs, ys, 'lightcoral')

        # show current horizon
        fig.plot([CURRENT_HORIZON], [0], 'white')

        # plot predicted values
        if args.use_tvf:
            xs = model.tvf_fixed_head_horizons
            ys = buffer["values"][t] / REWARD_SCALE  # model learned scaled rewards
            fig.plot(xs, ys, 'greenyellow')
        else:
            # white dot representing value estimate
            xs = [round(args.tvf_max_horizon*0.95)-1, args.tvf_max_horizon]
            y_value = buffer["model_values"][t] / REWARD_SCALE
            ys = [y_value, y_value]
            fig.plot(xs, ys, 'white')

            # if args.use_uac:
            #     xs = [round(args.tvf_max_horizon*0.98)-1, args.tvf_max_horizon]
            #     y_uac = buffer["uac_value"][t] / REWARD_SCALE
            #     ys = [y_uac, y_uac]
            #     fig.plot(xs, ys, [1.0, 0.5, 0.0])
            #     ys = [y_value, y_uac]
            #     xs = [round(np.mean(xs)), round(np.mean(xs))]
            #     fig.plot(xs, ys, [1.0, 0.5, 0.0])



        if needs_rediscount():
            # plot originally predicted values (without rediscounting)
            xs = list(range(len(buffer["tvf_discounted_values"][t])))
            ys = buffer["tvf_discounted_values"][t] / REWARD_SCALE  # model learned scaled rewards
            fig.plot(xs, ys, 'green')
            # also plot true score (with tvf_gamma)
            xs = list(range(len(true_tvf_discounted_returns)))
            ys = true_tvf_discounted_returns
            fig.plot(xs, ys, 'purple')

        frame[:plot_height, -plot_width:] = fig.buffer

        # show clock
        frame_height, frame_width, frame_channels = frame.shape
        utils.draw_numbers(frame, frame_width-100, frame_height-20, t, color=(255, 0, 0), zero_pad=4, size=3)

        # show action and distribution
        probs = buffer["probs"][t]
        action_taken = buffer["actions"][t]
        for i, prob in enumerate(probs):
            x = i*8
            y = 0
            utils.draw_pixel(frame, x, y+8, c=tuple(int(x) for x in [255*prob, 255*prob, 255*prob]), sx=8, sy=8)
            if i == action_taken:
                utils.draw_pixel(frame, x, y, c=(255, 255, 255), sx=8, sy=8)
            utils.draw_pixel(frame, x, y+16, c=tuple(int(x) for x in [32, i % 2 * 128, 0]), sx=8, sy=8)

        video_out.write(frame)

        frame_time = clock.time() - start_frame_time

        fps = 1 / frame_time
        fps_list.append(fps)
        max_fps = max(fps, max_fps)
        min_fps = min(fps, min_fps)

        # every 60 second print time
        # if (clock.time() - marker_time) > 60:
        #     print(f"<{t}/{len(rewards)} - {np.mean(fps_list[-100:]):.1f} FPS> ", end='', flush=True)
        #     marker_time = clock.time()

    video_out.release()

    end_time = clock.time()
    print(f"completed at {len(rewards) / (end_time - very_start_time):.1f} FPS")

    # rename temp file...
    try:
        shutil.move(temp_filename, video_filename)
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
    parser.add_argument("--max_frames", type=int, default=30*60*15, help="maximum number of frames to generate for videos.")
    parser.add_argument("--verbose", type=config.str2bool, default=False,
                        help="Enables extra logging.")
    parser.add_argument("--eval_horizons", type=str, default="debug",
                        help="Which horizons to include when evaluating model, [last|debug|full]. For multiverse use debug.")
    parser.add_argument("--mv_return_samples", type=int, default=0,
                        help="Number samples in a multiverse return distribution. If > 0 enables multiverse return estimation.")
    parser.add_argument("--mv_samples", type=int, default=100,
                        help="Number of checkpoints to generate return samples from.")
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
                max_frames=eval_args.max_frames,
            )
        elif eval_args.mode == "video_nt":
            video_filename = export_movie(
                model,
                os.path.splitext(eval_args.output_file)[0],
                include_score_in_filename=True,
                temperature=temperature,
                zero_time=True,
                max_frames=eval_args.max_frames,
            )
        elif eval_args.mode == "eval":
            evaluate_model(
                model,
                os.path.splitext(eval_args.output_file)[0],
                samples=samples,
                temperature=temperature,
                max_frames=eval_args.max_frames,
            )
        else:
            raise Exception(f"Invalid mode {args.experiment_name}")
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        print(e)
        traceback.print_exc()
        sys.exit(-1)

    sys.exit(0)
