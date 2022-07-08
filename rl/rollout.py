import os

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import ast
import time as clock
import json
import gzip
from collections import deque
from typing import Union, Optional
import math

from .logger import Logger
from . import utils, atari, mujoco, hybridVecEnv, wrappers, models, compression, config
from .returns import get_return_estimate
from .config import args
from .mutex import Mutex
from .replay import ExperienceReplayBuffer

import collections

def add_relative_noise(X:np.ndarray, rel_error:float):
    # does not change the expectation.
    if rel_error <= 0:
        return X
    factors = np.asarray(1 - (rel_error / 2) + (np.random.rand(*X.shape) * rel_error), dtype=np.float32)
    return X * factors

def _test_interpolate():
    horizons = np.asarray([0, 1, 2, 10, 100])
    values = np.asarray([0, 5, 10, -1, 2])[None, :].repeat(11, axis=0)
    results = interpolate(horizons, values, np.asarray([-100, -1, 0, 1, 2, 3, 4, 99, 100, 101, 200]))
    expected_results = [0, 0, 0, 5, 10, (7/8)*10+(1/8)*-1, (6/8)*10+(2/8)*-1, 1.96666667, 2, 2, 2]
    if np.max(np.abs(np.asarray(expected_results) - results)) > 1e-6:
        print("Expected:", expected_results)
        print("Found:", results)
        raise ValueError("Interpolation check failed")


def interpolate(horizons: np.ndarray, values: np.ndarray, target_horizons: np.ndarray):
    """
    Returns linearly interpolated value from source_values

    horizons: sorted ndarray of shape [K] of horizons, must be in *strictly* ascending order
    values: ndarray of shape [*shape, K] where values[...,h] corresponds to horizon horizons[h]
    target_horizons: np array of dims [*shape], the horizon we would like to know the interpolated value of for each
        example

    """

    # I did this custom, as I could not find a way to get numpy to interpolate the way I needed it to.
    # the issue is we interpolate nd data with non-uniform target x's.

    assert len(set(horizons)) == len(horizons), f"Horizons duplicates not supported {horizons}"
    assert np.all(np.diff(horizons) > 0), f"Horizons must be sorted and unique horizons:{horizons}, targets:{target_horizons}"

    assert horizons[0] == 0, "first horizon must be 0"

    # we do not extrapolate...
    target_horizons = np.clip(target_horizons, min(horizons), max(horizons))

    *shape, K = values.shape
    shape = tuple(shape)
    assert horizons.shape == (K,)
    assert target_horizons.shape == shape, f"{target_horizons.shape} != {shape}"

    # put everything into 1d
    N = np.prod(shape)
    values = values.reshape(N, K)
    target_horizons = target_horizons.reshape(N)

    post_index = np.searchsorted(horizons, target_horizons, side='left')

    # select out our values
    pre_index = np.maximum(post_index-1, 0)
    value_pre = values[range(N), pre_index]
    value_post = values[range(N), post_index]

    dx = (horizons[post_index] - horizons[pre_index])
    dx[dx == 0] = 1.0 # this only happens when we are at the boundaries, in which case we have 0/dx, and we just want 0.
    factor = (target_horizons - horizons[pre_index]) / dx
    result = value_pre * (1 - factor) + value_post * factor
    result[post_index == 0] = 0 # these have h<=0 which by definition has value 0
    result = result.reshape(*shape)

    return result


def get_value_head_horizons(n_heads: int, max_horizon: int, spacing: str="geometric"):
    """
    Provides a set of horizons spaces (approximately) geometrically, with the H[0] = 1 and H[-1] = max_horizon.
    Some early horizons may be duplicated due to rounding.
    """

    result = []
    target_n_heads = n_heads

    # special case for even distribution
    if spacing.startswith("even"):
        # format should be even_x, where x is an integer
        # we take the first n heads, then every n after that.
        n = int(spacing.split("_")[1])
        result = []
        result.extend(range(n))
        current = n
        while current <= max_horizon:
            result.append(current)
            current += n
        return np.asarray(result, dtype=np.int32)

    # the idea here is to remove duplicates.
    while len(result) < target_n_heads:
        if spacing == "geometric":
            result = np.asarray(np.round(np.geomspace(1, max_horizon+1, n_heads))-1, dtype=np.int32)
        elif spacing == "linear":
            result = np.asarray(np.round(np.linspace(0, max_horizon, n_heads)), dtype=np.int32)
        else:
            raise ValueError(f"Invalid spacing value {spacing}")
        result = np.asarray(sorted(set(result)))
        n_heads += 1

    if len(result) != target_n_heads:
        # this can fail sometimes...
        print("Warning, head distribution not even, trying to fix...")
        return get_value_head_horizons(n_heads-1, max_horizon, spacing)

    return result

def save_progress(log: Logger):
    """ Saves some useful information to progress.txt. """

    details = {}
    details["max_epochs"] = args.epochs
    details["completed_epochs"] = log["env_step"] / 1e6  # include the current completed step.
    # ep_score could be states, or a float (population based is the group mean which is a float)
    if type(log["ep_score"]) is float:
        details["score"] = log["ep_score"]
    else:
        details["score"] = log["ep_score"][0]
    details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
    try:
        details["fps"] = log["fps"]
    except:
        details["fps"] = 0
    frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
    details["eta"] = (frames_remaining / details["fps"]) if details["fps"] > 0 else 0
    details["host"] = args.hostname
    details["device"] = args.device
    details["last_modified"] = clock.time()
    with open(os.path.join(args.log_folder, "progress.txt"), "w") as f:
        json.dump(details, f, indent=4)

def calculate_tp_returns(dones: np.ndarray, final_tp_estimate: np.ndarray):
    """
    Calculate terminal prediction estimates using bootstrapping
    """

    # todo: make this td(\lambda) style...)

    N, A = dones.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_tp_estimate

    gamma = 0.99 # this is a very interesting idea, discount the terminal time.

    for i in reversed(range(N)):
        returns[i] = current_return = 1 + current_return * gamma * (1.0 - dones[i])

    return returns


def calculate_bootstrapped_returns(rewards, dones, final_value_estimate, gamma) -> np.ndarray:
    """
    Calculates returns given a batch of rewards, dones, and a final value estimate.

    Input is vectorized so it can calculate returns for multiple agents at once.
    :param rewards: nd array of dims [N,A]
    :param dones:   nd array of dims [N,A] where 1 = done and 0 = not done.
    :param final_value_estimate: nd array [A] containing value estimate of final state after last action.
    :param gamma:   discount rate.
    :return: np array of dims [N,A]
    """

    N, A = rewards.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_value_estimate

    if type(gamma) is float:
        gamma = np.ones([N, A], dtype=np.float32) * gamma

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma[i] * (1.0 - dones[i])

    return returns

def td_lambda(
        batch_rewards,
        batch_value,
        final_value_estimate,
        batch_terminal,
        gamma: float,
        lamb=0.95,
):
    advantages = gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma, lamb, normalize=False)
    return advantages + batch_value

def ed_gae(
    rewards,
    values,
    normalization_factors,
    batch_terminal,
    lamb=0.95,
):

    N_plus_one, A = values.shape
    N = N_plus_one - 1
    advantages = np.zeros([N, A], dtype=np.float32)

    def get_g_weights(max_n: int):
        """
        Returns the weights for each G estimate, given lambda.
        """
        weight = (1-lamb)
        weights = []
        for _ in range(max_n):
            weights.append(weight)
            weight *= lamb
        weights = np.asarray(weights) / np.sum(weights)
        return weights

    def calculate_g(t:int, n:int):
        """ Calculate G^(n) (s_t) """
        # we use the rewards first, then use expected rewards from 'pivot' state onwards.
        # pivot state is either the final state in rollout, or t+n, which ever comes first.
        sum_of_rewards = np.zeros([N], dtype=np.float32)
        discount = np.ones([N], dtype=np.float32) # just used to handle terminals
        if (t+n) >= N:
            n = (N - t)
        for i in range(n):
            sum_of_rewards += rewards[t+i, :]
            discount *= 1-batch_terminal[t+i, :]
        bootstrap = values[t+n, :] * normalization_factors[t+n]
        normed_return = (sum_of_rewards + bootstrap * discount) / normalization_factors[t]
        normed_value_estimate = values[t, :]
        advantage = normed_return-normed_value_estimate
        return advantage

    for t in range(N):
        max_n = N - t
        weights = get_g_weights(max_n)
        for n, weight in zip(range(1, max_n+1), weights):
            if weight <= 1e-6:
                # ignore small or zero weights.
                continue
            advantages[t, :] += weight * calculate_g(t, n)
    return advantages



def gae(
        batch_rewards,
        batch_value,
        final_value_estimate,
        batch_terminal,
        gamma: float,
        lamb=0.95,
        normalize=False
    ):
    """
    Calculates GAE based on rollout data.
    """
    N, A = batch_rewards.shape

    batch_advantage = np.zeros_like(batch_rewards, dtype=np.float32)
    prev_adv = np.zeros([A], dtype=np.float32)
    for t in reversed(range(N)):
        is_next_new_episode = batch_terminal[
            t] if batch_terminal is not None else False  # batch_terminal[t] records if prev_state[t] was terminal state)
        value_next_t = batch_value[t + 1] if t != N - 1 else final_value_estimate
        delta = batch_rewards[t] + gamma * value_next_t * (1.0 - is_next_new_episode) - batch_value[t]
        batch_advantage[t] = prev_adv = delta + gamma * lamb * (
                1.0 - is_next_new_episode) * prev_adv
    if normalize:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
    return batch_advantage


def calculate_gae_tvf(
        batch_reward: np.ndarray,
        batch_value: np.ndarray,
        final_value_estimate: np.ndarray,
        batch_terminal: np.ndarray,
        discount_fn = lambda t: 0.999**t,
        lamb: float = 0.95):

    """
    A modified version of GAE that uses truncated value estimates to support any discount function.
    This works by extracting estimated rewards from the value curve via a finite difference.

    batch_reward: [N, A] rewards for each timestep
    batch_value: [N, A, H] value at each timestep, for each horizon (0..max_horizon)
    final_value_estimate: [A, H] value at timestep N+1
    batch_terminal [N, A] terminal signals for each timestep
    discount_fn A discount function in terms of t, the number of timesteps in the future.
    lamb: lambda, as per GAE lambda.
    """

    N, A, H = batch_value.shape

    advantages = np.zeros([N, A], dtype=np.float32)
    values = np.concatenate([batch_value, final_value_estimate[None, :, :]], axis=0)

    # get expected rewards. Note: I'm assuming here that the rewards have not been discounted
    assert args.tvf_gamma == 1, "General discounting function requires TVF estimates to be undiscounted (might fix later)"
    expected_rewards = values[:, :, 0] - batch_value[:, :, 1]

    def calculate_g(t, n:int):
        """ Calculate G^(n) (s_t) """
        # we use the rewards first, then use expected rewards from 'pivot' state onwards.
        # pivot state is either the final state in rollout, or t+n, which ever comes first.
        sum_of_rewards = np.zeros([N], dtype=np.float32)
        discount = np.ones([N], dtype=np.float32) # just used to handle terminals
        pivot_state = min(t+n, N)
        for i in range(H):
            if t+i < pivot_state:
                reward = batch_reward[t+i, :]
                discount *= 1-batch_terminal[t+i, :]
            else:
                reward = expected_rewards[pivot_state, :, (t+i)-pivot_state]
            sum_of_rewards += reward * discount * discount_fn(i)
        return sum_of_rewards

    def get_g_weights(max_n: int):
        """
        Returns the weights for each G estimate, given some lambda.
        """

        # weights are assigned with exponential decay, except that the weight of the final g_return uses
        # all remaining weight. This is the same as assuming that g(>max_n) = g(n)
        # because of this 1/(1-lambda) should be a fair bit larger than max_n, so if a window of length 128 is being
        # used, lambda should be < 0.99 otherwise the final estimate carries a significant proportion of the weight

        weight = (1-lamb)
        weights = []
        for _ in range(max_n-1):
            weights.append(weight)
            weight *= lamb
        weights.append(lamb**max_n)
        weights = np.asarray(weights)

        assert abs(weights.sum() - 1.0) < 1e-6
        return weights

    # advantage^(n) = -V(s_t) + r_t + r_t+1 ... + r_{t+n-1} + V(s_{t+n})

    for t in range(N):
        max_n = N - t
        weights = get_g_weights(max_n)
        for n, weight in zip(range(1, max_n+1), weights):
            if weight <= 1e-6:
                # ignore small or zero weights.
                continue
            advantages[t, :] += weight * calculate_g(t, n)
        advantages[t, :] -= batch_value[t, :, -1]

    return advantages


def calculate_tvf_lambda(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        final_value_estimates: np.ndarray,
        gamma: float,
        lamb: float = 0.95,
):
    # this is a little slow, but calculate each n_step return and combine them.
    # also.. this is just an approximation

    params = (rewards, dones, values, final_value_estimates, gamma)

    if lamb == 0:
        return calculate_tvf_td(*params)
    if lamb == 1:
        return calculate_tvf_mc(*params)

    # can be slow for high n_steps... so we cap it at 100, and use effective horizon as a cap too
    N = int(min(1 / (1 - lamb), args.n_steps, 100))

    g = []
    for i in range(N):
        g.append(calculate_tvf_n_step(*params, n_step=i + 1))

    # this is totally wrong... please fix.

    result = g[0] * (1 - lamb)
    for i in range(1, N):
        result += g[i] * (lamb ** i) * (1 - lamb)

    return result


def calculate_tvf_n_step(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        final_value_estimates: np.ndarray,
        gamma: float,
        n_step: int,
):
    """
    Returns the n_step value estimate.
    This is the old, non sampled version
    """

    N, A, H = values.shape

    returns = np.zeros([N, A, H], dtype=np.float32)

    values = np.concatenate([values, final_value_estimates[None, :, :]], axis=0)

    for t in range(N):

        # first collect the rewards
        discount = np.ones([A], dtype=np.float32)
        reward_sum = np.zeros([A], dtype=np.float32)
        steps_made = 0

        for n in range(1, n_step + 1):
            if (t + n - 1) >= N:
                break
            # n_step is longer than horizon required
            if n >= H:
                break
            this_reward = rewards[t + n - 1]
            reward_sum += discount * this_reward
            discount *= gamma * (1 - dones[t + n - 1])
            steps_made += 1

            # the first n_step returns are just the discounted rewards, no bootstrap estimates...
            returns[t, :, n] = reward_sum

        # note: if we are near the end we might not be able to do a full n_steps, so just a shorter n_step for these

        # next update the remaining horizons based on the bootstrap estimates
        # we do all the horizons in one go, which quite fast for long horizons
        discounted_bootstrap_estimates = discount[:, None] * values[t + steps_made, :, 1:H - steps_made]
        returns[t, :, steps_made + 1:] += reward_sum[:, None] + discounted_bootstrap_estimates

        # this is the non-vectorized code, for reference.
        # for h in range(steps_made+1, H):
        #    bootstrap_estimate = discount * values[t + steps_made, :, h - steps_made] if (h - steps_made) > 0 else 0
        #    returns[t, :, h] = reward_sum + bootstrap_estimate

    return returns


def calculate_tvf_mc(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: None,  # note: values is ignored...
        final_value_estimates: np.ndarray,
        gamma: float
):
    """
    This is really just the largest n_step that will work, but does not require values
    """

    N, A = rewards.shape
    H = final_value_estimates.shape[-1]

    returns = np.zeros([N, A, H], dtype=np.float32)

    n_step = N

    for t in range(N):

        # first collect the rewards
        discount = np.ones([A], dtype=np.float32)
        reward_sum = np.zeros([A], dtype=np.float32)
        steps_made = 0

        for n in range(1, n_step + 1):
            if (t + n - 1) >= N:
                break
            # n_step is longer than horizon required
            if n >= H:
                break
            this_reward = rewards[t + n - 1]
            reward_sum += discount * this_reward
            discount *= gamma * (1 - dones[t + n - 1])
            steps_made += 1

            # the first n_step returns are just the discounted rewards, no bootstrap estimates...
            returns[t, :, n] = reward_sum

        # note: if we are near the end we might not be able to do a full n_steps, so just a shorter n_step for these

        # next update the remaining horizons based on the bootstrap estimates
        # we do all the horizons in one go, which quite fast for long horizons
        discounted_bootstrap_estimates = discount[:, None] * final_value_estimates[:, 1:-steps_made]
        returns[t, :, steps_made + 1:] += reward_sum[:, None] + discounted_bootstrap_estimates

    return returns


def calculate_tvf_td(
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        final_value_estimates: np.ndarray,
        gamma: float,
):
    """
    Calculate return targets using value function horizons.
    This involves finding targets for each horizon being learned

    rewards: np float32 array of shape [N, A]
    dones: np float32 array of shape [N, A]
    values: np float32 array of shape [N, A, H]
    final_value_estimates: np float32 array of shape [A, H]

    returns: returns for each time step and horizon, np array of shape [N, A, H]

    """

    N, A, H = values.shape

    returns = np.zeros([N, A, H], dtype=np.float32)

    # note: this webpage helped a lot with writing this...
    # https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html

    values = np.concatenate([values, final_value_estimates[None, :, :]], axis=0)

    for t in range(N):
        for h in range(1, H):
            reward_sum = rewards[t + 1 - 1]
            discount = gamma * (1 - dones[t + 1 - 1])
            bootstrap_estimate = discount * values[t + 1, :, h - 1] if (h - 1) > 0 else 0
            estimate = reward_sum + bootstrap_estimate
            returns[t, :, h] = estimate
    return returns


class Runner:

    def __init__(self, model: models.TVFModel, log, name="agent", action_dist='discrete'):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model
        self.step = 0

        self.previous_rollout = None

        def make_optimizer(params, cfg: config.OptimizerConfig):
            optimizer_params = {
                'lr': cfg.lr,
            }
            if cfg.optimizer == "adam":
                optimizer = torch.optim.Adam
                optimizer_params.update({
                    'eps': cfg.adam_epsilon,
                    'betas': (cfg.adam_beta1, cfg.adam_beta2),
                })
            elif cfg.optimizer == "sgd":
                optimizer = torch.optim.SGD
            else:
                raise ValueError(f"Invalid Optimizer {cfg.optimizer}")
            return optimizer(params, **optimizer_params)

        # special case for policy optimizer
        self.policy_optimizer = make_optimizer(model.policy_net.parameters(), args.policy)
        self.value_optimizer = make_optimizer(model.value_net.parameters(), args.value)
        if args.distil.epochs > 0:
            self.distil_optimizer = make_optimizer(model.policy_net.parameters(), args.distil)
        else:
            self.distil_optimizer = None
        if args.aux.epochs > 0:
            self.aux_optimizer = make_optimizer(model.parameters(), args.aux)
        else:
            self.aux_optimizer = None

        if args.use_rnd:
            self.rnd_optimizer = make_optimizer(model.prediction_net.parameters(), args.rnd)
        else:
            self.rnd_optimizer = None

        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents
        self.VH = VH = len(self.value_heads)
        if args.use_tvf:
            self.K = K = len(self.tvf_horizons)
        else:
            self.K = K = 0

        self.action_dist = action_dist

        self.state_shape = model.input_dims
        self.policy_shape = [model.actions]

        self.batch_counter = 0

        self.noise_stats = {}
        self.vars = {}

        self.grad_accumulator = {}

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.obs = np.zeros([A, *self.state_shape], dtype=np.uint8)
        self.done = np.zeros([A], dtype=np.bool)

        if args.mutex_key:
            log.info(f"Using mutex key <yellow>{args.get_mutex_key}<end>")

        # includes final state as well, which is needed for final value estimate
        if args.obs_compression:
            # states must be decompressed with .decompress before use.
            log.info(f"Compression <green>enabled<end>")
            self.all_obs = np.zeros([N + 1, A], dtype=np.object)
        else:
            FORCE_PINNED = args.device != "cpu"
            if FORCE_PINNED:
                # make the memory pinned...
                all_obs = torch.zeros(size=[N + 1, A, *self.state_shape], dtype=torch.uint8)
                all_obs = all_obs.pin_memory()
                self.all_obs = all_obs.numpy()
            else:
                self.all_obs = np.zeros([N + 1, A, *self.state_shape], dtype=np.uint8)

        if args.upload_batch and not args.obs_compression:
            # in batch upload mode we can just keep all_obs on the GPU
            self.all_obs = torch.zeros(size=[N + 1, A, *self.state_shape], dtype=torch.uint8, device=self.model.device)

        if self.action_dist == "discrete":
            self.actions = np.zeros([N, A], dtype=np.int64)
        elif self.action_dist == "gaussian":
            self.actions = np.zeros([N, A, self.model.actions], dtype=np.float32)
        else:
            raise ValueError(f"Invalid distribution {self.action_dist}")
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.raw_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.raw_advantage = np.zeros([N, A], dtype=np.float32) # advantages before normalization
        self.terminals = np.zeros([N, A], dtype=np.bool)  # indicates prev_state was a terminal state.
        self.advantage = np.zeros([N, A], dtype=np.float32)  # advantage estimates

        self.all_time = np.zeros([N+1, A], dtype=np.int32)  # time for each step in rollout
        self.time = np.zeros([A], dtype=np.int32)  # current step for all agents

        self.replay_value_estimates = np.zeros([N, A], dtype=np.float32) # what is this?

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)

        self.intrinsic_reward_norm_scale: float = 1

        # value and returns
        self.value = np.zeros([N+1, A, VH], dtype=np.float32)
        self.tvf_value = np.zeros([N + 1, A, K, VH], dtype=np.float32)
        self.returns = np.zeros([N, A, VH], dtype=np.float32)
        self.tvf_returns = np.zeros([N, A, K, VH], dtype=np.float32)

        # returns generation
        self.advantage = np.zeros([N, A], dtype=np.float32)

        self.intrinsic_returns_rms = utils.RunningMeanStd(shape=())
        self.ems_norm = np.zeros([args.agents])

        # outputs tensors when clip loss is very high.
        self.log_high_grad_norm = True

        self.stats = {
            'reward_clips': 0,
            'game_crashes': 0,
            'action_repeats': 0,
            'batch_action_repeats': 0,
        }
        self.ep_count = 0
        self.episode_length_buffer = collections.deque(maxlen=1000)

        # create the replay buffer if needed
        self.replay_buffer: Optional[ExperienceReplayBuffer] = None
        if type(self.prev_obs) is torch.Tensor:
            replay_dtype = self.prev_obs[0].cpu().numpy().dtype
        else:
            replay_dtype = self.prev_obs.dtype

        if args.replay_size > 0:
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=args.replay_size,
                obs_shape=self.prev_obs.shape[2:],
                obs_dtype=replay_dtype,
                mode=args.replay_mode,
                thinning=args.replay_thinning,
            )

    @property
    def ext_value(self):
        return self.value[:, :, self.value_heads.index('ext')]

    @property
    def int_value(self):
        return self.value[:, :, self.value_heads.index('int')]

    @property
    def ext_returns(self):
        return self.returns[:, :, self.value_heads.index('ext')]

    @property
    def int_returns(self):
        return self.returns[:, :, self.value_heads.index('int')]

    def anneal(self, x, mode: str = "linear"):

        anneal_epoch = args.anneal_target_epoch or args.epochs
        factor = 1.0

        assert mode in ["off", "linear", "cos", "cos_linear", "linear_inc", "quad_inc"], f"invalid mode {mode}"

        if mode in ["linear", "cos_linear"]:
            factor *= np.clip(1-(self.step / (anneal_epoch * 1e6)), 0, 1)
        if mode in ["linear_inc"]:
            factor *= np.clip((self.step / (anneal_epoch * 1e6)), 0, 1)
        if mode in ["quad_inc"]:
            factor *= np.clip((self.step / (anneal_epoch * 1e6))**2, 0, 1)
        if mode in ["cos", "cos_linear"]:
            factor *= (1 + math.cos(math.pi * 2 * self.step / 20e6)) / 2

        return x * factor

    # todo: generalize this
    @property
    def value_lr(self):
        return self.anneal(args.value.lr, mode="linear" if args.value.lr_anneal else "off")

    @property
    def distil_lr(self):
        return self.anneal(args.distil.lr, mode="linear" if args.distil.lr_anneal else "off")

    @property
    def policy_lr(self):
        return self.anneal(args.policy.lr, mode="linear" if args.policy.lr_anneal else "off")

    @property
    def ppo_epsilon(self):
        return self.anneal(args.ppo_epsilon, mode="linear" if args.ppo_epsilon_anneal else "off")

    @property
    def rnd_lr(self):
        return args.rnd.lr

    def update_learning_rates(self):
        """
        Update learning rates for all optimizers
        Also log learning rates
        """

        self.log.watch("lr_policy", self.policy_lr, display_width=0)
        for g in self.policy_optimizer.param_groups:
            g['lr'] = self.policy_lr

        self.log.watch("lr_value", self.value_lr, display_width=0)
        for g in self.value_optimizer.param_groups:
            g['lr'] = self.value_lr

        if self.distil_optimizer is not None:
            self.log.watch("lr_distil", self.distil_lr, display_width=0)
            for g in self.distil_optimizer.param_groups:
                g['lr'] = self.distil_lr

        if self.rnd_optimizer is not None:
            self.log.watch("lr_rnd", self.rnd_lr, display_width=0)
            for g in self.rnd_optimizer.param_groups:
                g['lr'] = self.rnd_lr


    def create_envs(self, N=None, verbose=True, monitor_video=False):
        """ Creates (vectorized) environments for runner"""
        N = N or args.agents
        base_seed = args.seed
        if base_seed is None or base_seed < 0:
            base_seed = np.random.randint(0, 9999)
        env_fns = [lambda i=i: make_env(args.env_type, env_id=args.get_env_name(i), args=args, seed=base_seed+(i*997), monitor_video=monitor_video) for i in range(N)]

        if args.sync_envs:
            self.vec_env = gym.vector.SyncVectorEnv(env_fns)
        else:
            self.vec_env = hybridVecEnv.HybridAsyncVectorEnv(
                env_fns,
                copy=False,
                max_cpus=args.workers,
                verbose=True
            )

        if args.reward_normalization != "off":
            self.vec_env = wrappers.VecNormalizeRewardWrapper(
                self.vec_env,
                gamma=args.reward_normalization_gamma,
                ed_type=args.ed_mode if args.use_ed else None,
                mode=args.reward_normalization,
                clip=args.reward_normalization_clipping,
            )

        if args.max_repeated_actions > 0:
            self.vec_env = wrappers.VecRepeatedActionPenalty(self.vec_env, args.max_repeated_actions, args.repeated_action_penalty)

        if verbose:
            model_total_size = self.model.model_size(trainable_only=False)/1e6
            self.log.important("Generated {} agents ({}) using {} ({:.2f}M params) {} model.".
                           format(args.agents, "async" if not args.sync_envs else "sync", self.model.name,
                                  model_total_size, self.model.dtype))

    def save_checkpoint(self, filename, step, disable_replay=False, disable_optimizer=False, disable_log=False, disable_env_state=False):

        data = {
            'step': step,
            'ep_count': self.ep_count,
            'episode_length_buffer': self.episode_length_buffer,
            'model_state_dict': self.model.state_dict(),
            'batch_counter': self.batch_counter,
            'reward_scale': self.reward_scale,
            'stats': self.stats,
            'vars': self.vars,
        }

        if not disable_optimizer:
            data['policy_optimizer_state_dict'] = self.policy_optimizer.state_dict()
            data['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            if args.use_rnd:
                data['rnd_optimizer_state_dict'] = self.rnd_optimizer.state_dict()
            if args.distil.epochs > 0:
                data['distil_optimizer_state_dict'] = self.distil_optimizer.state_dict()
            if args.aux.epochs > 0:
                data['aux_optimizer_state_dict'] = self.aux_optimizer.state_dict()

        if not disable_log:
            data['logs'] = self.log
        if not disable_env_state:
            data['env_state'] = utils.save_env_state(self.vec_env)

        if args.use_sns:
            data['noise_stats'] = self.noise_stats

        if self.replay_buffer is not None and not disable_replay:
            data["replay_buffer"] = self.replay_buffer.save_state(force_copy=False)

        if args.use_intrinsic_rewards:
            data['ems_norm'] = self.ems_norm
            data['intrinsic_returns_rms'] = self.intrinsic_returns_rms

        if args.observation_normalization:
            data['obs_rms'] = self.model.obs_rms

        def torch_save(f):
            # protocol >= 4 allows for >4gb files
            torch.save(data, f, pickle_protocol=4)

        if args.checkpoint_compression:
            # torch will compress the weights, but not the additional data.
            # checkpoint compression makes a substantial difference to the filesize, especially if an uncompressed
            # replay buffer is being used.
            with self.open_fn(filename+".gz", "wb") as f:
                torch_save(f)
        else:
            with self.open_fn(filename, "wb") as f:
                torch_save(f)

    @property
    def open_fn(self):
        # level 5 compression is good enough.
        return lambda fn, mode: (gzip.open(fn, mode, compresslevel=5) if args.checkpoint_compression else open(fn, mode))

    def get_checkpoints(self, path):
        """ Returns list of (epoch, filename) for each checkpoint in given folder. """
        results = []
        if not os.path.exists(path):
            return []
        for f in os.listdir(path):
            if f.startswith("checkpoint") and (f.endswith(".pt") or f.endswith(".pt.gz")):
                epoch = int(f[11:14])
                results.append((epoch, f))
        results.sort(reverse=True)
        return results

    def load_checkpoint(self, checkpoint_path):
        """ Restores model from checkpoint. Returns current env_step"""

        checkpoint = _open_checkpoint(checkpoint_path, map_location=args.device)

        if not models.JIT:
            # remove tracemodule if jit is disabled.
            #print("debug:", list(checkpoint['model_state_dict'].keys()))
            checkpoint['model_state_dict'] = {
                k: v for k, v in checkpoint['model_state_dict'].items() if "trace_module" not in k
            }

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        if args.use_rnd:
            self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        if args.distil.epochs > 0:
            self.distil_optimizer.load_state_dict(checkpoint['distil_optimizer_state_dict'])
        if args.aux.epochs > 0:
            self.aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])

        step = checkpoint['step']
        self.log = checkpoint['logs']
        self.step = step
        self.ep_count = checkpoint.get('ep_count', 0)
        self.episode_length_buffer = checkpoint['episode_length_buffer']
        self.batch_counter = checkpoint.get('batch_counter', 0)
        self.stats = checkpoint.get('stats', 0)
        self.noise_stats = checkpoint.get('noise_stats', {})
        self.vars = checkpoint.get('vars', {})

        if self.replay_buffer is not None:
            self.replay_buffer.load_state(checkpoint["replay_buffer"])

        if args.use_intrinsic_rewards:
            self.ems_norm = checkpoint['ems_norm']
            self.intrinsic_returns_rms = checkpoint['intrinsic_returns_rms']

        if args.observation_normalization:
            self.model.obs_rms = checkpoint['obs_rms']
            self.model.refresh_normalization_constants()

        utils.restore_env_state(self.vec_env, checkpoint['env_state'])

        return step

    def reset(self):

        assert self.vec_env is not None, "Please call create_envs first."

        # initialize agent
        self.obs = self.vec_env.reset()
        self.done = np.zeros_like(self.done)
        self.episode_score *= 0
        self.episode_len *= 0
        self.step = 0
        self.time *= 0

        # reset stats
        for k in list(self.stats.keys()):
            v = self.stats[k]
            if type(v) in [float, int]:
                self.stats[k] *= 0

        self.batch_counter = 0

        self.episode_length_buffer.clear()
        # so that there is something in the buffer to start with.
        self.episode_length_buffer.append(1000)

    @torch.no_grad()
    def detached_batch_forward(self, obs:np.ndarray, **kwargs):
        """ Forward states through model, returns output, which is a dictionary containing
            "log_policy" etc.
            obs: np array of dims [B, *state_shape]

            Large inputs will be batched.
            Never computes gradients

            Output is always a tensor on the cpu.
        """

        # state_shape will be empty_list if compression is enabled
        B, *state_shape = obs.shape
        assert type(obs) in [np.ndarray, torch.Tensor], f"Obs was of type {type(obs)}, expecting np.ndarray"
        assert tuple(state_shape) in [tuple(), tuple(self.state_shape)]

        max_batch_size = args.max_micro_batch_size

        # break large forwards into batches
        if B > max_batch_size:

            batches = math.ceil(B / max_batch_size)
            batch_results = []
            for batch_idx in range(batches):
                batch_start = batch_idx * max_batch_size
                batch_end = min((batch_idx + 1) * max_batch_size, B)
                batch_result = self.detached_batch_forward(
                    obs[batch_start:batch_end],
                    **kwargs
                )
                batch_results.append(batch_result)
            keys = batch_results[0].keys()
            result = {}
            for k in keys:
                result[k] = torch.cat(tensors=[batch_result[k] for batch_result in batch_results], dim=0)
            return result
        else:
            if obs.dtype == np.object:
                obs = np.asarray([obs[i].decompress() for i in range(len(obs))])
            results = self.model.forward(obs, **kwargs)
            return results

    @property
    def tvf_horizons(self):
        assert args.use_tvf
        return self.model.tvf_fixed_head_horizons

    def calculate_tvf_returns(
            self,
            value_head: str = "ext", # ext|int
            obs=None,
            rewards=None,
            dones=None,
            tvf_return_mode=None,
            tvf_n_step=None,
    ):
        """
        Calculates and returns the (tvf_gamma discounted) (transformed) return estimates for given rollout.

        prev_states: ndarray of dims [N+1, B, *state_shape] containing prev_states
        rewards: float32 ndarray of dims [N, B] containing reward at step n for agent b
        value_sample_horizons: int32 ndarray of dims [K] indicating horizons to generate value estimates at.
        value_head: which head to use for estimate, i.e. ext_value, int_value, ext_sqr_value etc

        """

        # setup
        obs = obs if obs is not None else self.all_obs
        rewards = rewards if rewards is not None else self.ext_rewards
        dones = dones if dones is not None else self.terminals
        tvf_return_mode = tvf_return_mode or args.tvf_return_mode
        tvf_n_step = tvf_n_step or args.tvf_return_n_step

        N, A, *state_shape = obs[:-1].shape

        assert obs.shape == (N + 1, A, *state_shape)
        assert rewards.shape == (N, A)
        assert dones.shape == (N, A)

        # step 2: calculate the returns
        start_time = clock.time()

        # setup return estimator mode, but only verify occasionally.
        re_mode = args.tvf_return_estimator_mode
        if re_mode == "verify" and self.batch_counter % 31 != 1:
            re_mode = "default"

        # episodic discounting correction
        if args.use_ed:
            # time was time before action, we want time after action
            normalization_factors = wrappers.EpisodicDiscounting.get_normalization_constant(
                self.all_time,
                discount_type=args.ed_mode,
                discount_bias=args.ed_bias,
            )[:, :, None]
        else:
            normalization_factors = 1

        # we must unnormalize the value estimates, then renormalize after
        values = self.tvf_value[..., self.value_heads.index(value_head)] * normalization_factors

        returns = get_return_estimate(
            mode=tvf_return_mode,
            gamma=self.tvf_gamma,
            rewards=rewards,
            dones=dones,
            required_horizons=np.asarray(self.tvf_horizons),
            value_sample_horizons=np.asarray(self.tvf_horizons),
            value_samples=values,
            n_step=tvf_n_step,
            max_samples=args.tvf_return_samples,
            estimator_mode=re_mode,
            log=self.log,
            use_log_interpolation=args.tvf_return_use_log_interpolation,
        )

        if args.use_ed:
            returns = returns / normalization_factors[:N]

        return_estimate_time = clock.time() - start_time
        self.log.watch_mean(
            "time_return_estimate",
            return_estimate_time,
            display_precision=3,
            display_name="t_re",
        )
        return returns

    @torch.no_grad()
    def get_diversity(self, obs, buffer:np.ndarray, reduce_fn=np.nanmean, mask=None):
        """
        Returns an estimate of the feature-wise distance between input obs, and the given buffer.
        Only a sample of buffer is used.
        @param mask: list of indexes where obs[i] should not match with buffer[mask[i]]
        """

        samples = len(buffer)
        sample = np.random.choice(len(buffer), [samples], replace=False)
        sample.sort()

        if mask is not None:
            assert len(mask) == len(obs)
            assert max(mask) < len(buffer)

        buffer_output = self.detached_batch_forward(
            buffer[sample],
            output="value",
            include_features=True,
        )

        obs_output = self.detached_batch_forward(
            obs,
            output="value",
            include_features=True,
        )

        replay_features = buffer_output["raw_features"]
        obs_features = obs_output["raw_features"]

        distances = torch.cdist(replay_features[None, :, :], obs_features[None, :, :], p=2)
        distances = distances[0, :, :].cpu().numpy()

        # mask out any distances where buffer matches obs
        index_lookup = {index: i for i, index in enumerate(sample)}
        if mask is not None:
            for i, idx in enumerate(mask):
                if idx in index_lookup:
                    distances[index_lookup[idx], i] = float('NaN')

        reduced_values = reduce_fn(distances, axis=0)
        is_nan = np.isnan(reduced_values)
        if np.any(is_nan):
            self.log.warn("NaNs found in diversity calculation. Setting to zero.")
            reduced_values[is_nan] = 0
        return reduced_values

    def export_debug_frames(self, filename, obs, marker=None):
        # obs will be [N, 4, 84, 84]
        if type(obs) is torch.Tensor:
            obs = obs.cpu().detach().numpy()
        N, C, H, W = obs.shape
        import matplotlib.pyplot as plt
        obs = np.concatenate([obs[:, i] for i in range(4)], axis=-2)
        # obs will be [N, 4*84, 84]
        obs = np.concatenate([obs[i] for i in range(N)], axis=-1)
        # obs will be [4*84, N*84]
        if marker is not None:
            obs[:, marker*W] = 255
        plt.figure(figsize=(N, 4), dpi=84*2)
        plt.imshow(obs, interpolation='nearest')
        plt.savefig(filename)
        plt.close()

    def export_debug_value(self, filename, value):
        pass

    def get_current_action_std(self):

        if self.action_dist == "discrete":
            return 0.0
        elif self.action_dist == "gaussian":
            # hard coded for the moment (switch to log scale)
            return np.exp(np.clip(-0.7 + (0.7-1.6) * (self.step / 1e6), -1.6, -0.7))
        else:
            raise ValueError(f"invalid action distribution {self.action_dist}")


    def sample_actions(self, model_out, train:bool=True):
        """
        Returns action sampled from the output of the given policy.
        """
        if self.action_dist == "discrete":
            log_policy = model_out["log_policy"].cpu().numpy()
            return np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)
        elif self.action_dist == "gaussian":
            mu = np.tanh(model_out["raw_policy"].cpu().numpy())*1.1
            std = self.get_current_action_std() if train else 0.0
            return np.clip(np.random.randn(*mu.shape) * std + mu, -1, 1)
        else:
            raise ValueError(f"invalid action distribution {self.action_dist}")

    def trim_horizons(self, tvf_value_estimates, time, method: str = "timelimit", mode:str = "interpolate"):
        """
        Adjusts horizons by reducing horizons that extend over the timeout back to the timeout.
        This is for a few reasons.
        1. it makes use of the other heads, which means errors might average out
        2. it can be that shorter horizons get learned more quickly, and this will make use of them earlier on
        so long as the episodes are fairly long. If episodes are short compared to max_horizon this might not
        help at all.

        @param tvf_value_estimates: np array of dims [A, K]
        @param time: np array of dims [A] containing time associated with the states that generated these estimates.

        @returns new trimmed estimates of [A, K]
        """

        tvf_value_estimates = tvf_value_estimates.copy() # don't modify input

        # by definition h=0 is 0.0
        assert self.tvf_horizons[0] == 0, "First horizon must be zero"
        tvf_value_estimates[:, 0] = 0 # always make sure h=0 is fixed to zero.

        # step 1: work out the trimmed horizon
        # output is [A]
        if method == "off":
            return tvf_value_estimates
        elif method == "timelimit":
            time_till_termination = max((args.timeout / args.frame_skip) - time, self.N)
        elif method == "av_term":
            time_till_termination = np.maximum(np.percentile(self.episode_length_buffer, 95).astype(int) - time, 0) + 64
            self.log.watch_mean("*ttt_ep_length", np.percentile(self.episode_length_buffer, 95).astype(int))
            self.log.watch_mean("*ttt_ep_std", np.std(self.episode_length_buffer))
            self.log.watch_stats("ttt", time_till_termination, display_width=0)
        elif method == "est_term":
            # todo implement per state estimate of remaining time
            raise NotImplementedError()
        else:
            raise ValueError(f"Invalid trimming method {method}")

        # step 2: calculate new estimates
        if mode == "interpolate":
            # this can work a little bit better if trimming is only minimal.
            # however, all horizons still end up sharing the same estimate.
            old_tvf_value_estimates = tvf_value_estimates.copy()
            A, K, VH = tvf_value_estimates.shape
            trimmed_ks = np.searchsorted(self.tvf_horizons, time_till_termination)
            trimmed_value_estimate = interpolate(
                    np.asarray(self.tvf_horizons),
                    old_tvf_value_estimates[..., 0], # select final value head
                    time_till_termination
                )
            for a in range(A):
                tvf_value_estimates[a, trimmed_ks[a]:] = trimmed_value_estimate[a]
            return tvf_value_estimates
        elif mode == "average":
            # we can use any horizon with h > remaining_time interchangeably with h.
            # so may as well average over them.

            # first compute the averaged values
            # output is averaged_tvf_value_estimate of dims [A, K] where averaged_tvf_value_estimate[:, k] is the
            # sum over the final K-k horizons. For example averaged_tvf_value_estimate[:,-1] is the final horizon
            # and averaged_tvf_value_estimate[:,-2] is the average of the final two horizon estimates.
            averaged_tvf_value_estimates = tvf_value_estimates.copy()
            accumulator = averaged_tvf_value_estimates[:, -1].copy()
            counter = 1
            for k in reversed(range(len(self.tvf_horizons)-1)):
                accumulator += averaged_tvf_value_estimates[:, k]
                counter += 1
                averaged_tvf_value_estimates[:, k] = accumulator / counter

            trimmed_ks = np.searchsorted(self.tvf_horizons, time_till_termination)

            for a, trimmed_k in enumerate(trimmed_ks):
                if trimmed_k >= len(self.tvf_horizons)-1:
                    # this means no trimming
                    continue
                # using this method all horizons longer than ttt can be trimmed to the same value
                # it might be better to use the average up to the h rather than up to h_max
                tvf_value_estimates[a, trimmed_k:] = averaged_tvf_value_estimates[a, trimmed_k]

            return tvf_value_estimates
        elif mode == "average2":
            # average up to h but no further
            # implementation is a bit slow, drop if it's not better.
            old_value_estimates = tvf_value_estimates.copy()
            trimmed_ks = np.searchsorted(self.tvf_horizons, time_till_termination)
            for a, trimmed_k in enumerate(trimmed_ks):
                if trimmed_k >= len(self.tvf_horizons)-1:
                    # this means no trimming
                    continue
                acc = 0
                counter = 0
                for k in range(trimmed_k, self.K):
                    # note: this could be tvf_value_estimate, but I want to make it explicit that we're using
                    # the original values.
                    acc += old_value_estimates[a, k]
                    counter += 1
                    tvf_value_estimates[a, k] = acc / counter
            return tvf_value_estimates
        elif mode == "average3":
            # average up to but not including h but no further
            # this is based on the following ideas
            # 1. average over as many horizons as we can.
            # 2. try to not refer to ourself (bootstrap) or any future horizons.
            trimmed_ks = np.searchsorted(self.tvf_horizons, time_till_termination)
            v2 = tvf_value_estimates.copy()
            # note, this is going to be quite slow...
            for a, trimmed_k in enumerate(trimmed_ks):
                if trimmed_k >= len(self.tvf_horizons)-1:
                    # this means no trimming
                    continue
                # if we can trim to head x then we want...
                # head_x = head_x
                # head_{x+1} = head_x
                # head_{x+2} = (head_x+head_{x+1}) / 2
                # ....
                # the final head is never used, as we only use heads *less* than ourselves, and the last head can
                # never be less than any head (unless no trimming is required)
                averages = np.cumsum(tvf_value_estimates[a, trimmed_k:-1, 0]) / np.arange(1, self.K-trimmed_k, dtype=np.float32)
                tvf_value_estimates[a, trimmed_k+1:, 0] = averages

                # old method
                # acc = float(tvf_value_estimates[a, trimmed_k, 0])
                # counter = 1
                # for k in range(trimmed_k+1, self.K):
                #     old_value = tvf_value_estimates[a, k, 0]
                #     tvf_value_estimates[a, k, 0] = acc / counter
                #     acc += old_value
                #     counter += 1

            return tvf_value_estimates
        elif mode == "substitute":
            # just use the smallest h we can, very simple.
            untrimmed_ks = np.arange(self.K)[None, :]
            trimmed_ks = np.searchsorted(self.tvf_horizons, time_till_termination)[:, None]
            trimmed_ks = np.minimum(trimmed_ks, untrimmed_ks)
            tvf_value_estimates = np.take_along_axis(tvf_value_estimates[:, :, 0], trimmed_ks, axis=1)
            return tvf_value_estimates[:, :, None]
        else:
            raise ValueError(f"Invalid trimming mode {mode}")

    @torch.no_grad()
    def generate_rollout(self):

        assert self.vec_env is not None, "Please call create_envs first."

        def upload_if_needed(x):
            if type(self.all_obs) is torch.Tensor:
                x = torch.from_numpy(x).to(self.all_obs.device)
            return x

        self.model.train()

        reward_scale_before_rollout = self.reward_scale

        self.int_rewards *= 0
        self.ext_rewards *= 0
        self.value *= 0
        self.tvf_value *= 0
        self.all_time *= 0

        for k in self.stats.keys():
            if k.startswith("batch_"):
                self.stats[k] *= 0

        for t in range(self.N):

            prev_obs = self.obs.copy()
            prev_time = self.time.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.detached_batch_forward(
                self.obs,
                output="default",
                include_rnd=args.use_rnd,
                update_normalization=True
            )

            # sample actions and run through environment.
            actions = self.sample_actions(model_out)
            self.obs, ext_rewards, dones, infos = self.vec_env.step(actions)
            self.time = np.asarray([info["time"] for info in infos], dtype=np.int32)

            if args.use_rnd:
                # update the intrinsic rewards
                self.int_rewards[t] += model_out["rnd_error"].detach().cpu().numpy()

            # per step reward noise
            if args.noisy_reward > 0:
                ext_rewards = add_relative_noise(ext_rewards, args.noisy_reward)

            # save raw rewards for monitoring the agents progress
            raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(ext_rewards, infos)],
                                     dtype=np.float32)

            if args.noisy_zero >= 0:
                ext_rewards = np.random.normal(0, args.noisy_zero, size=ext_rewards.shape).astype(np.float32)
                raw_rewards *= 0

            self.episode_score += raw_rewards
            self.episode_len += 1

            # log repeated action stats
            if 'max_repeats' in infos[0]:
                self.log.watch_mean('max_repeats', infos[0]['max_repeats'], display_name="reps", display_width=7)
            if 'mean_repeats' in infos[0]:
                self.log.watch_mean('mean_repeats', infos[0]['mean_repeats'], display_width=0)

            # process each environment, check if they have finished
            for i, (done, info) in enumerate(zip(dones, infos)):
                if "reward_clips" in info:
                    self.stats['reward_clips'] += info["reward_clips"]
                if "game_freeze" in info:
                    self.stats['game_crashes'] += 1
                if "repeated_action" in info:
                    self.stats['action_repeats'] += 1
                if "repeated_action" in info:
                    self.stats['batch_action_repeats'] += 1

                if done:
                    # this should be always updated, even if it's just a loss of life terminal
                    self.episode_length_buffer.append(info["ep_length"])

                    if "fake_done" in info:
                        # this is a fake reset on loss of life...
                        continue

                    # reset is handled automatically by vectorized environments
                    # so just need to keep track of book-keeping
                    self.ep_count += 1
                    self.log.watch_full("ep_score", info["ep_score"], history_length=100)
                    self.log.watch_full("ep_length", info["ep_length"])
                    self.log.watch_mean("ep_count", self.ep_count, history_length=1)

                    self.episode_score[i] = 0
                    self.episode_len[i] = 0

            # compress observations if needed
            if args.obs_compression:
                prev_obs = np.asarray([compression.BufferSlot(prev_obs[i]) for i in range(len(prev_obs))])

            # take advantage of the fact that V_h = V_min(h, remaining_time).
            if args.use_tvf:
                start_time = clock.time()
                tvf_values = model_out["tvf_value"].cpu().numpy()
                self.tvf_value[t] = self.trim_horizons(
                    tvf_values,
                    prev_time,
                    method=args.tvf_trimming,
                    mode=args.tvf_trimming_mode
                )
                ms = (clock.time() - start_time) * 100
                self.log.watch_mean("*t_trim", ms)

            # get all the information we need from the model
            self.all_obs[t] = upload_if_needed(prev_obs)
            self.all_time[t] = prev_time
            self.value[t] = model_out["value"].cpu().numpy()
            self.actions[t] = actions
            self.ext_rewards[t] = ext_rewards
            self.log_policy[t] = model_out["log_policy"].cpu().numpy()
            self.raw_policy[t] = model_out["raw_policy"].cpu().numpy()
            self.terminals[t] = dones
            self.done = dones

        # process the final state
        if args.obs_compression:
            last_obs = np.asarray([compression.BufferSlot(self.obs[i]) for i in range(len(self.obs))])
        else:
            last_obs = self.obs
        self.all_obs[-1] = upload_if_needed(last_obs)
        self.all_time[-1] = self.time
        final_model_out = self.detached_batch_forward(self.obs, output="default")
        self.value[-1] = final_model_out["value"].cpu().numpy()

        if args.use_tvf:
            self.tvf_value[-1] = self.trim_horizons(
                final_model_out["tvf_value"].cpu().numpy(),
                self.time,
                method=args.tvf_trimming,
                mode=args.tvf_trimming_mode
            )

        # turn off train mode (so batch norm doesn't update more than once per example)
        self.model.eval()

        self.int_rewards = np.clip(self.int_rewards, -5, 5) # just in case there are extreme values here

        aux_fields = {}

        # calculate targets for ppg
        if args.aux.epochs > 0:
            v_target = td_lambda(
                self.ext_rewards,
                self.ext_value[:self.N],
                self.ext_value[self.N],
                self.terminals,
                self.gamma,
                args.lambda_value
            )
            aux_fields['vtarg'] = utils.merge_down(v_target)

        # add data to replay buffer if needed
        steps = (np.arange(args.n_steps * args.agents) + self.step)
        if self.replay_buffer is not None:
            self.replay_buffer.add_experience(
                utils.merge_down(self.prev_obs),
                self.replay_buffer.create_aux_buffer(
                    (len(steps),),
                    reward=utils.merge_down(self.ext_rewards),
                    time=utils.merge_down(self.prev_time),
                    action=utils.merge_down(self.actions),
                    step=steps,
                    **aux_fields,
                )
            )

        # remember what reward scale was used when we generated this rollout
        # note, we ignore the first update as reward scale may initialize to 0.
        if args.auto_weight_scaling and self.step > 0:
            ratio = self.reward_scale / reward_scale_before_rollout
            self.log.watch_mean("rs_ratio", ratio)
            self.model.adjust_value_scale(ratio, process_value=args.tvf_include_ext)
            # just a quick check... really just want to make sure nothing is exploding or zeroing out.
            self.log.watch_mean("rs_tmag", self.model.value_net.tvf_head.weight.data[-1].std()) # just look at final head
            self.log.watch_mean("rs_vmag", self.model.value_net.value_head.weight.data.std())


    def get_ema_constant(self, required_horizon: int, updates_every: int = 1):
        """
        Returns an ema coefficent to match given horizon (in environment interactions), when updates will be applied
        every "updates_every" rollouts
        """
        if required_horizon == 0:
            return 0
        updates_every_steps = (updates_every * self.N * self.A)
        ema_horizon = required_horizon / updates_every_steps
        return 1 - (1 / ema_horizon)

    def process_noise_scale(
            self,
            g_b_small_squared: float,
            g_b_big_squared: float,
            label: str,
            verbose: bool = True,
            b_big = None
        ):
        """
        Logs noise levels using provided gradients
        """

        b_small = args.sns_b_small
        b_big = b_big or args.sns_b_big

        est_s = (g_b_small_squared - g_b_big_squared) / (1 / b_small - 1 / b_big)
        est_g2 = (b_big * g_b_big_squared - b_small * g_b_small_squared) / (b_big - b_small)

        if args.sns_smoothing_mode == "avg":
            # add these samples to the mix
            for var_name, var_value in zip(['s', 'g2'], [est_s, est_g2]):
                if f'{label}_{var_name}_history' not in self.noise_stats:
                    history_frames = int(args.sns_smoothing_horizon_avg) # 5 million frames should be about right
                    ideal_updates_length = history_frames / (self.N*self.A)
                    buffer_len = int(max(10, math.ceil(ideal_updates_length / args.sns_period)))
                    self.noise_stats[f'{label}_{var_name}_history'] = deque(maxlen=buffer_len)
                self.noise_stats[f'{label}_{var_name}_history'].append(var_value)
                self.noise_stats[f'{label}_{var_name}'] = np.mean(self.noise_stats[f'{label}_{var_name}_history'])
        elif args.sns_smoothing_mode == "ema":
            ema_s = self.get_ema_constant(args.sns_smoothing_horizon_s, args.sns_period)
            g2_horizon = args.sns_smoothing_horizon_policy if label == "policy" else args.sns_smoothing_horizon_g2
            ema_g2 = self.get_ema_constant(g2_horizon, args.sns_period)
            # question: we do we need to smooth both of these? which is more noisy? I think it's just g2 right?
            utils.dictionary_ema(self.noise_stats, f'{label}_s', est_s, ema_s)
            utils.dictionary_ema(self.noise_stats, f'{label}_g2', est_g2, ema_g2)
        else:
            raise ValueError(f"Invalid smoothing mode {args.sns_smoothing_mode}.")

        smooth_s = float(self.noise_stats[f'{label}_s'])
        smooth_g2 = float(self.noise_stats[f'{label}_g2'])

        # g2 estimate is frequently negative. If ema average bounces below 0 the ratio will become negative.
        # to avoid this we clip the *smoothed* g2 to epsilon.
        # alternative include larger batch_sizes, and / or larger EMA horizon.
        # noise levels above 1000 will not be very accurate, but around 20 should be fine.
        epsilon = 1e-4 # we can't really measure noise above this level anyway (which is around a ratio of 10k:1)
        ratio = (smooth_s) / (max(0.0, smooth_g2) + epsilon)

        self.noise_stats[f'{label}_ratio'] = ratio
        if 'head' in label:
            # keep track of which heads we have results for
            try:
                idx = int(label.split("_")[-1])
                if 'active_heads' not in self.noise_stats:
                    self.noise_stats['active_heads'] = set()
                self.noise_stats['active_heads'].add(idx)
            except:
                # this is fine
                pass

        # maybe this is too much logging?
        self.log.watch(f'sns_{label}_smooth_s', smooth_s, display_precision=0, display_width=0, display_name=f"sns_{label}_s")
        self.log.watch(f'sns_{label}_smooth_g2', smooth_g2, display_precision=0, display_width=0, display_name=f"sns_{label}_g2")
        self.log.watch(f'sns_{label}_s', est_s, display_precision=0, display_width=0)
        self.log.watch(f'sns_{label}_g2', est_g2, display_precision=0, display_width=0)
        self.log.watch(f'sns_{label}_b', ratio, display_precision=0, display_width=0)
        self.log.watch(
            f'sns_{label}',
            np.clip(ratio, 0, float('inf')) ** 0.5,
            display_precision=0,
            display_width=8 if verbose else 0,
        )

        return self.noise_stats[f'{label}_ratio']

    def estimate_noise_scale(
            self,
            batch_data,
            mini_batch_func,
            optimizer: torch.optim.Optimizer,
            label,
            verbose: bool = True,
    ):
        """
        Estimates the critical batch size using the gradient magnitude of a small batch and a large batch

        ema smoothing produces cleaner results, but is biased.

        new version...

        See: https://arxiv.org/pdf/1812.06162.pdf
        """

        b_small = args.sns_b_small

        if label == "policy":
            # always use full batch for policy (it's required to get the precision needed)
            b_big = self.N * self.A
        else:
            b_big = args.sns_b_big

        # resample data
        # this also shuffles order
        data = {}
        samples = np.random.choice(range(len(batch_data["prev_state"])), b_big, replace=False)
        for k, v in batch_data.items():
            data[k] = batch_data[k][samples]

        assert b_big % b_small == 0, "b_small must divide b_big"
        mini_batches = b_big // b_small

        small_norms_sqr = []
        big_grad = None

        for i in range(mini_batches):
            optimizer.zero_grad(set_to_none=True)
            segment = slice(i * b_small, (i + 1) * b_small)
            mini_batch_data = {}
            for k, v in data.items():
                mini_batch_data[k] = data[k][segment]
            # todo: make this a with no log...
            self.log.mode = self.log.LM_MUTE
            mini_batch_func(mini_batch_data)
            self.log.mode = self.log.LM_DEFAULT
            # get small grad
            small_norms_sqr.append(utils.optimizer_grad_norm(optimizer) ** 2)
            if i == 0:
                big_grad = [x.clone() for x in utils.list_grad(optimizer)]
            else:
                for acc, p in zip(big_grad, utils.list_grad(optimizer)):
                    acc += p

        optimizer.zero_grad(set_to_none=True)
        g_b_big_squared = float((utils.calc_norm(big_grad) / mini_batches) ** 2)
        g_b_small_squared = float(np.mean(small_norms_sqr))
        self.process_noise_scale(g_b_small_squared, g_b_big_squared, label, verbose, b_big=b_big)


    @torch.no_grad()
    def log_dna_value_quality(self):
        targets = calculate_bootstrapped_returns(
            self.ext_rewards, self.terminals, self.ext_value[self.N], self.gamma
        )
        values = self.ext_value[:self.N]
        ev = utils.explained_variance(values.ravel(), targets.ravel())
        self.log.watch_mean("ev_ext", ev, history_length=1)

        # use ev_ext for ev_average when not using tvf
        self.log.watch_mean(
            f"ev_average", ev,
            display_width=8,
            display_name="ev_avg",
            history_length=1
        )

        self.log.watch_mean(
            f"z_value_bias",
            np.mean(values),
            display_width=0,
            history_length=1
        )
        self.log.watch_mean(
            f"z_target_bias",
            np.mean(targets),
            display_width=0,
            history_length=1
        )
        self.log.watch_mean(
            f"z_value_var",
            np.var(values),
            display_width=0,
            history_length=1
        )
        self.log.watch_mean(
            f"z_target_var",
            np.var(targets),
            display_width=0,
            history_length=1
        )


    def _log_curve_quality(self, estimates, targets, postfix: str = ''):
        """
        Calculates explained variance at each of the debug horizons
        @param estimates: np array of dims[N,A,K]
        @param targets: np array of dims[N,A,K]
        @param postfix: postfix to add after the name during logging.
        where K is the length of tvf_debug_horizons

        """

        def log_head(head_index:int, name:str = None):

            if name is None:
                name = str(head_index)

            value = estimates[:, :, head_index].reshape(-1)
            target = targets[:, :, head_index].reshape(-1)

            var = np.var(target)
            not_explained_var = np.var(target - value)

            ev = 0 if (var == 0) else np.clip(1 - not_explained_var / var, -1, 1)

            self.log.watch_mean(
                f"ev_{name}" + postfix,
                ev,
                display_width=0,
                history_length=1
            )

            if args.noisy_zero >= 0:
                # special stats for learning zero rewards
                self.log.watch_mean(
                    f"z_value_bias_{name}" + postfix,
                    np.mean(value),
                    display_width=0,
                    history_length=1
                )
                self.log.watch_mean(
                    f"z_target_bias_{name}" + postfix,
                    np.mean(target),
                    display_width=0,
                    history_length=1
                )
                self.log.watch_mean(
                    f"z_value_var_{name}" + postfix,
                    np.var(value),
                    display_width=0,
                    history_length=1
                )
                self.log.watch_mean(
                    f"z_target_var_{name}" + postfix,
                    np.var(target),
                    display_width=0,
                    history_length=1
                )
            return var, not_explained_var

        total_not_explained_var = 0
        total_var = 0
        heads_to_log = utils.even_sample_down(range(len(self.tvf_horizons)), args.sns_max_heads)
        for i, head_index in enumerate(heads_to_log):
            this_var, this_not_explained_var = log_head(head_index)
            total_var += this_var
            total_not_explained_var += this_not_explained_var

        # log first, last, and mid
        log_head(heads_to_log[0], "first")
        log_head(heads_to_log[-1], "last")
        log_head(heads_to_log[len(heads_to_log)//2], "mid")

        self.log.watch_mean(
            f"ev_average"+postfix,
            0 if (total_var == 0) else np.clip(1 - total_not_explained_var / total_var, -1, 1),
            display_width=8,
            display_name="ev_avg"+postfix,
            history_length=1
        )

        if args.noisy_zero >= 0:
            # todo: clean this up..
            targets = calculate_bootstrapped_returns(
                self.ext_rewards, self.terminals, self.ext_value[self.N], self.gamma
            )
            values = self.ext_value[:self.N]
            self.log.watch_mean(
                f"z_value_bias",
                np.mean(values),
                display_width=0,
                history_length=1
            )
            self.log.watch_mean(
                f"z_target_bias",
                np.mean(targets),
                display_width=0,
                history_length=1
            )
            self.log.watch_mean(
                f"z_value_var",
                np.var(values),
                display_width=0,
                history_length=1
            )
            self.log.watch_mean(
                f"z_target_var",
                np.var(targets),
                display_width=0,
                history_length=1
            )



    @torch.no_grad()
    def log_tvf_curve_quality(self):
        """
        Writes value quality stats to log
        """

        N, A, *state_shape = self.prev_obs.shape
        K = len(self.tvf_horizons)

        targets = self.calculate_tvf_returns(
            value_head='ext',
            obs=self.all_obs,
            rewards=self.ext_rewards,
            dones=self.terminals,
            tvf_return_mode="fixed",  # <-- MC is the least bias method we can do...
            tvf_n_step=args.n_steps,
        )

        first_moment_targets = targets
        first_moment_estimates = self.tvf_value[:N, :, :, 0].reshape(N, A, K)
        self._log_curve_quality(first_moment_estimates, first_moment_targets)

        # also log ev_ext
        targets = calculate_bootstrapped_returns(
            self.ext_rewards, self.terminals, self.ext_value[self.N], self.gamma
        )
        values = self.ext_value[:self.N]
        ev = utils.explained_variance(values.ravel(), targets.ravel())
        self.log.watch_mean("ev_ext", ev, history_length=1)

    @property
    def prev_obs(self):
        """
        Returns prev_obs with size [N,A] (i.e. missing final state)
        """
        return self.all_obs[:-1]

    @property
    def final_obs(self):
        """
        Returns final observation
        """
        return self.all_obs[-1]

    @property
    def prev_time(self):
        """
        Returns prev_time with size [N,A] (i.e. missing final state)
        """
        return self.all_time[:-1]

    def final_time(self):
        """
        Returns final time
        """
        return self.all_time[-1]

    def generate_sampled_return_targets(self, ext_value_estimates: np.ndarray):
        """
        Generates targets for value function, used only in PPO and DNA.
        """
        assert not args.use_tvf

        N_plus_one, A = ext_value_estimates.shape
        N = N_plus_one - 1

        SAMPLES = 10

        # really trying to make samples be different between runs here.
        h = 1/(1 - args.lambda_value)
        lambdas = [1 - (1 / (factor * h)) for factor in np.geomspace(0.25, 4.0, SAMPLES)]

        advantage_estimate = np.zeros([SAMPLES, N, A], dtype=np.float32)

        for i, lamb in enumerate(lambdas):
            advantage_estimate[i] = gae(
                self.ext_rewards,
                ext_value_estimates[:N],
                ext_value_estimates[N],
                self.terminals,
                self.gamma,
                lamb,
            )

        sample = np.random.randint(size=[N, A], low=0, high=SAMPLES)
        values = (advantage_estimate + ext_value_estimates[:N])
        return np.take_along_axis(values, sample[None, :, :], axis=0)

    @torch.no_grad()
    def estimate_horizon_from_rediscounting(self):

        VALUE_HEAD_INDEX = 0

        def rve(h, source=None):
            if source is None:
                source = self.tvf_value[..., VALUE_HEAD_INDEX]
            # for debuging
            N, A, K = source.shape
            return get_rediscounted_value_estimate(
                values=source.reshape([N * A, K]),
                old_gamma=self.tvf_gamma,
                new_gamma=1-(1/h),
                horizons=self.tvf_horizons,
            ).reshape([-1])

        hs = np.geomspace(args.ag_min_h, args.ag_max_h, 128)

        def get_ratios(source: str):

            N, A = self.N, self.A

            vars = []
            means = []
            ratios = []
            for h in hs:

                if source == "returns":
                    _rve = rve(h, source=self.tvf_returns[..., VALUE_HEAD_INDEX])
                elif source == "value":
                    _rve = rve(h)
                elif source == "td":
                    _rve = rve(h)
                    _rve = td_lambda(self.ext_rewards, _rve[:N], _rve[N], self.terminals, 1 - (1 / h), args.lambda_policy)
                elif source == "advantages":
                    _ve = rve(h)
                    _rve = td_lambda(self.ext_rewards, _ve[:N], _ve[N], self.terminals, 1 - (1 / h), args.lambda_policy)
                    _rve += _ve[:N]
                else:
                    raise ValueError(f"Invalid ratio source {args.ag_ratio_source}")

                # I fell this is the correct way to do it, go std over each trajectory, but std over the entire
                # thing works a lot better in practice.

                var = _rve.var()
                l2s = _rve.mean()**2 # squared mean
                ratio = var / (l2s + 1e-6)

                # old method..
                # std = _rve.std()
                # mean = _rve.mean()
                # ratio = std / (np.abs(mean) + 1e-6)

                vars.append(var)
                means.append(l2s)
                ratios.append(ratio)

            return np.asarray(ratios), np.asarray(vars), np.asarray(means),

        def score(ratios, i: int):
            ratio = ratios[i]
            return 0.2 * np.log(100 + hs[i]) - np.log(ratio+2e-1)

        if args.debug_log_rediscount_curve and self.batch_counter % 64 == 0:

            import matplotlib.pyplot as plt

            data = []
            keys = ['returns', 'value', 'td', 'advantages']
            for source in keys:
                data.append(get_ratios(source))

            plt.figure(figsize=(12, 6))
            cm = plt.get_cmap('tab10')
            for i, (ratio, vars, means) in enumerate(data):
                key = keys[i]
                plt.plot(hs, vars, label=f'{key}_var', color=cm(i), ls="-")
                plt.plot(hs, means, label=f'{key}_norm', color=cm(i), ls="--")
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(alpha=0.25)
            plt.legend()
            epoch = self.step / 1e6
            plt.savefig(args.log_folder + f"/rediscount_{epoch:05.2f}.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            for key, (ratios, vars, means) in zip(keys, data):
                plt.plot(hs, np.minimum(ratios, 5), label=f'ratio_{key}')
            plt.xscale('log')
            plt.yscale('log')
            plt.hlines(args.ag_ratio_threshold, args.ag_min_h, args.ag_max_h, ls="--", color="gray")
            plt.ylim(1e-3, 1e1)
            plt.grid(alpha=0.25)
            plt.legend()

            plt.savefig(args.log_folder + f"/rediscount_ratio_{epoch:05.2f}.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            for key, (ratios, vars, means) in zip(keys, data):
                scores = [score(ratios, i) for i in range(len(ratios))]
                plt.plot(hs, scores, label=f'score_{key}')
            plt.legend()
            plt.xscale('log')
            plt.grid(alpha=0.25)
            plt.savefig(args.log_folder + f"/rediscount_score_{epoch:05.2f}.png")
            plt.close()

            # save data for later...
            with open(args.log_folder + f"/data_{epoch:05.2f}.dat", 'wb') as f:
                import pickle
                save = {
                    'data': data,
                    'returns': self.tvf_returns[..., VALUE_HEAD_INDEX], # just in case we need these...
                    'horizons': self.tvf_horizons
                }
                pickle.dump(save, f)


        ratios, vars, means = get_ratios(args.ag_ratio_source)

        scores = [score(ratios, i) for i in range(len(ratios))]

        # find and log the gamma the minimized variance
        min_h = hs[np.argmin(ratios)]
        max_h = 0
        for ratio, h in zip(ratios, hs):
            if ratio < args.ag_ratio_threshold:
                max_h = h

        best_h = max(min_h, max_h)
        self.log.watch("*dc_min", min_h)
        self.log.watch("*dc_max", max_h)
        self.log.watch("*dc_best", best_h)
        self.vars['dc_best'] = best_h

        if args.ag_ratio_algorithm == "min":
            target_h = min_h
        elif args.ag_ratio_algorithm == "best":
            target_h = best_h
        elif args.ag_ratio_algorithm == "fixed":
            target_h = args.ag_initial_h
        elif args.ag_ratio_algorithm == "adv":
            best_i = np.argmax(scores)
            target_h = hs[best_i]
        elif args.ag_ratio_algorithm == "adv2":
            best_ratio = np.min(ratios)
            if best_ratio < args.ag_ratio_threshold:
                target_h = args.ag_max_h
            else:
                target_h = hs[np.argmin(ratios)]
        else:
            raise ValueError(f"Invalid ag_ratio_algorithm {args.ag_ratio_algorithm}")

        utils.dictionary_ema(self.vars, 'dc_h', target_h, 0.99, default=args.ag_initial_h, log=True)
        self.log.watch("dc_h", self.vars['dc_h'])
        self.log.watch("*dc_target", target_h)

        # plt.figure(figsize=(12, 6))
        # plt.plot(hs, td_stds, label='std')
        # plt.plot(hs, td_means, label='mean')
        # plt.xscale('log')
        # plt.grid(alpha=0.25)
        # plt.legend()
        # epoch = self.step / 1e6
        # plt.savefig(args.log_folder+f"/td_{epoch:05.2f}.png")
        # plt.close()
        #
        # td_ratios = td_stds / (np.abs(td_means) + 1e-6)
        # plt.figure(figsize=(12, 6))
        # plt.plot(hs, np.minimum(td_ratios, 5))
        # plt.xscale('log')
        # plt.grid(alpha=0.25)
        # plt.savefig(args.log_folder + f"/td_ratio_{epoch:05.2f}.png")
        # plt.close()


    @torch.no_grad()
    def get_tvf_ext_value_estimate(self, new_gamma: float):
        """

        Returns rediscounted value estimate for given rollout (i.e. rewards + value if using given gamma)
        Usually this is just GAE, but if gamma != tvf_gamma, then rediscounting is applied.

        We expect:
        self.tvf_value: np array of dims [N+1, A, K, VH] containing value estimates for each horizon K and each value head VH

        @returns value estimate for gamma=gamma for example [N+1, A]
        """

        assert args.use_tvf
        N, A, K, VH = self.tvf_value[:self.N].shape

        VALUE_HEAD_INDEX = self.value_heads.index('ext')

        # [N, A, K]
        tvf_values = self.tvf_value[:, :, :, VALUE_HEAD_INDEX]

        if abs(new_gamma - self.tvf_gamma) < 1e-8:
            return tvf_values[:, :, -1]

        # otherwise... we need to rediscount...
        return get_rediscounted_value_estimate(
            values=tvf_values.reshape([(N + 1) * A, -1]),
            old_gamma=self.tvf_gamma,
            new_gamma=new_gamma,
            horizons=self.tvf_horizons,
        ).reshape([(N + 1), A])

    def calculate_intrinsic_returns(self):

        if not args.use_intrinsic_rewards:
            return 0

        N, A, *state_shape = self.prev_obs.shape

        if args.normalize_intrinsic_rewards:
            # normalize returns using EMS
            # this is this how openai did it (i.e. forward rather than backwards)
            for t in range(self.N):
                terminals = (not args.intrinsic_reward_propagation) * self.terminals[t, :]
                self.ems_norm = (1 - terminals) * args.gamma_int * self.ems_norm + self.int_rewards[t, :]
                self.intrinsic_returns_rms.update(self.ems_norm.reshape(-1))

            # normalize the intrinsic rewards
            # the 0.4 means that the returns average around 1, which is similar to where the
            # extrinsic returns should average. I used to have a justification for this being that
            # the clipping during normalization means that the input has std < 1 and this accounts for that
            # but since we multiply by EMS normalizing scale this shouldn't happen.
            # in any case, the 0.4 is helpful as it means |return_int| ~ |return_ext|, which is good when
            # we try to set the intrinsic_reward_scale hyperparameter.
            # One final thought, this also means that return_ratio, under normal circumstances, should be
            # about 1.0
            self.intrinsic_reward_norm_scale = (1e-5 + self.intrinsic_returns_rms.var ** 0.5)
            self.int_rewards = (self.int_rewards / self.intrinsic_reward_norm_scale)

        int_advantage = gae(
            self.int_rewards,
            self.int_value[:N],
            self.int_value[N],
            (not args.intrinsic_reward_propagation) * self.terminals,
            gamma=args.gamma_int,
            lamb=args.lambda_policy
        )

        self.int_returns[:] = int_advantage + self.int_value[:N]
        return int_advantage

    def calculate_returns(self):
        """
        Calculates return targets for all value heads as required
        """

        self.returns *= 0
        self.tvf_returns *= 0
        N, A, *state_shape = self.prev_obs.shape

        self.model.eval()

        # 1. first we calculate 'ext_value' estimate, which is the primarily value estimate
        if args.use_tvf:
            ext_value_estimates = self.get_tvf_ext_value_estimate(new_gamma=self.gamma)
        else:
            # in this case just use the value networks value estimate
            ext_value_estimates = self.ext_value

        if args.use_ed:
            # most of these requirements aren't strictly needed, I just can' be bothered coding them up.
            assert self.gamma == 1.0, "ed requires gamma=1.0 for the moment."
            assert args.use_tvf, "ed requires tvf enabled for the moment."
            normalization_factors = wrappers.EpisodicDiscounting.get_normalization_constant(
                self.all_time,
                discount_type=args.ed_mode,
                discount_bias=args.ed_bias,
            )
            ext_advantage = ed_gae(
                rewards=self.ext_rewards,
                values=ext_value_estimates,
                normalization_factors=normalization_factors,
                batch_terminal=self.terminals,
                lamb=args.lambda_policy,
            )
            # ext_returns should probably not be used... they might work I guess...
            self.ext_returns[:] = ext_advantage + ext_value_estimates[:N]
        else:
            ext_advantage = gae(
                self.ext_rewards,
                ext_value_estimates[:N],
                ext_value_estimates[N],
                self.terminals,
                self.gamma,
                args.lambda_policy
            )
            # calculate ext_returns.
            self.ext_returns[:] = td_lambda(
                self.ext_rewards,
                ext_value_estimates[:N],
                ext_value_estimates[N],
                self.terminals,
                self.gamma,
                args.lambda_value,
            )

        self.advantage = ext_advantage
        if args.use_intrinsic_rewards:
            int_advantage = args.intrinsic_reward_scale * self.calculate_intrinsic_returns()
            self.advantage += int_advantage
            self.log.watch_mean_std("adv_int", int_advantage, display_width=0)

        # tvf
        if args.use_tvf:
            # only ext enabled at the moment...
            self.tvf_returns[..., 0] = self.calculate_tvf_returns(value_head='ext')

        # logging
        if args.observation_normalization:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        self.log.watch_mean_std("adv_ext", ext_advantage, display_width=0)

        # for i, head in enumerate(self.value_heads):
        #     self.log.watch_mean_std(f"return_{head}", self.returns[..., i], display_width=0)
        #     self.log.watch_mean_std(f"value_{head}", self.value[..., i], display_name="ret_ext")
        self.log.watch_mean_std(f"*return_ext", self.ext_returns)
        self.log.watch_mean_std(f"value_ext", ext_value_estimates, display_name="ve")

        self.log.watch_mean("reward_scale", self.reward_scale, display_width=0, history_length=1)
        self.log.watch_mean("entropy_bonus", self.current_entropy_bonus, display_width=0, history_length=1)

        for k, v in self.stats.items():
            self.log.watch(k, v, display_width=0)

        self.log.watch("gamma", self.gamma, display_width=0)
        if args.use_tvf:
            self.log.watch("tvf_gamma", self.tvf_gamma)
            # just want to know th max horizon std, should be about 3 I guess, but also the max.
            self.log.watch_stats("*tvf_return_ext", self.tvf_returns[:, :, -1])

        if self.batch_counter % 4 == 0:
            # this can be a little slow, ~2 seconds, compared to ~40 seconds for the rollout generation.
            # so under normal conditions we do it every other update.
            if args.replay_size > 0:
                self.replay_buffer.log_stats(self.log)

        if not args.disable_ev and self.batch_counter % 4 == 3:
            # only about 3% slower with this on.
            if args.use_tvf:
                self.log_tvf_curve_quality()
            else:
                self.log_dna_value_quality()

        if args.noisy_return > 0:
            self.returns = add_relative_noise(self.returns, args.noisy_return)
            self.tvf_returns = add_relative_noise(self.tvf_returns, args.noisy_return)
            self.tvf_returns[:, :, 0] = 0 # by definition...

        if (self.batch_counter * self.N * self.A) >= args.ag_delay and args.use_tvf:
            if self.batch_counter % 4 == 0:
                self.estimate_horizon_from_rediscounting()
        else:
            self.vars['dc_h'] = args.ag_initial_h
            self.log.watch("dc_h", self.vars['dc_h'])

    def optimizer_step(self, optimizer: torch.optim.Optimizer, label: str = "opt"):

        # get parameters
        parameters = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parameters.append(p)

        def calc_grad_norm(parameters):
            # even if we don't clip the gradient we should at least log the norm. This is probably a bit slow though.
            # we could do this every 10th step, but it's important that a large grad_norm doesn't get missed.
            grad_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            return grad_norm ** 0.5


        grad_norm = None

        if args.grad_clip_mode == "off":
            pass
        elif args.grad_clip_mode == "global_norm":
            grad_norm = nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        else:
            raise ValueError("Invalid clip_mode.")

        if grad_norm is None:
            # generate gradient l2 norm for debugging (if not already done...)
            grad_norm = calc_grad_norm(parameters)
        self.log.watch_mean(f"grad_{label}", grad_norm, display_name=f"gd_{label}", display_width=10)

        optimizer.step()

        return float(grad_norm)

    def train_distil_minibatch(self, data, loss_scale=1.0, **kwargs):

        # todo: make sure heads all line up... I think they might be offset sometimes. Perhpas make sure that
        # we always pass in all heads, and maybe just generate them all the time aswell?

        if args.use_tvf and not args.distil_force_ext:
            # the following is used to only apply distil to every nth head, which can be useful as multi value head involves
            # learning a very complex function. We go backwards so that the final head is always included.
            head_sample = utils.even_sample_down(np.arange(len(self.tvf_horizons)), max_values=args.distil_max_heads)
        else:
            head_sample = None

        if args.distil_reweighing and head_sample is not None:
            weights = [args.gamma**self.tvf_horizons[i]/args.tvf_gamma**self.tvf_horizons[i] for i in head_sample]
            weights = np.clip(weights, 0, 1).astype(np.float32)[None, :]
            weights = torch.from_numpy(weights).to(self.model.device)
            weights = weights ** 2 # square as we want to manage the squared loss.
            # that is to say, we want the loss as if we were learning the discounted return.
        else:
            weights = 1

        model_out = self.model.forward(
            data["prev_state"],
            output="policy",
            exclude_tvf=not args.use_tvf or args.distil_force_ext,
            required_tvf_heads=head_sample,
        )

        targets = data["distil_targets"] # targets are [B or B, K]

        if args.use_tvf and not args.distil_force_ext:
            predictions = model_out["tvf_value"][:, :, 0] # [B, K, VH] -> [B, K]
            if head_sample is not None:
                targets = targets[:, head_sample]
        else:
            predictions = model_out["value"][:, 0]

        loss_value = 0.5 * torch.square(targets - predictions) # [B, K]

        # apply discount reweighing
        loss_value = loss_value * weights

        # normalize the loss
        # this is required as return magntiude can differ by a factor of 10x or 0.1x,
        # which can happen if we apply different discounts to the environment. This makes
        # beta hard to tune.

        if args.distil_loss_value_target is not None and 'context' in data:
            # calibrate distil loss to be roughly the same each time
            # if no context then this implies we are doing sns.

            if args.distil_lvt_mode == "first":
                # we do this only during the first few updates though, as loss will get very small after that.
                if data['context']['epoch'] == 0 and data['context']['mini_batch'] == 0:
                    with torch.no_grad():
                        loss_value_norm2 = float(torch.norm(loss_value.view(-1), p=2).detach().cpu().numpy())
                    self.vars['distil_loss_scale'] = dls = self.vars.get('distil_loss_scale', 1.0) * 0.9 + 0.1 * loss_value_norm2
                    self.log.watch_mean("distil_loss_scale", dls, history_length=64 * args.distil.epochs,
                                        display_width=8, display_name="dls")
                else:
                    dls = self.vars['distil_loss_scale']
            elif args.distil_lvt_mode == "mean":
                # still, just over first epoch
                if data['context']['epoch'] == 0:
                    with torch.no_grad():
                        loss_value_norm2 = float(torch.norm(loss_value.view(-1), p=2).detach().cpu().numpy())
                    self.vars['distil_loss_scale'] = dls = self.vars.get('distil_loss_scale', 1.0) * 0.9 + 0.1 * loss_value_norm2
                    self.log.watch_mean("distil_loss_scale", dls, history_length=64 * args.distil.epochs,
                                        display_width=8, display_name="dls")
                else:
                    dls = self.vars['distil_loss_scale']
            else:
                raise ValueError(f'invalid distil_lvt_mode {args.distil_lvt_mode}')

            loss_value = loss_value * args.distil_loss_value_target / (dls+1e-3)

        if len(loss_value.shape) == 2:
            loss_value = loss_value.mean(axis=-1) # mean across final dim if targets / predictions were vector.
        loss = loss_value

        # note: mse on logits is a bad idea. The reason is we might get logits of -40 for settings where a policy
        # must be determanistic. The reality is there isn't much difference between exp(-40) and exp(-30) so don't do
        # mse on it.

        if args.distil_loss == "mse_logit":
            loss_policy = args.distil_beta * 0.5 * torch.square(data["old_raw_policy"] - model_out["raw_policy"]).mean(dim=-1)
        elif args.distil_loss == "mse_policy":
            loss_policy = args.distil_beta * 0.5 * torch.square(data["old_log_policy"] - model_out["log_policy"]).mean(dim=-1)
        elif args.distil_loss == "kl_policy":
            loss_policy = args.distil_beta * F.kl_div(data["old_log_policy"], model_out["log_policy"], log_target=True, reduction="none").sum(dim=-1)
        else:
            raise ValueError(f"Invalid distil_loss {args.distil_loss}")

        loss = loss + loss_policy

        pred_var = torch.var(predictions*weights)
        targ_var = torch.var(targets*weights)

        # some debugging stats
        with torch.no_grad():
            self.log.watch_mean("distil_targ_var", targ_var, history_length=64 * args.distil.epochs, display_width=0)
            self.log.watch_mean("distil_pred_var", pred_var, history_length=64 * args.distil.epochs,
                                display_width=0)
            delta = (predictions - targets) * weights
            mse = torch.square(delta).mean()
            ev = 1 - torch.var(delta) / (torch.var(targets * weights) + 1e-8)
            self.log.watch_mean("distil_ev", ev, history_length=64 * args.distil.epochs,
                                display_name="ev_dist",
                                display_width=8)
            self.log.watch_mean("distil_mse", mse, history_length=64 * args.distil.epochs,
                                display_width=0)

        # check model sparsity
        def log_model_sparsity(model, label):
            # quick check out weight sparsity is correct
            total_edges = np.prod(model.tvf_head.weight.data.shape)
            active_edges = torch.ge(model.tvf_head.weight.data.abs(), 1e-6).sum().detach().cpu().numpy()
            self.log.watch(f"*ae_{label}", active_edges / total_edges)

        if args.use_tvf:
            log_model_sparsity(self.model.value_net, "value")
            log_model_sparsity(self.model.policy_net, "policy")

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        loss = loss * loss_scale
        loss.mean().backward()

        self.log.watch_mean("loss_distil_policy", loss_policy.mean(), history_length=64 * args.distil.epochs, display_width=0)
        self.log.watch_mean("loss_distil_value", loss_value.mean(), history_length=64 * args.distil.epochs, display_width=0)
        self.log.watch_mean("loss_distil", loss.mean(), history_length=64*args.distil.epochs, display_name="ls_distil", display_width=8)

        # this is a lot, just do this for the moment...
        if 'context' in data:
            epoch = data['context']['epoch']
            mini_batch = data['context']['mini_batch']
            key = f"{epoch}_{mini_batch:02d}"
            self.log.watch_mean(f"ldp_{key}", loss_policy.mean(), history_length=10,
                                display_width=0)
            self.log.watch_mean(f"ldv_{key}", loss_value.mean(), history_length=10,
                                display_width=0)

        return {
            'losses': loss.detach()
        }

    def train_aux_minibatch(self, data, loss_scale=1.0, **kwargs):

        model_out = self.model.forward(data["prev_state"], output="full")

        if args.aux_target == "vtarg":
            # train actual value predictions on vtarg
            targets = data["aux_vtarg"]
        elif args.aux_target == "reward":
            targets = data["aux_reward"]
        else:
            raise ValueError(f"Invalid aux target, {args.aux_target}.")

        if args.aux_source == "aux":
            value_predictions = model_out["value_aux"][..., 0]
            policy_predictions = model_out["policy_aux"][..., 0]
            value_constraint = 1.0 * torch.square(
                model_out["value_ext_value"] - data['old_value']).mean()
        elif args.aux_source == "value":
            value_predictions = model_out["value_ext_value"]
            policy_predictions = model_out["policy_ext_value"]
            value_constraint = 0
        else:
            raise ValueError(f"Invalid aux target, {args.aux_source}")

        value_loss = torch.square(targets - value_predictions).mean()
        policy_loss = torch.square(targets - policy_predictions).mean()

        value_ev = 1 - torch.var(value_predictions - targets) / (torch.var(targets) + 1e-8)
        policy_ev = 1 - torch.var(policy_predictions - targets) / (torch.var(targets) + 1e-8)

        # we do a lot of minibatches, so makes sure we average over them all.
        history_length = 2 * args.aux.epochs*args.distil_batch_size // args.distil.mini_batch_size

        self.log.watch_mean("aux_value_ev", value_ev, history_length=history_length, display_width=0)
        self.log.watch_mean("aux_policy_ev", policy_ev, history_length=history_length, display_width=0)

        policy_constraint = 1.0 * (F.kl_div(data['old_log_policy'], model_out["policy_log_policy"], log_target=True, reduction="batchmean")) # todo find good constant

        loss = value_loss + policy_loss + value_constraint + policy_constraint

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        opt_loss = loss * loss_scale
        opt_loss.backward()

        self.log.watch_mean("loss_aux_value", value_loss , history_length=history_length, display_width=0)
        self.log.watch_mean("loss_aux_policy", policy_loss, history_length=history_length, display_width=0)
        self.log.watch_mean("loss_aux_value_constraint", value_constraint, history_length=history_length, display_width=0)
        self.log.watch_mean("loss_aux_policy_constraint", policy_constraint, history_length=history_length, display_width=0)

    @property
    def value_heads(self):
        """
        Returns a list containing value heads that need to be calculated.
        """
        value_heads = ['ext']
        if args.use_intrinsic_rewards:
            value_heads.append("int")
        return value_heads

    def log_fake_accumulated_gradient_norms(self, optimizer: torch.optim.Optimizer):

        required_heads = utils.even_sample_down(range(len(self.tvf_horizons)), args.sns_max_heads)
        b_small = args.sns_b_small
        b_big = args.sns_b_big

        # get dims for this optimizer
        d = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    d += np.prod(p.data.shape)

        mini_batches = b_big // b_small

        for required_head in required_heads:

            small_norms_sqr = []
            big_grad = 0

            for i in range(mini_batches):
                # note: we do not use fake noise on final horizon, this is because I want to check if final head
                # and value noise estimate match, which they should as they measure the same thing.
                # note: we split half the noise as decreasing signal and the other half as increasing noise
                target_noise_level = self.tvf_horizons[abs(required_head)] / 10
                if target_noise_level > 0:
                    noise_level = math.sqrt(target_noise_level)
                    signal_level = 1/math.sqrt(target_noise_level)
                else:
                    noise_level = target_noise_level
                    signal_level = 1
                grad = np.random.randn(d).astype(np.float32)
                norm2 = d ** 0.5 # a bit more fair than taking the true norm I guess
                # normalize so our noise vector is required length
                # the divide by b_small is because we would mean over these samples, so noise should be less
                renorm_factor = noise_level / norm2 / math.sqrt(b_small)
                grad *= renorm_factor
                grad[0] += signal_level # true signal is unit vector on first dim

                small_norms_sqr.append(np.linalg.norm(grad, ord=2) ** 2)
                if i == 0:
                    big_grad = grad.copy()
                else:
                    big_grad += grad

            g_small_sqr = float(np.mean(small_norms_sqr))
            g_big_sqr = (np.linalg.norm(big_grad, ord=2) / mini_batches) ** 2

            self.process_noise_scale(g_small_sqr, g_big_sqr, label=f"fake_head_{required_head}", verbose=False)

    def get_value_head_accumulated_gradient_norms(self, optimizer, prev_state, targets, required_head:int):
        """
        Calculate big and small gradient from given batch of data
        prev_state and targets should be in shuffled order.
        """

        B, K = targets.shape

        b_small = args.sns_b_small

        assert B % b_small == 0, "b_small must divide b_big"
        mini_batches = B // b_small

        small_norms_sqr = []
        big_grad = None

        for i in range(mini_batches):

            segment = slice(i*b_small, (i+1)*b_small)
            data = {"tvf_returns": targets[segment], "prev_state": prev_state[segment]}

            self.log.mode = self.log.LM_MUTE
            optimizer.zero_grad(set_to_none=True)
            self.train_value_minibatch(data, single_value_head=-required_head)
            self.log.mode = self.log.LM_DEFAULT
            # get small grad
            small_norms_sqr.append(utils.optimizer_grad_norm(optimizer) ** 2)
            if i == 0:
                big_grad = [x.clone() for x in utils.list_grad(optimizer)]
            else:
                for acc, p in zip(big_grad, utils.list_grad(optimizer)):
                    acc += p

        optimizer.zero_grad(set_to_none=True)

        # delete comment
        big_norm_sqr = (utils.calc_norm(big_grad)/mini_batches)**2

        return float(np.mean(small_norms_sqr)), float(big_norm_sqr)

    def train_value_minibatch(self, data, loss_scale=1.0, single_value_head: Optional[int] = None):
        """
        @param single_value_head: if given trains on just this indexed tvf value head.
        """

        required_tvf_heads = []
        if single_value_head is None:
            required_tvf_heads = None
        elif single_value_head >= 0:
            required_tvf_heads = [single_value_head]
        elif single_value_head < 0:
            required_tvf_heads = list(range(-single_value_head+1))

        model_out = self.model.forward(
            data["prev_state"],
            output="value",
            # saves a bit of time to only fetch one head when only one is needed.
            required_tvf_heads=required_tvf_heads,
        )

        B = len(data["prev_state"])

        # -------------------------------------------------------------------------
        # Calculate loss_value_function_horizons
        # -------------------------------------------------------------------------

        loss = torch.zeros(size=[B], dtype=torch.float32, device=self.model.device, requires_grad=True)

        if "tvf_returns" in data:
            # targets "tvf_returns" are [B, K]
            # predictions "tvf_value" are [B, K, VH]
            # predictions need to be generated... this could take a lot of time so just sample a few..
            targets = data["tvf_returns"] # locked to "ext" head for the moment [B, K]
            predictions = model_out["tvf_value"][:, :, 0] # locked to ext for the moment [B, K, VH] -> [B, K]

            if required_tvf_heads is not None:
                targets = targets[:, required_tvf_heads]

            tvf_loss = 0.5 * torch.square(targets - predictions) * args.tvf_coef

            # h_weighting adjustment
            if args.tvf_head_weighting == "h_weighted" and single_value_head is None:
                def h_weight(h):
                    # roughly the number of times an error will be copied, plus the original error
                    return 1 + ((args.tvf_max_horizon - h) / args.tvf_return_n_step)
                weighting = np.asarray([h_weight(h) for h in self.tvf_horizons], dtype=np.float32)[None, :]
                adjustment = 2 / (np.min(weighting) + np.max(weighting)) # try to make MSE roughly the same scale as before
                tvf_loss = tvf_loss * torch.tensor(weighting).to(device=tvf_loss.device) * adjustment

            if args.tvf_horizon_dropout > 0:
                # note: we weight the mask so that after the average the loss per example will be approximately the same
                # magnitude.
                keep_prob = (1-args.tvf_horizon_dropout)
                mask = torch.bernoulli(torch.ones_like(tvf_loss)*keep_prob) / keep_prob
                tvf_loss = tvf_loss * mask

            tvf_loss = tvf_loss.mean(dim=-1) # mean over horizons
            loss = loss + tvf_loss

            self.log.watch_mean("loss_tvf", tvf_loss.mean(), history_length=64*args.value.epochs, display_name="ls_tvf", display_width=8)

        # -------------------------------------------------------------------------
        # Calculate loss_value
        # -------------------------------------------------------------------------

        if "returns" in data:
            loss = loss + self.train_value_heads(model_out, data)

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        loss = loss * loss_scale
        loss.mean().backward()

        # -------------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------------

        self.log.watch_mean("loss_value", loss.mean(), display_name=f"ls_value")

        return {
            'losses': loss.detach()
        }

    @property
    def current_entropy_bonus(self):
        return args.entropy_bonus

    def train_value_heads(self, model_out, data):
        """
        Calculates loss for each value head, then returns their sum.
        This can be done by either the policy of value update.
        """
        loss = torch.zeros([len(data["prev_state"])], dtype=torch.float32, device=self.model.device)
        for i, value_head in enumerate(self.value_heads):
            value_prediction = model_out["value"][:, i]
            returns = data["returns"][:, i]
            value_loss = args.ppo_vf_coef * torch.square(value_prediction - returns)
            self.log.watch_mean("loss_v_" + value_head, value_loss.mean(), history_length=64, display_name="ls_v_" + value_head)
            loss = loss + value_loss
        return loss

    def train_policy_minibatch(self, data, loss_scale=1.0):

        def calc_entropy(x):
            return -(x.exp() * x).sum(axis=1)

        mini_batch_size = len(data["prev_state"])

        prev_states = data["prev_state"]
        actions = data["actions"].to(torch.long)
        old_log_pac = data["log_pac"]
        advantages = data["advantages"]

        model_out = self.model.forward(prev_states, output="policy", exclude_tvf=True)

        gain = torch.scalar_tensor(0, dtype=torch.float32, device=prev_states.device)

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        if self.action_dist == "discrete":
            old_log_policy = data["log_policy"]
            logps = model_out["log_policy"]
            logpac = logps[range(mini_batch_size), actions]
            ratio = torch.exp(logpac - old_log_pac)

            clip_frac = torch.gt(torch.abs(ratio - 1.0), self.ppo_epsilon).float().mean()
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

            self.log.watch_mean("ppo_epsilon", self.ppo_epsilon, display_width=0)

            loss_clip = torch.min(ratio * advantages, clipped_ratio * advantages)
            gain = gain + loss_clip

            with torch.no_grad():
                # record kl...
                kl_approx = (old_log_pac - logpac).mean()
                kl_true = F.kl_div(old_log_policy, logps, log_target=True, reduction="batchmean")

            entropy = calc_entropy(logps)
            original_entropy = calc_entropy(old_log_policy)

            gain = gain + entropy * self.current_entropy_bonus

            self.log.watch_mean("entropy", entropy.mean())
            self.log.watch_stats("entropy", entropy, display_width=0)  # super useful...
            self.log.watch_mean("entropy_bits", entropy.mean() * (1 / math.log(2)), display_width=0)
            self.log.watch_mean("loss_ent", entropy.mean() * self.current_entropy_bonus, display_name=f"ls_ent",
                                display_width=8)
            self.log.watch_mean("kl_approx", kl_approx, display_width=0)
            self.log.watch_mean("kl_true", kl_true, display_width=8)

        elif self.action_dist == "gaussian":
            mu = torch.clip(torch.tanh(model_out["raw_policy"])*1.1, -1, 1)
            logpac = torch.distributions.normal.Normal(mu, self.get_current_action_std()).log_prob(actions)
            ratio = torch.exp(logpac - old_log_pac)

            clip_frac = torch.gt(torch.abs(ratio - 1.0), self.ppo_epsilon).float().mean()
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

            loss_clip = torch.min(ratio * advantages[:, None], clipped_ratio * advantages[:, None])
            gain = gain + loss_clip.mean(dim=-1) # mean over actions..

            # todo kl for gaussian
            kl_approx = torch.zeros(1)
            kl_true = torch.zeros(1)

        else:
            raise ValueError(f"Invalid action distribution type {self.action_dist}")

        # -------------------------------------------------------------------------
        # Value learning for PPO mode
        # -------------------------------------------------------------------------

        if args.architecture == "single":
            # negative because we're doing gradient ascent.
            gain = gain - self.train_value_heads(model_out, data)

        # -------------------------------------------------------------------------
        # Calculate gradients
        # -------------------------------------------------------------------------

        loss = (-gain) * loss_scale
        loss.mean().backward()

        # -------------------------------------------------------------------------
        # Generate log values
        # -------------------------------------------------------------------------

        self.log.watch_mean("loss_pg", loss_clip.mean(), history_length=64*args.policy.epochs, display_name=f"ls_pg", display_width=8)
        self.log.watch_mean("clip_frac", clip_frac, display_width=8, display_name="clip")
        self.log.watch_mean("loss_policy", gain.mean(), display_name=f"ls_policy")

        return {
            'losses': loss.detach(),
            'kl_approx': float(kl_approx.detach()),  # make sure we don't pass the graph through.
            'kl_true': float(kl_true.detach()),
            'clip_frac': float(clip_frac.detach()),
        }

    @property
    def training_fraction(self):
        return (self.step / 1e6) / args.epochs

    @property
    def episode_length_mean(self):
        return np.mean(self.episode_length_buffer)

    @property
    def episode_length_std(self):
        return np.std(self.episode_length_buffer)

    @property
    def agent_age(self):
        """
        Approximate age of agent in terms of environment steps.
        Measure individual agents age, so if 128 agents each walk 10 steps, agents will be 10 steps old, not 1280.
        """
        return self.step / args.agents

    @property
    def _auto_horizon(self):
        if args.ag_mode == "episode_length":
            if len(self.episode_length_buffer) == 0:
                auto_horizon = 0
            else:
                auto_horizon = self.episode_length_mean + (2 * self.episode_length_std)
            return auto_horizon
        elif args.ag_mode == "training":
            return (1/1000) * self.step # todo make this a parameter
        elif args.ag_mode == "sns":
            return self.noise_stats.get('ag_sns_horizon', args.ag_initial_h)
        elif args.ag_mode == "h_best":
            return np.clip(self.vars.get('dc_h', args.ag_initial_h), args.ag_min_h, args.ag_max_h)
        else:
            raise ValueError(f"Invalid auto_strategy {args.ag_mode}")

    @property
    def _auto_gamma(self):
        horizon = float(np.clip(self._auto_horizon, 10, float("inf")))
        return 1 - (1 / horizon)

    @property
    def gamma(self):
        if args.use_ag and (args.ag_target in ["policy", "both"]) and args.ag_mode != "shadow":
            return self._auto_gamma
        else:
            return args.gamma

    @property
    def tvf_gamma(self):
        if args.use_ag and (args.ag_target in ["value", "both"]) and args.ag_mode != "shadow":
            return self._auto_gamma
        else:
            return args.tvf_gamma

    @property
    def reward_scale(self):
        """ The amount rewards have been multiplied by. """
        if args.noisy_zero > 0:
            # no reward scaling for noisy zero rewards.
            return 1.0
        if args.reward_normalization != "off":
            norm_wrapper = wrappers.get_wrapper(self.vec_env, wrappers.VecNormalizeRewardWrapper)
            return 1.0 / norm_wrapper.std
        else:
            return 1.0

    def train_rnd_minibatch(self, data, loss_scale: float = 1.0, **kwargs):

        # -------------------------------------------------------------------------
        # Random network distillation update
        # -------------------------------------------------------------------------
        # note: we include this here so that it can be used with PPO. In practice, it does not matter if the
        # policy network or the value network learns this, as the parameters for the prediction model are
        # separate anyway.

        loss_rnd = self.model.rnd_prediction_error(data["prev_state"]).mean()
        self.log.watch_mean("loss_rnd", loss_rnd)

        self.log.watch_mean("feat_mean", self.model.rnd_features_mean, display_width=0)
        self.log.watch_mean("feat_var", self.model.rnd_features_var, display_width=10)
        self.log.watch_mean("feat_max", self.model.rnd_features_max, display_width=10, display_precision=1)

        loss = loss_rnd * loss_scale
        loss.backward()


    def train_rnd(self):

        batch_data = {}
        B = args.batch_size
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([B, *state_shape])[:round(B*args.rnd_experience_proportion)]

        for epoch in range(args.rnd.epochs):
            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_rnd_minibatch,
                mini_batch_size=args.rnd.mini_batch_size,
                optimizer=self.rnd_optimizer,
                epoch=epoch,
                label="rnd",
            )

    def upload_batch(self, batch_data):
        """
        Uploads entire batch to GPU.
        """
        for k, v in batch_data.items():
            if type(v) is np.ndarray:
                v = torch.from_numpy(v)
            # todo: handle decompression...
            batch_data[k] = v.to(device=self.model.device, non_blocking=True)

    def train_policy(self):

        # ----------------------------------------------------
        # policy phase

        start_time = clock.time()

        if args.policy.epochs == 0:
            return

        batch_data = {}
        B = args.batch_size
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([B, *state_shape])

        if self.action_dist == "discrete":
            batch_data["actions"] = self.actions.reshape(B).astype(np.long)
            batch_data["log_policy"] = self.log_policy.reshape([B, *self.policy_shape])
            batch_data["log_pac"] = batch_data["log_policy"][range(B), self.actions.reshape([B])]
        elif self.action_dist == "gaussian":
            assert self.actions.dtype == np.float32, f"actions should be float32, but were {type(self.actions)}"
            mu = np.clip(np.tanh(self.raw_policy)*1.1, -1, 1)
            batch_data["actions"] = self.actions.reshape(B, self.model.actions).astype(np.float32)
            batch_data["log_pac"] = torch.distributions.normal.Normal(
                torch.from_numpy(mu),
                self.get_current_action_std()
            ).log_prob(torch.from_numpy(self.actions)).reshape(B, self.model.actions)

        if args.architecture == "single":
            # ppo trains value during policy update
            batch_data["returns"] = self.returns.reshape([B, self.VH])

        # sort out advantages
        advantages = self.advantage.reshape(B).copy()
        self.log.watch_stats("advantages_raw", advantages, display_width=0, history_length=1)

        # we should normalize at the mini_batch level, but it's so much easier to do this at the batch level.
        advantages = (advantages - advantages.mean()) / (advantages.std() + args.advantage_epsilon)
        self.log.watch_stats("advantages_norm", advantages, display_width=0, history_length=1)

        if args.advantage_clipping is not None:
            advantages = np.clip(advantages, -args.advantage_clipping, +args.advantage_clipping)
            self.log.watch_stats("advantages_clipped", advantages, display_width=0, history_length=1)

        self.log.watch_stats("advantages", advantages, display_width=0, history_length=1)
        batch_data["advantages"] = advantages

        epochs = 0
        for epoch in range(args.policy.epochs):
            results = self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_policy_minibatch,
                mini_batch_size=args.policy.mini_batch_size,
                optimizer=self.policy_optimizer,
                label="policy",
                epoch=epoch,
            )
            expected_mini_batches = (args.batch_size / args.policy.mini_batch_size)
            epochs += results["mini_batches"] / expected_mini_batches
            if "did_break" in results:
                break

        self.log.watch(f"time_train_policy", (clock.time() - start_time),
                       display_width=6, display_name='t_pol', display_precision=3)

    def wants_noise_estimate(self, label:str):
        """
        Returns if given label wants a noise update on this step.
        """

        if not args.use_sns:
            return False
        if self.batch_counter % args.sns_period != args.sns_period-1:
            # only evaluate every so often.
            return False
        if label.lower() not in ast.literal_eval(args.sns_labels):
            return False
        return True

    def log_accumulated_gradient_norms(self, batch_data):

        required_heads = utils.even_sample_down(range(len(self.tvf_horizons)), args.sns_max_heads)

        start_time = clock.time()
        for i, head_id in enumerate(required_heads):

            # select a different sample for each head (why not)
            prev_state = batch_data["prev_state"]
            targets = batch_data["tvf_returns"]
            if args.sns_b_big > self.N * self.A:
                raise ValueError(f"Can not take {args.sns_b_big} samples from rollout of size {self.N}x{self.A}")

            # we sample even when we need all examples, as it's important to shuffle the order
            sample = np.random.choice(range(self.N * self.A), args.sns_b_big, replace=False)
            prev_state = prev_state[sample]
            targets = targets[sample]

            g_small_sqr, g_big_sqr = self.get_value_head_accumulated_gradient_norms(
                optimizer=self.value_optimizer,
                prev_state=prev_state,
                targets=targets,
                required_head=head_id,
            )
            self.process_noise_scale(
                g_small_sqr, g_big_sqr, label=f"acc_head_{head_id}", verbose=False)
        s = clock.time() - start_time
        self.log.watch_mean("t_s_heads", s / args.sns_period)

    def train_value(self):

        # ----------------------------------------------------
        # value phase

        start_time = clock.time()

        if args.value.epochs == 0:
            return

        batch_data = {}
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([N*A, *state_shape])

        if not args.use_tvf or args.tvf_include_ext:
            # these are not really needed, maybe they provide better features, I don't know.
            # one issue is that they will be the wrong scale if rediscounting is applied.
            # e.g. if gamma defaults to 0.99997, but these are calculated at 0.999 they might be extremly large
            batch_data["returns"] = self.returns.reshape(N*A, self.VH)

        if args.use_tvf:
            # just train ext heads for the moment
            batch_data["tvf_returns"] = self.tvf_returns[:, :, :, -1].reshape(N*A, self.K)

            # per horizon noise estimates
            # note: it's about 2x faster to generate accumulated noise all at one go, but this means
            # the generic code for noise estimation no longer works well.
            if self.wants_noise_estimate('value_heads') and args.sns_max_heads > 0:
                if args.upload_batch:
                    self.upload_batch(batch_data)
                # generate our per-horizon estimates
                self.log_accumulated_gradient_norms(batch_data)
                if args.sns_fake_noise:
                    self.log_fake_accumulated_gradient_norms(optimizer=self.value_optimizer)

        for value_epoch in range(args.value.epochs):
            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_value_minibatch,
                mini_batch_size=args.value.mini_batch_size,
                optimizer=self.value_optimizer,
                label="value",
                epoch=value_epoch,
            )

        self.log.watch(f"time_train_value", (clock.time() - start_time),
                       display_width=6, display_name='t_val', display_precision=3)

    def generate_aux_buffer(self):
        """
        Output will be [N, A, 16] of type float64
        """
        return ExperienceReplayBuffer.create_aux_buffer(
            shape=self.ext_rewards.shape,
            reward=self.ext_rewards,
            action=self.actions,
        )

    def get_replay_sample(self, samples_wanted:int):
        """
        Samples from our replay buffer. If no buffer is present samples from rollout. Also supports mixing...
        """
        # work out what to use to train distil on
        if self.replay_buffer is not None and len(self.replay_buffer.data) > 0:
            # buffer is 1D, need to reshape to 2D
            _, *state_shape = self.replay_buffer.data.shape

            if args.replay_mixing:
                # use mixture of replay buffer and current batch of data
                obs = np.concatenate([
                    self.replay_buffer.data,
                    utils.merge_down(self.prev_obs),
                ], axis=0)
                aux = np.concatenate([
                    self.replay_buffer.aux,
                    utils.merge_down(self.generate_aux_buffer()),
                ], axis=0)
            else:
                obs = self.replay_buffer.data
                aux = self.replay_buffer.aux
        else:
            obs = utils.merge_down(self.prev_obs)
            aux = utils.merge_down(self.generate_aux_buffer())

        # filter down to n samples (if needed)
        if samples_wanted < len(obs):
            sample = np.random.choice(len(obs), samples_wanted, replace=False)
            obs = obs[sample]
            aux = aux[sample]

        return obs, aux

    def rediscount_horizons(self, old_value_estimates):
        """
        Input is [B, K]
        Output is [B, K]
        """
        if self.tvf_gamma == self.gamma:
            return old_value_estimates

        # old_distil_targets = batch_data["distil_targets"].copy()  # B, K
        new_value_estimates = old_value_estimates.copy()
        B, K = old_value_estimates.shape
        for k in range(K):
            new_value_estimates[:, k] = get_rediscounted_value_estimate(
                old_value_estimates[:, :k+1],
                self.tvf_gamma,
                self.gamma,
                self.tvf_horizons[:k+1]
            )
        return new_value_estimates

    def get_distil_batch(self, samples_wanted:int):
        """
        Creates a batch of data to train on during distil phase.
        Also generates any required targets.

        If no replay buffer is being used then uses the rollout data instead.

        @samples_wanted: The number of samples requested. Might be smaller if the replay buffer is too small, or
            has not seen enough data yet.

        """

        # todo: tidy this up so there's only one path.

        if self.replay_buffer is None and samples_wanted == args.batch_size:

            # fast path... only requires policy module to evaluate, can reuse value estimates from rollout.
            obs = utils.merge_down(self.prev_obs)
            batch_data = {"prev_state": obs}

            # get targets from rollout
            if args.use_tvf and not args.distil_force_ext: # tvf_value is [N, A, K, VH]
                batch_data["distil_targets"] = utils.merge_down(self.tvf_value[:self.N, :, :, 0]) # N*A, K
                if args.distil_rediscount:
                    batch_data["distil_targets"] = self.rediscount_horizons(batch_data["distil_targets"])
            else:
                batch_data["distil_targets"] = utils.merge_down(self.ext_value[:self.N])

            # returns should have unit variance, if they do not, divide by the standard deviation.
            # this can occur, for example, when rediscounting.
            if args.distil_renormalize:
                targets = batch_data["distil_targets"]
                std = np.std(targets)
                smooth_std = utils.dictionary_ema(self.vars, "distil_target_std", std, 0.9)
                batch_data["distil_targets"] /= (smooth_std+1e-1) # 1e-1 so we don't multiply by more than 10.
                self.log.watch("distil_target_std", std, display_width=0)

            if args.distil_order == "before_policy":
                # in this case we can just use the rollout policy
                batch_data["old_raw_policy"] = utils.merge_down(self.raw_policy)
                batch_data["old_log_policy"] = utils.merge_down(self.log_policy)
            else:
                # otherwise, policy has changed so we need to update it
                model_out = self.detached_batch_forward(
                    obs=obs,
                    output="policy",
                    exclude_tvf=True,
                )
                batch_data["old_raw_policy"] = model_out["raw_policy"].detach().cpu().numpy()
                batch_data["old_log_policy"] = model_out["log_policy"].detach().cpu().numpy()

            return batch_data

        # slower path, for when rollout is needed and we need to regenerate all targets
        assert not args.distil_renormalize, "renormalization only supported under a complete rollout distil batch."
        obs, distil_aux = self.get_replay_sample(samples_wanted)

        batch_data = {}

        batch_data["prev_state"] = obs

        # forward through model to get targets from model
        model_out = self.detached_batch_forward(
            obs=obs,
            output="full",
        )

        if args.use_tvf and not args.distil_force_ext:
            # we could skip this if we trained on rollout rather then replay
            batch_data["distil_targets"] = model_out["value_tvf_value"][:, :, 0].detach().cpu().numpy()
            if args.distil_rediscount:
                batch_data["distil_targets"] = self.rediscount_horizons(batch_data["distil_targets"])
        else:
            batch_data["distil_targets"] = model_out["value_value"][:, 0].detach().cpu().numpy()

        # get old policy
        batch_data["old_raw_policy"] = model_out["policy_raw_policy"].detach().cpu().numpy()
        batch_data["old_log_policy"] = model_out["policy_log_policy"].detach().cpu().numpy()

        return batch_data

    def train_distil(self):

        # ----------------------------------------------------
        # distil phase

        start_time = clock.time()

        if args.distil.epochs == 0:
            return

        batch_data = self.get_distil_batch(args.distil_batch_size)

        for distil_epoch in range(args.distil.epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_distil_minibatch,
                mini_batch_size=args.distil.mini_batch_size,
                optimizer=self.distil_optimizer,
                label="distil",
                epoch=distil_epoch,
            )

        self.log.watch(f"time_train_distil", (clock.time() - start_time) / args.distil_period,
                       display_width=6, display_name='t_dis', display_precision=3)

    def train_aux(self):

        # ----------------------------------------------------
        # aux phase
        # borrows a lot of hyperparameters from distil

        start_time = clock.time()

        if args.aux.epochs == 0:
            return

        # we could train on terminals, or reward.
        # time would be policy dependant, and is aliased.

        replay_obs, replay_aux = self.get_replay_sample(args.distil_batch_size)
        batch_data = {}
        batch_data['prev_state'] = replay_obs
        batch_data['aux_reward'] = replay_aux[:, ExperienceReplayBuffer.AUX_REWARD].astype(np.float32)
        batch_data['aux_action'] = replay_aux[:, ExperienceReplayBuffer.AUX_ACTION].astype(np.float32)
        batch_data['aux_vtarg'] = replay_aux[:, ExperienceReplayBuffer.AUX_VTARG].astype(np.float32)

        # calculate value required for constraints
        model_out = self.detached_batch_forward(
            replay_obs,
            output='full',
        )

        batch_data['old_value'] = model_out['value_ext_value'].cpu().numpy()
        batch_data['old_log_policy'] = model_out['policy_log_policy'].cpu().numpy()

        for aux_epoch in range(args.aux.epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_aux_minibatch,
                mini_batch_size=args.distil.mini_batch_size,
                optimizer=self.aux_optimizer,
                epoch=aux_epoch,
                label="aux",
            )

        self.log.watch(f"time_train_aux", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_aux', display_precision=1)

    def update_sns_horizon_target(self):
        """
        New version of sns_horizon estimation

        The idea is to work out the noise level for training up to k heads, then force it to be montonic, then
        use interpolation to find the first horizon that has a noise level above some threshold.
        """

        assert args.use_sns, "SNS must be enabled."
        assert args.use_tvf, "TVF must be enabled"

        if self.step < args.ag_delay:
            # it's too early to modify gamma, but log what we can anyway.
            self.noise_stats['ag_sns_horizon'] = args.ag_initial_h
            self.log.watch_mean('ag_sns_target', args.ag_initial_h, display_width=0)
            self.log.watch_mean('*ag_sns_horizon', args.ag_initial_h)
            return

        if len(self.noise_stats.get('active_heads', [])) <= 0:
            # no noise levels logged yes
            return

        # data used to interpolate noise levels
        logged_heads = np.asarray(sorted(self.noise_stats['active_heads']))

        clean_and_sqrt = lambda x: max(x, 0) ** 0.5

        logged_noise_levels = np.asarray([clean_and_sqrt(self.noise_stats.get(f'acc_head_{i}_ratio', float('inf'))) for i in logged_heads])

        # force plot to be monotonic
        #logged_noise_levels = np.asarray([np.max(logged_noise_levels[:i+1]) for i in range(len(logged_noise_levels))])

        # step 1: work out our target (with a cap at min_h)
        new_target = args.ag_min_h
        for i, h in enumerate(self.tvf_horizons):
            noise_level = interpolate(logged_heads, logged_noise_levels[None, :], np.asarray([i]))[0]
            if noise_level < args.ag_sns_threshold:
                new_target = max(h, new_target)

        # step 2: move towards (clipped) log target
        alpha = 1-(1/(args.ag_sns_ema_horizon / (self.N * self.A)))
        old_log_horizon = math.log(1 + self.noise_stats.get('ag_sns_horizon', args.ag_initial_h))
        target_log_horizon = math.log(1+new_target)
        new_log_horizon = alpha * old_log_horizon + (1-alpha) * target_log_horizon
        self.noise_stats['ag_sns_horizon'] = np.clip(math.exp(new_log_horizon) - 1, args.ag_min_h, args.ag_max_h)

        self.log.watch_mean('ag_sns_target', new_target, display_width=0)
        self.log.watch_mean('ag_sns_horizon', self.noise_stats['ag_sns_horizon'], display_name="auto_horizon")


    def wants_distil_update(self, location=None):
        location_match = location is None or location == args.distil_order
        return \
            args.architecture == "dual" and  \
            args.distil.epochs > 0 and \
            self.batch_counter % args.distil_period == args.distil_period - 1 and \
            location_match

    def train(self):

        if args.disable_logging:
            self.log.mode = self.log.LM_MUTE

        self.log.watch("*device", args.device)
        self.log.watch("*host", args.hostname)

        self.model.eval()

        self.update_learning_rates()

        if self.wants_distil_update("before_policy"):
            self.train_distil()

        with Mutex(args.get_mutex_key) as mx:
            self.log.watch_mean(
                "mutex_wait", round(1000 * mx.wait_time), display_name="mutex",
                type="int",
                display_width=0,
            )
            self.train_policy()

        if args.architecture == "dual":
            # value learning is handled with policy in PPO mode.
            self.train_value()
            if self.wants_distil_update("after_policy"):
                self.train_distil()

        if args.aux.epochs > 0 and (args.aux_period == 0 or self.batch_counter % args.aux_period == args.aux_period-1):
            self.train_aux()

        if args.use_rnd:
            self.train_rnd()

        if args.use_tvf and args.ag_mode in ["sns", "shadow"]:
            self.update_sns_horizon_target()

        self.batch_counter += 1

    def train_batch(
            self,
            batch_data,
            mini_batch_func,
            mini_batch_size,
            optimizer: torch.optim.Optimizer,
            label,
            epoch: Optional[int] = None,
            hooks: Union[dict, None] = None,
            thinning: float = 1.0,
            force_micro_batch_size = None,
        ) -> dict:
        """
        Trains agent on current batch of experience

        Thinning: uses this proportion of the batch_data.

        Returns context with
            'mini_batches' number of mini_batches completed
            'outputs' output from each mini_batch update
            'did_break'=True (only if training terminated early)
        """

        if args.upload_batch:
            assert batch_data["prev_state"].dtype != object, "obs_compression can no be enabled with upload_batch."
            self.upload_batch(batch_data)

        if epoch == 0 and self.wants_noise_estimate(label): # check noise of first update only
            start_time = clock.time()
            self.estimate_noise_scale(batch_data, mini_batch_func, optimizer, label)
            s = clock.time()-start_time
            self.log.watch_mean(f"sns_time_{label}", s / args.sns_period, display_width=8, display_name=f"t_s{label[:3]}")

        assert "prev_state" in batch_data, "Batches must contain 'prev_state' field of dims (B, *state_shape)"
        batch_size, *state_shape = batch_data["prev_state"].shape

        for k, v in batch_data.items():
            assert len(v) == batch_size, f"Batch input must all match in entry count. Expecting {batch_size} but found {len(v)} on {k}"
            if type(v) is np.ndarray:
                assert v.dtype in [np.uint8, np.int64, np.float32, np.object], \
                    f"Batch input should [uint8, int64, or float32] but {k} was {type(v.dtype)}"
            elif type(v) is torch.Tensor:
                assert v.dtype in [torch.uint8, torch.int64, torch.float32], \
                    f"Batch input should [uint8, int64, or float32] but {k} was {type(v.dtype)}"

        assert batch_size % mini_batch_size == 0
        mini_batches = batch_size // mini_batch_size
        if force_micro_batch_size is not None:
            micro_batch_size = force_micro_batch_size
        else:
            micro_batch_size = min(args.max_micro_batch_size, mini_batch_size)
        assert mini_batch_size % micro_batch_size == 0
        micro_batches = mini_batch_size // micro_batch_size

        ordering = list(range(batch_size))
        np.random.shuffle(ordering)

        micro_batch_counter = 0
        outputs = []

        context = {}

        for j in range(mini_batches):

            optimizer.zero_grad(set_to_none=True)

            for k in range(micro_batches):
                # put together a micro_batch.
                batch_start = micro_batch_counter * micro_batch_size
                batch_end = (micro_batch_counter + 1) * micro_batch_size
                sample = ordering[batch_start:batch_end]
                micro_batch_counter += 1

                # context for the minibatch.
                micro_batch_context = {
                    'epoch': epoch,
                    'mini_batch': j,
                    'micro_batch': k,
                    'is_first': j == 0,
                    'is_last': j == mini_batches-1,
                }
                micro_batch_data = {}
                micro_batch_data['context'] = micro_batch_context

                for var_name, var_value in batch_data.items():
                    data = var_value[sample]

                    if thinning < 1.0:
                        samples_to_use = int(micro_batch_size * thinning)
                        data = data[:samples_to_use]

                    if data.dtype == np.object:
                        # handle decompression
                        data = np.asarray([data[i].decompress() for i in range(len(data))])

                    if type(data) is np.ndarray:
                         data = torch.from_numpy(data)

                    # upload to gpu
                    data = data.to(self.model.device, non_blocking=True)

                    micro_batch_data[var_name] = data

                result = mini_batch_func(micro_batch_data, loss_scale=1 / micro_batches)

                if hooks is not None and "after_micro_batch" in hooks:
                    hooks["after_micro_batch"](micro_batch_context)

                outputs.append(result)

            context = {
                'mini_batches': j + 1,
                'outputs': outputs
            }

            if hooks is not None and "after_mini_batch" in hooks:
                if hooks["after_mini_batch"](context):
                    context["did_break"] = True
                    break

            self.optimizer_step(optimizer=optimizer, label=label)

        # free up memory by releasing grads.
        optimizer.zero_grad(set_to_none=True)

        return context


def get_rediscounted_value_estimate(
        values: Union[np.ndarray, torch.Tensor],
        old_gamma: float,
        new_gamma: float,
        horizons,
        clipping=10,
):
    """
    Returns rediscounted return at horizon h

    values: float tensor of shape [B, K]
    horizons: int tensor of shape [K] giving horizon for value [:, k]
    returns float tensor of shape [B]
    """

    B, K = values.shape

    if old_gamma == new_gamma:
        return values[:, -1]

    assert K == len(horizons), f"missmatch {K} {horizons}"
    assert horizons[0] == 0, 'first horizon must be 0'

    if type(values) is np.ndarray:
        values = torch.from_numpy(values)
        is_numpy = True
    else:
        is_numpy = False

    prev = values[:, 0] # should be zero
    prev_h = 0
    discounted_reward_sum = torch.zeros([B], dtype=torch.float32, device=values.device)
    for i_minus_one, h in enumerate(horizons[1:]):
        i = i_minus_one + 1
        # rewards occurred at some point after prev_h and before h, so just average them. Remembering that
        # v_h includes up to and including h timesteps.
        # also, we subtract 1 as the reward given by V_h=1 occurs at t=0
        mid_h = ((prev_h+1 + h) / 2) - 1
        discounted_reward = (values[:, i] - prev)
        prev = values[:, i]
        prev_h = h
        # a clipping of 10 gets us to about 2.5k horizon before we start introducing bias. (going from 1k to 10k discounting)
        ratio = min((new_gamma ** mid_h) / (old_gamma ** mid_h), clipping) # clipped ratio
        discounted_reward_sum += discounted_reward * ratio

    #print(old_gamma, new_gamma, values[:, 0].std(), values[:, -1].std(), discounted_reward_sum.std())

    return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum


def expand_to_na(n, a, x):
    """
    takes 1d input and returns it duplicated [N,A] times
    in form [n, a, *]
    """
    x = x[None, None, :]
    x = np.repeat(x, n, axis=0)
    x = np.repeat(x, a, axis=1)
    return x

def expand_to_h(h,x):
    """
    takes 2d input and returns it duplicated [H] times
    in form [*, *, h]
    """
    x = x[:, :, None]
    x = np.repeat(x, h, axis=2)
    return x


def make_env(env_type, env_id, **kwargs):
    if env_type == "atari":
        make_fn = atari.make
    elif env_type == "mujoco":
        make_fn = mujoco.make
    else:
        raise ValueError(f"Invalid environment type {env_type}")
    return make_fn(env_id, **kwargs)

def _open_checkpoint(checkpoint_path: str, **pt_args):
    """
    Load checkpoint. Supports zip format.
    """
    # gzip support

    try:
        with gzip.open(os.path.join(checkpoint_path, ".gz"), 'rb') as f:
            return torch.load(f, **pt_args)
    except:
        pass

    try:
        # unfortunately some checkpoints were saved without the .gz so just try and fail to load them...
        with gzip.open(checkpoint_path, 'rb') as f:
            return torch.load(f, **pt_args)
    except:
        pass

    try:
        # unfortunately some checkpoints were saved without the .gz so just try and fail to load them...
        with open(checkpoint_path, 'rb') as f:
            return torch.load(f, **pt_args)
    except:
        pass

    raise Exception(f"Could not open checkpoint {checkpoint_path}")
