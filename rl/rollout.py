import logging
import os

import numpy as np
import gym
import scipy.stats
from tqdm import tqdm
import blosc
from ast import literal_eval

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import time as clock
import json
import cv2
import pickle
import gzip
from collections import defaultdict, deque
from typing import Union, Optional
import math

import train
from .logger import Logger
from . import utils, atari, mujoco, procgen, hybridVecEnv, wrappers, models, compression
from .lfr import compute_band_pass
from .returns import get_return_estimate
from .config import args
from .mutex import Mutex
from .replay import ExperienceReplayBuffer

import collections

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


def calculate_gae(
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


class SimulatedAnnealing:
    """
    Handles simulated annealing

    usage

    sa = SimulatedAnnealing

    for i in range(100):
        eval_score = eval(sa.value)
        sa.process(eval_score)

    """
    def __init__(self, initial_value:float=0):
        self.value = initial_value
        self.neighbour = initial_value
        self.prev_score = float('-inf')
        self._generate_neighbour()
        self.accepts = 0
        self.rejects = 0

    def _generate_neighbour(self):
        """
        Returns new candidate value
        """

        # we do the random walk in log-space.
        # this makes sure that applying n jumps, where n-> inf has an expected change of 0 (but infinte variance)

        theta = math.log(self.value, 2)

        new_theta = theta + np.random.normal(args.sa_mu, args.sa_sigma)

        self.neighbour = 2**new_theta

    @property
    def acceptance_rate(self):
        iterations = self.accepts + self.rejects
        if iterations > 0:
            return self.accepts / (iterations)
        else:
            return 0

    def process(self, score, prev_score=None):
        """
        If previous score is given not given it uses the previously stored score.
        """

        if prev_score is None:
            prev_score = self.prev_score

        if (score > prev_score) or (np.random.rand() < 0.05):
            # accept
            self.value = self.neighbour
            self.prev_score = score
            self.accepts += 1
        else:
            # reject
            self.rejects += 1
        self._generate_neighbour()

class Runner:

    def __init__(self, model: models.TVFModel, log, name="agent", action_dist='discrete'):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model
        self.step = 0
        self.horizon_sa = SimulatedAnnealing(1000) # this ends up being one third, so a horizon of ~300

        self.previous_rollout = None

        def make_optimizer(opt, name, params, **kwargs):
            name = name.lower()
            optimizer_params = {}
            if name == "adam":
                optimizer = torch.optim.Adam
                optimizer_params = {
                    'eps': args.adam_epsilon,
                    'betas': (args.adam_beta1, args.adam_beta2),
                    'weight_decay': args.policy_weight_decay if opt == "policy" else 0.0,
                }
            elif name == "sgd":
                optimizer = torch.optim.SGD
            else:
                raise ValueError(f"Invalid Optimizer {name}")
            return optimizer(params, **kwargs, **optimizer_params)

        # special case for policy optimizer
        self.policy_optimizer = make_optimizer('policy', args.policy_optimizer or args.optimizer, model.policy_net.parameters(), lr=self.policy_lr)
        self.value_optimizer = make_optimizer('value', args.optimizer, model.value_net.parameters(), lr=self.value_lr)
        if args.distil_epochs > 0:
            self.distil_optimizer = make_optimizer('distil', args.optimizer, model.policy_net.parameters(), lr=self.distil_lr)
        else:
            self.distil_optimizer = None
        if args.aux_epochs > 0:
            self.aux_optimizer = make_optimizer('aux', args.optimizer, model.parameters(), lr=self.distil_lr)
        else:
            self.aux_optimizer = None

        if args.use_rnd:
            self.rnd_optimizer = make_optimizer('rnd', args.optimizer, model.prediction_net.parameters(), lr=self.rnd_lr)
        else:
            self.rnd_optimizer = None

        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents

        self.action_dist = action_dist

        self.state_shape = model.input_dims
        self.rnn_state_shape = [2, 512]  # records h and c for LSTM units.
        self.policy_shape = [model.actions]

        self.batch_counter = 0
        self.erp_stats = {}
        self.are_stats = {}
        self.are_stats["value_h"] = 1/(1-args.td_lambda-1e-6)
        self.are_stats["policy_h"] = 1/(1-args.gae_lambda-1e-6)

        self.grad_accumulator = {}

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.obs = np.zeros([A, *self.state_shape], dtype=np.uint8)
        self.time = np.zeros([A], dtype=np.float32)
        self.done = np.zeros([A], dtype=np.bool)

        if args.mutex_key:
            log.info(f"Using mutex key <yellow>{args.get_mutex_key}<end>")

        # includes final state as well, which is needed for final value estimate
        if args.use_compression:
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

        if args.upload_batch and not args.use_compression:
            # in batch upload mode we can just keep all_obs on the GPU
            self.all_obs = torch.zeros(size=[N + 1, A, *self.state_shape], dtype=torch.uint8, device=self.model.device)

        self.all_time = np.zeros([N + 1, A], dtype=np.float32)
        if self.action_dist == "discrete":
            self.actions = np.zeros([N, A], dtype=np.int64)
        elif self.action_dist == "gaussian":
            self.actions = np.zeros([N, A, self.model.actions], dtype=np.float32)
        else:
            raise ValueError(f"Invalid distribution {self.action_dist}")
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.raw_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.ext_advantage = np.zeros([N, A], dtype=np.float32) # unnormalized extrinsic reward advantages
        self.raw_advantage = np.zeros([N, A], dtype=np.float32) # advantages before normalization
        self.terminals = np.zeros([N, A], dtype=np.bool)  # indicates prev_state was a terminal state.

        self.replay_value_estimates = np.zeros([N, A], dtype=np.float32)

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)
        self.int_value = np.zeros([N+1, A], dtype=np.float32)
        self.int_advantage = np.zeros([N, A], dtype=np.float32)
        self.int_returns = np.zeros([N, A], dtype=np.float32)
        self.intrinsic_reward_norm_scale: float = 1

        # log optimal
        self.sqr_value = np.zeros([N+1, A], dtype=np.float32)
        self.sqr_returns = np.zeros([N, A], dtype=np.float32)

        # returns generation
        self.ext_returns = np.zeros([N, A], dtype=np.float32)
        self.advantage = np.zeros([N, A], dtype=np.float32)
        self.ext_value = np.zeros([N+1, A], dtype=np.float32)
        self.uni_value = np.zeros([N+1, A], dtype=np.float32)

        # terminal prediction
        self.tp_final_value_estimate = np.zeros([A], dtype=np.float32)
        self.tp_returns = np.zeros([N, A], dtype=np.float32)  # terminal state predictions are generated the same was as returns.

        self.intrinsic_returns_rms = utils.RunningMeanStd(shape=())
        self.ems_norm = np.zeros([args.agents])

        # outputs tensors when clip loss is very high.
        self.log_high_grad_norm = True

        self.stats = {
            'reward_clips': 0,
            'game_crashes': 0,
            'big_red_button_pushes': 0,
            'action_repeats': 0,
            'batch_action_repeats': 0,
            'batch_big_red_button_pushes': 0,
        }
        self.ep_count = 0
        self.episode_length_buffer = collections.deque(maxlen=1000)
        self.episode_brb_buffer = collections.deque(maxlen=1000)

        # create the replay buffer if needed
        self.replay_buffer: Optional[ExperienceReplayBuffer] = None
        if args.replay_size > 0:
            self.replay_buffer = ExperienceReplayBuffer(
                max_size=args.replay_size,
                obs_shape=self.prev_obs.shape[2:],
                obs_dtype=self.prev_obs.dtype,
                mode=args.replay_mode,
                thinning=args.replay_thinning,
            )


        #  these horizons will always be generated and their scores logged.
        self.tvf_debug_horizons = [0] + self.get_standard_horizon_sample(args.tvf_max_horizon)


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

    @property
    def value_lr(self):
        return self.anneal(args.value_lr, mode="linear" if args.value_lr_anneal else "off")

    @property
    def intrinsic_reward_scale(self):
        return self.anneal(args.ir_scale, mode=args.ir_anneal)

    @property
    def extrinsic_reward_scale(self):
        return args.er_scale

    @property
    def policy_lr(self):
        return self.anneal(args.policy_lr, mode="linear" if args.policy_lr_anneal else "off")

    @property
    def ppo_epsilon(self):
        return self.anneal(args.ppo_epsilon, mode="linear" if args.ppo_epsilon_anneal else "off")

    @property
    def distil_lr(self):
        return self.anneal(args.distil_lr, mode="linear" if args.distil_lr_anneal else "off")


    def get_standard_horizon_sample(self, max_horizon: int):
        """
        Provides a set of horizons spaces (approximately) geometrically, with the H[0] = 1 and H[-1] = current_horizon.
        These may change over time (if current horizon changes). Use debug_horizons for a fixed set.
        """
        assert max_horizon <= 30000, "horizons over 30k not yet supported."
        horizons = [h for h in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000] if
                                   h <= max_horizon]
        if max_horizon not in horizons:
            horizons.append(max_horizon)
        return horizons

    @property
    def rnd_lr(self):
        return args.rnd_lr

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

        if args.reward_normalization:
            self.vec_env = wrappers.VecNormalizeRewardWrapper(
                self.vec_env,
                gamma=args.reward_normalization_gamma,
                scale=args.reward_scale,
            )

        if args.max_repeated_actions > 0:
            self.vec_env = wrappers.VecRepeatedActionPenalty(self.vec_env, args.max_repeated_actions, args.repeated_action_penalty)

        if verbose:
            model_total_size = self.model.model_size(trainable_only=False)/1e6
            self.log.important("Generated {} agents ({}) using {} ({:.1f}M params) {} model.".
                           format(args.agents, "async" if not args.sync_envs else "sync", self.model.name,
                                  model_total_size, self.model.dtype))

    def save_checkpoint(self, filename, step, disable_replay=False, disable_optimizer=False, disable_log=False, disable_env_state=False):

        data = {
            'step': step,
            'ep_count': self.ep_count,
            'episode_length_buffer': self.episode_length_buffer,
            'episode_brb_buffer': self.episode_brb_buffer,
            'current_horizon': self.current_horizon,
            'model_state_dict': self.model.state_dict(),
            'batch_counter': self.batch_counter,
            'reward_scale': self.reward_scale,
            'stats': self.stats,
        }

        if not disable_optimizer:
            data['policy_optimizer_state_dict'] = self.policy_optimizer.state_dict()
            data['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            if args.use_rnd:
                data['rnd_optimizer_state_dict'] = self.rnd_optimizer.state_dict()
            if args.distil_epochs > 0:
                data['distil_optimizer_state_dict'] = self.distil_optimizer.state_dict()
            if args.aux_epochs > 0:
                data['aux_optimizer_state_dict'] = self.aux_optimizer.state_dict()

        if not disable_log:
            data['logs'] = self.log
        if not disable_env_state:
            data['env_state'] = utils.save_env_state(self.vec_env)

        if args.use_erp:
            data['erp_stats'] = self.erp_stats

        if args.are_mode != "off":
            data['are_stats'] = self.are_stats

        if self.replay_buffer is not None and not disable_replay:
            data["replay_buffer"] = self.replay_buffer.save_state(force_copy=False)

        if args.auto_strategy[:2] == "sa":
            data['horizon_sa'] = self.horizon_sa

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
            print("debug:", list(checkpoint['model_state_dict'].keys()))
            checkpoint['model_state_dict'] = {
                k: v for k, v in checkpoint['model_state_dict'].items() if "trace_module" not in k
            }

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        if args.use_rnd:
            self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        if args.distil_epochs > 0:
            self.distil_optimizer.load_state_dict(checkpoint['distil_optimizer_state_dict'])
        if args.aux_epochs > 0:
            self.aux_optimizer.load_state_dict(checkpoint['aux_optimizer_state_dict'])

        if args.auto_strategy[:2] == "sa":
            self.horizon_sa = checkpoint['horizon_sa']

        step = checkpoint['step']
        self.log = checkpoint['logs']
        self.step = step
        self.ep_count = checkpoint.get('ep_count', 0)
        self.episode_length_buffer = checkpoint['episode_length_buffer']
        self.episode_brb_buffer = checkpoint['episode_brb_buffer']
        self.batch_counter = checkpoint.get('batch_counter', 0)
        self.stats = checkpoint.get('stats', 0)

        if args.use_erp:
            self.erp_stats = checkpoint['erp_stats']

        self.are_stats = checkpoint.get('are_stats', {})

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
        self.time *= 0
        self.done = np.zeros_like(self.done)
        self.episode_score *= 0
        self.episode_len *= 0
        self.step = 0

        # reset stats
        for k in list(self.stats.keys()):
            v = self.stats[k]
            if type(v) in [float, int]:
                self.stats[k] *= 0

        self.batch_counter = 0

        self.episode_length_buffer.clear()
        self.episode_brb_buffer.clear()
        # so that there is something in the buffer to start with.
        self.episode_length_buffer.append(1000)

    @torch.no_grad()
    def detached_batch_forward(self, obs:np.ndarray, aux_features=None, **kwargs):
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
                    aux_features=aux_features[batch_start:batch_end] if aux_features is not None else None,
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
            results = self.model.forward(obs, aux_features=aux_features, **kwargs)
            return results

    def calculate_sampled_returns(
            self,
            value_sample_horizons: Union[list, np.ndarray],
            required_horizons: Union[list, np.ndarray, int],
            head: str = "ext", # ext|int
            obs=None,
            time=None,
            rewards=None,
            dones=None,
            tvf_mode=None,
            tvf_n_step=None,
            include_second_moment: bool = False,
    ):
        """
        Calculates and returns the (tvf_gamma discounted) (transformed) return estimates for given rollout.

        prev_states: ndarray of dims [N+1, B, *state_shape] containing prev_states
        rewards: float32 ndarray of dims [N, B] containing reward at step n for agent b
        value_sample_horizons: int32 ndarray of dims [K] indicating horizons to generate value estimates at.
        required_horizons: int32 ndarray of dims [K] indicating the horizons for which we want a return estimate.
        head: which head to use for estimate, i.e. ext_value, int_value, ext_value_square etc
        sqrt_m2: returns a tuple containing (first moment, sqrt second moment)
        """

        assert utils.is_sorted(required_horizons), f"Required horizons must be sorted but found {required_horizons}"

        if type(value_sample_horizons) is list:
            value_sample_horizons = np.asarray(value_sample_horizons)
        if type(required_horizons) is list:
            required_horizons = np.asarray(required_horizons)
        if type(required_horizons) in [float, int]:
            required_horizons = np.asarray([required_horizons])

        # setup
        obs = obs if obs is not None else self.all_obs
        time = time if time is not None else self.all_time
        rewards = rewards if rewards is not None else self.ext_rewards
        dones = dones if dones is not None else self.terminals
        tvf_mode = tvf_mode or args.tvf_return_mode
        tvf_n_step = tvf_n_step or args.tvf_return_n_step

        N, A, *state_shape = obs[:-1].shape

        assert obs.shape == (N + 1, A, *state_shape)
        assert rewards.shape == (N, A)
        assert dones.shape == (N, A)

        # step 1:
        # use our model to generate the value estimates required
        # for MC this is just an estimate at the end of the window
        assert value_sample_horizons[0] == 0 and value_sample_horizons[-1] == self.current_horizon, "First and value horizon are required."

        value_samples = self.get_value_estimates(
            obs=obs,
            time=time,
            horizons=value_sample_horizons,
            head=f"{head}_value",
        )

        if include_second_moment:
            sqrt_m2_value_samples = self.get_value_estimates(
                obs=obs,
                time=time,
                horizons=value_sample_horizons,
                head=f"{head}_value_m2",
            )
            # model predicts sqrt of sqr, so need to square it here
            value_samples_m2 = np.maximum(sqrt_m2_value_samples, 0) ** 2
        else:
            value_samples_m2 = None

        # step 2: calculate the returns
        start_time = clock.time()

        # setup return estimator mode, but only verify occasionally.
        re_mode = args.return_estimator_mode
        if re_mode == "verify" and self.batch_counter % 31 != 1:
            re_mode = "default"

        returns = get_return_estimate(
            mode=tvf_mode,
            gamma=args.tvf_gamma,
            rewards=rewards,
            dones=dones,
            required_horizons=required_horizons,
            value_sample_horizons=value_sample_horizons,
            value_samples=value_samples,
            value_samples_m2=value_samples_m2,
            n_step=tvf_n_step,
            max_samples=args.tvf_return_samples,
            estimator_mode=re_mode,
            log=self.log,
            use_log_interpolation=args.tvf_return_use_log_interpolation,
        )

        if include_second_moment:
            # we always return the sqrt of the second moment.
            m1, m2 = returns
            returns = (m1, np.maximum(m2, 0) ** 0.5)

        return_estimate_time = clock.time() - start_time
        self.log.watch_mean(
            "time_return_estimate",
            return_estimate_time,
            display_precision=2,
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

        samples = min(args.erp_samples, len(buffer))
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

        features_code = "features" if args.erp_relu else "raw_features"
        replay_features = buffer_output[features_code]
        obs_features = obs_output[features_code]

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

    @property
    def erp_internal_distance(self):
        return self.erp_stats.get("erp_internal_distance", None)

    @erp_internal_distance.setter
    def erp_internal_distance(self, value):
        self.erp_stats["erp_internal_distance"] = value

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



    @torch.no_grad()
    def generate_rollout(self):

        assert self.vec_env is not None, "Please call create_envs first."

        def upload_if_needed(x):
            if type(self.all_obs) is torch.Tensor:
                x = torch.from_numpy(x).to(self.all_obs.device)
            return x

        # times are...
        # Forward: 1.1ms
        # Step: 33ms
        # Compress: 2ms
        # everything else should be minimal.

        self.model.train()

        self.int_rewards *= 0
        for k in self.stats.keys():
            if k.startswith("batch_"):
                self.stats[k] *= 0

        needs_value_output = True

        for t in range(self.N):

            prev_obs = self.obs.copy()
            prev_time = self.time.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.detached_batch_forward(
                self.obs,
                output="default" if needs_value_output else "policy", # would make sense to do both here? especially for batch norm?
                include_rnd=args.use_rnd,
                update_normalization=True
            )

            log_policy = model_out["log_policy"].cpu().numpy()
            raw_policy = model_out["raw_policy"].cpu().numpy()

            value_estimate = model_out["ext_value"].cpu().numpy()

            if args.use_rnd:
                # update the intrinsic rewards
                self.int_rewards[t] += model_out["rnd_error"].detach().cpu().numpy()
                self.int_value[t] = model_out["int_value"].detach().cpu().numpy()

            # sample actions and run through environment.
            actions = self.sample_actions(model_out)

            self.obs, ext_rewards, dones, infos = self.vec_env.step(actions)

            # time fraction
            self.time = np.asarray([info["time"] for info in infos])

            # per step reward noise
            if args.per_step_reward_noise > 0:
                ext_rewards += np.random.normal(0, args.per_step_reward_noise, size=ext_rewards.shape)

            # save raw rewards for monitoring the agents progress
            raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(ext_rewards, infos)],
                                     dtype=np.float32)

            self.episode_score += raw_rewards
            self.episode_len += 1

            for i, (done, info) in enumerate(zip(dones, infos)):
                if "reward_clips" in info:
                    self.stats['reward_clips'] += info["reward_clips"]
                if "game_freeze" in info:
                    self.stats['game_crashes'] += 1
                if "repeated_action" in info:
                    self.stats['action_repeats'] += 1
                if "repeated_action" in info:
                    self.stats['batch_action_repeats'] += 1
                if "pushed_button" in info:
                    self.stats['big_red_button_pushes'] += 1
                if "pushed_button" in info:
                    self.stats['batch_big_red_button_pushes'] += 1

                # log probability of pushing button
                if "button" in info:
                    correct_action = info["button"]
                    noop_probability = np.exp(log_policy[i, correct_action])
                    self.log.watch_mean("brb_prob", 1 - noop_probability, history_length=100, display_width=8, display_precision=5)

                if done:
                    # this should be always updated, even if it's just a loss of life terminal
                    self.episode_length_buffer.append(info["ep_length"])
                    self.episode_brb_buffer.append("pushed_button" in info)

                    if "fake_done" in info:
                        # this is a fake reset on loss of life...
                        continue

                    # reset is handled automatically by vectorized environments
                    # so just need to keep track of book-keeping
                    self.ep_count += 1
                    self.log.watch_full("ep_score", info["ep_score"], history_length=100)
                    self.log.watch_full("ep_length", info["ep_length"])
                    self.log.watch_mean("ep_count", self.ep_count, history_length=1)
                    #self.log.watch_mean("ep_brb", 100.0*self.episode_brb_fraction, history_length=1)

                    self.episode_score[i] = 0
                    self.episode_len[i] = 0

            if args.use_compression:
                prev_obs = np.asarray([compression.BufferSlot(prev_obs[i]) for i in range(len(prev_obs))])

            self.all_obs[t] = upload_if_needed(prev_obs)
            self.done = dones

            self.all_time[t] = prev_time
            self.actions[t] = actions

            self.ext_rewards[t] = ext_rewards
            self.ext_value[t] = value_estimate
            self.log_policy[t] = log_policy
            self.raw_policy[t] = raw_policy
            self.terminals[t] = dones

        # save the last state
        if args.use_compression:
            last_obs = np.asarray([compression.BufferSlot(self.obs[i]) for i in range(len(self.obs))])
        else:
            last_obs = self.obs
        self.all_obs[-1] = upload_if_needed(last_obs)
        self.all_time[-1] = self.time

        # work out final value if needed
        if args.use_rnd:
            final_state_out = self.detached_batch_forward(self.obs, output="policy")
            self.int_value[-1] = final_state_out["int_value"].detach().cpu().numpy()

        # required for PPG
        if args.aux_epochs > 0:
            final_state_out = self.detached_batch_forward(self.obs, output="value")
            self.ext_value[-1] = final_state_out["ext_value"].detach().cpu().numpy()

        # turn off train mode (so batch norm doesn't update more than once per example)
        self.model.eval()

        # check if environments are in sync or not...
        rollout_rvs = self.time / max(self.time) # todo: fix time divide error...
        ks = scipy.stats.kstest(rvs=rollout_rvs, cdf=scipy.stats.uniform.cdf)
        self.log.watch("t_ks", ks.statistic, display_width=0)

        # calculate int_value for intrinsic motivation (RND does not need this as it was done during rollout)
        # note: RND generates int_value during rollout, however in dual mode these need to (and should) come
        # from the value network, so we redo them here.
        if args.use_ebd or args.use_erp or (args.use_rnd and args.architecture == "dual"):
            N, A = self.prev_time.shape
            aux_features = None
            if args.use_tvf:
                aux_features = package_aux_features(
                    np.asarray(self.get_standard_horizon_sample(self.current_horizon)),
                    self.all_time.reshape([(N + 1) * A])
                )
            output = self.detached_batch_forward(
                utils.merge_down(self.all_obs),
                aux_features,
                output="full"
            )

            # note: in single mode value_int_value = policy_int_value
            self.int_value[:, :] = output["value_int_value"].reshape([(N + 1), A]).detach().cpu().numpy()

            if args.use_erp:
                # calculate intrinsic reward via replay diversity

                if args.erp_reduce == "mean":
                    reduce_fn = np.nanmean
                elif args.erp_reduce == "min":
                    reduce_fn = np.nanmin
                elif args.erp_reduce == "top5":
                    def top5(x, axis):
                        x_sorted = np.sort(x, axis=axis)[:5]
                        return np.nanmean(x_sorted, axis=axis)
                    reduce_fn = top5
                else:
                    raise ValueError(f"Invalid erp_reduce {args.erp_reduce}")

                def get_distances(target: np.ndarray, enable_mask=False):
                    samples = min(args.erp_samples, len(target))
                    samples = np.random.choice(len(target), [samples], replace=False)
                    internal_distance = self.get_diversity(target[samples], target, reduce_fn, mask=samples).mean()
                    target_distances = self.get_diversity(
                        utils.merge_down(self.prev_obs), target, reduce_fn,
                        mask=np.arange(len(target)) if enable_mask else None
                    )
                    return internal_distance, target_distances

                def get_intrinsic_rewards(mode:str):
                    if mode == "rollout" or self.replay_buffer.current_size == 0:
                        internal_distance, rollout_distances = get_distances(utils.merge_down(self.prev_obs), enable_mask=True)
                    elif mode == "replay":
                        internal_distance, rollout_distances = get_distances(self.replay_buffer.data[:self.replay_buffer.current_size])
                    else:
                        raise ValueError(f"Invalid mode {mode}")

                    id_code = f"{mode}_internal_distance"
                    self.erp_stats[id_code] = internal_distance

                    self.log.watch_mean(f"{mode}_internal_distance", internal_distance, display_width=0)

                    # calculate intrinsic reward
                    self.log.watch_mean_std(f"{mode}_distance", rollout_distances, display_width=0)
                    if args.erp_bias == "centered":
                        bias = rollout_distances.mean()
                    elif args.erp_bias == "none":
                        bias = 0
                    elif args.erp_bias == "internal":
                        bias = self.erp_stats[id_code]
                    else:
                        raise ValueError(f"Invalid erp_bias {args.erp_bias}")

                    return (rollout_distances.reshape([N, A]) - bias) / self.erp_stats[id_code]

                if args.erp_source == "rollout":
                    self.int_rewards += get_intrinsic_rewards("rollout")
                elif args.erp_source == "replay":
                    self.int_rewards += get_intrinsic_rewards("replay")
                elif args.erp_source == "both":
                    self.int_rewards += (get_intrinsic_rewards("replay") + get_intrinsic_rewards("rollout")) / 2
                else:
                    raise ValueError(f"Invalid erp_source {args.erp_source}")

            if args.use_ebd:
                if args.use_tvf:
                    if args.tvf_force_ext_value_distil:
                        # in this case it is just policy_ext_value that has been trained
                        # note: there might be a problem here as value gets time, but policy does not, using
                        # a shorter horizon might work better...
                        policy_prediction = output["policy_ext_value"][..., np.newaxis]
                        value_prediction = output["value_tvf_value"][..., -1][..., np.newaxis]
                    else:
                        # use samples from entire curve
                        policy_prediction = output["policy_tvf_value"]
                        value_prediction = output["value_tvf_value"]
                else:
                    policy_prediction = output["policy_ext_value"][..., np.newaxis]
                    value_prediction = output["value_ext_value"][..., np.newaxis]
                errors = (policy_prediction - value_prediction).mean(dim=-1)
                int_rewards = torch.square(errors).reshape([N+1, A])
                self.int_rewards += int_rewards[:N].cpu().numpy()

        self.int_rewards = np.clip(self.int_rewards, -5, 5) # just in case there are extreme values here

        aux_fields = {}

        # calculate targets for ppg
        if args.aux_epochs > 0:
            v_target = calculate_gae(
                self.ext_rewards,
                self.ext_value[:self.N],
                self.ext_value[self.N],
                self.terminals,
                self.gamma,
                self.lambda_value
            ) + self.ext_value[:self.N]
            aux_fields['vtarg'] = utils.merge_down(v_target)

        # add data to replay buffer if needed
        steps = (np.arange(args.n_steps * args.agents) + self.step)
        if self.replay_buffer is not None:
            self.replay_buffer.add_experience(
                utils.merge_down(self.prev_obs),
                self.replay_buffer.create_aux_buffer(
                    (len(steps),),
                    time=utils.merge_down(self.prev_time),
                    reward=utils.merge_down(self.ext_rewards),
                    action=utils.merge_down(self.actions),
                    step=steps,
                    **aux_fields,
                )
            )

        # log the expected return, and expected squared return for each horizon
        # this is just for debugging second moment learning
        for h in self.tvf_debug_horizons:
            # this will be a bit noisy, as we only use the first transition within the rollout.
            if h < self.N:
                self.log.watch_mean(
                    f"av_r_{h}",
                    self.ext_rewards[:h].sum(axis=0).mean(),
                    display_width=0,
                )
                self.log.watch_mean(
                    f"av_r2_{h}",
                    (self.ext_rewards[:h].sum(axis=0)**2).mean(),
                    display_width=0,
                )

        if args.debug_terminal_logging:
            for t in range(self.N):
                first_frame = max(t-2, 0)
                last_frame = t+2
                for i in range(self.A):
                    if self.terminals[t, i]:
                        time_1 = round(self.all_time[t, i]*args.timeout)
                        time_2 = round(self.all_time[t+1, i] * args.timeout)
                        self.export_debug_frames(
                            f"{args.log_folder}/{self.batch_counter:04}-{i:04}-{t:03} [{time_1:04}-{time_2:04}].png",
                            self.all_obs[first_frame:last_frame + 1, i].decompress(),
                            marker=t-first_frame+1
                        )

    def estimate_noise_level(self, batch_data, mini_batch_func, optimizer: torch.optim.Optimizer, label):
        """
        Estimates the critical batch size using the gradient magnitude of a small batch and a large batch
        See: https://arxiv.org/pdf/1812.06162.pdf

        """

        self.log.mode = self.log.LM_MUTE
        result = {}

        def process_gradient(context):
            # calculate norm of gradient
            parameters = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        parameters.append(p)

            result['grad_magnitude'] = nn.utils.clip_grad_norm_(parameters, 99999)
            optimizer.zero_grad(set_to_none=True)
            return True  # make sure to not apply the gradient!

        hook = {'after_mini_batch': process_gradient}

        b_small = 32
        b_big = len(batch_data["prev_state"])

        # take 16 samples of the small gradient.
        # ideally we would do this using a shuffle, but with replacement should be fine too
        g_b_small_squared = 0
        for sample in range(16):
            self.train_batch(batch_data, mini_batch_func, b_small, optimizer, label, hooks=hook)
            g_b_small_squared += float(result['grad_magnitude']) / 16
        g_b_small_squared *= g_b_small_squared

        self.train_batch(batch_data, mini_batch_func, b_big, optimizer, label, hooks=hook)
        g_b_big_squared = float(result['grad_magnitude'])
        g_b_big_squared *= g_b_big_squared

        g2 = (b_big * g_b_big_squared - b_small * g_b_small_squared) / (b_big - b_small)
        s = (g_b_small_squared - g_b_big_squared) / (1 / b_small - 1 / b_big)
        self.log.mode = self.log.LM_DEFAULT

        self.are_stats[f'{label}_s'] = 0.9 * self.are_stats.get(f'{label}_s', s) + 0.1 * s
        self.are_stats[f'{label}_g2'] = 0.9 * self.are_stats.get(f'{label}_g2', g2) + 0.1 * g2

        # g2 estimate is frequently negative. If ema average bounces below 0 the ratio will become negative.
        # to avoid this we clip the *smoothed* g2 to epsilon.
        # alternative include larger batch_sizes, and / or larger EMA horizon.
        epsilon = 1e-4
        target_smooth_ratio = self.are_stats[f'{label}_s'] / np.clip(self.are_stats[f'{label}_g2'], epsilon,
                                                                     float('inf'))

        # a second layer of EMA makes sure that mini-batch size doesn't flick between 8k, and 32 every other update.
        # slow changes in the ratio will come through quickly, but large flucations will be averaged out.
        self.are_stats[f'{label}_ratio'] = self.are_stats.get(f'{label}_ratio',
                                                              target_smooth_ratio) * 0.9 + target_smooth_ratio * 0.1

        self.log.watch(f'are_{label}_s', s, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_g2', g2, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_b', target_smooth_ratio, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_smooth_b', self.are_stats[f'{label}_ratio'], display_precision=0, display_width=0)
        self.log.watch(
            f'are_{label}_sqrt_b',
            np.clip(target_smooth_ratio, 0, float('inf')) ** 0.5,
            display_precision=0,
            display_name=f"{label}_sns"

        )

        return self.are_stats[f'{label}_ratio']

    def estimate_noise_level_v2(self, batch_data, mini_batch_func, optimizer: torch.optim.Optimizer, label):
        """
        Estimates the critical batch size using the gradient magnitude of a small batch and a large batch
        See: https://arxiv.org/pdf/1812.06162.pdf

        This is an improved version that should be less biased, but was developed after the experiments.
        """

        self.log.mode = self.log.LM_MUTE
        result = {}

        def process_gradient(context):
            # calculate norm of gradient
            parameters = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        parameters.append(p)

            result['grad_magnitude'] = nn.utils.clip_grad_norm_(parameters, 99999)
            optimizer.zero_grad(set_to_none=True)
            return True  # make sure to not apply the gradient!

        hook = {'after_mini_batch': process_gradient}

        SMALL_SAMPLES = 32
        b_small = 32
        b_big = len(batch_data["prev_state"])

        g2s = []
        ss = []

        self.train_batch(batch_data, mini_batch_func, b_big, optimizer, label, hooks=hook)
        g_b_big_squared = float(result['grad_magnitude']) ** 2

        for sample in range(SMALL_SAMPLES):
            self.train_batch(batch_data, mini_batch_func, b_small, optimizer, label, hooks=hook)
            g_b_small_squared = float(result['grad_magnitude']) ** 2
            g2s.append((b_big * g_b_big_squared - b_small * g_b_small_squared) / (b_big - b_small))
            ss.append((g_b_small_squared - g_b_big_squared) / (1 / b_small - 1 / b_big))

        s_mean = np.mean(ss)
        g2_mean = np.mean(g2s)

        self.log.mode = self.log.LM_DEFAULT

        # add these samples to the mix
        for var_name, var_value in zip(['s', 'g2'], [s_mean, g2_mean]):
            if f'{label}_{var_name}_history' not in self.are_stats:
                self.are_stats[f'{label}_{var_name}_history'] = deque(maxlen=100)
            self.are_stats[f'{label}_{var_name}_history'].append(var_value)
            self.are_stats[f'{label}_{var_name}'] = np.mean(self.are_stats[f'{label}_{var_name}_history'])

        av_s = np.mean(self.are_stats[f'{label}_s_history'])
        av_g2 = np.mean(self.are_stats[f'{label}_g2_history'])

        # g2 estimate is frequently negative. If ema average bounces below 0 the ratio will become negative.
        # to avoid this we clip the *smoothed* g2 to epsilon.
        # alternative include larger batch_sizes, and / or larger EMA horizon.
        epsilon = 1e-4
        ratio = max(av_s, epsilon) / max(av_g2, epsilon)

        self.are_stats[f'{label}_ratio'] = ratio

        self.log.watch(f'are_{label}_smooth_s', av_s, display_precision=0, display_width=8, display_name=f"sns_{label}_s")
        self.log.watch(f'are_{label}_smooth_g2', av_g2, display_precision=0, display_width=8, display_name=f"sns_{label}_g2")

        self.log.watch(f'are_{label}_s', s_mean, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_g2', g2_mean, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_b', ratio, display_precision=0, display_width=0)
        self.log.watch(f'are_{label}_smooth_b', self.are_stats[f'{label}_ratio'], display_precision=0, display_width=0)
        self.log.watch(
            f'are_{label}_sqrt_b',
            np.clip(ratio, 0, float('inf')) ** 0.5,
            display_precision=0,
            display_name=f"sns_{label}"
        )

        return self.are_stats[f'{label}_ratio']

    @torch.no_grad()
    def get_value_estimates(self, obs: Union[np.ndarray, torch.Tensor], time: Union[None, np.ndarray]=None,
                            horizons: Union[None, np.ndarray, int] = None,
                            include_model_out: bool = False,
                            head: str = "ext_value",
                            ) -> Union[np.ndarray, tuple]:
        """
        Returns value estimates for each given observation
        If horizons are none current_horizon is used.
        obs: np array of dims [N, A, *state_shape]
        time: np array of dims [N, A]
        horizons:
            ndarray of dims [K] (returns NAK)
            integer (returns NA for horizon given
            none (returns NA for current horizon

        returns: ndarray of dims [N, A, K] if horizons was array else [N, A]
        """

        N, A, *state_shape = obs.shape

        if args.use_tvf:
            horizons = horizons if horizons is not None else self.current_horizon

            assert time is not None and time.shape == (N, A)
            if type(horizons) == int:
                pass
            elif type(horizons) is np.ndarray:
                assert len(horizons.shape) == 1, f"Invalid horizon shape {horizons.shape}"
            else:
                raise ValueError("Invalid horizon type {type(horizons)}")

            if type(horizons) in [int, float]:
                scalar_output = True
                horizons = np.asarray([horizons])
            else:
                scalar_output = False

            time = time.reshape(N*A)

            model_out = self.detached_batch_forward(
                obs=obs.reshape([N * A, *state_shape]),
                aux_features=package_aux_features(horizons, time),
                output="value",
            )

            values = model_out[f"tvf_{head}"]

            if scalar_output:
                result = values.reshape([N, A]).cpu().numpy()
            else:
                result = values.reshape([N, A, horizons.shape[-1]]).cpu().numpy()
        else:
            assert horizons is None, "PPO only supports max horizon value estimates"
            model_out = self.detached_batch_forward(
                obs=obs.reshape([N * A, *state_shape]),
                output="value",
            )
            result = model_out[head].reshape([N, A]).cpu().numpy()

        if include_model_out:
            return result, model_out
        else:
            return result

    @torch.no_grad()
    def log_detailed_value_quality(self):
        """

        This function generates detailed logging information about quality of each of the return estimation methods.
        It is *very* slow, and logs *a lot* of information but can give helpful insights into which estimation
        methods work best when, and what hyperparameters they should use.

        The function uses the actual return distribution for a each horizon < args.n_steps. This is achived by
        generating the rollout multiple times using different seeds to get the actual return distribution for the
        starting state for each of the horizons.

        One disadvantage of this approach is that the value estimates used are taken from the agent used durning
        training, and therefore if these estimates are poor (or biased) it might effect the other estimates.
        For this reason it's normally best to set return estimation to fixed with tvf_return_n_step=tvf_n_step so
        that the return estimates are unbiased. (not 100% sure about this actually... maybe just use exp, which seems
        good enough.)

        Suggestion is to run this with 128 agents and n_steps=512, with a 10k desync
        """

        # step 1: get ground truth by regenerating the rollout 100 times
        root_env_state = utils.save_env_state(self.vec_env) # save current state of environments

        # helpful variables for later

        h = 1
        HORIZONS = []
        while h <= args.dvq_rollout_length:
            HORIZONS.append(h)
            h *= 2
        S = args.dvq_samples    # number of samples (required number depends on env + agent stochasticity)
        N = args.dvq_rollout_length  # N here need not be n_steps
        A = args.agents          # agents must be the same as the number of training agents
        K = len(HORIZONS)

        # unfortunately we can not get estimates for horizons longer than N, so just generate the return estimates
        # up to N.
        VALUE_SAMPLE_HORIZONS = self.generate_horizon_sample(
            # I really want 128 samples, but need to reduce the space requirements a little.
            N, 64, distribution="fixed_geometric", force_first_and_last=True
        )
        VALUE_SAMPLE_HORIZONS = list(set(VALUE_SAMPLE_HORIZONS)) # remove any duplicates
        VALUE_SAMPLE_HORIZONS.sort()
        V = len(VALUE_SAMPLE_HORIZONS)

        # rollout stuff
        discount = np.ones([A], dtype=np.float32)

        # need these for all samples
        rewards = np.zeros([N, A], dtype=np.float32)
        mask = np.zeros([N, A], dtype=np.bool)
        dones = np.zeros([N, A], dtype=np.bool)

        # value estimates are for each transition in rollout, store in float16 to save some space
        # (otherwise his is 1.3G per save, even with compression)
        value_estimates = np.zeros([N+1, A, V], dtype=np.float16)
        value_estimates_m2 = np.zeros([N + 1, A, V], dtype=np.float16)
        all_times = np.zeros([N + 1, A], dtype=np.uint32)  # not needed, just for debugging

        # this is required as otherwise when we call .step it will simply overwrite the old self.obs (which we need for later)
        self.obs = self.obs.copy()

        print("Generating rollout:")

        for seed in tqdm(range(S)):

            seed_base = self.step + (seed * 191)

            # set general seeds
            torch.manual_seed(seed_base)
            np.random.seed(seed_base)

            # init rollout stuff
            discount = np.ones_like(discount)
            obs = self.obs  # [A, *state_shape]
            infos = []

            # reset and reseed environments
            utils.restore_env_state(self.vec_env, root_env_state)
            #self.vec_env.seed([seed_base + i * 7 for i in range(A)])
            time = self.time

            steps_completed = 0

            still_running = np.ones([A], dtype=np.bool)

            for t in range(N):
                # forward observation through model
                aux_features = package_aux_features(
                    np.asarray(VALUE_SAMPLE_HORIZONS),
                    time)
                model_out = self.detached_batch_forward(obs, output='full', aux_features=aux_features)
                log_probs = model_out["policy_log_policy"].cpu().numpy()

                # sample action and act
                # note: we mask out actions of completed agents, this saves a bit of time as the wrapper will not
                # forward the action on (i.e. it will not be simulated)
                actions = utils.sample_action_from_logp(log_probs).astype("int32")
                actions = np.asarray([a if running else -1 for a, running in zip(actions, still_running)], np.int32) # a bit faster
                obs, rews, done, infos = self.vec_env.step(actions)

                # book keeping
                time = np.asarray([info["time"] for info in infos])
                rewards[t, :] = rews * still_running  # all we need are the rewards, dones, and the value estimates
                dones[t, :] = done
                all_times[t, :] = time * still_running
                mask[t, :] = still_running
                value_estimates[t, :, :] = model_out["value_tvf_ext_value"].cpu().numpy() * still_running[:, None]
                value_estimates_m2[t, :, :] = model_out["value_tvf_ext_value_m2"].cpu().numpy() * still_running[:, None]

                # update discount
                discount *= args.tvf_gamma
                discount *= (1-done)

                steps_completed += 1

                still_running *= (1-done).astype(np.bool)

                if sum(still_running) == 0:
                    # this occurs when all agents have died.
                    break

            # final value estimate
            time = np.asarray([info["time"] for info in infos])
            aux_features = package_aux_features(
                np.asarray(VALUE_SAMPLE_HORIZONS),
                time
            )
            model_out = self.detached_batch_forward(obs, output='full', aux_features=aux_features)
            value_estimates[steps_completed, :, :] = model_out["value_tvf_ext_value"].cpu().numpy() * still_running[:, None]
            value_estimates_m2[steps_completed, :, :] = model_out["value_tvf_ext_value_m2"].cpu().numpy() * still_running[:, None]
            all_times[steps_completed, :] = time * still_running

            # save the data
            with open(f"{args.log_folder}/rollouts_{self.step//args.batch_size}_{seed}.dat", "wb") as f:
                data = {
                    'seed': seed,
                    'rewards': rewards,
                    'dones': dones,
                    'required_horizons': np.asarray(HORIZONS),
                    'value_sample_horizons': VALUE_SAMPLE_HORIZONS,
                    'value_samples': value_estimates,
                    'value_samples_m2': value_estimates_m2,
                    'reward_scale': self.reward_scale,
                    'mask': mask,
                    'all_times': all_times,
                    'gamma': args.tvf_gamma,
                }

                # zlip + blosc is the best compression I could find, it should be about 3:1 compared to
                # gzip which is 2:1 and very slow
                def compress(x):
                    return blosc.pack_array(x, cname='zlib', shuffle=blosc.SHUFFLE, clevel=5)

                for k in data.keys():
                    v = data[k]
                    if type(v) is np.ndarray:
                        data[k] = compress(v)
                pickle.dump(data, f)

        # restore final state so training can pick up where we left off.
        # the seeding means that results should be consistent, even if we add additional evaluations
        seed_base = self.step + 187
        utils.restore_env_state(self.vec_env, root_env_state)
        #self.vec_env.seed([seed_base + i * 7 for i in range(A)])
        torch.manual_seed(seed_base)
        np.random.seed(seed_base)

        print()

    @torch.no_grad()
    def log_dna_value_quality(self, head="ext_value"):
        targets = calculate_bootstrapped_returns(
            self.ext_rewards, self.terminals, self.ext_value[self.N], self.gamma
        )
        values = self.ext_value[:self.N]
        self.log.watch_mean("ev_ext", utils.explained_variance(values.ravel(), targets.ravel()), history_length=1)

    def log_curve_quality(self, estimates, targets, postfix: str = ''):
        """
        Calculates explained variance at each of the debug horizons
        @param estimates: np array of dims[N,A,K]
        @param targets: np array of dims[N,A,K]
        @param postfix: postfix to add after the name during logging.
        where K is the length of tvf_debug_horizons

        """
        total_not_explained_var = 0
        total_var = 0
        for h_index, h in enumerate(self.tvf_debug_horizons):
            value = estimates[:, :, h_index].reshape(-1)
            target = targets[:, :, h_index].reshape(-1)

            this_var = np.var(target)
            this_not_explained_var = np.var(target - value)
            total_var += this_var
            total_not_explained_var += this_not_explained_var

            ev = 0 if (this_var == 0) else np.clip(1 - this_not_explained_var / this_var, -1, 1)

            self.log.watch_mean(
                f"ev_{h}"+postfix,
                ev,
                display_width=8 if (10 <= h <= 30) or h == args.tvf_max_horizon else 0,
                history_length=1
            )

            # this is just for debugging sml
            self.log.watch_mean(
                f"v_{h}" + postfix,
                value.mean(),
                display_width=0,
                history_length=1
            )

        self.log.watch_mean(
            f"ev_average"+postfix,
            0 if (total_var == 0) else np.clip(1 - total_not_explained_var / total_var, -1, 1),
            display_width=8,
            display_name="ev_avg"+postfix,
            history_length=1
        )



    @torch.no_grad()
    def log_tvf_value_quality(self):
        """
        Writes value quality stats to log
        """

        N, A, *state_shape = self.prev_obs.shape
        K = len(self.tvf_debug_horizons)

        # first we generate the value estimates, then we calculate the returns required for each debug horizon
        # because we use sampling it is not guaranteed that these horizons will be included, so we need to
        # recalculate everything

        model_out = self.detached_batch_forward(
            obs=utils.merge_down(self.prev_obs),
            aux_features=package_aux_features(np.asarray(self.tvf_debug_horizons), utils.merge_down(self.prev_time)),
            output="value",
        )

        value_samples = self.generate_horizon_sample(
            self.current_horizon,
            args.tvf_value_samples,
            distribution=args.tvf_value_distribution,
            force_first_and_last=True,
        )

        targets = self.calculate_sampled_returns(
            value_sample_horizons=value_samples,
            required_horizons=self.tvf_debug_horizons,
            obs=self.all_obs,
            time=self.all_time,
            rewards=self.ext_rewards,
            dones=self.terminals,
            tvf_mode="fixed",  # <-- MC is the least bias method we can do...
            tvf_n_step=args.n_steps,
            include_second_moment=args.learn_second_moment
        )

        if args.learn_second_moment:

            first_moment_targets, sqrt_second_moment_targets = targets
            first_moment_estimates = model_out["tvf_ext_value"].reshape(N, A, K).cpu().numpy()
            sqrt_second_moment_estimates = model_out["tvf_ext_value_m2"].reshape(N, A, K).cpu().numpy()
            self.log_curve_quality(first_moment_estimates, first_moment_targets)
            self.log_curve_quality(sqrt_second_moment_estimates, sqrt_second_moment_targets, postfix='_m2')

            # also log the variance estimates
            for h_index, h in enumerate(self.tvf_debug_horizons):
                var_est = sqrt_second_moment_estimates[:, :, h_index] ** 2 - (first_moment_estimates[:, :, h_index] ** 2)
                var_est = np.clip(var_est, 0, float('inf'))
                self.log.watch_mean(
                    f"var_{h}",
                    var_est.mean(),
                    display_width=0,
                    history_length=1
                )
        else:
            first_moment_targets = targets
            first_moment_estimates = model_out["tvf_ext_value"].reshape(N, A, K).cpu().numpy()
            self.log_curve_quality(first_moment_estimates, first_moment_targets)



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
        h = 1/(1 - self.lambda_value)
        lambdas = [1 - (1 / (factor * h)) for factor in np.geomspace(0.25, 4.0, SAMPLES)]

        advantage_estimate = np.zeros([SAMPLES, N, A], dtype=np.float32)

        for i, lamb in enumerate(lambdas):
            advantage_estimate[i] = calculate_gae(
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
    def get_tvf_rediscounted_value_estimates(self, rollout, new_gamma:float):
        """
        Returns rediscounted value estimate for given rollout (i.e. rewards + value if using given gamma)
        """

        N, A, *state_shape = rollout.all_obs[:-1].shape

        if (abs(new_gamma - args.tvf_gamma) < 1e-8):
            # no rediscounting is required...
            return self.get_value_estimates(
                obs=rollout.all_obs,
                time=rollout.all_time
            )

        # these makes sure error is less than 1%
        c_1 = 7
        c_2 = 400

        # work out a range and skip to use so that we never use more than around 400 samples and we don't waste
        # samples on heavily discounted rewards
        if new_gamma >= 1:
            effective_horizon = round(rollout.current_horizon)
        else:
            effective_horizon = min(round(c_1 / (1 - new_gamma)), rollout.current_horizon)

        # note: we could down sample the value estimates and adjust the gamma calculations if
        # this ends up being too slow..
        step_skip = math.ceil(effective_horizon / c_2)

        # going backwards makes sure that final horizon is always included
        horizons = np.asarray(range(effective_horizon, 0, -step_skip))[::-1]

        # these are the value estimates at each horizon under current tvf_gamma
        value_estimates = self.get_value_estimates(
            obs=rollout.all_obs,
            time=rollout.all_time,
            horizons=horizons
        )

        # convert to new gamma
        return get_rediscounted_value_estimate(
            values=value_estimates.reshape([(N + 1) * A, -1]),
            old_gamma=self.tvf_gamma,
            new_gamma=new_gamma,
            horizons=horizons
        ).reshape([(N + 1), A])


    def calculate_returns(self):

        self.old_time = time.time()

        N, A, *state_shape = self.prev_obs.shape

        self.model.eval()

        # generate a candidate gamma (if needed) that will be used to calculate this rounds advantages...
        # for score I'm using average reward. Using some return estimate is problematic as it will include gamma
        # and gamma has changed. This might make the algorithm prefer long horizons over short. Maybe this is ok though?
        if args.auto_strategy[:3] == "sa_":
            if args.auto_strategy == "sa_reward":
                # estimate of gain (i.e. average reward)
                score = np.mean(self.ext_rewards)
                prev_score = self.horizon_sa.prev_score
            elif args.auto_strategy == "sa_return":
                # note: we can't use the return estimates below as these require a decision on gamma, which we need
                # to make before we calculate them. So instead we run through the rewards, discount then, then
                # add the final value.
                def get_discounted_score(rollout: RolloutBuffer):
                    if args.use_tvf:
                        batch_value_estimates = self.get_tvf_rediscounted_value_estimates(rollout, new_gamma=self.gamma)
                    else:
                        # in this case just generate ext value estimates from model, which will be using self.gamma
                        batch_value_estimates = self.get_value_estimates(rollout.all_obs)

                    ext_advantage = calculate_gae(
                        rollout.ext_rewards,
                        batch_value_estimates[:N],
                        batch_value_estimates[N],
                        rollout.terminals,
                        self.gamma,
                        self.lambda_policy
                    )

                    batch_returns = ext_advantage + batch_value_estimates[:N]
                    return np.mean(batch_returns)

                this_rollout = RolloutBuffer(self)
                score = get_discounted_score(this_rollout)
                if self.previous_rollout is None:
                    self.previous_rollout = this_rollout
                prev_score = get_discounted_score(self.previous_rollout)
                self.previous_rollout = this_rollout
            else:
                raise ValueError(f"Invalid SA strategy {args.auto_strategy}")

            self.horizon_sa.process(score, prev_score)
            self.log.watch_mean("sa_acceptance", self.horizon_sa.acceptance_rate, history_length=10, display_width=8)
            self.log.watch_mean("sa_score", score, history_length=10, display_width=0)
            self.log.watch_mean("sa_score_delta", score-prev_score, history_length=10, display_width=0)

        # 1. first we calculate the ext_value estimate
        if args.use_tvf:
            ext_value_estimates = self.get_tvf_rediscounted_value_estimates(
                RolloutBuffer(self, copy=False),
                new_gamma=self.gamma
            )
            # todo: uni value estimates for tvf
        else:
            # in this case just generate ext value estimates from model
            ext_value_estimates, model_out = self.get_value_estimates(obs=self.all_obs, include_model_out=True)
            self.uni_value = model_out["uni_value"].detach().cpu().numpy().reshape([N+1, A])

        self.ext_value = ext_value_estimates

        # GAE requires inputs to be true value not transformed value...
        if args.tvf_gae and args.use_tvf:
            self.ext_advantage = calculate_gae_tvf(
                self.ext_rewards,
                ext_value_estimates[:N],
                ext_value_estimates[N],
                self.terminals,
                self.gamma,
                self.lambda_policy
            )
        else:
            self.ext_advantage = calculate_gae(
                self.ext_rewards,
                ext_value_estimates[:N],
                ext_value_estimates[N],
                self.terminals,
                self.gamma,
                self.lambda_policy
            )

        # calculate ext_returns for PPO targets, and for debugging
        # note, we use a different lambda for these.
        temp_advantage_estimate = calculate_gae(
            self.ext_rewards,
            ext_value_estimates[:N],
            ext_value_estimates[N],
            self.terminals,
            self.gamma,
            self.lambda_value,
        )
        self.ext_returns = temp_advantage_estimate + ext_value_estimates[:N]

        if args.use_intrinsic_rewards:

            if args.normalize_intrinsic_rewards:
                # normalize returns using EMS
                # this is this how openai did it (i.e. forward rather than backwards)
                for t in range(self.N):
                    terminals = (not args.intrinsic_reward_propagation) * self.terminals[t, :]
                    self.ems_norm = (1-terminals) * args.gamma_int * self.ems_norm + self.int_rewards[t, :]
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

            self.int_advantage = calculate_gae(
                self.int_rewards,
                self.int_value[:N],
                self.int_value[N],
                (not args.intrinsic_reward_propagation) * self.terminals,
                gamma=args.gamma_int,
                lamb=self.lambda_policy
            )

            self.int_returns = self.int_advantage + self.int_value[:N]

        self.advantage = self.extrinsic_reward_scale * self.ext_advantage
        if args.use_intrinsic_rewards:
            self.advantage += self.intrinsic_reward_scale * self.int_advantage

        if args.observation_normalization:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        self.log.watch_mean("reward_scale", self.reward_scale, display_width=0, history_length=1)
        self.log.watch_mean("entropy_bonus", self.current_entropy_bonus, display_width=0, history_length=1)

        self.log.watch_mean_std("reward_ext", self.ext_rewards, display_name="rew_ext", display_width=0)
        self.log.watch_mean_std("return_ext", self.ext_returns, display_name="ret_ext")
        self.log.watch_mean_std("value_ext", self.ext_value, display_name="est_v_ext", display_width=0)

        for k, v in self.stats.items():
            self.log.watch(k, v, display_width=0)

        if args.use_tvf:
            self.log.watch("tvf_horizon", self.current_horizon)
            self.log.watch("tvf_gamma", self.tvf_gamma)

        if self.batch_counter % 4 == 0:
            # this can be a little slow, ~2 seconds, compared to ~40 seconds for the rollout generation.
            # so under normal conditions we do it every other update.
            if args.replay_size > 0:
                self.replay_buffer.log_stats(self.log)

        self.log.watch("gamma", self.gamma, display_width=0)

        if not args.disable_ev and self.batch_counter % 2 == 1:
            # only about 3% slower with this on.
            if not args.use_tvf:
                self.log_dna_value_quality()
            else:
                self.log_tvf_value_quality()

        if args.log_detailed_value_quality and self.batch_counter % args.dvq_freq == 0:
            self.log_detailed_value_quality()

        self.log.watch_mean_std("adv_ext", self.ext_advantage, display_width=0)

        if args.use_intrinsic_rewards:
            self.log.watch_mean_std("reward_int", self.int_rewards, display_name="rew_int", display_width=0)
            self.log.watch_mean_std("return_int", self.int_returns, display_name="ret_int", display_width=0)
            self.log.watch_mean_std("value_int", self.int_value, display_name="est_v_int", display_width=0)
            ev_int = utils.explained_variance(self.int_value[:-1].ravel(), self.int_returns.ravel())
            self.log.watch_mean("ev_int", ev_int)
            if ev_int <= 0:
                pass

            self.log.watch_mean_std("adv_int", self.int_advantage, display_width=0)
            self.log.watch_mean("ir_scale", self.intrinsic_reward_scale, display_width=0)
            self.log.watch_mean(
                "return_ratio",
                np.mean(np.abs(self.ext_returns)) / np.mean(np.abs(self.int_returns)),
                display_name="return_ratio",
                display_width=10
            )

        if args.normalize_intrinsic_rewards:
            self.log.watch_mean("reward_scale_int", self.intrinsic_reward_norm_scale, display_width=0)

        self.flags = {}

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

    def calculate_value_loss(self, targets:torch.tensor, predictions:torch.tensor, horizons:torch.tensor = None):
        """
        Calculate loss between predicted value and actual value

        targets: tensor of dims [N, A, H] or [N, A] or [N]
        predictions: tensor of dims matching target
        horizons: tensor of dims matching targets of type int (optional)

        returns: loss as a tensor of dims [N, A]

        """

        assert predictions.shape == targets.shape
        if horizons is not None:
            assert horizons.shape == targets.shape

        if args.tvf_loss_fn == "MSE":
            # MSE loss, sum across samples
            return torch.square(targets - predictions)
        elif args.tvf_loss_fn == "huber":
            if args.tvf_huber_loss_delta == 0:
                loss = torch.abs(targets - predictions)
            else:
                # Smooth huber loss
                # see https://en.wikipedia.org/wiki/Huber_loss
                errors = targets - predictions
                loss = args.tvf_huber_loss_delta ** 2 * (
                        torch.sqrt(1 + (errors / args.tvf_huber_loss_delta) ** 2) - 1
                )
            return loss
        elif args.tvf_loss_fn == "h_weighted":
            if horizons is None:
                # assume h = args.tvf_max_horizon for all elements.
                return torch.square(targets - predictions) / args.tvf_max_horizon

            # we need to remove all zero horizons from this calculation as they will result in NaN
            horizons = horizons[0]
            assert torch.abs(horizons - horizons[np.newaxis, :]).sum() == 0, "Batch entries must have matched horizons."
            non_zero_horizon_ids = [i for i in range(len(horizons)) if horizons[i] != 0]
            horizons = horizons[non_zero_horizon_ids]

            av_targets = targets[:, non_zero_horizon_ids] / (horizons / args.tvf_max_horizon)
            av_predictions = predictions[:, non_zero_horizon_ids] / (horizons / args.tvf_max_horizon)
            # scale so that loss scale is roughly the same as MSE loss
            return torch.square(av_targets - av_predictions) / args.tvf_max_horizon
        else:
            raise ValueError(f"Invalid tvf_loss_fn {args.tvf_loss_fn}")

    def get_distil_target_name(self):
        if args.distil_mode == "value":
            return 'ext_value'
        elif args.distil_mode == "features":
            return 'raw_features'
        else:
            raise ValueError(f"Invalid distil mode {args.distil_mode}")

    def train_distil_minibatch(self, data, loss_scale=1.0, **kwargs):

        model_out = self.model.forward(data["prev_state"], output="policy", include_features=args.distil_mode=='features')
        targets = data["distil_targets"]
        predictions = model_out[self.get_distil_target_name()]
        loss_value = 0.5 * torch.square(targets - predictions)
        if len(loss_value.shape) == 2:
            loss_value = loss_value.mean(axis=-1) # mean accross final dim if targets / predictions were vector.
        loss = loss_value

        # MSE loss on the logits...
        loss_policy = args.distil_beta * 0.5 * self.calculate_value_loss(data["old_raw_policy"], model_out["raw_policy"]).mean(dim=-1)
        loss = loss + loss_policy

        pred_var = torch.var(predictions)
        targ_var = torch.var(targets)

        # some debugging stats
        with torch.no_grad():
            self.log.watch_mean("distil_targ_var", targ_var, history_length=64 * args.distil_epochs, display_width=0)
            self.log.watch_mean("distil_pred_var", pred_var, history_length=64 * args.distil_epochs,
                                display_width=0)
            mse = torch.square(predictions - targets).mean()
            ev = 1 - torch.var(predictions-targets) / (torch.var(targets) + 1e-8)
            self.log.watch_mean("distil_ev", ev, history_length=64 * args.distil_epochs,
                                display_name="ev_dist",
                                display_width=8)
            self.log.watch_mean("distil_mse", mse, history_length=64 * args.distil_epochs,
                                display_width=0)

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        loss = loss * loss_scale
        loss.mean().backward()

        self.log.watch_mean("loss_distil_policy", loss_policy.mean(), history_length=64 * args.distil_epochs, display_width=0)
        self.log.watch_mean("loss_distil_value", loss_value.mean(), history_length=64 * args.distil_epochs, display_width=0)
        self.log.watch_mean("loss_distil", loss.mean(), history_length=64*args.distil_epochs, display_name="ls_distil", display_width=8)

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

        value_loss = self.calculate_value_loss(targets, value_predictions).mean()
        policy_loss = self.calculate_value_loss(targets, policy_predictions).mean()

        value_ev = 1 - torch.var(value_predictions - targets) / (torch.var(targets) + 1e-8)
        policy_ev = 1 - torch.var(policy_predictions - targets) / (torch.var(targets) + 1e-8)

        # we do a lot of minibatches, so makes sure we average over them all.
        history_length = 2 * args.aux_epochs*args.distil_batch_size // args.distil_mini_batch_size

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
        value_heads = []
        if not args.use_tvf:
            value_heads.append("ext")
        if args.use_intrinsic_rewards:
            value_heads.append("int")
        return value_heads

    def train_value_minibatch(self, data, loss_scale=1.0):

        # create additional args if needed
        kwargs = {}
        if args.use_tvf:
            # horizons are B, H, and times is B so we need to adjust them
            kwargs['aux_features'] = package_aux_features(
                data["tvf_horizons"],
                data["tvf_time"]
            )

        model_out = self.model.forward(data["prev_state"], output="value", include_features=True, **kwargs)

        # -------------------------------------------------------------------------
        # Calculate loss_value_function_horizons
        # -------------------------------------------------------------------------

        loss = None

        if args.use_tvf:
            # targets "tvf_returns" are [B, K]
            # predictions "tvf_value" are [B, K]
            # predictions need to be generated... this could take a lot of time so just sample a few..
            targets = data["tvf_returns"]
            value_predictions = model_out["tvf_ext_value"]
            tvf_loss = self.calculate_value_loss(targets, value_predictions, data["tvf_horizons"])
            per_horizon_loss = tvf_loss.detach().mean(dim=0).cpu().numpy()
            tvf_loss = tvf_loss.mean(dim=-1)
            tvf_loss = 0.5 * args.tvf_coef * tvf_loss
            loss = tvf_loss

            self.log.watch_mean("loss_tvf", tvf_loss.mean(), history_length=64*args.value_epochs, display_name="ls_tvf", display_width=8)
            # horizons = data["tvf_horizons"].float()
            # horizons_mu = horizons.mean(axis=0).cpu().numpy().astype('uint32')
            # horizons_var = horizons.var(axis=0).cpu().numpy().mean()
            # per horizon loss does not work yet with randomized distributions
            # if horizons_var < 1e-6:
            #     # log per horizon loss
            #     for h_index, h in enumerate(horizons_mu):
            #         self.log.watch_mean(f"loss_tvf_{h}", per_horizon_loss[h_index], history_length=64 * args.value_epochs, display_width=0)
            # else:
            #     if 'horizon_missmatch' not in self.flags:
            #         self.log.warn("Horizon missmatch, logging of TVF error disabled.")
            #         self.flags['horizon_missmatch'] = True

            if args.learn_second_moment:
                targets = data["tvf_returns_m2"]
                value_predictions = model_out["tvf_ext_value_m2"]
                sqr_loss = self.calculate_value_loss(targets, value_predictions, data["tvf_horizons"])
                sqr_loss = sqr_loss.mean(dim=-1)
                sqr_loss = 0.5 * args.tvf_coef * sqr_loss.mean()
                loss = loss + sqr_loss

                self.log.watch_mean("loss_m2", sqr_loss, history_length=64*args.value_epochs, display_name="ls_m2", display_width=8)

        # -------------------------------------------------------------------------
        # Calculate loss_value
        # -------------------------------------------------------------------------

        loss = (0.0 if loss is None else loss) + self.train_value_heads(model_out, data)

        # -------------------------------------------------------------------------
        # Uniform value
        # -------------------------------------------------------------------------

        # handle uniform value as a special case
        if args.use_uac != 0:
            assert not args.use_tvf, "TVF not supported with uniform value learning yet."

            # find the magic ratio
            log_uniform_prob = math.log(1.0 / self.policy_shape[0])
            log_policy = data["log_pac"]
            log_ratio = torch.clip(log_uniform_prob - log_policy, float('-inf'), 0)
            ratio = torch.exp(log_ratio)

            uni_predicction = model_out["uni_value"]
            targets = (1-ratio*ratio) * uni_predicction + (ratio * ratio) * data["ext_returns"]
            uni_value_loss = args.vf_coef * torch.square(uni_predicction - targets)

            self.log.watch_mean("loss_v_uni", uni_value_loss.mean(), history_length=64,
                                display_name="ls_v_uni")
            loss = loss + uni_value_loss

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        loss = loss * loss_scale
        loss.mean().backward()

        # -------------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------------

        self.log.watch_stats("value_features", model_out["features"], display_width=0)
        self.log.watch_mean("loss_value", loss.mean(), display_name=f"ls_value")

        return {
            'losses': loss.detach()
        }

    def generate_horizon_sample(
            self,
            max_value: int,
            samples: int,
            distribution: str = "linear",
            force_first_and_last: bool = False) -> np.ndarray:
        """
        generates random samples from 0 to max (inclusive) using sampling with replacement
        and always including the first and last value
        distribution is the distribution to sample from
        force_first_and_last: if true horizon 0 and horizon max_value will always be included.
        output is always sorted.

        Note: fixed_geometric may return less than the expected number of samples.

        """

        if samples == -1 or samples >= (max_value + 1):
            return np.arange(0, max_value + 1)

        # these distributions don't require random sampling, and always include first and last by default.
        if distribution == "fixed_linear":
            samples = np.linspace(0, max_value, num=samples, endpoint=True)
        elif distribution == "fixed_geometric":
            samples = np.geomspace(1, 1+max_value, num=samples, endpoint=True)-1
        elif distribution == "linear":
            samples = np.random.choice(range(1, max_value), size=samples, replace=False)
        elif distribution == "geometric":
            samples = np.random.uniform(np.log(1), np.log(max_value+1), size=samples)
            samples = np.exp(samples)-1
        elif distribution == "saturated_geometric":
            samples1 = np.random.uniform(np.log(1), np.log(min(self.N, max_value) + 1), size=samples//2)
            samples2 = np.random.uniform(np.log(1), np.log(max_value + 1), size=samples//2)
            samples = np.exp(np.concatenate([samples1, samples2])) - 1
        elif distribution == "saturated_fixed_geometric":
            samples1 = np.geomspace(1, min(self.N, max_value) + 1, num=samples//2, endpoint=False)-1
            samples2 = np.geomspace(1, 1+max_value, num=samples//2, endpoint=True)-1
            samples = np.concatenate([samples1, samples2])
        else:
            raise Exception(f"Invalid distribution {distribution}")

        samples.sort()
        if force_first_and_last:
            samples[0] = 0
            samples[-1] = max_value
        return np.rint(samples).astype(int)

    @torch.no_grad()
    def generate_return_sample(self, force_first_and_last: bool = False):
        """
        Generates return estimates for current batch of data.

        force_first_and_last: if true always includes first and last horizon

        Note: could roll this into calculate_sampled_returns, and just have it return the horizons as well?

        returns:
            returns: ndarray of dims [N,A,K] containing the return estimates using tvf_gamma discounting
            returns_m2: ndarray of dims [N,A,K] containing the squared return estimate using tvf_gamma discounting (or None)
            horizon_samples: ndarray of dims [N,A,K] containing the horizons used
        """

        # max horizon to train on
        H = self.current_horizon
        N, A, *state_shape = self.prev_obs.shape

        horizon_samples = self.generate_horizon_sample(
            H,
            args.tvf_horizon_samples,
            distribution=args.tvf_horizon_distribution,
            force_first_and_last=force_first_and_last,
        )

        value_samples = self.generate_horizon_sample(
            H,
            args.tvf_value_samples,
            distribution=args.tvf_value_distribution,
            force_first_and_last=True
        )

        if args.learn_second_moment:
            if args.sqr_return_mode == "joint":
                # both first and second moment are calculated together, using the same estimator / sampling etc.
                returns, returns_m2 = self.calculate_sampled_returns(
                    value_sample_horizons=value_samples,
                    required_horizons=horizon_samples,
                    include_second_moment=True
                )
            else:
                # we are using different estimators for the first and second moment, so run it through twice
                returns = self.calculate_sampled_returns(
                    value_sample_horizons=value_samples,
                    required_horizons=horizon_samples,
                    include_second_moment=False
                )
                _, returns_m2 = self.calculate_sampled_returns(
                    value_sample_horizons=value_samples,
                    required_horizons=horizon_samples,
                    tvf_mode=args.sqr_return_mode,
                    tvf_n_step=args.sqr_return_n_step,
                    include_second_moment=True
                )
        else:
            returns = self.calculate_sampled_returns(
                value_sample_horizons=value_samples,
                required_horizons=horizon_samples,
                include_second_moment=False
            )
            returns_m2 = None

        horizon_samples = horizon_samples[None, None, :]
        horizon_samples = np.repeat(horizon_samples, N, axis=0)
        horizon_samples = np.repeat(horizon_samples, A, axis=1)
        return returns, returns_m2, horizon_samples

    @property
    def current_entropy_bonus(self):
        if args.entropy_scaling:
            # this is just so that entropy_bonus parameter can be left roughly the same when entropy scaling is enabled.
            typical_advantage_std = 0.05
            return args.entropy_bonus / ((self.advantage.std()/typical_advantage_std) + args.advantage_epsilon)
        else:
            # standard anneal...
            t = self.step / 10e6
            return args.entropy_bonus * 10 ** (args.eb_alpha * -math.cos(args.eb_theta*t*math.pi*2) + args.eb_beta * t)

    def train_value_heads(self, model_out, data):
        """
        Calculates loss for each value head, then returns their sum.
        """
        loss = torch.zeros([len(data["prev_state"])], dtype=torch.float32, device=self.model.device)
        for value_head in self.value_heads:
            value_prediction = model_out["{}_value".format(value_head)]
            returns = data["{}_returns".format(value_head)]
            value_loss = args.vf_coef * torch.square(value_prediction - returns)
            self.log.watch_mean("loss_v_" + value_head, value_loss.mean(), history_length=64, display_name="ls_v_" + value_head)
            loss = loss + value_loss
        return loss

    def generate_band_logs(self, label, x, y):

        if self.step == 0:
            # we need normalization constants to settle down a bit before starting, otherwise
            # the 1/C part will generate NANs.
            return

        # if x is too large reduce down.
        if len(x) > args.lfr_samples:
            sample = np.random.choice(range(len(x)), args.lfr_samples, replace=False)
            x = x[sample]
            y = y[sample]
        elif len(x) < args.lfr_samples:
            raise Exception(f"Too few samples for band estimate, needed {args.lfr_samples}, found {len(x)}")

        # convert to float
        x = self.model.prep_for_model(x)
        y = y.to(device=x.device)
        if args.lfr_normalize:
            x = self.model.perform_normalization(x) / 3.0 # this will get int to unit norm.

        bands = list(2 ** np.linspace(0, 20, 81))
        errors = compute_band_pass(x, y, bands)
        rms_scores = []
        for band, error in zip(bands, errors):
            rms = ((error**2).mean() ** 0.5).cpu()
            rms_scores.append(rms)
            log_band = np.log2(band)
            if abs(np.round(log_band) - log_band) < 0.01:
                # just log the powers of two.
                self.log.watch_mean(f"band_{label}_{log_band}", rms, history_length=10, display_width=0,
                                display_name=f"b{label[0]}_{log_band}")

        max_loss = (y.square().mean().detach().cpu()**0.5)
        self.log.watch_mean(f"band_{label}_max", max_loss, history_length=10, display_width=0)

        # find delta that splits error equally between high and low.
        target = max_loss * (2/4)
        mid_point_index = np.argmin(np.abs(target - np.asarray(rms_scores)))
        mid_point = bands[mid_point_index]
        self.log.watch_mean(f"band_{label}_midpoint", np.log2(mid_point), history_length=10, display_width=6,
                        display_name=f"mp_{label[0]}")
        target = max_loss * (3/4)
        high_point_index = np.argmin(np.abs(target - np.asarray(rms_scores)))
        high_point = bands[high_point_index]
        self.log.watch_mean(f"band_{label}_highpoint", np.log2(high_point), history_length=10, display_width=6,
                            display_name=f"hp_{label[0]}")

        target = max_loss * (1/4)
        low_point_index = np.argmin(np.abs(target - np.asarray(rms_scores)))
        low_point = bands[low_point_index]
        self.log.watch_mean(f"band_{label}_lowpoint", np.log2(low_point), history_length=10, display_width=6,
                            display_name=f"lp_{label[0]}")

    def train_policy_minibatch(self, data, loss_scale=1.0):

        def calc_entropy(x):
            return -(x.exp() * x).sum(axis=1)

        mini_batch_size = len(data["prev_state"])

        prev_states = data["prev_state"]
        actions = data["actions"].to(torch.long)
        old_log_pac = data["log_pac"]
        advantages = data["advantages"]

        model_out = self.model.forward(prev_states, output="policy")

        gain: torch.Tensor = 0

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

            if args.eb_clip >= 0:
                entropy_delta = (entropy - original_entropy).mean()
                entropy_clip_frac = torch.gt(entropy, original_entropy + args.eb_clip).float().mean()
                entropy_clipped = torch.min(entropy, original_entropy + args.eb_clip)
                self.log.watch_mean("entropy_clip_frac", entropy_clip_frac, display_width=8, display_name="e_clip")
                self.log.watch_stats("entropy_delta", entropy_delta.detach().cpu().numpy(), display_width=0,
                                     display_name="e_delta")
            else:
                entropy_clipped = entropy

            if args.use_uac:

                # the simplest solution.
                # sort states into high cost and low cost, and apply two different entropy bonus levels.
                entropy_cost = data["ext_value"] - data["uni_value"]
                threshold = torch.mean(entropy_cost) + torch.std(entropy_cost) * 1.0
                mask = torch.gt(entropy_cost, threshold)  # not normal, so masked will be few...
                inv_mask = torch.logical_not(mask)

                bonus_safe = self.current_entropy_bonus * 1.1
                bonus_risky = self.current_entropy_bonus / 10.0

                eb = bonus_safe * inv_mask + bonus_risky * mask
                self.log.watch_mean("eb", torch.mean(eb), display_width=8, display_precision=4)
                self.log.watch_stats("ec", entropy_cost, display_width=0)

                gain = gain + entropy_clipped * eb

            else:
                gain = gain + entropy_clipped * self.current_entropy_bonus

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

        self.log.watch_mean("loss_pg", loss_clip.mean(), history_length=64*args.policy_epochs, display_name=f"ls_pg", display_width=8)
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
    def episode_brb_fraction(self):
        # perctange of recent episodes that ended with button push.
        if len(self.episode_brb_buffer) == 0:
            return 0.0
        else:
            return np.mean(self.episode_brb_buffer)

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
        if args.auto_strategy == "episode_length":
            if len(self.episode_length_buffer) == 0:
                auto_horizon = 0
            else:
                auto_horizon = self.episode_length_mean + (2 * self.episode_length_std)
            return auto_horizon
        elif args.auto_strategy == "agent_age_slow":
            return (1/1000) * self.step # todo make this a parameter
        elif args.auto_strategy in ["sa_return", "sa_reward"]:
            return self.horizon_sa.neighbour
        else:
            raise ValueError(f"Invalid auto_strategy {args.auto_strategy}")

    @property
    def _auto_gamma(self):
        horizon = float(np.clip(self._auto_horizon, 10, float("inf")))
        return 1 - (1 / horizon)

    @property
    def current_horizon(self):
        if args.auto_horizon:
            min_horizon = max(128, args.tvf_horizon_samples, args.tvf_value_samples)
            return int(np.clip(self._auto_horizon*3, min_horizon, args.tvf_max_horizon))
        else:
            return int(args.tvf_max_horizon)

    @property
    def gamma(self):
        if args.auto_gamma in ["gamma", "both"]:
            return self._auto_gamma
        else:
            return args.gamma

    @property
    def tvf_gamma(self):
        if args.auto_gamma in ["tvf", "both"]:
            return self._auto_gamma
        else:
            return args.tvf_gamma

    @property
    def reward_scale(self):
        """ The amount rewards have been multiplied by. """
        if args.reward_normalization:
            norm_wrapper = wrappers.get_wrapper(self.vec_env, wrappers.VecNormalizeRewardWrapper)
            return args.reward_scale / norm_wrapper.std
        else:
            return args.reward_scale

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

        for epoch in range(args.rnd_epochs):
            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_rnd_minibatch,
                mini_batch_size=args.rnd_mini_batch_size,
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
            v.to(device=self.model.device, non_blocking=True)
            batch_data[k] = v

    def wants_are_update(self, network=None):
        """
        returns if this batch step wants an adaptive batch_size update or not.
        """
        if args.are_mode == "off":
            return False
        elif args.are_mode in ["on", "shadow"]:
            return True
        else:
            return args.are_mode == network

    def train_policy(self):

        # ----------------------------------------------------
        # policy phase

        start_time = clock.time()

        if args.policy_epochs == 0:
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
            batch_data["ext_returns"] = self.ext_returns.reshape([B])

        if args.use_uac:
            batch_data["ext_value"] = self.ext_value[:N].reshape([B])
            batch_data["uni_value"] = self.uni_value[:N].reshape([B])

        # sort out advantages
        advantages = self.advantage.reshape(B).copy()
        self.log.watch_stats("advantages_raw", advantages, display_width=0, history_length=1)

        if args.normalize_advantages != "off":
            # we should normalize at the mini_batch level, but it's so much easier to do this at the batch level.
            if args.normalize_advantages == "center":
                advantages = advantages - advantages.mean()
            elif args.normalize_advantages in ["norm", "True", "true", "on"]:
                advantages = (advantages - advantages.mean()) / (advantages.std() + args.advantage_epsilon)
            else:
                raise ValueError(f"Invalid normalize_advantages mode {args.normalize_advantages}")
            self.log.watch_stats("advantages_norm", advantages, display_width=0, history_length=1)

        if args.advantage_clipping is not None:
            advantages = np.clip(advantages, -args.advantage_clipping, +args.advantage_clipping)
            self.log.watch_stats("advantages_clipped", advantages, display_width=0, history_length=1)

        self.log.watch_stats("advantages", advantages, display_width=0, history_length=1)
        batch_data["advantages"] = advantages

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(B)
            batch_data["int_value"] = self.int_value[:N].reshape(B)

        policy_epochs = 0
        for epoch in range(args.policy_epochs):
            results = self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_policy_minibatch,
                mini_batch_size=args.policy_mini_batch_size,
                optimizer=self.policy_optimizer,
                label="policy",
                epoch=epoch,
                hooks={
                    'after_mini_batch': lambda x: x["outputs"][-1]["kl_approx"] > 1.5 * args.target_kl
                } if args.target_kl > 0 else {}
            )
            expected_mini_batches = (args.batch_size / args.policy_mini_batch_size)
            policy_epochs += results["mini_batches"] / expected_mini_batches
            if "did_break" in results:
                break

        self.log.watch_full("policy_epochs", policy_epochs,
                            display_width=9 if args.target_kl >= 0 else 0,
                            display_name="epochs_p"
                            )

        self.log.watch(f"time_train_policy", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_policy', display_precision=1)

    def train_value(self):

        # ----------------------------------------------------
        # value phase

        start_time = clock.time()

        if args.value_epochs == 0:
            return

        batch_data = {}
        B = args.batch_size
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([B, *state_shape])
        batch_data["ext_returns"] = self.ext_returns.reshape(B)

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(B)
            batch_data["int_value"] = self.int_value[:N].reshape(B) # only needed for clipped objective

        if (not args.use_tvf) and args.use_clipped_value_loss:
            # these are needed if we are using the clipped value objective...
            batch_data["ext_value"] = self.get_value_estimates(obs=self.prev_obs).reshape(B)

        if args.use_uac:
            log_policy = self.log_policy.reshape(B, *self.policy_shape)
            actions = self.actions.reshape(B)
            batch_data["log_pac"] = log_policy[range(B), actions]

        def get_value_estimate(prev_states, times):
            assert args.use_tvf, "replay_restraint require use_tvf=true (for the moment)"
            aux_features = package_aux_features(np.asarray([self.current_horizon]), times)
            return self.detached_batch_forward(
                prev_states, output="value", aux_features=aux_features
            )["tvf_value"][..., 0].detach().cpu().numpy()

        if args.use_tvf:
            # we do this once at the start, generate one returns estimate for each epoch on different horizons then
            # during epochs take a random mixture from this. This helps shuffle the horizons, and also makes sure that
            # we don't drift, as the updates will modify our model and change the value estimates.
            # it is possible that instead we should be updating our return estimates as we go though
            returns, returns_m2, horizons = self.generate_return_sample(force_first_and_last=True)
            batch_data["tvf_returns"] = returns.reshape([B, -1])
            batch_data["tvf_horizons"] = horizons.reshape([B, -1])
            batch_data["tvf_time"] = self.prev_time.reshape([B])

            if args.learn_second_moment:
                batch_data["tvf_returns_m2"] = returns_m2.reshape([B, -1])

        N, A = self.prev_time.shape
        B = N * A

        if args.value_resampling:
            # generate the return estimates all at once, should be quick
            ext_value_estimates = self.get_value_estimates(obs=self.all_obs)
        else:
            ext_value_estimates = None

        for value_epoch in range(args.value_epochs):

            if args.value_resampling:
                assert not args.learn_second_moment, "second moment not supported with value resampling."
                assert not args.use_tvf, "tvf not supported with value resampling."
                # each time we call this we will get different estimates.
                batch_data["ext_returns"] = self.generate_sampled_return_targets(ext_value_estimates).reshape(B)

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_value_minibatch,
                mini_batch_size=args.value_mini_batch_size,
                optimizer=self.value_optimizer,
                label="value",
                epoch=value_epoch,
                thinning=args.value_thinning,
            )

        self.log.watch(f"time_train_value", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_value', display_precision=1)

    def generate_aux_buffer(self):
        """
        Output will be [N, A, 16] of type float64
        """
        return ExperienceReplayBuffer.create_aux_buffer(
            shape=self.prev_time.shape,
            time=self.prev_time,
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

    def get_distil_batch(self, samples_wanted:int):
        """
        Creates a batch of data to train on during distil phase.
        Also generates any required targets.

        If no replay buffer is being used then uses the rollout data instead.

        @samples_wanted: The number of samples requested. Might be smaller if the replay buffer is too small, or
            has not seen enough data yet.

        """

        obs, distil_aux = self.get_replay_sample(samples_wanted)
        time = distil_aux[:, ExperienceReplayBuffer.AUX_TIME].astype(np.float32)

        batch_data = {}
        B, *state_shape = obs.shape

        batch_data["prev_state"] = obs

        if args.use_tvf:
            # this might not be needed anymore... ?
            horizons = self.generate_horizon_sample(
                self.current_horizon, args.tvf_horizon_samples,
                args.tvf_horizon_distribution
            )
            H = len(horizons)
            batch_data["tvf_horizons"] = expand_to_na(1, B, horizons).reshape([B, H])
            batch_data["tvf_time"] = time.reshape([B])
            aux_features = package_aux_features(horizons, time)
        else:
            aux_features = None

        assert not args.use_tvf, "tvf not working with distil yet..."

        # if args.use_tvf and not args.tvf_force_ext_value_distil:
        #     # maybe drop this and just use aux features?
        #     horizons = self.generate_horizon_sample(self.current_horizon, args.tvf_horizon_samples,
        #                                             args.tvf_horizon_distribution)
        #     H = len(horizons)
        #     output = self.get_value_estimates(
        #         obs=distil_obs[np.newaxis], # change from [B,*obs_shape] to [1,B,*obs_shape]
        #         time=distil_time[np.newaxis],
        #         horizons=horizons,
        #         include_model_out=args.use_intrinsic_rewards,
        #     )
        #     batch_data["tvf_horizons"] = expand_to_na(1, B, horizons).reshape([B, H])
        #     batch_data["tvf_time"] = distil_time.reshape([B])
        # else:
        #     output = self.get_value_estimates(
        #         obs=distil_obs[np.newaxis], time=distil_time[np.newaxis],
        #         include_model_out=args.use_intrinsic_rewards,
        #     )
        #

        # if args.use_intrinsic_rewards:
        #     target_values, model_out = output
        #     target_int_values = model_out["int_value"].cpu().numpy()
        #     batch_data["int_value_targets"] = target_int_values.reshape([B])
        # else:
        #     target_values = output

        # forward through model to get targets from model
        model_out = self.detached_batch_forward(
            obs=obs,
            aux_features=aux_features,
            output="full",
            include_features=args.distil_mode == "features"
        )

        batch_data["distil_targets"] = model_out['value_'+self.get_distil_target_name()]

        # get old policy
        batch_data["old_raw_policy"] = model_out["policy_raw_policy"].detach().cpu().numpy()

        return batch_data

    def get_adaptive_mbs(self, noise_scale: float, r=50):
        """
        Returns an appropriate mini-batch size based on the cbs estimate.
        """

        # todo: factor in the number of epochs

        min_size = 128  # partly for performance, partly for safety
        max_size = args.batch_size
        if noise_scale < 0:
            # this happens quite a bit...
            noise_scale = 0
        target_batch_size = math.sqrt(r*noise_scale)
        clipped_target_batch_size = np.clip(target_batch_size, min_size, max_size)
        quantised_target_batch_size = 2 ** round(math.log2(clipped_target_batch_size))
        return quantised_target_batch_size

    @property
    def lambda_value(self):
        if args.are_mode in ["on"]:
            return 1 - (1 / self.are_stats["value_h"])
        else:
            return args.td_lambda

    @property
    def lambda_policy(self):
        if args.are_mode in ["on", "policy"]:
            return 1 - (1 / self.are_stats["policy_h"])
        else:
            return args.gae_lambda

    def train_distil(self):

        # ----------------------------------------------------
        # distil phase

        start_time = clock.time()

        if args.distil_epochs == 0:
            return

        batch_data = self.get_distil_batch(args.distil_batch_size)

        for distil_epoch in range(args.distil_epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_distil_minibatch,
                mini_batch_size=args.distil_mini_batch_size,
                optimizer=self.distil_optimizer,
                label="distil",
                epoch=distil_epoch,
            )

        self.log.watch(f"time_train_distil", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_distil', display_precision=1)

    def train_aux(self):

        # ----------------------------------------------------
        # aux phase
        # borrows a lot of hyperparameters from distil

        start_time = clock.time()

        if args.aux_epochs == 0:
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

        for aux_epoch in range(args.aux_epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_aux_minibatch,
                mini_batch_size=args.distil_mini_batch_size,
                optimizer=self.aux_optimizer,
                epoch=aux_epoch,
                label="aux",
            )

        self.log.watch(f"time_train_aux", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_aux', display_precision=1)

    def train(self):

        if args.disable_logging:
            self.log.mode = self.log.LM_MUTE

        self.model.eval()

        self.update_learning_rates()

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
            if args.distil_epochs > 0 and self.batch_counter % args.distil_period == 0 and \
                    self.step > args.distil_delay:
                self.train_distil()

        if args.aux_epochs > 0 and (args.aux_period == 0 or self.batch_counter % args.aux_period == args.aux_period-1):
            self.train_aux()

        if args.use_rnd:
            self.train_rnd()

        # adaptive return estimation
        # takes time for noise scale to warm up...
        if args.are_mode != "off":
            # note: due to noise, ratios can be negative early on in training.
            if 'policy_ratio' in self.are_stats and self.are_stats['policy_ratio'] > 0:
                rho_a = (self.are_stats['policy_ratio'] ** 0.5) / args.are_target_p
                if self.step >= args.are_warmup and args.are_mode in ['on', 'policy']:
                    if rho_a > 1 + args.are_epsilon:
                        self.are_stats["policy_h"] = np.clip(self.are_stats["policy_h"] * args.are_alpha, args.are_min_h, args.are_max_h)
                    if rho_a < 1 - args.are_epsilon:
                        self.are_stats["policy_h"] = np.clip(self.are_stats["policy_h"] / args.are_alpha, args.are_min_h, args.are_max_h)
                self.log.watch_mean("are_pr", rho_a)
                self.log.watch_mean("are_ph", self.are_stats["policy_h"])

            if 'value_ratio' in self.are_stats and self.are_stats['value_ratio'] > 0:
                rho_v = (self.are_stats['value_ratio'] ** 0.5) / args.are_target_v
                if self.step >= args.are_warmup and args.are_mode in ['on']:
                    if rho_v > 1 + args.are_epsilon:
                        self.are_stats["value_h"] = np.clip(self.are_stats["value_h"] * args.are_alpha, args.are_min_h, args.are_max_h)
                    if rho_v < 1 - args.are_epsilon:
                        self.are_stats["value_h"] = np.clip(self.are_stats["value_h"] / args.are_alpha, args.are_min_h, args.are_max_h)
                self.log.watch_mean("are_vr", rho_v)
                self.log.watch_mean("are_vh", self.are_stats["value_h"])

        self.log.watch("lambda_policy", self.lambda_policy)
        self.log.watch("lambda_value", self.lambda_value)

        self.batch_counter += 1

    def train_batch(
            self,
            batch_data,
            mini_batch_func,
            mini_batch_size,
            optimizer: torch.optim.Optimizer,
            label,
            epoch=None,
            hooks: Union[dict, None] = None,
            thinning:float = 1.0,
        ) -> dict:
        """
        Trains agent on current batch of experience

        Thinning: uses this proportion of the batch_data.

        Returns context with
            'mini_batches' number of mini_batches completed
            'outputs' output from each mini_batch update
            'did_break'=True (only if training terminated early)
        """

        start_time = clock.time()

        if args.upload_batch:
            self.upload_batch(batch_data)

        if epoch == 0 and self.wants_are_update(label):
            self.estimate_noise_level(batch_data, mini_batch_func, optimizer, label)

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

        mini_batches = batch_size // mini_batch_size
        micro_batch_size = min(args.max_micro_batch_size, mini_batch_size)
        micro_batches = mini_batch_size // micro_batch_size

        ordering = list(range(batch_size))
        np.random.shuffle(ordering)

        micro_batch_counter = 0
        outputs = []

        context = {}

        if args.log_frequency_response:
            assert not args.use_compression, 'compression not supported with DLP'
            assert args.lfr_samples % (micro_batches * mini_batches) == 0, f"lrf_samples must be a multiple of {(micro_batches * mini_batches)}"
            logging_sample_per_microbatch = int(args.lfr_samples / (micro_batches * mini_batches))
            log_x = torch.zeros([args.lfr_samples, *self.state_shape], dtype=torch.uint8)
            log_y = torch.zeros([args.lfr_samples], dtype=torch.float32)
        else:
            logging_sample_per_microbatch = 0
            log_x = None
            log_y = None

        for j in range(mini_batches):

            optimizer.zero_grad(set_to_none=True)

            for k in range(micro_batches):
                # put together a micro_batch.
                batch_start = micro_batch_counter * micro_batch_size
                batch_end = (micro_batch_counter + 1) * micro_batch_size
                sample = ordering[batch_start:batch_end]
                micro_batch_counter += 1

                # context for the minibatch.
                micro_batch_data = {}
                micro_batch_data['context'] = {
                    'epoch': j,
                    'micro_batch': micro_batch_counter-1,
                }

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
                outputs.append(result)

                if args.log_frequency_response:
                    log_sample = np.random.choice(range(micro_batch_size), logging_sample_per_microbatch, replace=False)
                    s = slice((micro_batch_counter-1) * logging_sample_per_microbatch, micro_batch_counter * logging_sample_per_microbatch)
                    log_x[s] = micro_batch_data['prev_state'][log_sample]
                    log_y[s] = result['losses'][log_sample]

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

        # apply loss bandpass logging needed.
        if args.log_frequency_response:
            self.generate_band_logs(label, log_x, log_y)

        time_per_example = (clock.time() - start_time) / batch_size * 1000

        self.log.watch_mean(f"time_train_{label}_bms", time_per_example, display_width=0, display_name=f"t_{label}")

        return context


def get_rediscounted_value_estimate(
        values: Union[np.ndarray, torch.Tensor],
        old_gamma: float,
        new_gamma: float, horizons
):
    """
    Returns rediscounted return at horizon H

    values: float tensor of shape [B, H]
    returns float tensor of shape [B]
    """

    B, H = values.shape

    if old_gamma == new_gamma:
        return values[:, -1]

    if type(values) is np.ndarray:
        values = torch.from_numpy(values)
        is_numpy = True
    else:
        is_numpy = False

    device = values.device
    prev = torch.zeros([B], dtype=torch.float32, device=device)
    discounted_reward_sum = torch.zeros([B], dtype=torch.float32, device=device)
    for i, h in enumerate(horizons):
        reward = (values[:, i] - prev) / (old_gamma ** h)
        prev = values[:, i]
        discounted_reward_sum += reward * (new_gamma ** h)

    return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum

def package_aux_features(horizons: Union[np.ndarray, torch.Tensor], time: Union[np.ndarray, torch.Tensor]):
    """
    Return aux features for given horizons and time fraction.
    Output is [B, H, 2] where final dim is (horizon, time)

    Horizons should be floats from 0..tvf_max_horizons
    Time should be floats from 0..tvf_max_horizons

    horizons: [B, H] or [H]
    time: [B]

    """

    if len(horizons.shape) == 1:
        # duplicate out horizon list for each batch entry
        assert type(horizons) is np.ndarray, "Horizon duplication only implemented with numpy arrays for the moment sorry."
        horizons = np.repeat(horizons[None, :], len(time), axis=0)

    B, H = horizons.shape
    assert time.shape == (B,)

    # horizons might be int16, so cast it to float.
    if type(horizons) is np.ndarray:
        assert type(time) == np.ndarray
        horizons = horizons.astype(np.float32)
        aux_features = np.concatenate([
            horizons.reshape([B, H, 1]),
            np.repeat(time.reshape([B, 1, 1]), H, axis=1)
        ], axis=-1)
    elif type(horizons) is torch.Tensor:
        assert type(time) == torch.Tensor
        horizons = horizons.to(dtype=torch.float32)
        aux_features = torch.cat([
            horizons.reshape([B, H, 1]),
            torch.repeat_interleave(time.reshape([B, 1, 1]), H, dim=1)
        ], dim=-1)
    else:
        raise TypeError("Input must be of type np.ndarray or torch.Tensor")

    return aux_features


def _scale_function(x, method):
    if method == "default":
        return x / args.tvf_max_horizon
    elif method == "zero":
        return x*0
    elif method == "log":
        return (10+x).log10()-1
    elif method == "sqrt":
        return x.sqrt()
    elif method == "centered":
        # this will be roughly unit normal
        return ((x / args.tvf_max_horizon) - 0.5) * 3.0
    elif method == "wide":
        # this will be roughly 10x normal
        return ((x / args.tvf_max_horizon) - 0.5) * 30.0
    elif method == "wider":
        # this will be roughly 30x normal
        return ((x / args.tvf_max_horizon) - 0.5) * 100.0
    else:
        raise ValueError(f"Invalid scale mode {method}. Please use [zero|log|sqrt|centered|wide|wider]")

def horizon_scale_function(x):
    return _scale_function(x, args.tvf_horizon_scale)

def time_scale_function(x):
    return _scale_function(x, args.tvf_time_scale)

def expand_to_na(n,a,x):
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


class RolloutBuffer():
    """ Saves a rollout """
    def __init__(self, runner:Runner, copy:bool = True):

        copy_fn = np.copy if copy else lambda x: x

        self.all_obs = copy_fn(runner.all_obs)
        self.all_time = copy_fn(runner.all_time)
        self.ext_rewards = copy_fn(runner.ext_rewards)
        self.terminals = copy_fn(runner.terminals)
        self.tvf_gamma = runner.tvf_gamma
        self.gamma = runner.gamma
        self.current_horizon = runner.current_horizon

def make_env(env_type, env_id, **kwargs):
    if env_type == "atari":
        make_fn = atari.make
    elif env_type == "mujoco":
        make_fn = mujoco.make
    elif env_type == "procgen":
        make_fn = procgen.make
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
