import os
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import json
import cv2
import pickle
import gzip
from collections import defaultdict
from typing import Union
import bisect
import math

from .logger import Logger
from . import utils, atari, hybridVecEnv, wrappers, models
from .config import args


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
    details["fps"] = log["fps"]
    frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
    details["eta"] = frames_remaining / details["fps"]
    details["host"] = args.hostname
    details["last_modified"] = time.time()
    with open(os.path.join(args.log_folder, "progress.txt"), "w") as f:
        json.dump(details, f, indent=4)


def calculate_returns(rewards, dones, final_value_estimate, gamma):
    """
    Calculates returns given a batch of rewards, dones, and a final value estimate.
    Input is vectorized so it can calculate returns for multiple agents at once.
    :param rewards: nd array of dims [N,A]
    :param dones:   nd array of dims [N,A] where 1 = done and 0 = not done.
    :param final_value_estimate: nd array [A] containing value estimate of final state after last action.
    :param gamma:   discount rate.
    :return:
    """

    N, A = rewards.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_value_estimate

    if type(gamma) is float:
        gamma = np.ones([N, A], dtype=np.float32) * gamma

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma[i] * (1.0 - dones[i])

    return returns


def calculate_gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma: float, lamb=1.0,
                  normalize=False):
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
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        final_value_estimates: np.ndarray,
        gamma,
        lamb: float = 0.95):
    N, A, H = values.shape

    advantages = np.zeros([N, A], dtype=np.float32)

    # note: this webpage helped a lot with writing this...
    # https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html

    values = np.concatenate([values, final_value_estimates[None, :, :]], axis=0)

    # note, we could calculate advantage for all horizons if we want.

    for t in range(N):
        h = H - 1
        total_weight = np.zeros([A], dtype=np.float32)
        current_weight = np.ones([A], dtype=np.float32) * (1 - lamb)
        discount = np.ones([A], dtype=np.float32)
        advantage = np.zeros([A], dtype=np.float32)
        weighted_estimate = np.zeros([A], dtype=np.float32)
        # for the moment limit n_step to 10 for performance reasons (slow python, slow algorithm)
        # I have a NH version coming (instead of this NNH version)
        # if lamb is 0 we just do TD(0), which is much faster.
        advantage -= values[t, :, h]
        for n in range(1, 60):  # 60 should be enough for lamb=0.95
            # here we calculate the n_step estimate for V(s_t, h) (i.e. G(n))
            if t + n - 1 >= N:
                # we reached the end, so best we can do is stop here and use the estimates we have already
                # created
                weighted_estimate += current_weight * advantage
                total_weight += current_weight
                current_weight *= lamb
                continue
            this_reward = rewards[t + n - 1]
            advantage += discount * this_reward

            terminals = (1 - dones[t + n - 1])
            discount *= terminals * gamma

            bootstrap_estimate = discount * values[t + n, :, h - n] if (h - n) > 0 else 0
            weighted_estimate += current_weight * (advantage + bootstrap_estimate)
            total_weight += current_weight
            # note, I'm not sure if we should multiply by (1-dones[t+n]) here.
            # I think it's best not to, otherwise short n_steps get more weight close to a terminal...
            # actually maybe this is right?
            # actually terminal should get all remaining weight.
            current_weight *= lamb

        advantages[t, :] = weighted_estimate / total_weight

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
            # n_step is longer that horizon required
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


def _interpolate(horizons, values, target_horizon):
    """
    Returns linearly interpolated value from source_values

    horizons: sorted ndarray of shape[K] of horizons, must be in *strictly* ascending order
    values: ndarray of shape [*shape, K] where values[...,h] corresponds to horizon horizons[h]
    target_horizon: the horizon we would like to know the interpolated value of

    """

    if target_horizon <= 0:
        # by definition value of a 0 horizon is 0.
        return values[..., 0] * 0

    index = bisect.bisect_left(horizons, target_horizon)
    if index == 0:
        return values[..., 0]
    if index == len(horizons):
        return values[..., -1]
    value_pre = values[..., index - 1]
    value_post = values[..., index]
    factor = (target_horizon - horizons[index - 1]) / (horizons[index] - horizons[index - 1])
    return value_pre * (1 - factor) + value_post * factor


class Runner():

    def __init__(self, model: models.TVFModel, log, name="agent"):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model

        self.policy_optimizer = torch.optim.Adam(model.policy_net.parameters(), lr=args.policy_lr, eps=args.adam_epsilon)
        self.value_optimizer = torch.optim.Adam(model.value_net.parameters(), lr=args.value_lr, eps=args.adam_epsilon)

        if args.use_rnd:
            self.rnd_optimizer = torch.optim.Adam(model.prediction_net.parameters(), lr=args.value_lr, eps=args.adam_epsilon)

        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents

        self.state_shape = model.input_dims
        self.rnn_state_shape = [2, 512]  # records h and c for LSTM units.
        self.policy_shape = [model.actions]

        self.batch_counter = 0

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.obs = np.zeros([A, *self.state_shape], dtype=np.uint8)
        self.time = np.zeros([A], dtype=np.float32)
        # includes final state as well, which is needed for final value estimate
        self.all_obs = np.zeros([N + 1, A, *self.state_shape], dtype=np.uint8)
        self.all_time = np.zeros([N + 1, A], dtype=np.float32)
        self.actions = np.zeros([N, A], dtype=np.int64)
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool)  # indicates prev_state was a terminal state.

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)
        self.int_value = np.zeros([N, A], dtype=np.float32)

        # returns generation
        self.ext_returns = np.zeros([N, A], dtype=np.float32)
        self.int_returns_raw = np.zeros([N, A], dtype=np.float32)
        self.advantage = np.zeros([N, A], dtype=np.float32)

        self.int_final_value_estimate = np.zeros([A], dtype=np.float32)

        self.intrinsic_returns_rms = utils.RunningMeanStd(shape=())
        self.ems_norm = np.zeros([args.agents])

        # outputs tensors when clip loss is very high.
        self.log_high_grad_norm = True

        self.step = 0
        self.game_crashes = 0
        self.ep_count = 0

        #  these horizons will always be generated and their scores logged.
        self.tvf_debug_horizons = [h for h in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000] if
                                   h <= args.tvf_max_horizon]
        if args.tvf_max_horizon not in self.tvf_debug_horizons:
            self.tvf_debug_horizons.append(args.tvf_max_horizon)
        self.tvf_debug_horizons.sort()

    def create_envs(self):
        """ Creates environments for runner"""
        env_fns = [lambda: atari.make() for _ in range(args.agents)]

        if args.sync_envs:
            self.vec_env = gym.vector.SyncVectorEnv(env_fns)
        else:
            self.vec_env = hybridVecEnv.HybridAsyncVectorEnv(
                env_fns,
                copy=False,
                shared_memory=True,
                max_cpus=args.workers,
                verbose=True
            )

        if args.reward_normalization:
            self.vec_env = wrappers.VecNormalizeRewardWrapper(
                self.vec_env,
                initial_state=atari.get_env_state("returns_norm_state"),
                gamma=args.gamma
            )

        self.log.important("Generated {} agents ({}) using {} ({}) model.".
                           format(args.agents, "async" if not args.sync_envs else "sync", self.model.name,
                                  self.model.dtype))

    def save_checkpoint(self, filename, step):

        data = {
            'step': step,
            'ep_count': self.ep_count,
            'model_state_dict': self.model.state_dict(),
            'logs': self.log,
            'env_state': atari.ENV_STATE,
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict()
        }

        if args.use_rnd:
            data['rnd_optimizer_state_dict'] = self.rnd_optimizer.state_dict()

        if args.use_intrinsic_rewards:
            data['ems_norm'] = self.ems_norm
            data['intrinsic_returns_rms'] = self.intrinsic_returns_rms

        if args.normalize_observations:
            data["observation_norm_state"] = self.model.obs_rms.save_state()

        torch.save(data, filename)

    def get_checkpoints(self, path):
        """ Returns list of (epoch, filename) for each checkpoint in given folder. """
        results = []
        if not os.path.exists(path):
            return []
        for f in os.listdir(path):
            if f.startswith("checkpoint") and f.endswith(".pt"):
                epoch = int(f[11:14])
                results.append((epoch, f))
        results.sort(reverse=True)
        return results

    def load_checkpoint(self, checkpoint_path):
        """ Restores model from checkpoint. Returns current env_step"""
        checkpoint = torch.load(checkpoint_path, map_location=args.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        if "rnd_optimizer_state_dict" in checkpoint:
            self.rnd_optimizer = checkpoint['rnd_optimizer_state_dict']

        step = checkpoint['step']
        self.log = checkpoint['logs']
        self.step = step
        self.ep_count = checkpoint.get('ep_count', 0)

        if args.use_intrinsic_rewards:
            self.ems_norm = checkpoint['ems_norm']
            self.intrinsic_returns_rms = checkpoint['intrinsic_returns_rms']

        atari.ENV_STATE = checkpoint['env_state']

        if args.normalize_observations:
            self.model.obs_rms.restore_state(checkpoint["observation_norm_state"])

        return step

    def reset(self):

        assert self.vec_env is not None, "Please call create_envs first."

        # initialize agent
        self.obs = self.vec_env.reset()
        self.episode_score *= 0
        self.episode_len *= 0
        self.step = 0
        self.game_crashes = 0
        self.batch_counter = 0

    def run_random_agent(self, iterations):
        self.log.info("Warming up model with random agent...")

        # collect experience
        self.reset()

        for iteration in range(iterations):
            self.generate_rollout(is_warmup=True)

    def forward(self, obs, aux_features=None, max_batch_size=None, **kwargs):
        """ Forward states through model, returns output, which is a dictionary containing
            "log_policy" etc.
        """

        max_batch_size = max_batch_size or args.max_micro_batch_size

        B, *state_shape = obs.shape
        assert tuple(state_shape) == tuple(self.state_shape)

        # break large forwards into batches (note: would be better to just run multiple max_size batches + one last
        # small one than to subdivide)
        if B > max_batch_size:

            mid_point = B // 2

            if aux_features is not None:
                a = self.forward(
                    obs[:mid_point],
                    aux_features=aux_features[:mid_point],
                    max_batch_size=max_batch_size,
                    **kwargs
                )
                b = self.forward(
                    obs[mid_point:],
                    aux_features=aux_features[mid_point:],
                    max_batch_size=max_batch_size,
                    **kwargs
                )
            else:
                a = self.forward(obs[:mid_point], max_batch_size=max_batch_size, **kwargs)
                b = self.forward(obs[mid_point:], max_batch_size=max_batch_size, **kwargs)
            result = {}
            for k in a.keys():
                result[k] = torch.cat(tensors=[a[k], b[k]], dim=0)
            return result
        else:
            return self.model.forward(obs, aux_features=aux_features, **kwargs)

    def _calculate_n_step_sampled_returns(
            self,
            n_step:int,
            gamma:float,
            rewards: np.ndarray,
            dones: np.ndarray,
            required_horizons: np.ndarray,
            value_sample_horizons: np.ndarray,
            value_samples: np.ndarray,
        ):
        """
        This is a fancy n-step sampled reutrns calculation

        n_step: n-step to use in calculation
        gamma: discount to use
        reward: nd array of dims [N, A]
        dones: nd array of dims [N, A]
        requred_horizons: nd array of dims [K]


        If n_step td_lambda is negative it is taken as
        """

        N, A = rewards.shape
        H = args.tvf_max_horizon
        K = len(required_horizons)

        # this allows us to map to our 'sparse' returns table
        h_lookup = {}
        for index, h in enumerate(required_horizons):
            h_lookup[h] = index

        returns = np.zeros([N, A, K], dtype=np.float32)

        # generate return estimates using n-step returns
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
                if n in h_lookup:
                    returns[t, :, h_lookup[n]] = reward_sum

            for h in required_horizons:
                if h < steps_made + 1:
                    # these are just the accumulated sums and don't need horizon bootstrapping
                    continue
                interpolated_value = _interpolate(value_sample_horizons, value_samples[t + steps_made], h - steps_made)
                returns[t, :, h_lookup[h]] = reward_sum + interpolated_value * discount

        return returns

    def _calculate_lambda_sampled_returns(
            self,
            dims:tuple,
            td_lambda: float,
            n_step_func: callable,
    ):
        """
        Calculate td_lambda returns using sampling
        """

        N, A, K = dims

        if td_lambda == 0:
            return n_step_func(1)

        if td_lambda == 1:
            return n_step_func(N)

        # first calculate the weight for each return
        current_weight = (1-td_lambda)
        weights = np.zeros([N], dtype=np.float32)
        for n in range(N):
            weights[n] = current_weight
            current_weight *= td_lambda
        # use last n-step for remaining weight
        weights[-1] = 1.0 - np.sum(weights)

        if args.tvf_lambda_samples == -1:
            # if we have disabled sampling just generate them all and weight them accordingly
            returns = np.zeros([N, A, K], dtype=np.float32)
            for n, weight in zip(range(N), weights):
                returns += weight * n_step_func(n+1)
            return returns
        else:
            # otherwise sample randomly from n_steps with replacement, but always include last horizon
            # as this will reduce variance a lot for small window lengths where most of the probability
            # mass is on the final step
            weights_missing_last = weights[:-1]
            weights_missing_last /= np.sum(weights_missing_last) # just due to rounding error..
            returns = weights[-1] * n_step_func(N)
            for _ in range(args.tvf_lambda_samples-1):
                sampled_n_step = np.random.choice(range(N-1), p=weights_missing_last) + 1
                returns += n_step_func(sampled_n_step) / (args.tvf_lambda_samples-1) * (1-weights[-1])
            return returns

    def calculate_sampled_returns(
            self,
            value_sample_horizons: Union[list, np.ndarray],
            required_horizons: Union[list, np.ndarray, int],
            obs=None,
            time=None,
            rewards=None,
            dones=None,
    ):
        """
        Calculates and returns the (tvf_gamma discounted) return estimates for given rollout.

        prev_states: ndarray of dims [N+1, B, *state_shape] containing prev_states
        rewards: float32 ndarray of dims [N, B] containing reward at step n for agent b
        value_sample_horizons: int32 ndarray of dims [K] indicating horizons to generate value estimates at.
        required_horizons: int32 ndarray of dims [K] indicating the horizons for which we want a return estimate.
        """

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

        N, A, *state_shape = obs[:-1].shape

        assert obs.shape == (N + 1, A, *state_shape)
        assert rewards.shape == (N, A)
        assert dones.shape == (N, A)

        # step 1:
        # use our model to generate the value estimates required
        # for MC this is just an estimate at the end of the window
        horizons = value_sample_horizons[None, None, :]
        horizons = np.repeat(horizons, repeats=N + 1, axis=0)
        horizons = np.repeat(horizons, repeats=A, axis=1)
        horizons = horizons.reshape([(N + 1) * A, -1])
        time = time.reshape([(N + 1) * A])

        with torch.no_grad():
            model_out = self.forward(
                obs=obs.reshape([(N + 1) * A, *state_shape]),
                aux_features=self.package_aux_features(horizons, time),
                output="value",
            )
        value_samples = model_out["tvf_value"].reshape([(N + 1), A, len(value_sample_horizons)]).cpu().numpy()

        def plot_true_value(n, a):
            xs = np.arange(args.tvf_max_horizon+1)
            horizons = xs[None, None, :]

            with torch.no_grad():
                model_out = self.forward(
                    obs=obs[n, a].reshape([1, *state_shape]),
                    horizons=horizons.reshape([1, -1]),
                    output="value",
                )
            ys = model_out["tvf_value"].reshape([1, 1, args.tvf_max_horizon+1]).cpu().numpy()[0, 0]
            import matplotlib.pyplot as plt
            print(ys)
            plt.plot(xs, ys)
            plt.show()


        # stub show value curve
        def plot_debug_curve(n,a, interpolate_before_scale=True):
            xs = []
            ys = []
            print(value_sample_horizons)
            for h in range(args.tvf_max_horizon+1):
                xs.append(h)
                if interpolate_before_scale:
                    values = _interpolate(value_sample_horizons, value_samples[n, a], h)
                    values = self.model.value_net.apply_tvf_transform(values, h)
                else:
                    values = _interpolate(
                        value_sample_horizons,
                        self.model.value_net.apply_tvf_transform(value_samples[n, a], value_sample_horizons),
                        h
                    )

                ys.append(values)
            import matplotlib.pyplot as plt

            # show curve
            plt.plot(xs, ys)
            # show samples
            plt.scatter(value_sample_horizons, [ys[x] for x in value_sample_horizons], marker='x')
            plt.show()

        if args.tvf_lambda >= 0:

            n_step_func = lambda x: self._calculate_n_step_sampled_returns(
                    n_step=x,
                    gamma=args.tvf_gamma,
                    rewards=rewards,
                    dones=dones,
                    required_horizons=required_horizons,
                    value_sample_horizons=value_sample_horizons,
                    value_samples=value_samples,
                )

            return self._calculate_lambda_sampled_returns(
                dims=(N, A, required_horizons),
                td_lambda=args.tvf_lambda,
                n_step_func=n_step_func,
            )
        else:
            return self._calculate_n_step_sampled_returns(
                n_step=-int(args.tvf_lambda),
                gamma=args.tvf_gamma,
                rewards=rewards,
                dones=dones,
                required_horizons=required_horizons,
                value_sample_horizons=value_sample_horizons,
                value_samples=value_samples,
            )


    @torch.no_grad()
    def export_movie(self, filename, include_rollout=False, include_video=True, max_frames=60 * 60 * 15):
        """ Exports a movie of agent playing game.
            include_rollout: save a copy of the rollout (may as well include policy, actions, value etc)
        """

        scale = 2

        env = atari.make(monitor_video=True)
        _ = env.reset()
        action = 0
        state, reward, done, info = env.step(0)
        rendered_frame = info.get("monitor_obs", state)

        # work out our height
        first_frame = utils.compose_frame(state, rendered_frame)
        height, width, channels = first_frame.shape
        width = (width * scale) // 4 * 4  # make sure these are multiples of 4
        height = (height * scale) // 4 * 4

        # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
        if include_video:
            video_out = cv2.VideoWriter(filename + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height),
                                        isColor=True)
        else:
            video_out = None

        state = env.reset()

        frame_count = 0

        history = defaultdict(list)

        # play the game...
        while not done:

            additional_params = {}

            model_out = self.model.forward(state[np.newaxis], **additional_params)

            log_probs = model_out["log_policy"][0].detach().cpu().numpy()
            action = utils.sample_action_from_logp(log_probs)

            if include_rollout:
                history["logprobs"].append(log_probs)
                history["actions"].append(action)
                history["states"].append(state)

            state, reward, done, info = env.step(action)

            channels = info.get("channels", None)
            rendered_frame = info.get("monitor_obs", state)

            agent_layers = state.copy()

            frame = utils.compose_frame(agent_layers, rendered_frame, channels)

            if frame.shape[0] != width or frame.shape[1] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            # show current state
            assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
                (width, height, 3), frame.shape)

            if video_out is not None:
                video_out.write(frame)

            frame_count += 1

            if frame_count >= max_frames:
                break

        if video_out is not None:
            video_out.release()

        if include_rollout:
            np_history = {}
            for k, v in history.items():
                np_history[k] = np.asarray(v)
            pickle.dump(np_history, gzip.open(filename + ".hst.gz", "wb", compresslevel=9))

    @property
    def tvf_requires_full_horizon_at_rollout(self):
        return args.tvf_gamma != args.gamma

    @torch.no_grad()
    def generate_rollout(self, is_warmup=False):

        assert self.vec_env is not None, "Please call create_envs first."

        max_h = self.current_max_horizon

        # calculate a good batch_size to use

        max_rollout_batch_size = args.max_micro_batch_size
        max_final_batch_size = args.max_micro_batch_size

        if args.use_tvf:
            horizon_factor = np.clip(int(max_h / 128), 1, args.max_micro_batch_size)
            if self.tvf_requires_full_horizon_at_rollout:
                max_rollout_batch_size //= horizon_factor
            max_final_batch_size //= horizon_factor

        infos = None

        for t in range(self.N):

            prev_obs = self.obs.copy()
            prev_time = self.time.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.forward(self.obs, output="policy")

            log_policy = model_out["log_policy"].cpu().numpy()

            # during warm-up we simply collect experience through a uniform random policy.
            if is_warmup:
                actions = np.random.randint(0, self.model.actions, size=[self.A], dtype=np.int32)
            else:
                # sample actions and run through environment.
                actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)

            self.obs, ext_rewards, dones, infos = self.vec_env.step(actions)

            # time fraction
            self.time = np.asarray([info["time_frac"] for info in infos])

            # per step reward
            ext_rewards += args.per_step_reward

            # work out our intrinsic rewards
            if args.use_intrinsic_rewards:
                value_int = model_out["int_value"].detach().cpu().numpy()

                int_rewards = np.zeros_like(ext_rewards)

                if args.use_rnd:
                    if is_warmup:
                        # in random mode just update the normalization constants
                        self.model.perform_normalization(self.obs)
                    else:
                        # reward is prediction error on state we land inn.
                        loss_rnd = self.model.prediction_error(self.obs).detach().cpu().numpy()
                        int_rewards += loss_rnd
                else:
                    assert False, "No intrinsic rewards set."

                self.int_rewards[t] = int_rewards
                self.int_value[t] = value_int

            # save raw rewards for monitoring the agents progress
            raw_rewards = np.asarray([info.get("raw_reward", ext_rewards) for reward, info in zip(ext_rewards, infos)],
                                     dtype=np.float32)

            self.episode_score += raw_rewards
            self.episode_len += 1

            for i, (done, info) in enumerate(zip(dones, infos)):
                if done:
                    # reset is handled automatically by vectorized environments
                    # so just need to keep track of book-keeping
                    if not is_warmup:
                        self.ep_count += 1
                        self.log.watch_full("ep_score", info["ep_score"])
                        self.log.watch_full("ep_length", info["ep_length"])
                        self.log.watch_mean("ep_count", self.ep_count, history_length=1)
                        if "game_freeze" in infos[i]:
                            self.game_crashes += 1
                    self.episode_score[i] = 0
                    self.episode_len[i] = 0

            self.all_obs[t] = prev_obs
            self.all_time[t] = prev_time
            self.actions[t] = actions

            self.ext_rewards[t] = ext_rewards
            self.log_policy[t] = log_policy
            self.terminals[t] = dones

        # save the last state
        self.all_obs[-1] = self.obs
        self.all_time[-1] = self.time

        #  save a copy of the observation normalization statistics.
        if args.normalize_observations:
            atari.ENV_STATE["observation_norm_state"] = self.model.obs_rms.save_state()

        # save a copy of the reward normalizion statistics
        # returns_norm_state is always put in info 0.
        if infos is not None and "returns_norm_state" in infos[0]:
            atari.ENV_STATE["returns_norm_state"] = infos[0]["returns_norm_state"]
            norm_mean, norm_var, norm_count = infos[0]["returns_norm_state"]
            self.log.watch("returns_norm_mu", norm_mean, display_width=0)
            self.log.watch("returns_norm_std", norm_var ** 0.5, display_width=0)

    def package_aux_features(self, horizons: np.ndarray, time: np.ndarray):
        """
        Return aux features for given horizons and time fraction.

        horizons: [B, H]
        time: [B]

        """

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


    @torch.no_grad()
    def get_value_estimates(self, obs: np.ndarray, time: np.ndarray, horizons: Union[None, np.ndarray, int] = None):
        """
        Returns value estimates for each given observation
        If horizons is none max_horizon is used.
        obs: np array of dims [N, A, *state_shape]
        time: np array of dims [N, A]
        horizons: ndarray of dims [K]

        returns: ndarray of dims [N, A, K] of horizons was array else [N, A]
        """

        N, A, *state_shape = obs.shape

        if args.use_tvf:
            horizons = horizons if horizons is not None else args.tvf_max_horizon

            if type(horizons) in [int, float]:
                scalar_output = True
                horizons = np.asarray([horizons])
            else:
                scalar_output = False

            horizons = np.repeat(horizons[None, :], N * A, axis=0)
            time = time.reshape(N*A)

            model_out = self.forward(
                obs=obs.reshape([N * A, *state_shape]),
                output="value",
                aux_features=self.package_aux_features(horizons, time),
            )

            values = model_out["tvf_value"]

            if scalar_output:
                return values.reshape([N, A]).cpu().numpy()
            else:
                return values.reshape([N, A, horizons.shape[-1]]).cpu().numpy()
        else:
            raise Exception("Not implemented yet")

    def log_value_quality(self):
        """
        Writes value quality stats to log
        """

        if args.use_tvf:

            # first we generate the value estimates, then we calculate the returns required for each debug horizon
            # because we use sampling it is not guaranteed that these horizons will be included so we need to
            # recalculate everything

            agent_sample_count = int(np.clip(args.agents // 4, 4, float('inf')))
            agent_filter = np.random.choice(args.agents, agent_sample_count, replace=False)

            values = self.get_value_estimates(
                obs=self.prev_obs[:, agent_filter],
                time=self.prev_time[:, agent_filter],
                horizons=np.asarray(self.tvf_debug_horizons)
            )

            value_samples = self.generate_horizon_sample(
                args.tvf_max_horizon,
                args.tvf_value_samples,
                distribution=args.tvf_value_distribution
            )

            targets = self.calculate_sampled_returns(
                value_sample_horizons=value_samples,
                required_horizons=self.tvf_debug_horizons,
                obs=self.all_obs[:, agent_filter],
                time=self.all_time[:, agent_filter],
                rewards=self.ext_rewards[:, agent_filter],
                dones=self.terminals[:, agent_filter],
            )

            for index, h in enumerate(self.tvf_debug_horizons):
                value = values[:, :, index].reshape(-1)
                target = targets[:, :, index].reshape(-1)
                self.log.watch_mean(
                    f"ev_{h:04d}",
                    utils.explained_variance(value, target),
                    display_width=8 if h < 100 or h == args.tvf_max_horizon else 0
                )
                # raw is RMS on unscaled error
                self.log.watch_mean(f"raw_{h:04d}", np.mean(np.square(self.reward_scale * (value - target)) ** 0.5),
                                    display_width=0)
                self.log.watch_mean(f"mse_{h:04d}", np.mean(np.square(value - target)), display_width=0)

            # do ev over random horizons
            random_horizon_samples = np.random.choice(args.tvf_max_horizon+1, [100], replace=True)

            targets = self.calculate_sampled_returns(
                value_sample_horizons=value_samples,
                required_horizons=random_horizon_samples,
                obs=self.all_obs[:, agent_filter],
                time=self.all_time[:, agent_filter],
                rewards=self.ext_rewards[:, agent_filter],
                dones=self.terminals[:, agent_filter],
            )

            values = self.get_value_estimates(
                obs=self.prev_obs[:, agent_filter],
                time=self.prev_time[:, agent_filter],
                horizons=random_horizon_samples,
            )

            value = values.reshape(-1)
            target = targets.reshape(-1)

            self.log.watch_mean(f"ev_av", utils.explained_variance(value, target), display_width=8)
            self.log.watch_mean(f"raw_av", np.mean(np.square(self.reward_scale * (value - target)) ** 0.5), display_width=0)
            self.log.watch_mean(f"mse_av", np.mean(np.square(value - target)), display_width=0)
        else:
            raise Exception("PPO not supported yet")
            # self.log.watch_mean("ev_ext", utils.explained_variance(self.ext_value.ravel(), self.ext_returns.ravel()))

    @property
    def prev_obs(self):
        """
        Returns prev_obs with size [N,A] (i.e. missing final state)
        """
        return self.all_obs[:-1]

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

    def calculate_returns(self):

        N, A, *state_shape = self.prev_obs.shape

        # 1. first we calculate the ext_value estimate

        if args.use_tvf:
            # if gamma's match we only need to generate the final horizon
            # if they don't we need to generate them all and rediscount
            if args.tvf_gamma == args.gamma:
                ext_value_estimates = self.get_value_estimates(obs=self.all_obs, time=self.all_time)
            else:
                # note: we could down sample the value estimates and adjust the gamma calculations if
                # this ends up being too slow..
                step_skip = math.ceil(args.tvf_max_horizon / 100)

                value_estimates = self.get_value_estimates(
                    obs=self.all_obs,
                    time=self.all_time,
                    # going backwards makes sure that final horizon is always included
                    horizons=np.asarray(range(args.tvf_max_horizon, 0, -step_skip))
                )
                # add h=0
                value_estimates = np.concatenate((value_estimates, np.zeros_like(value_estimates[..., 0:1])), axis=-1)
                value_estimates = value_estimates[..., ::-1].copy() # reverse order

                ext_value_estimates = get_rediscounted_value_estimate(
                    values=value_estimates.reshape([(N+1)*A, -1]),
                    old_gamma=args.tvf_gamma**step_skip,
                    new_gamma=args.gamma**step_skip
                ).reshape([(N+1), A])
        else:
            # in this case just generate ext value estimates from model
            raise Exception("Not Implemented Yet")
            # with torch.no_grad():
            #     ext_value_estimates = self.forward(
            #         obs=np.concatenate((self.prev_obs, self.obs), axis=0),
            #         output="value",
            #         max_batch_size=args.max_micro_batch_size
            #     )["ext_value"]

        self.ext_advantage = calculate_gae(
            self.ext_rewards,
            ext_value_estimates[:N],
            ext_value_estimates[N],
            self.terminals,
            args.gamma,
            args.gae_lambda
        )
        # calculate ext_returns for PPO targets
        self.ext_returns = self.ext_advantage + ext_value_estimates[:N]

        if args.use_intrinsic_rewards:
            # calculate the returns, but let returns propagate through terminal states.
            self.int_returns_raw = calculate_returns(
                self.int_rewards,
                args.intrinsic_reward_propagation * self.terminals,
                self.int_final_value_estimate,
                args.gamma_int
            )

            if args.normalize_intrinsic_rewards:

                # normalize returns using EMS
                for t in range(self.N):
                    self.ems_norm = 0.99 * self.ems_norm + self.int_rewards[t, :]
                    self.intrinsic_returns_rms.update(self.ems_norm.reshape(-1))

                # normalize the intrinsic rewards
                # we multiply by 0.4 otherwise the intrinsic returns sit around 1.0, and we want them to be more like 0.4,
                # which is approximately where normalized returns will sit.
                self.intrinsic_reward_norm_scale = (1e-5 + self.intrinsic_returns_rms.var ** 0.5)
                self.int_rewards = self.int_rewards / self.intrinsic_reward_norm_scale * 0.4
            else:
                self.intrinsic_reward_norm_scale = 1

            self.int_returns = calculate_returns(
                self.int_rewards,
                args.intrinsic_reward_propagation * self.terminals,
                self.int_final_value_estimate,
                args.gamma_int
            )

            self.int_advantage = calculate_gae(self.int_rewards, self.int_value, self.int_final_value_estimate, None,
                                               args.gamma_int)

        self.advantage = args.extrinsic_reward_scale * self.ext_advantage
        if args.use_intrinsic_rewards:
            self.advantage += args.intrinsic_reward_scale * self.int_advantage

        if args.normalize_observations:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        self.log.watch_mean("reward_scale", self.reward_scale, display_width=0)

        self.log.watch_mean("adv_mean", np.mean(self.advantage), display_width=0)
        self.log.watch_mean("adv_std", np.std(self.advantage), display_width=0)
        self.log.watch_mean("adv_max", np.max(self.advantage), display_width=0)
        self.log.watch_mean("adv_min", np.min(self.advantage), display_width=0)
        self.log.watch_mean("batch_reward_ext", np.mean(self.ext_rewards), display_name="rew_ext", display_width=0)
        self.log.watch_mean("batch_return_ext", np.mean(self.ext_returns), display_name="ret_ext")
        self.log.watch_mean("batch_return_ext_std", np.std(self.ext_returns), display_name="ret_ext_std",
                            display_width=0)
        # self.log.watch_mean("value_est_ext", np.mean(self.ext_value), display_name="est_v_ext", display_width=0)
        # self.log.watch_mean("value_est_ext_std", np.std(self.ext_value), display_name="est_v_ext_std", display_width=0)

        self.log.watch("game_crashes", self.game_crashes, display_width=0 if self.game_crashes == 0 else 8)

        self.log.watch("tvf_horizon", self.current_max_horizon)

        # this is a big slow as it recalculates values so it might be a good idea to only do it every so often
        # current time taken is 0.24 for 10k horizons
        if self.batch_counter % 2 == 0:
            self.log_value_quality()

        if args.use_intrinsic_rewards:
            self.log.watch_mean("batch_reward_int", np.mean(self.int_rewards), display_name="rew_int", display_width=0)
            self.log.watch_mean("batch_reward_int_std", np.std(self.int_rewards), display_name="rew_int_std",
                                display_width=0)
            self.log.watch_mean("batch_return_int", np.mean(self.int_returns), display_name="ret_int")
            self.log.watch_mean("batch_return_int_std", np.std(self.int_returns), display_name="ret_int_std",
                                display_width=0)
            self.log.watch_mean("batch_return_int_raw_mean", np.mean(self.int_returns_raw),
                                display_name="ret_int_raw_mu",
                                display_width=0)
            self.log.watch_mean("batch_return_int_raw_std", np.std(self.int_returns_raw),
                                display_name="ret_int_raw_std",
                                display_width=0)

            self.log.watch_mean("value_est_int", np.mean(self.int_value), display_name="est_v_int", display_width=0)
            self.log.watch_mean("value_est_int_std", np.std(self.int_value), display_name="est_v_int_std",
                                display_width=0)
            self.log.watch_mean("ev_int", utils.explained_variance(self.int_value.ravel(), self.int_returns.ravel()))
            if args.use_rnd:
                self.log.watch_mean("batch_reward_int_unnorm", np.mean(self.int_rewards), display_name="rew_int_unnorm",
                                    display_width=0, display_priority=-2)
                self.log.watch_mean("batch_reward_int_unnorm_std", np.std(self.int_rewards),
                                    display_name="rew_int_unnorm_std",
                                    display_width=0)

        if args.normalize_intrinsic_rewards:
            self.log.watch_mean("norm_scale_int", self.intrinsic_reward_norm_scale, display_width=0)

    def train_rnd_minibatch(self, data, zero_grad=True, apply_update=True, loss_scale=1.0):

        raise Exception("Not implemented yet")

        # mini_batch_size = len(data["prev_state"])
        #
        # # -------------------------------------------------------------------------
        # # Calculate loss_rnd
        # # -------------------------------------------------------------------------
        #
        # if args.use_rnd:
        #     # learn prediction slowly by only using some of the samples... otherwise it learns too quickly.
        #     predictor_proportion = np.clip(32 / args.agents, 0.01, 1)
        #     n = int(len(prev_states) * predictor_proportion)
        #     loss_rnd = -self.model.prediction_error(prev_states[:n]).mean()
        #     loss = loss + loss_rnd
        #
        #     self.log.watch_mean("loss_rnd", loss_rnd)
        #
        #     self.log.watch_mean("feat_mean", self.model.features_mean, display_width=0)
        #     self.log.watch_mean("feat_var", self.model.features_var, display_width=10)
        #     self.log.watch_mean("feat_max", self.model.features_max, display_width=10, display_precision=1)

    def optimizer_step(self, optimizer: torch.optim.Optimizer, label: str = "opt"):

        # get parameters
        parameters = []
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    parameters.append(p)

        if args.max_grad_norm is not None and args.max_grad_norm != 0:
            grad_norm = nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        else:
            # even if we don't clip the gradient we should at least log the norm. This is probably a bit slow though.
            # we could do this every 10th step, but it's important that a large grad_norm doesn't get missed.
            grad_norm = 0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

        self.log.watch_mean(f"grad_{label}", grad_norm)
        optimizer.step()
        return float(grad_norm)

    def train_value_minibatch(self, data, loss_scale=1.0):

        mini_batch_size = len(data["prev_state"])

        loss: torch.Tensor = torch.tensor(0, dtype=torch.float32, device=self.model.device)

        prev_states = data["prev_state"]

        # create additional args if needed
        kwargs = {}
        if args.use_tvf:
            # horizons are B, H, and times is B so we need to adjust them
            kwargs['aux_features'] = self.package_aux_features(
                data["tvf_horizons"],
                data["tvf_time"]
            )

        model_out = self.forward(prev_states, output="value", **kwargs)

        # -------------------------------------------------------------------------
        # Calculate loss_value_function_horizons
        # -------------------------------------------------------------------------

        if args.use_tvf:
            # targets "tvf_returns" are [B, K]
            # predictions "tvf_value" are [B, K]
            # predictions need to be generated... this could take a lot of time so just sample a few..
            targets = data["tvf_returns"]
            value_predictions = model_out["tvf_value"]

            if args.tvf_loss_weighting == "default":
                weighting = 1
            elif args.tvf_loss_weighting == "advanced":
                if args.tvf_lambda < 0:
                    # n step returns
                    effective_n_step = -args.tvf_lambda
                    # td updates
                elif args.tvf_lambda == 0:
                    effective_n_step = 1
                elif args.tvf_lambda == 1:
                    # MC style
                    effective_n_step = args.n_steps / 2
                else:
                    effective_n_step = min(1 / (1 - args.tvf_lambda), args.n_steps)
                weighting = (self.current_max_horizon - data["tvf_horizons"] + effective_n_step) / effective_n_step
                # normalize so weights average out to be 1
                weighting = weighting / weighting.mean()
            else:
                raise ValueError("Invalid tvf_loss_weighting value.")

            if args.tvf_soft_anchor != 0:
                assert torch.all(data["tvf_horizons"][:, 0] == 0)
                anchor_loss = args.tvf_soft_anchor * torch.sum(torch.square(value_predictions[:,  0]))
                self.log.watch_mean("anchor_loss", loss, display_width=8)
                loss = loss + anchor_loss

            # MSE loss
            tvf_loss = 0.5 * weighting * args.tvf_coef * torch.square(targets - value_predictions)
            tvf_loss = tvf_loss.sum() / mini_batch_size
            loss = loss + tvf_loss

            self.log.watch_mean("loss_tvf", tvf_loss, history_length=64, display_width=8)

        # -------------------------------------------------------------------------
        # Calculate loss_value
        # -------------------------------------------------------------------------

        value_heads = []

        if not args.use_tvf:
            value_heads.append("ext")

        if args.use_intrinsic_rewards:
            value_heads.append("int")

        for value_head in value_heads:
            value_prediction = model_out["{}_value".format(value_head)]
            returns = data["{}_returns".format(value_head)]
            old_pred_values = data["{}_value".format(value_head)]

            if args.use_clipped_value_loss:
                # is is essentially trust region for value learning, and seems to help a lot.
                value_prediction_clipped = old_pred_values + torch.clamp(value_prediction - old_pred_values,
                                                                         -args.ppo_epsilon, +args.ppo_epsilon)
                vf_losses1 = (value_prediction - returns).pow(2)
                vf_losses2 = (value_prediction_clipped - returns).pow(2)
                loss_value = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            else:
                # simpler version, just use MSE.
                vf_losses1 = (value_prediction - returns).pow(2)
                loss_value = 0.5 * torch.mean(vf_losses1)
            loss_value = loss_value * args.vf_coef
            self.log.watch_mean("loss_v_" + value_head, loss_value, history_length=64)
            loss = loss + loss_value

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        opt_loss = loss * loss_scale
        opt_loss.backward()

        # -------------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------------

        self.log.watch_mean("value_loss", loss)

        return {}

    def generate_horizon_sample(self, max_value: int, samples: int, distribution: str = "uniform") -> np.ndarray:
        """
        generates random samples from 0 to max (inclusive) using sampling with replacement
        and always including the first and last value
        distribution is the distribution to sample from
        """
        if samples == -1 or samples >= (max_value + 1):
            return np.arange(0, max_value + 1)

        # make sure first and last horizons are always included.
        f_and_l = math.ceil(samples * args.tvf_first_and_last)
        required = []
        for i in range(f_and_l):
            required.append(i)
            required.append(max_value-i)
        required = np.asarray(required, dtype=np.int32)

        sample_range = range(f_and_l, max_value-f_and_l+1)

        if distribution in ["uniform", "constant"]: # constant was the old name for this
            p = None
        elif distribution == "linear":
            p = np.asarray([max_value-h for h in sample_range], dtype=np.float32)
            p /= np.sum(p)
        elif distribution == "hyperbolic":
            p = np.asarray([1/(h+1) for h in sample_range])
            p /= np.sum(p)
        elif distribution == "exponential":
            # adjust exponential so mean is half horizon
            p = np.asarray([np.exp(-(2/max_value*h)) for h in sample_range])
            p /= np.sum(p)
        else:
            raise Exception("invalid distribution")

        sampled = np.random.choice(sample_range, samples-(f_and_l*2), replace=False, p=p)
        result = list(np.concatenate((required, sampled)))
        result.sort()
        return np.asarray(result)

    def generate_return_sample(self):
        """
        Generates return estimates for current batch of data.

        Note: could roll this into calculate_sampled_returns, and just have it return the horizions aswell?

        returns:
            returns: ndarray of dims [N,A,K] containing the return estimates using tvf_gamma discounting
            horizon_samples: ndarray of dims [N,A,K] containing the horizons used
        """

        # max horizon to train on
        H = self.current_max_horizon
        N, A, *state_shape = self.prev_obs.shape

        value_samples = self.generate_horizon_sample(H, args.tvf_value_samples, distribution=args.tvf_value_distribution)
        horizon_samples = self.generate_horizon_sample(H, args.tvf_horizon_samples)

        returns = self.calculate_sampled_returns(
            value_sample_horizons=value_samples,
            required_horizons=horizon_samples,
        )

        horizon_samples = horizon_samples[None, None, :]
        horizon_samples = np.repeat(horizon_samples, N, axis=0)
        horizon_samples = np.repeat(horizon_samples, A, axis=1)
        return returns, horizon_samples

    def train_policy_minibatch(self, data, loss_scale=1.0):

        mini_batch_size = len(data["prev_state"])

        loss = torch.tensor(0, dtype=torch.float32, device=self.model.device)

        prev_states = data["prev_state"]
        actions = data["actions"].to(torch.long)
        old_policy_logprobs = data["log_policy"]
        advantages = data["advantages"]

        model_out = self.forward(prev_states, output="policy")

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        logps = model_out["log_policy"]

        logpac = logps[range(mini_batch_size), actions]
        old_logpac = old_policy_logprobs[range(mini_batch_size), actions]
        ratio = torch.exp(logpac - old_logpac)
        clipped_ratio = torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon)

        loss_clip = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss_clip_mean = loss_clip.mean()
        loss = loss + loss_clip_mean

        # approx kl
        # this is from https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/ppo2/ppo2.html
        # but https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
        # uses approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean() which I think is wrong
        # anyway, why not just calculate the true kl?

        # ok, I figured this out, we want
        # sum_x P(x) log(P/Q)
        # our actions were sampled from the policy so we have
        # pi(expected_action) = sum_x P(x) then just mulitiply this by log(P/Q) which is log(p)-log(q)
        # this means the spinning up version is right.

        with torch.no_grad():
            clip_frac = torch.gt(torch.abs(ratio - 1.0), args.ppo_epsilon).float().mean()
            kl_approx = (old_logpac - logpac).mean()
            kl_true = F.kl_div(old_policy_logprobs, logps, log_target=True, reduction="batchmean")

        # -------------------------------------------------------------------------
        # Calculate loss_entropy
        # -------------------------------------------------------------------------

        loss_entropy = -(logps.exp() * logps).mean(axis=1)
        loss_entropy = loss_entropy * args.entropy_bonus
        loss_entropy = loss_entropy.mean()
        loss = loss + loss_entropy

        # -------------------------------------------------------------------------
        # Calculate gradients
        # -------------------------------------------------------------------------

        opt_loss = -loss * loss_scale
        opt_loss.backward()

        # -------------------------------------------------------------------------
        # Generate log values
        # -------------------------------------------------------------------------

        self.log.watch_mean("loss_pg", loss_clip_mean, history_length=64)
        self.log.watch_mean("kl_approx", kl_approx, display_width=0)
        self.log.watch_mean("kl_true", kl_true, display_width=8)
        self.log.watch_mean("clip_frac", clip_frac, display_width=8)
        self.log.watch_mean("loss_ent", loss_entropy)
        self.log.watch_mean("policy_loss", loss)

        return {
            'kl_approx': float(kl_approx.detach()),  # make sure we don't pass the graph through.
            'kl_true': float(kl_true.detach()),
            'clip_frac': float(clip_frac.detach()),
        }

    @property
    def training_fraction(self):
        return (self.step / 1e6) / args.epochs

    @property
    def current_max_horizon(self):
        if args.tvf_horizon_warmup > 0:
            # 100 seems safe to learn so make that the minimum
            return int(np.clip(self.training_fraction * args.tvf_max_horizon / args.tvf_horizon_warmup, 100,
                               args.tvf_max_horizon))
        else:
            return int(args.tvf_max_horizon)

    @property
    def reward_scale(self):
        """ The amount rewards have been scaled by. """
        if args.reward_normalization:
            return atari.get_env_state("returns_norm_state")[1] ** 0.5
        else:
            return 1.0

    def train(self, step):

        self.step = step

        # ----------------------------------------------------
        # policy phase

        B = args.batch_size

        batch_data = {}
        batch_data["prev_state"] = self.prev_obs.reshape([B, *self.state_shape])
        batch_data["actions"] = self.actions.reshape(B).astype(np.long)
        batch_data["log_policy"] = self.log_policy.reshape([B, *self.policy_shape])

        if args.normalize_advantages:
            # we should normalize at the mini_batch level, but it's so much easier to do this at the batch level.
            advantages = self.advantage.reshape(B)
            batch_data["advantages"] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            batch_data["advantages"] = self.advantage.reshape(B)

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(B)
            batch_data["int_value"] = self.int_value.reshape(B)

        policy_epochs = 0
        for _ in range(args.policy_epochs):
            results = self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_policy_minibatch,
                mini_batch_size=args.policy_mini_batch_size,
                optimizer=self.policy_optimizer,
                label="train",
                hooks={
                    # 'after_mini_batch': lambda x: x["outputs"][-1]["kl_true"] > args.target_kl
                    'after_mini_batch': lambda x: x["outputs"][-1]["kl_approx"] > 1.5 * args.target_kl
                }
            )
            expected_mini_batches = (args.batch_size / args.policy_mini_batch_size)
            policy_epochs += results["mini_batches"] / expected_mini_batches
            if results["mini_batches"] < expected_mini_batches:
                break
        self.log.watch_full("policy_epochs", policy_epochs, display_width=8)

        # ----------------------------------------------------
        # value phase

        batch_data = {}
        batch_data["prev_state"] = self.prev_obs.reshape([B, *self.state_shape])
        batch_data["ext_returns"] = self.ext_returns.reshape(B)

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(B)
            batch_data["int_value"] = self.int_value.reshape(B)

        for _ in range(args.value_epochs):

            if args.use_tvf:
                # we do this once at the start, generate one returns estimate for each epoch on different horizons then
                # during epochs take a random mixture from this. This helps shuffle the horizons, and also makes sure that
                # we don't drift, as the updates will modify our model and change the value estimates.
                # it is possible that instead we should be updating our return estimates as we go though
                returns, horizons = self.generate_return_sample()
                batch_data["tvf_returns"] = returns.reshape([B, -1])
                batch_data["tvf_horizons"] = horizons.reshape([B, -1])
                batch_data["tvf_time"] = self.prev_time.reshape([B])

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_value_minibatch,
                mini_batch_size=args.value_mini_batch_size,
                optimizer=self.value_optimizer,
                label="value",
            )

        # aux phase is not implemented yet...
        # we also want to include rnd here somewhere..
        pass

        self.batch_counter += 1

    def train_batch(
            self,
            batch_data,
            mini_batch_func,
            mini_batch_size,
            optimizer: torch.optim.Optimizer,
            label,
            hooks: Union[dict, None] = None) -> dict:
        """
        Trains agent policy on current batch of experience
        Returns context with
            'mini_batches' number of mini_batches completed
            'outputs' output from each mini_batch update
        """

        mini_batches = args.batch_size // mini_batch_size
        micro_batch_size = min(args.max_micro_batch_size, mini_batch_size)
        micro_batches = mini_batch_size // micro_batch_size

        ordering = list(range(args.batch_size))
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

                minibatch_data = {}
                for var_name, var_value in batch_data.items():
                    minibatch_data[var_name] = torch.from_numpy(var_value[sample]).to(self.model.device)

                outputs.append(mini_batch_func(
                    minibatch_data, loss_scale=1 / micro_batches
                ))

            context = {
                'mini_batches': j + 1,
                'outputs': outputs
            }

            if hooks is not None and "after_mini_batch" in hooks:
                if hooks["after_mini_batch"](context):
                    break

            self.optimizer_step(optimizer=optimizer, label=label)

        return context


def get_rediscounted_value_estimate(values: Union[np.ndarray, torch.Tensor], old_gamma: float, new_gamma: float):
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
    old_discount = 1
    discount = 1
    for h in range(H):
        reward = (values[:, h] - prev) / old_discount
        prev = values[:, h]
        discounted_reward_sum += reward * discount
        old_discount *= old_gamma
        discount *= new_gamma

    return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum
