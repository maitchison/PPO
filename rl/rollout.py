import os
import numpy as np
import gym
import torch
import torch.nn as nn
import time
import json
import math
import cv2
import pickle
import gzip
from collections import defaultdict
from typing import Union

from .logger import Logger
from . import utils, atari, hybridVecEnv
from .config import args

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.learning_rate_decay == 1.0:
        return args.learning_rate
    else:
        lr = args.learning_rate * (args.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

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


def calculate_gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma:float, lamb=1.0, normalize=False):

    N, A = batch_rewards.shape

    batch_advantage = np.zeros_like(batch_rewards, dtype=np.float32)
    prev_adv = np.zeros([A], dtype=np.float32)
    for t in reversed(range(N)):
        is_next_terminal = batch_terminal[
            t] if batch_terminal is not None else False  # batch_terminal[t] records if t+1 is a terminal state)
        value_next_t = batch_value[t + 1] if t != N - 1 else final_value_estimate
        delta = batch_rewards[t] + gamma * value_next_t * (1.0 - is_next_terminal) - batch_value[t]
        batch_advantage[t] = prev_adv = delta + gamma * lamb * (
                1.0 - is_next_terminal) * prev_adv
    if normalize:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
    return batch_advantage

def calculate_gae_tvf(
        rewards:np.ndarray,
        dones:np.ndarray,
        values:np.ndarray,
        final_value_estimates:np.ndarray,
        gamma,
        lamb:float=0.95):

    N, A, H = values.shape

    advantages = np.zeros([N, A], dtype=np.float32)

    # note: this webpage helped a lot with writing this...
    # https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html

    values = np.concatenate([values, final_value_estimates[None, :, :]], axis=0)

    # note, we could calculate advantage for all horizons if we want.

    for t in range(N):
        h = H-1
        total_weight = np.zeros([A], dtype=np.float32)
        current_weight = np.ones([A], dtype=np.float32) * (1-lamb)
        discount = np.ones([A], dtype=np.float32)
        advantage = np.zeros([A], dtype=np.float32)
        weighted_estimate = np.zeros([A], dtype=np.float32)
        # for the moment limit n_step to 10 for performance reasons (slow python, slow algorithm)
        # I have a NH version coming (instead of this NNH version)
        # if lamb is 0 we just do TD(0), which is much faster.
        advantage -= values[t, :, h]
        for n in range(1, 60): # 60 should be enough for lamb=0.95
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
        rewards:np.ndarray,
        dones:np.ndarray,
        values:np.ndarray,
        final_value_estimates:np.ndarray,
        gamma:float,
        lamb:float=0.95,
        ):
    # this is a little slow, but calculate each n_step return and combine them.
    # also.. this is just an approximation

    params = (rewards, dones, values, final_value_estimates, gamma)

    if lamb == 0:
        return calculate_tvf_td(*params)
    if lamb == 1:
        return calculate_tvf_mc(*params)

    # can be slow for high n_steps... so we cap it at 100, and use effective horizon as a cap too
    N = int(min(1/(1-lamb), args.n_steps, 100))

    g = []
    for i in range(N):
        g.append(calculate_tvf_n_step(*params, n_step=i+1))

    result = g[0] * (1-lamb)
    for i in range(1, N):
        result += g[i] * (lamb**i) * (1-lamb)

    return result


def calculate_tvf_n_step(
        rewards:np.ndarray,
        dones:np.ndarray,
        values:np.ndarray,
        final_value_estimates:np.ndarray,
        gamma:float,
        n_step:int,
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
        discounted_bootstrap_estimates = discount[:, None] * values[t + steps_made, :, 1:H-steps_made]
        returns[t, :, steps_made+1:H] += reward_sum[:, None] + discounted_bootstrap_estimates

        # this is the non-vectorized code, for reference.
        #for h in range(steps_made+1, H):
        #    bootstrap_estimate = discount * values[t + steps_made, :, h - steps_made] if (h - steps_made) > 0 else 0
        #    returns[t, :, h] = reward_sum + bootstrap_estimate

    return returns


def calculate_tvf_mc(
        rewards:np.ndarray,
        dones:np.ndarray,
        values:None, #note: values is ignored...
        final_value_estimates:np.ndarray,
        gamma:float
        ):
    """
    This is really just the largest n_step that will work, but does not require values
    """

    N, A = rewards.shape
    H = final_value_estimates.shape[-1]

    returns = np.zeros([N, A, H], dtype=np.float32)

    n_step = N-1

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
        discounted_bootstrap_estimates = discount[:, None] * final_value_estimates[:, 1:H-steps_made]
        returns[t, :, steps_made+1:H] += reward_sum[:, None] + discounted_bootstrap_estimates

    return returns


def calculate_tvf_td(
        rewards:np.ndarray,
        dones:np.ndarray,
        values:np.ndarray,
        final_value_estimates:np.ndarray,
        gamma:float,
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
            reward_sum = rewards[t+1-1]
            discount = gamma * (1 - dones[t + 1 - 1])
            bootstrap_estimate = discount * values[t+1, :, h-1] if (h-1) > 0 else 0
            estimate = reward_sum + bootstrap_estimate
            returns[t, :, h] = estimate
    return returns


class Runner():

    def __init__(self, model, optimizer, log, name="agent"):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents

        self.state_shape = model.input_dims
        self.rnn_state_shape = [2, 512] #records h and c for LSTM units.
        self.policy_shape = [model.actions]

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.states = np.zeros([A, *self.state_shape], dtype=np.uint8)
        self.prev_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.actions = np.zeros([N, A], dtype=np.int64)
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool)
        self.ext_value = np.zeros([N, A], dtype=np.float32)

        # value function horizons
        if args.use_tvf:
            self.tvf_returns = np.zeros([N, A, args.tvf_max_horizon + 1], dtype=np.float32)
            if self.tvf_requires_full_horizon_at_rollout:
                self.tvf_values = np.zeros([N, A, args.tvf_max_horizon + 1], dtype=np.float32)
            # we need to have a final value estimate for each horizon, as we don't know which ones will be used.
            self.tvf_final_value_estimates = np.zeros([A, args.tvf_max_horizon + 1], dtype=np.float32)

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)
        self.int_value = np.zeros([N, A], dtype=np.float32)

        # returns generation
        self.ext_returns = np.zeros([N, A], dtype=np.float32)
        self.int_returns_raw = np.zeros([N, A], dtype=np.float32)
        self.advantage = np.zeros([N, A], dtype=np.float32)

        self.ext_final_value_estimate = np.zeros([A], dtype=np.float32)
        self.int_final_value_estimate = np.zeros([A], dtype=np.float32)

        self.intrinsic_returns_rms = utils.RunningMeanStd(shape=())
        self.ems_norm = np.zeros([args.agents])

        # outputs tensors when clip loss is very high.
        self.log_high_grad_norm = True

        self.step=0

        # create rediscounting ratio and debug horizon list (these are always generated)
        self.tvf_rediscount_ratios = np.asarray([(args.gamma ** h) / (args.tvf_gamma ** h) for h in range(args.tvf_max_horizon)], dtype=np.float32)
        self.tvf_debug_horizons = [h for h in [0, 1, 10, 30, 100, 300, 500, 1000, 2000, 4000] if h <= args.tvf_max_horizon]


    def create_envs(self):
        """ Creates environments for runner"""
        env_fns = [lambda : atari.make() for _ in range(args.agents)]

        if args.sync_envs:
            self.vec_env = gym.vector.SyncVectorEnv(env_fns)
        else:
            self.vec_env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=args.workers, verbose=True)
        self.log.important("Generated {} agents ({}) using {} ({}) model.".
                           format(args.agents, "async" if not args.sync_envs else "sync", self.model.name,
                                  self.model.dtype))

    def save_checkpoint(self, filename, step):

        data = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'logs': self.log,
            'env_state': atari.ENV_STATE
        }

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
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        self.log = checkpoint['logs']
        self.step = step

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
        self.states = self.vec_env.reset()
        self.episode_score *= 0
        self.episode_len *= 0
        self.step = 0

    def run_random_agent(self, iterations):
        self.log.info("Warming up model with random agent...")

        # collect experience
        self.reset()

        for iteration in range(iterations):
            self.generate_rollout(is_warmup=True)

    def forward(self, states=None, **kwargs):
        """ Forward states through model, returns output, which is a dictionary containing
            "log_policy" etc.
        """

        if states is None:
            states = self.states

        return self.model.forward(self.states if states is None else states, **kwargs)

    def export_movie(self, filename, include_rollout=False, include_video=True, max_frames = 30*60*15):
        """ Exports a movie of agent playing game.
            include_rollout: save a copy of the rollout (may as well include policy, actions, value etc)
        """

        scale = 2

        env = atari.make()
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
            video_out = cv2.VideoWriter(filename+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)
        else:
            video_out = None

        state = env.reset()

        frame_count = 0

        history = defaultdict(list)

        # play the game...
        while not done:

            prev_state = state.copy()
            prev_action = action

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
            for k, v in history.items():
                history[k] = np.asarray(v)
            pickle.dump(history, gzip.open(filename+".hst.gz", "wb", compresslevel=9))

    @property
    def tvf_requires_full_horizon_at_rollout(self):
        return args.tvf_gamma != args.gamma or args.tvf_lambda != 1

    @torch.no_grad()
    def generate_rollout(self, is_warmup=False):

        assert self.vec_env is not None, "Please call create_envs first."

        additional_params = {}

        max_h = self.current_max_horizon
        all_horizons = np.repeat(np.arange(args.tvf_max_horizon+1, dtype=np.int16)[None, :], repeats=self.A, axis=0)
        # also include horizons for debugging
        required_horizons = np.asarray(self.tvf_debug_horizons + [max_h], dtype=np.int16)
        far_horizons = np.repeat(required_horizons[None, :], repeats=self.A, axis=0)

        for t in range(self.N):

            prev_states = self.states.copy()

            # forward state through model, then detach the result and convert to numpy.
            if args.use_tvf:
                if self.tvf_requires_full_horizon_at_rollout:
                    model_out = self.forward(horizons=all_horizons)
                    tvf_values = model_out["tvf_value"].cpu().numpy()
                    self.tvf_values[t] = tvf_values
                    ext_value = self.get_rediscounted_value_estimate(tvf_values[:, :max_h], args.gamma)
                else:
                    model_out = self.forward(horizons=far_horizons)
                    tvf_values = model_out["tvf_value"].cpu().numpy()
                    # map across all the required horizons
                    for h in required_horizons:
                        self.tvf_values[t, :, h] = tvf_values[:, h]
                    ext_value = tvf_values[..., -1]
            else:
                model_out = self.forward()
                ext_value = model_out["ext_value"].cpu().numpy()

            log_policy = model_out["log_policy"].cpu().numpy()

            # during warm-up we simply collect experience through a uniform random policy.
            if is_warmup:
                actions = np.random.randint(0, self.model.actions, size=[self.A], dtype=np.int32)
            else:
                # sample actions and run through environment.
                actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)

            self.states, ext_rewards, dones, infos = self.vec_env.step(actions)

            # it's a bit silly to have this here...
            if "returns_norm_state" in infos[0]:
                atari.ENV_STATE["returns_norm_state"] = infos[0]["returns_norm_state"]
                norm_mean, norm_var, norm_count = infos[0]["returns_norm_state"]
                self.log.watch("returns_norm_mu", norm_mean, display_width=0)
                self.log.watch("returns_norm_std", norm_var**0.5, display_width=0)

            # work out our intrinsic rewards
            if args.use_intrinsic_rewards:
                value_int = model_out["int_value"].detach().cpu().numpy()

                int_rewards = np.zeros_like(ext_rewards)

                if args.use_rnd:
                    if is_warmup:
                        # in random mode just update the normalization constants
                        self.model.perform_normalization(self.states)
                    else:
                        # reward is prediction error on state we land inn.
                        loss_rnd = self.model.prediction_error(self.states).detach().cpu().numpy()
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

            for i, done in enumerate(dones):
                if done:
                    # reset is handled automatically by vectorized environments
                    # so just need to keep track of book-keeping
                    if not is_warmup:
                        self.log.watch_full("ep_score", self.episode_score[i])
                        self.log.watch_full("ep_length", self.episode_len[i])
                    self.episode_score[i] = 0
                    self.episode_len[i] = 0

            self.prev_state[t] = prev_states
            self.actions[t] = actions

            self.ext_rewards[t] = ext_rewards
            self.log_policy[t] = log_policy
            self.terminals[t] = dones
            self.ext_value[t] = ext_value

        #  save a copy of the normalization statistics.
        if args.normalize_observations:
            atari.ENV_STATE["observation_norm_state"] = self.model.obs_rms.save_state()

        # get value estimates for final state.
        if args.use_tvf:
            model_out = self.forward(horizons=all_horizons)
            final_tvf_values = model_out["tvf_value"].cpu().numpy()
            self.tvf_final_value_estimates[:] = final_tvf_values
            self.ext_final_value_estimate = self.get_rediscounted_value_estimate(final_tvf_values[:, :max_h], args.gamma)
        else:
            model_out = self.forward()
            self.ext_final_value_estimate = model_out["ext_value"].cpu().numpy()

        if "int_value" in model_out:
            self.int_final_value_estimate = model_out["int_value"].cpu().numpy()

    def calculate_returns(self):

        if args.use_tvf:
            params = (
                self.ext_rewards,
                self.terminals,
                self.tvf_values,
                self.tvf_final_value_estimates,
                args.tvf_gamma,
            )

            # negative values are assumed to be n_step, positive are td_lambda
            # 0, and 1 have special cases implemented which are faster.

            # also, we copy into returns just to make sure shape is right, and to insure the type is right.
            if args.tvf_lambda < 0:
                self.tvf_returns[:] = calculate_tvf_n_step(*params, n_step=-int(args.tvf_lambda))
            else:
                # tvf_lambda has special cases for lambda=0, and lambda=1 which are more efficient.
                self.tvf_returns[:] = calculate_tvf_lambda(*params, lamb=args.tvf_lambda)

        if args.use_tvf:
            self.log.watch("tvf_horizon", self.current_max_horizon)

        self.ext_advantage = calculate_gae(
            self.ext_rewards,
            self.ext_value,
            self.ext_final_value_estimate,
            self.terminals,
            args.gamma,
            args.gae_lambda
        )
        self.ext_returns = self.ext_advantage + self.ext_value

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
        if args.normalize_advantages:
            self.advantage = (self.advantage - self.advantage.mean()) / (
                        self.advantage.std() + 1e-8)

        if args.normalize_observations:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        self.log.watch_mean("adv_mean", np.mean(self.advantage), display_width=0)
        self.log.watch_mean("adv_std", np.std(self.advantage), display_width=0)
        self.log.watch_mean("adv_max", np.max(self.advantage), display_width=0)
        self.log.watch_mean("adv_min", np.min(self.advantage), display_width=0)
        self.log.watch_mean("batch_reward_ext", np.mean(self.ext_rewards), display_name="rew_ext", display_width=0)
        self.log.watch_mean("batch_return_ext", np.mean(self.ext_returns), display_name="ret_ext")
        self.log.watch_mean("batch_return_ext_std", np.std(self.ext_returns), display_name="ret_ext_std",
                            display_width=0)
        self.log.watch_mean("value_est_ext", np.mean(self.ext_value), display_name="est_v_ext", display_width=0)
        self.log.watch_mean("value_est_ext_std", np.std(self.ext_value), display_name="est_v_ext_std", display_width=0)


        if args.use_tvf:
            for h in self.tvf_debug_horizons:
                value = self.tvf_values[:, :, h].ravel().astype(np.float32)
                target = self.tvf_returns[:, :, h].ravel().astype(np.float32)
                self.log.watch_mean(f"ev_{h:04d}", utils.explained_variance(value, target), display_width=8)
                self.log.watch_mean(f"mse_{h:04d}", np.mean((value - target)**2), display_width=0)
        else:
            self.log.watch_mean("ev_ext", utils.explained_variance(self.ext_value.ravel(), self.ext_returns.ravel()))

        if args.use_intrinsic_rewards:
            self.log.watch_mean("batch_reward_int", np.mean(self.int_rewards), display_name="rew_int", display_width=0)
            self.log.watch_mean("batch_reward_int_std", np.std(self.int_rewards), display_name="rew_int_std",
                                display_width=0)
            self.log.watch_mean("batch_return_int", np.mean(self.int_returns), display_name="ret_int")
            self.log.watch_mean("batch_return_int_std", np.std(self.int_returns), display_name="ret_int_std", display_width=0)
            self.log.watch_mean("batch_return_int_raw_mean", np.mean(self.int_returns_raw),
                                display_name="ret_int_raw_mu",
                                display_width=0)
            self.log.watch_mean("batch_return_int_raw_std", np.std(self.int_returns_raw),
                                display_name="ret_int_raw_std",
                                display_width=0)

            self.log.watch_mean("value_est_int", np.mean(self.int_value), display_name="est_v_int", display_width=0)
            self.log.watch_mean("value_est_int_std", np.std(self.int_value), display_name="est_v_int_std", display_width=0)
            self.log.watch_mean("ev_int", utils.explained_variance(self.int_value.ravel(), self.int_returns.ravel()))
            if args.use_rnd:
                self.log.watch_mean("batch_reward_int_unnorm", np.mean(self.int_rewards), display_name="rew_int_unnorm",
                                    display_width=0, display_priority=-2)
                self.log.watch_mean("batch_reward_int_unnorm_std", np.std(self.int_rewards),
                                    display_name="rew_int_unnorm_std",
                                    display_width=0)

        if args.normalize_intrinsic_rewards:
            self.log.watch_mean("norm_scale_int", self.intrinsic_reward_norm_scale, display_width=0)


    def get_rediscounted_value_estimate(self, values:Union[np.ndarray, torch.Tensor], gamma:float):
        """
        Returns rediscounted return at horizon H

        values: float tensor of shape [B, H]
        returns: float tensor of shape [B]
        """

        # todo: please unit test this...

        B, H = values.shape

        if gamma == args.tvf_gamma:
            return values[:, -1]

        if type(values) is np.ndarray:
            values = torch.from_numpy(values)
            is_numpy = True
        else:
            is_numpy = False

        if gamma == args.gamma and H <= len(self.tvf_rediscount_ratios):
            # fast path for standard gamma conversion
            values_copy = values[:] # make a copy (as we'll be editing values)
            values_copy[:, 1:] -= values_copy[:, 0:-1]
            values_copy *= torch.from_numpy(self.tvf_rediscount_ratios)[None, :H]
            discounted_reward_sum = torch.sum(values_copy, dim=1)
            return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum

        # general path
        device = values.device
        prev = torch.zeros([B], dtype=torch.float32, device=device)
        discounted_reward_sum = torch.zeros([B], dtype=torch.float32, device=device)
        old_discount = 1
        discount = 1
        for h in range(H):
            reward = (values[:, h] - prev) / old_discount
            prev = values[:, h]
            discounted_reward_sum += reward * discount
            old_discount *= args.tvf_gamma
            discount *= gamma

        return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum

    def train_minibatch(self, data, zero_grad=True, apply_update=True, initial_loss=None,
                        loss_scale=1.0):

        result = {}

        loss = initial_loss or torch.tensor(0, dtype=torch.float32, device=self.model.device)

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        prev_states = data["prev_state"]
        actions = data["actions"].to(torch.long)
        policy_logprobs = data["log_policy"]
        advantages = data["advantages"]
        weights = data["weights"] if "weights" in data else 1

        mini_batch_size = len(prev_states)

        # create additional args if needed
        kwargs = {}
        if args.use_tvf:
            kwargs['horizons'] = data["tvf_horizons"]

        model_out = self.forward(prev_states, **kwargs)
        logps = model_out["log_policy"]

        ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])
        clipped_ratio = torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon)

        loss_clip = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss_clip_mean = (weights*loss_clip).mean()

        self.log.watch_mean("loss_pg", loss_clip_mean, history_length=64)
        loss += loss_clip_mean

        # -------------------------------------------------------------------------
        # Calculate loss_value_function_horizons
        # -------------------------------------------------------------------------

        if args.use_tvf:
            # targets "tvf_returns" are [N, A, K]
            # predictions "tvf_value" are [N, A, K]
            # predictions need to be generated... this could take a lot of time so just sample a few..
            targets = data["tvf_returns"]
            mu = model_out["tvf_value"]

            # MSE loss (calculate for reference even when using distributional)
            # I wonder if huber loss is better here?
            tvf_loss_mse = -0.5 * args.tvf_coef * (targets - mu).pow(2)
            self.log.watch_mean("tvf_loss_mse", -tvf_loss_mse.mean(), history_length=64)

            if args.tvf_distributional:
                std = model_out["tvf_std"]
                # log prob loss on gaussian distribution
                # I think this is dodgy, as log_prob is really log_pdf, and therefore we will get log_probs > 0.
                # not sure how to solve this though? Maybe scale the prob of each individual sample based on the
                # std of the gaussian?? In the end it seems to work though
                dist = torch.distributions.Normal(mu, std)
                tvf_loss_nlp = args.tvf_coef * dist.log_prob(targets)
                self.log.watch_mean("tvf_loss_nlp", -tvf_loss_nlp.mean(), history_length=64)
                self.log.watch_mean("tvf_std_nlp", tvf_loss_nlp.std(), history_length=64)
                loss += tvf_loss_nlp.mean()
            else:
                loss += tvf_loss_mse.mean()

        # -------------------------------------------------------------------------
        # Calculate loss_value
        # -------------------------------------------------------------------------

        value_heads = ["ext"]

        if args.use_intrinsic_rewards:
            value_heads.append("int")

        loss_value = 0
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
                loss_value = -0.5 * torch.mean(torch.max(vf_losses1, vf_losses2) * weights)
            else:
                # simpler version, just use MSE.
                vf_losses1 = (value_prediction - returns).pow(2)
                loss_value = -0.5 * torch.mean(vf_losses1 * weights)
            loss_value *= args.vf_coef
            self.log.watch_mean("loss_v_" + value_head, loss_value, history_length=64)
            loss += loss_value

        # -------------------------------------------------------------------------
        # Calculate loss_entropy
        # -------------------------------------------------------------------------

        loss_entropy = -(logps.exp() * logps).sum(axis=1)
        loss_entropy *= weights * args.entropy_bonus / mini_batch_size
        loss_entropy = loss_entropy.mean()
        self.log.watch_mean("loss_ent", loss_entropy)
        loss += loss_entropy

        # -------------------------------------------------------------------------
        # Calculate loss_rnd
        # -------------------------------------------------------------------------

        if args.use_rnd:
            # learn prediction slowly by only using some of the samples... otherwise it learns too quickly.
            predictor_proportion = np.clip(32 / args.agents, 0.01, 1)
            n = int(len(prev_states) * predictor_proportion)
            loss_rnd = -self.model.prediction_error(prev_states[:n]).mean()
            loss += loss_rnd

            self.log.watch_mean("loss_rnd", loss_rnd)

            self.log.watch_mean("feat_mean", self.model.features_mean, display_width=0)
            self.log.watch_mean("feat_var", self.model.features_var, display_width=10)
            self.log.watch_mean("feat_max", self.model.features_max, display_width=10, display_precision=1)

        # -------------------------------------------------------------------------
        # Run optimizer
        # -------------------------------------------------------------------------

        self.log.watch_mean("loss", loss * loss_scale)

        result["loss"] = loss

        # todo, seperate this into another function (useful for RNN)
        if apply_update:

            if zero_grad:
                self.optimizer.zero_grad()

            (-loss * loss_scale).backward()

            if args.max_grad_norm is not None and args.max_grad_norm != 0:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), args.max_grad_norm)
            else:
                # even if we don't clip the gradient we should at least log the norm. This is probably a bit slow though.
                # we could do this every 10th step, but it's important that a large grad_norm doesn't get missed.
                grad_norm = 0
                parameters = list(filter(lambda p: p.grad is not None, self.model.parameters()))
                for p in parameters:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5

            self.log.watch_mean("opt_grad", grad_norm)
            self.optimizer.step()

            # -------------------------------------------------------------------------
            # -------------------------------------------------------------------------

            if self.log_high_grad_norm and grad_norm > 100:
                self.log.important("Extremely high grad norm ... outputting inputs.")
                self.log.important("Loss clip was " + str(loss_clip_mean))
                self.log.important("Loss value was " + str(loss_value))

                f_name = lambda x: os.path.join(args.log_folder,self.name+"-"+x+"-"+str(self.log["env_step"]))

                utils.dump_data(advantages, f_name("advantages"))
                utils.dump_data(loss_clip, f_name("loss_clip"))
                utils.dump_data(ratio, f_name("ratio"))
                utils.dump_data(clipped_ratio, f_name("clipped_ratio"))
                utils.dump_data(logps, f_name("logps"))
                utils.dump_data(policy_logprobs, f_name("policy_logprobs"))
                utils.dump_data(actions, f_name("actions"))
                utils.dump_data(data["ext_value"], f_name("values"))
                utils.dump_data(data["ext_returns"], f_name("returns"))

                if args.use_tvf:
                    targets = data["tvf_returns"]
                    mu = model_out["tvf_value"]
                    utils.dump_data(targets, f_name("tvf_targets"))
                    utils.dump_data(mu, f_name("tvf_values"))
                    utils.dump_data(targets-mu, f_name("tvf_errors"))
                    utils.dump_data(tvf_loss_mse, f_name("tvf_loss_mse"))
                    if args.tvf_distributional:
                        utils.dump_data(tvf_loss_nlp, f_name("tvf_loss_nlp"))

                self.log_high_grad_norm = False

        return result

    @property
    def training_fraction(self):
        return (self.step / 1e6) / args.epochs

    @property
    def current_max_horizon(self):
        if args.tvf_horizon_warmup > 0:
            # 100 seems safe to learn so make that the minimum
            return int(np.clip(self.training_fraction * args.tvf_max_horizon / args.tvf_horizon_warmup, 100, args.tvf_max_horizon))
        else:
            return int(args.tvf_max_horizon)


    def train(self, step):
        """ trains agent on it's own experience """

        self.step = step

        # organise our data...
        batch_data = {}
        batch_size = self.N * self.A

        batch_data["prev_state"] = self.prev_state.reshape([batch_size, *self.state_shape])
        batch_data["actions"] = self.actions.reshape(batch_size).astype(np.long)
        batch_data["ext_returns"] = self.ext_returns.reshape(batch_size)

        batch_data["log_policy"] = self.log_policy.reshape([batch_size, *self.policy_shape])
        batch_data["advantages"] = self.advantage.reshape(batch_size)
        batch_data["ext_value"] = self.ext_value.reshape(batch_size)

        if args.use_tvf:
            # number of sample in mini_batch
            B = args.mini_batch_size
            # max horizon to train on
            H = self.current_max_horizon
            # number of sample to use
            K = min(args.tvf_n_horizons, H)

            batch_data["tvf_returns"] = self.tvf_returns.reshape([batch_size, -1])

            if args.tvf_sample_dist == "uniform":
                p = [1 / H for _ in range(H)]
            elif args.tvf_sample_dist == "linear":
                p = [1 - (i / H) for i in range(H)]
            else:
                raise Exception("Invalid distribution.")

            p = np.asarray(p, dtype=np.float64)
            p /= p.sum()
        else:
            H,B,K = 0,0,0
            p=0

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(batch_size)
            batch_data["int_value"] = self.int_value.reshape(batch_size)


        for i in range(args.batch_epochs):

            ordering = list(range(batch_size))
            np.random.shuffle(ordering)

            n_batches = math.ceil(batch_size / args.mini_batch_size)

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * args.mini_batch_size
                batch_end = (j + 1) * args.mini_batch_size
                sample = ordering[batch_start:batch_end]

                minibatch_data = {}

                for k, v in batch_data.items():

                    if k == "tvf_returns":
                        # we want to down sample the horizons randomly each time
                        # this is because we run out of GPU memory if we don't down sample, and also it is too slow.

                        mb_returns = np.zeros([B, K], dtype=np.float32)
                        mb_horizons = np.zeros_like(mb_returns)
                        horizon_sample = np.zeros([B], dtype=np.int64)

                        # apply horizon sampling
                        for b in range(B):

                            # sample with replacement is a little slow so we only update this every 16 values
                            # this will still mix things up enough.
                            if b % 16 == 0:
                                horizon_sample = np.random.choice(H, size=[K], replace=False, p=p)

                            mb_horizons[b, :] = horizon_sample
                            mb_returns[b, :] = v[sample[b], horizon_sample]

                        minibatch_data["tvf_horizons"] = torch.from_numpy(mb_horizons).to(device=self.model.device, dtype=torch.int16)
                        minibatch_data["tvf_returns"] = torch.from_numpy(mb_returns).to(device=self.model.device, dtype=torch.float32)

                    else:
                        minibatch_data[k] = torch.from_numpy(v[sample]).to(self.model.device)

                self.train_minibatch(minibatch_data)
