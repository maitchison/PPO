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


def calculate_gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma:float, lamb=1.0, normalize=False):

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

    # this is totally wrong... please fix.

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
        returns[t, :, steps_made+1:] += reward_sum[:, None] + discounted_bootstrap_estimates

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
        returns[t, :, steps_made+1:] += reward_sum[:, None] + discounted_bootstrap_estimates

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

    def __init__(self, model:models.TVFModel, log, name="agent"):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model

        self.policy_optimizer = torch.optim.Adam(model.policy_net.parameters(), lr=args.policy_lr, eps=args.adam_epsilon)
        self.value_optimizer = torch.optim.Adam(model.value_net.parameters(), lr=args.policy_lr, eps=args.adam_epsilon)
        if args.use_rnd:
            self.rnd_optimizer = torch.optim.Adam(model.prediction_net.parameters(), lr=args.policy_lr, eps=args.adam_epsilon)

        # stub: show optimizer parameters
        def show_opt_params(opt:torch.optim.Optimizer):
            for param in opt.param_groups[0]['params']:
                print(param.shape)

        # print("Policy Optimizer")
        # show_opt_params(self.policy_optimizer)
        # print("Value Optimizer")
        # show_opt_params(self.value_optimizer)

        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents

        self.state_shape = model.input_dims
        self.rnn_state_shape = [2, 512] #records h and c for LSTM units.
        self.policy_shape = [model.actions]

        self.batch_counter = 0

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.obs = np.zeros([A, *self.state_shape], dtype=np.uint8)
        self.prev_obs = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.actions = np.zeros([N, A], dtype=np.int64)
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool) # indicates prev_state was a terminal state.
        self.ext_value = np.zeros([N, A], dtype=np.float32)

        # value function horizons
        if args.use_tvf:
            self.tvf_returns = np.zeros([N, A, args.tvf_max_horizon + 1], dtype=np.float32)
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

        self.step = 0
        self.game_crashes = 0
        self.ep_count = 0

        #  these horizons will always be generated and their scores logged.
        self.tvf_debug_horizons = [h for h in [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000] if h <= args.tvf_max_horizon]
        if args.tvf_max_horizon not in self.tvf_debug_horizons:
            self.tvf_debug_horizons.append(args.tvf_max_horizon)
        self.tvf_debug_horizons.sort()

    def create_envs(self):
        """ Creates environments for runner"""
        env_fns = [lambda : atari.make() for _ in range(args.agents)]

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
                initial_state=atari.get_env_state("returns_norm_state")
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

    def forward(self, obs, max_batch_size=None, **kwargs):
        """ Forward states through model, returns output, which is a dictionary containing
            "log_policy" etc.
        """

        # break large forwards into batches
        if max_batch_size is not None and len(obs) > max_batch_size:

            mid_point = len(obs) // 2

            if 'horizons' in kwargs:
                horizons = kwargs["horizons"]
                del kwargs["horizons"]
                a = self.forward(obs[:mid_point], horizons=horizons[:mid_point], max_batch_size=max_batch_size, **kwargs)
                b = self.forward(obs[mid_point:], horizons=horizons[mid_point:], max_batch_size=max_batch_size, **kwargs)
            else:
                a = self.forward(obs[:mid_point], **kwargs)
                b = self.forward(obs[mid_point:], **kwargs)
            result = {}
            for k in a.keys():
                result[k] = torch.cat(tensors=[a[k], b[k]], dim=0)
            return result
        else:
            return self.model.forward(obs, **kwargs)

    def export_movie(self, filename, include_rollout=False, include_video=True, max_frames = 60*60*15):
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

        max_h = self.current_max_horizon
        all_horizons = np.repeat(np.arange(max_h+1, dtype=np.int16)[None, :], repeats=self.A, axis=0)
        # also include horizons for debugging
        required_horizons = np.asarray([x for x in self.tvf_debug_horizons if x < max_h] + [max_h], dtype=np.int16)
        far_horizons = np.repeat(required_horizons[None, :], repeats=self.A, axis=0)

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

            # forward state through model, then detach the result and convert to numpy.
            if args.use_tvf:
                if self.tvf_requires_full_horizon_at_rollout:
                    model_out = self.forward(self.obs, horizons=all_horizons, max_batch_size=max_rollout_batch_size)
                    tvf_values = model_out["tvf_value"].cpu().numpy()
                    self.tvf_values[t,:max_h+1] = tvf_values
                    ext_value = self.get_rediscounted_value_estimate(tvf_values[:,:max_h+1], args.gamma)
                else:
                    model_out = self.forward(self.obs, horizons=far_horizons, max_batch_size=max_rollout_batch_size)
                    tvf_values = model_out["tvf_value"].cpu().numpy()
                    # map across all the required horizons
                    for index, h in enumerate(required_horizons):
                        self.tvf_values[t, :, h] = tvf_values[:, index]
                    ext_value = tvf_values[..., -1]
            else:
                model_out = self.forward(self.obs)
                ext_value = model_out["ext_value"].cpu().numpy()

            log_policy = model_out["log_policy"].cpu().numpy()

            # during warm-up we simply collect experience through a uniform random policy.
            if is_warmup:
                actions = np.random.randint(0, self.model.actions, size=[self.A], dtype=np.int32)
            else:
                # sample actions and run through environment.
                actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)

            self.obs, ext_rewards, dones, infos = self.vec_env.step(actions)

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

            #for i, info in enumerate(infos): #just check info 0
            #    if "unclipped_reward" in info:
            #        print(f"Env {i:03d} clipped reward from {info['unclipped_reward']:.1f}")

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

            self.prev_obs[t] = prev_obs
            self.actions[t] = actions

            self.ext_rewards[t] = ext_rewards
            self.log_policy[t] = log_policy
            self.terminals[t] = dones
            self.ext_value[t] = ext_value

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

        # get value estimates for final state.
        if args.use_tvf:

            def debug_plot():
                xs = range(3001)
                ys = [np.mean(self.tvf_final_value_estimates[:, x]) for x in xs]
                import matplotlib.pyplot as plt
                plt.plot(xs, ys)
                plt.show()

            model_out = self.forward(self.obs, horizons=all_horizons, max_batch_size=max_final_batch_size)
            final_tvf_values = model_out["tvf_value"].cpu().numpy()
            self.tvf_final_value_estimates[:, :max_h+1] = final_tvf_values
            self.ext_final_value_estimate = self.get_rediscounted_value_estimate(final_tvf_values[:, :max_h+1], args.gamma)
        else:
            model_out = self.forward(self.obs, max_batch_size=max_final_batch_size)
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
        self.log.watch_mean("value_est_ext", np.mean(self.ext_value), display_name="est_v_ext", display_width=0)
        self.log.watch_mean("value_est_ext_std", np.std(self.ext_value), display_name="est_v_ext_std", display_width=0)

        self.log.watch("game_crashes", self.game_crashes, display_width=0 if self.game_crashes==0 else 8)

        if args.use_tvf:
            for h in self.tvf_debug_horizons:
                value = self.tvf_values[:, :, h].ravel().astype(np.float32)
                target = self.tvf_returns[:, :, h].ravel().astype(np.float32)
                self.log.watch_mean(f"ev_{h:04d}", utils.explained_variance(value, target), display_width=8)
                # raw is RMS on unscaled error
                self.log.watch_mean(f"raw_{h:04d}", np.mean(np.square(self.reward_scale*(value - target)) ** 0.5), display_width=0)
                self.log.watch_mean(f"mse_{h:04d}", np.mean(np.square(value - target)), display_width=0)
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
        # faster version used cached rediscount ratios
        return get_rediscounted_value_estimate(values, old_gamma=args.tvf_gamma, new_gamma=gamma)

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

    def optimizer_step(self, optimizer:torch.optim.Optimizer, label: str = "opt"):

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
            kwargs['horizons'] = data["tvf_horizons"]

        model_out = self.forward(prev_states, output="value", **kwargs)

        # -------------------------------------------------------------------------
        # Calculate loss_value_function_horizons
        # -------------------------------------------------------------------------

        if args.use_tvf:
            # targets "tvf_returns" are [N, A, K]
            # predictions "tvf_value" are [N, A, K]
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
                    effective_n_step = min(1/(1-args.tvf_lambda), args.n_steps)
                weighting = (self.current_max_horizon - data["tvf_horizons"] + effective_n_step) / effective_n_step
                # normalize so weights average out to be 1
                weighting = weighting / weighting.mean()
            else:
                raise ValueError("Invalid tvf_loss_weighting value.")

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
        # Run optimizer
        # -------------------------------------------------------------------------

        self.log.watch_mean("value_loss", loss * loss_scale)

        loss = loss * loss_scale

        loss.backward()

        return {}


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

        self.log.watch_mean("loss_pg", loss_clip_mean, history_length=64)
        self.log.watch_mean("kl_approx", kl_approx, display_width=8)
        self.log.watch_mean("kl_true", kl_true, display_width=8)
        self.log.watch_mean("clip_frac", clip_frac, display_width=8)

        # -------------------------------------------------------------------------
        # Calculate loss_entropy
        # -------------------------------------------------------------------------

        loss_entropy = -(logps.exp() * logps).mean(axis=1)
        loss_entropy = loss_entropy * args.entropy_bonus
        loss_entropy = loss_entropy.mean()
        self.log.watch_mean("loss_ent", loss_entropy)
        loss = loss + loss_entropy

        # -------------------------------------------------------------------------
        # Run optimizer
        # -------------------------------------------------------------------------

        self.log.watch_mean("policy_loss", loss * loss_scale)

        loss = -loss * loss_scale

        loss.backward()

        return {
            'kl_approx': float(kl_approx.detach()), # make sure we don't pass the graph through.
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
            return int(np.clip(self.training_fraction * args.tvf_max_horizon / args.tvf_horizon_warmup, 100, args.tvf_max_horizon))
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
            # we should normalize at the mini_batch level, but it's so much easyer to do this at the batch level.
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
                    'after_mini_batch': lambda x : x["outputs"][-1]["kl_approx"] > 1.5 * args.target_kl
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
        batch_data["ext_value"] = self.ext_value.reshape(B)

        if args.use_tvf:

            # max horizon to train on
            H = self.current_max_horizon
            # number of sample to use
            K = min(args.tvf_n_horizons, H)

            required_horizons = np.asarray([0, H], dtype=np.int32)
            required_horizons = np.repeat(required_horizons[None, :], B, axis=0)
            all_horizons = np.asarray(range(H + 1), dtype=np.int32)
            all_horizons = np.repeat(all_horizons[None, :], B, axis=0)
            all_returns = self.tvf_returns.reshape([B, H+1])

            if K >= H and args.tvf_sample_dist=="uniform":
                # use all samples, this is more efficient as we won't sample the same horizon twice.
                batch_data["tvf_horizons"] = all_horizons
                batch_data["tvf_returns"] = all_returns
            else:
                # perform sub sampling
                if args.tvf_sample_dist == "uniform":
                    p = None
                elif args.tvf_sample_dist == "linear":
                    p = [1 - (i / H) for i in range(H)]
                else:
                    raise Exception("Invalid distribution.")

                horizon_sample = np.random.choice(H, size=[B, K - required_horizons.shape[-1]], replace=True, p=p)
                horizon_sample = np.concatenate([horizon_sample, required_horizons], axis=1)
                batch_data["tvf_horizons"] = horizon_sample
                batch_data["tvf_returns"] = np.take_along_axis(all_returns, horizon_sample, axis=1)


        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(B)
            batch_data["int_value"] = self.int_value.reshape(B)

        for _ in range(args.value_epochs):
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
            optimizer:torch.optim.Optimizer,
            label,
            hooks:Union[dict, None] = None) -> dict:
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


def get_rediscounted_value_estimate(values:Union[np.ndarray, torch.Tensor], old_gamma:float, new_gamma:float):
    """
    Returns rediscounted return at horizon H

    values: float tensor of shape [B, H]
    returns: float tensor of shape [B]
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
