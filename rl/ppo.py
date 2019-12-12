import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from .logger import Logger, LogVariable

import torch.multiprocessing

from . import utils, models, atari, hybridVecEnv, config
from .config import args

class Runner():

    def __init__(self, model, optimizer, log):
        """ Setup our rollout runner. """

        self.model = model
        self.optimizer = optimizer
        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents

        self.state_shape = model.input_dims
        self.policy_shape = [model.actions]
        self.policy_atn_shape = [25]

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.states = np.zeros([A, *self.state_shape], dtype=np.uint8)

        self.prev_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.next_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.actions = np.zeros([N, A], dtype=np.int32)
        self.rewards_ext = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool)
        self.value_ext = np.zeros([N, A], dtype=np.float32)

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)
        self.int_value = np.zeros([N, A], dtype=np.float32)

        # attention
        self.atn_rewards = np.zeros([N, A], dtype=np.float32)
        self.atn_actions = np.zeros([N, A], dtype=np.int32)
        self.atn_log_policy = np.zeros([N, A, *self.policy_atn_shape], dtype=np.float32)
        self.atn_value = np.zeros([N, A], dtype=np.float32)

        # returns generation
        self.ext_returns = np.zeros([N, A], dtype=np.float32)
        self.int_returns_raw = np.zeros([N, A], dtype=np.float32)
        self.atn_returns = np.zeros([N, A], dtype=np.float32)
        self.advantage = np.zeros([N, A], dtype=np.float32)

        self.final_value_estimate_ext = np.zeros([A], dtype=np.float32)
        self.final_value_estimate_int = np.zeros([A], dtype=np.float32)
        self.final_value_estimate_atn = np.zeros([A], dtype=np.float32)

        self.intrinsic_returns_rms = utils.RunningMeanStd(shape=())
        self.ems_norm = np.zeros([args.agents])

    def create_envs(self, env_name):
        """ Creates environments for runner"""
        env_fns = [lambda: atari.make(env_name) for _ in range(args.agents)]
        self.vec_env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=args.workers,
                                                    verbose=True) if not args.sync_envs else gym.vector.SyncVectorEnv(env_fns)
        self.log.important("Generated {} agents ({}) using {} ({}) model.".
                      format(args.agents, "async" if not args.sync_envs else "sync", self.model.name, self.model.dtype))

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
            data['intrinsic_returns_rms'] = self.intrinsic_returns_rms,

        if args.use_rnd:
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
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        step = checkpoint['step']
        self.log = checkpoint['logs']

        if args.use_intrinsic_rewards:
            self.ems_norm = checkpoint.get['ems_norm']
            self.intrinsic_returns_rms = checkpoint['intrinsic_returns_rms']

        atari.ENV_STATE = checkpoint['env_state']

        if args.use_rnd:
            self.model.obs_rms.restore_state(checkpoint["observation_norm_state"])

        return step


    def reset(self):

        assert self.vec_env is not None, "Please call create_envs first."

        # initialize agent
        self.states = self.vec_env.reset()
        self.episode_score *= 0
        self.episode_len *= 0

    def run_random_agent(self, iterations):
        """
        Runs agent through environment
        :param env_name:
        :param model:
        :param log:
        :return:
        """
        self.log.info("Warming up model with random agent...")

        # collect experience
        self.reset()

        for iteration in range(iterations):
            self.generate_rollout(is_warmup=True)

    def generate_rollout(self, is_warmup=False):

        assert self.vec_env is not None, "Please call create_envs first."

        for t in range(self.N):

            prev_states = self.states.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.model.forward(self.states)

            log_policy = model_out["log_policy"].detach().cpu().numpy()
            value_ext = model_out["ext_value"].detach().cpu().numpy()

            # during warm-up we simply collect experience through a uniform random policy.
            if is_warmup:
                actions = np.random.randint(0, self.model.actions, size=[self.A], dtype=np.int32)
            else:
                # sample actions and run through environment.
                actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)

            # perform the agents action.
            if args.use_atn:
                log_policy_atn = model_out["atn_log_policy"].detach().cpu().numpy()
                value_atn = model_out["atn_value"].detach().cpu().numpy()
                actions_atn = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy_atn], dtype=np.int32)
                self.states, rewards_ext, dones, infos = self.vec_env.step(list(zip(actions, actions_atn)))

                attention_cost = np.asarray([infos[i].get("attention_cost", 0) * -args.atn_movement_cost for i in range(args.agents)])
                self.log.watch_mean("atn_cost", attention_cost.mean())
                attention_rewards = rewards_ext + attention_cost

                self.atn_actions[t] = actions_atn
                self.atn_value[t] = value_atn
                self.atn_rewards[t] = attention_rewards
                self.atn_log_policy[t] = log_policy_atn
            else:
                self.states, rewards_ext, dones, infos = self.vec_env.step(actions)

            # it's a bit silly to have this here...
            if "returns_norm_state" in infos[0]:
                atari.ENV_STATE["returns_norm_state"] = infos[0]["returns_norm_state"]

            # work out our intrinsic rewards
            if args.use_intrinsic_rewards:
                value_int = model_out["int_value"].detach().cpu().numpy()
                rewards_int = np.zeros_like(rewards_ext)
                if is_warmup:
                    # in random mode just update the normalization constants
                    self.model.perform_normalization(self.states)
                else:
                    # reward is prediction error on state we land inn.
                    loss_rnd = self.model.prediction_error(self.states).detach().cpu().numpy()
                    rewards_int += loss_rnd
                self.int_rewards[t] = rewards_int
                self.int_value[t] = value_int

            # save raw rewards for monitoring the agents progress
            raw_rewards = np.asarray([info.get("raw_reward", rewards_ext) for reward, info in zip(rewards_ext, infos)],
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
            self.next_state[t] = self.states
            self.actions[t] = actions

            self.rewards_ext[t] = rewards_ext
            self.log_policy[t] = log_policy
            self.terminals[t] = dones
            self.value_ext[t] = value_ext

        #  save a copy of the normalization statistics.
        if args.use_rnd:
            atari.ENV_STATE["observation_norm_state"] = self.model.obs_rms.save_state()

        # get value estimates for final state.
        model_out = self.model.forward(self.states)
        self.final_value_estimate_ext = model_out["ext_value"].detach().cpu().numpy()
        if "int_value" in model_out:
            self.final_value_estimate_int = model_out["int_value"].detach().cpu().numpy()
        if "atn_value" in model_out:
            self.final_value_estimate_atn = model_out["atn_value"].detach().cpu().numpy()

    def calculate_returns(self):

        self.ext_returns = calculate_returns(self.rewards_ext, self.terminals, self.final_value_estimate_ext, args.gamma)
        self.advantage_ext = calculate_gae(self.rewards_ext, self.value_ext, self.final_value_estimate_ext,
                                           self.terminals, args.gamma, args.normalize_advantages)

        if args.use_intrinsic_rewards:
            # calculate the returns, but let returns propagate through terminal states.
            self.int_returns_raw = calculate_returns(self.int_rewards, 0 * self.terminals, self.final_value_estimate_int,
                                                     args.gamma_int)

            # normalize returns using EMS
            for t in range(self.N):
                self.ems_norm = 0.99 * self.ems_norm + self.int_rewards[t, :]
                self.intrinsic_returns_rms.update(self.ems_norm.reshape(-1))

            # normalize the intrinsic rewards
            # we multiply by 0.4 otherwise the intrinsic returns sit around 1.0, and we want them to be more like 0.4,
            # which is approximately where normalized returns will sit.
            self.intrinsic_reward_norm_scale = (1e-5 + self.intrinsic_returns_rms.var ** 0.5)
            self.batch_rewards_int = self.int_rewards / self.intrinsic_reward_norm_scale * 0.4

            # note: we zero all the terminals here so that intrinsic rewards propagate through episodes as per the RND paper.
            self.returns_int = calculate_returns(self.int_rewards, 0 * self.terminals, self.final_value_estimate_int,
                                                 args.gamma_int)
            self.advantage_int = calculate_gae(self.int_rewards, self.int_value, self.final_value_estimate_int, None,
                                               args.gamma_int)

        if args.use_atn:
            self.atn_returns = calculate_returns(self.atn_rewards, self.terminals, self.final_value_estimate_atn,
                                                 args.gamma)
            self.advantage_atn = calculate_gae(self.atn_rewards, self.atn_value, self.final_value_estimate_atn,
                                               self.terminals, args.gamma, args.normalize_advantages)

        self.advantage = args.extrinsic_reward_scale * self.advantage_ext
        if args.use_intrinsic_rewards:
            self.advantage += args.intrinsic_reward_scale * self.advantage_int

        if args.use_rnd:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)


        self.log.watch_mean("adv_mean", np.mean(self.advantage), display_width = 0 if args.normalize_advantages else 10)
        self.log.watch_mean("adv_std", np.std(self.advantage), display_width = 0 if args.normalize_advantages else 10)
        self.log.watch_mean("batch_reward_ext", np.mean(self.rewards_ext), display_name="rew_ext", display_width=0)
        self.log.watch_mean("batch_return_ext", np.mean(self.ext_returns), display_name="ret_ext")
        self.log.watch_mean("batch_return_ext_std", np.std(self.ext_returns), display_name="ret_ext_std", display_width=0)
        self.log.watch_mean("value_est_ext", np.mean(self.value_ext), display_name="est_v_ext")
        self.log.watch_mean("value_est_ext_std", np.std(self.value_ext), display_name="est_v_ext_std", display_width=0)
        self.log.watch_mean("ev_ext", utils.explained_variance(self.value_ext.ravel(), self.ext_returns.ravel()))

        if args.use_intrinsic_rewards:
            self.log.watch_mean("batch_reward_int", np.mean(self.int_rewards), display_name="rew_int", display_width=0)
            self.log.watch_mean("batch_reward_int_std", np.std(self.int_rewards), display_name="rew_int_std",
                                display_width=0)
            self.log.watch_mean("batch_return_int", np.mean(self.returns_int), display_name="ret_int")
            self.log.watch_mean("batch_return_int_std", np.std(self.returns_int), display_name="ret_int_std")
            self.log.watch_mean("batch_return_int_raw_mean", np.mean(self.int_returns_raw), display_name="ret_int_raw_mu",
                                display_width=0)
            self.log.watch_mean("batch_return_int_raw_std", np.std(self.int_returns_raw), display_name="ret_int_raw_std",
                                display_width=0)
            self.log.watch_mean("norm_scale_int", self.intrinsic_reward_norm_scale, display_width=12)
            self.log.watch_mean("value_est_int", np.mean(self.int_value), display_name="est_v_int")
            self.log.watch_mean("value_est_int_std", np.std(self.int_value), display_name="est_v_int_std")
            self.log.watch_mean("ev_int", utils.explained_variance(self.int_value.ravel(), self.returns_int.ravel()))
            self.log.watch_mean("batch_reward_int_unnorm", np.mean(self.int_rewards), display_name="rew_int_unnorm",
                                display_width=10, display_priority=-2)
            self.log.watch_mean("batch_reward_int_unnorm_std", np.std(self.int_rewards), display_name="rew_int_unnorm_std",
                                display_width=0)

    # this needs to be a dict...
    def train_minibatch(self, data):

        loss = torch.tensor(0, dtype=torch.float32, device=self.model.device)

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        prev_states = data["prev_state"]
        actions = data["actions"]
        policy_logprobs = data["log_policy"]
        advantages = data["advantages"]
        mini_batch_size = len(prev_states)

        model_out = self.model.forward(prev_states)
        logps = model_out["log_policy"]

        ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])
        loss_clip = torch.mean(
            torch.min(ratio * advantages, torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon) * advantages))
        self.log.watch_mean("loss_pg", loss_clip)
        loss += loss_clip

        # -------------------------------------------------------------------------
        # Calculate loss_value
        # -------------------------------------------------------------------------

        value_heads = ["ext"]

        if args.use_rnd:
            value_heads.append("int")
        if args.use_atn:
            value_heads.append("atn")

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
                loss_value = -0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))
            else:
                # simpler version, just use MSE.
                vf_losses1 = (value_prediction - returns).pow(2)
                loss_value = -0.5 * torch.mean(vf_losses1)
            loss_value *= args.vf_coef
            self.log.watch_mean("loss_v_" + value_head, loss_value)
            loss += loss_value

        # -------------------------------------------------------------------------
        # Calculate Attention losses
        # -------------------------------------------------------------------------

        if args.use_atn:
            logps_atn = model_out["atn_log_policy"]

            # policy gradient
            atn_actions = data["atn_actions"].long()
            atn_policy_logprobs = data["atn_log_policy"]
            atn_advantages = data["atn_advantages"]
            atn_ratio = torch.exp(logps_atn[range(mini_batch_size), atn_actions] - atn_policy_logprobs[
                range(mini_batch_size), atn_actions])

            loss_clip_atn = torch.mean(torch.min(atn_ratio * atn_advantages,
                                                 torch.clamp(atn_ratio, 1 - args.ppo_epsilon,
                                                             1 + args.ppo_epsilon) * atn_advantages))
            self.log.watch_mean("loss_atn_pg", loss_clip)
            loss += loss_clip_atn

            # entropy
            loss_entropy_atn = args.entropy_bonus * utils.log_entropy(logps_atn) / mini_batch_size
            self.log.watch_mean("loss_ent_atn", loss_entropy_atn)
            loss += loss_entropy_atn

        # -------------------------------------------------------------------------
        # Calculate loss_entropy
        # -------------------------------------------------------------------------

        loss_entropy = args.entropy_bonus * (utils.log_entropy(logps)) / mini_batch_size
        self.log.watch_mean("loss_ent", loss_entropy)
        loss += loss_entropy

        # -------------------------------------------------------------------------
        # Calculate loss_rnd
        # -------------------------------------------------------------------------

        if args.use_rnd:
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

        self.log.watch_mean("loss", loss)

        self.optimizer.zero_grad()
        (-loss).backward()

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

    def train(self):

        # organise our data...
        batch_data = {}
        batch_size = self.N * self.A

        batch_data["prev_state"] = self.prev_state.reshape([batch_size, *self.state_shape])
        batch_data["actions"] = self.actions.reshape(batch_size).astype(np.long)
        batch_data["ext_returns"] = self.ext_returns.reshape(batch_size)
        batch_data["log_policy"] = self.log_policy.reshape([batch_size, *self.policy_shape])
        batch_data["advantages"] = self.advantage.reshape(batch_size)
        batch_data["ext_value"] = self.value_ext.reshape(batch_size)

        if args.use_rnd:
            batch_data["int_returns"] = self.returns_int.reshape(batch_size)
            batch_data["int_value"] = self.int_value.reshape(batch_size)

        if args.use_atn:
            batch_data["atn_actions"] = self.atn_actions.reshape(batch_size)
            batch_data["atn_returns"] = self.atn_returns.reshape(batch_size).astype(np.long)
            batch_data["atn_log_policy"] = self.atn_log_policy.reshape([batch_size, *self.policy_atn_shape])
            batch_data["atn_advantages"] = self.advantage_atn.reshape(batch_size)
            batch_data["atn_value"] = self.atn_value.reshape(batch_size)

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
                    minibatch_data[k] = torch.tensor(v[sample]).to(self.model.device)

                self.train_minibatch(minibatch_data)


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

    N,A = rewards.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_value_estimate

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma * (1.0 - dones[i])

    return returns


def calculate_gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma, normalize=False):
    batch_advantage = np.zeros([args.n_steps, args.agents], dtype=np.float32)
    prev_adv = np.zeros([args.agents], dtype=np.float32)
    for t in reversed(range(args.n_steps)):
        is_next_terminal = batch_terminal[t] if batch_terminal is not None else False # batch_terminal[t] records if t+1 is a terminal state)
        value_next_t = batch_value[t + 1] if t != args.n_steps - 1 else final_value_estimate
        delta = batch_rewards[t] + gamma * value_next_t * (1.0 - is_next_terminal) - batch_value[t]
        batch_advantage[t] = prev_adv = delta + gamma * args.gae_lambda * (
                1.0 - is_next_terminal) * prev_adv
    if normalize:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
    return batch_advantage

def train(env_name, model: models.BaseModel, log:Logger):
    """
    Default parameters from stable baselines
    
    https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html
    
    gamma             0.99
    n_steps            128
    ent_coef          0.01
    learning_rate   2.5e-4
    vf_coef            0.5
    max_grad_norm      0.5
    lam               0.95
    nminibatches         4
    noptepoch            4
    cliprange          0.1 
    
    atari usually requires ~10M steps 
    
    """

    # setup logging
    log.add_variable(LogVariable("ep_score", 100, "stats"))   # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats"))  # time get get first score / length.

    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    # create an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    # create our runner
    runner = Runner(model, optimizer, log)

    # detect a previous experiment
    checkpoints = runner.get_checkpoints(args.log_folder)
    if len(checkpoints) > 0:
        log.info("Previous checkpoint detected.")
        checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
        restored_step = runner.load_checkpoint(checkpoint_path)
        log = runner.log
        log.info("  (resumed from step {:.0f}M)".format(restored_step/1000/1000))
        start_iteration = (restored_step // batch_size) + 1
        walltime = log["walltime"]
        did_restore = True
    else:
        start_iteration = 0
        walltime = 0
        did_restore = False

    runner.create_envs(env_name)

    if not did_restore and args.use_rnd:
        # this will get an initial estimate for the normalization constants.
        runner.run_random_agent(20)

    runner.reset()

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"),"w") as f:
        params = {k:v for k,v in args.__dict__.items()}
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations*batch_size/1000/1000))
    else:
        log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(n_iterations * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    log.info()


    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on
    checkpoints = [x // batch_size for x in range(0, n_iterations*batch_size+1, config.CHECKPOINT_EVERY_STEPS)]
    checkpoints += [x // batch_size for x in [1e6]] #add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    for iteration in range(start_iteration, n_iterations+1):

        step_start_time = time.time()

        env_step = iteration * batch_size

        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precision=2)
        log.watch("walltime", walltime,
                  display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        adjust_learning_rate(optimizer, env_step / 1e6)

        # generate the rollout
        rollout_start_time = time.time()

        runner.generate_rollout()
        rollout_time = (time.time() - rollout_start_time) / batch_size

        # calculate returns
        returns_start_time = time.time()
        runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size

        train_start_time = time.time()
        runner.train()
        train_time = (time.time() - train_start_time) / batch_size

        step_time = (time.time() - step_start_time) / batch_size

        log_start_time = time.time()

        fps = 1.0 / (step_time)

        # record some training stats
        log.watch_mean("fps", int(fps))
        log.watch_mean("time_train", train_time*1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_step", step_time*1000, display_postfix="ms", display_precision=2, display_width=10)
        log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_log", log_time*1000, display_postfix="ms", display_precision=2, display_width=0)

        log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_freq:
            save_progress(log)
            log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug_log_freq:
            utils.lock_job()
            log.export_to_csv()
            log.save_log()
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            log.info()
            log.important("Checkpoint: {}".format(args.log_folder))

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                runner.save_checkpoint(checkpoint_name, env_step)
                log.log("  -checkpoint saved")

            if args.export_video:
                video_name  = utils.get_checkpoint_path(env_step, env_name+".mp4")
                utils.export_movie(video_name, model, env_name)
                log.info("  -video exported")

            log.info()

        log_time = (time.time() - log_start_time) / batch_size

        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += (step_time * batch_size)

    # -------------------------------------
    # save final information

    save_progress(log)
    log.export_to_csv()
    log.save_log()

    log.info()
    log.important("Training Complete.")
    log.info()

