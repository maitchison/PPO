import os
import numpy as np
import gym
import torch
import torchvision
import torch.nn as nn
import time
import json
import math
import cv2
import csv
import pickle
import gzip, bz2, lzma
from collections import defaultdict

from .logger import Logger, LogVariable
from . import utils, models, atari, hybridVecEnv, config, logger
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

# todo: make this td(\lambda) style...)
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

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma * (1.0 - dones[i])

    return returns


def calculate_gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma, lamb=1.0, normalize=False):

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
        self.rnn_state_shape = [512]
        self.policy_shape = [model.actions]


        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.states = np.zeros([A, *self.state_shape], dtype=np.uint8)
        if args.use_rnn:
            self.rnn_states = np.zeros([A, *self.rnn_state_shape], dtype=np.uint8)
            self.prev_rnn_state = np.zeros([N, A, *self.rnn_state_shape], dtype=np.uint8)

        self.prev_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.next_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.actions = np.zeros([N, A], dtype=np.int64)
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool)
        self.ext_value = np.zeros([N, A], dtype=np.float32)

        # emi
        self.emi_prev_state = np.zeros([args.emi_test_size, *self.state_shape], dtype=np.uint8)
        self.emi_next_state = np.zeros([args.emi_test_size, *self.state_shape], dtype=np.uint8)
        self.emi_actions = np.zeros([args.emi_test_size], dtype=np.int64)

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
            self.ems_norm = checkpoint['ems_norm']
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

    def forward(self, states=None, rnn_states=None):
        """ Forward states through model, returns output, which is a dictionary containing
            "log_policy" etc.
        """
        if args.use_rnn:
            return self.model.forward(
                self.states if states is None else states,
                self.rnn_states if rnn_states is None else rnn_states
            )
        else:
            return self.model.forward(self.states if states is None else states)

    def export_movie(self, filename, include_rollout=False, max_frames = 30*60*15):
        """ Exports a movie of agent playing game.
            include_rollout: save a copy of the rollout (may as well include policy, actions, value etc)
        """

        scale = 2

        env = atari.make()
        _ = env.reset()
        state, reward, done, info = env.step(0)
        rendered_frame = info.get("monitor_obs", state)

        # work out our height
        first_frame = utils.compose_frame(state, rendered_frame)
        height, width, channels = first_frame.shape
        width = (width * scale) // 4 * 4  # make sure these are multiples of 4
        height = (height * scale) // 4 * 4

        # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
        video_out = cv2.VideoWriter(filename+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

        state = env.reset()

        frame_count = 0

        history = defaultdict(list)

        # todo: support rnn
        if args.use_rnn:
            print("Warning! RNN not supported for video export yet, results may not be correct.")

        # play the game...
        while not done:

            model_out = self.model.forward(state[np.newaxis])

            logprobs = model_out["log_policy"][0].detach().cpu().numpy()
            action = utils.sample_action_from_logp(logprobs)

            if include_rollout:
                history["logprobs"].append(logprobs)
                history["actions"].append(action)
                history["states"].append(state)

            if args.algo == "arl":
                arl_model_out = self.arl_model.forward(state[np.newaxis])
                arl_logprobs = arl_model_out["log_policy"][0].detach().cpu().numpy()
                arl_action = utils.sample_action_from_logp(arl_logprobs)

                # decode the actions
                concentrate = bool(action % 2)
                noop = (arl_action == 0)

                action = action // 2 if (concentrate or noop) else arl_action - 1

            state, reward, done, info = env.step(action)

            channels = info.get("channels", None)
            rendered_frame = info.get("monitor_obs", state)

            frame = utils.compose_frame(state, rendered_frame, channels)
            if frame.shape[0] != width or frame.shape[1] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            if args.algo == "arl":
                # show control
                col = None
                if concentrate:
                    col = [0, 0, 255]
                if not noop:
                    col = [255, 0, 0]
                if (not noop) and concentrate:
                    col = [255, 255, 0]
                if col is not None:
                    utils.draw_pixel(frame, 10, 10, col, sx=10, sy=10)

            # show current state
            assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
                (width, height, 3), frame.shape)

            video_out.write(frame)

            frame_count += 1

            if frame_count >= max_frames:
                break

        video_out.release()

        if include_rollout:
            for k, v in history.items():
                history[k] = np.asarray(v)
            pickle.dump(history, gzip.open(filename+".hst.gz", "wb", compresslevel=5))


    def generate_rollout(self, is_warmup=False):

        assert self.vec_env is not None, "Please call create_envs first."

        for t in range(self.N):

            prev_states = self.states.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.forward()

            log_policy = model_out["log_policy"].detach().cpu().numpy()
            ext_value = model_out["ext_value"].detach().cpu().numpy()

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
                self.log.watch("returns_norm_mu", norm_mean)
                self.log.watch("returns_norm_std", norm_var**0.5)

            # work out our intrinsic rewards
            if args.use_intrinsic_rewards:
                value_int = model_out["int_value"].detach().cpu().numpy()

                int_rewards = np.zeros_like(ext_rewards)

                if args.use_emi:
                    # relu makes sure that only positive rewards are counted.
                    # otherwise intrinsic reward may become negative causing agent to terminate the episode as quickly
                    # as possible.
                    int_rewards = torch.nn.functional.relu(
                        self.model.predict_model_improvement(self.states)).detach().cpu().numpy()

                if args.use_rnd:
                    if is_warmup:
                        # in random mode just update the normalization constants
                        self.model.perform_normalization(self.states)
                    else:
                        # reward is prediction error on state we land inn.
                        loss_rnd = self.model.prediction_error(self.states).detach().cpu().numpy()
                        int_rewards += loss_rnd

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
            self.next_state[t] = self.states
            self.actions[t] = actions

            self.ext_rewards[t] = ext_rewards
            self.log_policy[t] = log_policy
            self.terminals[t] = dones
            self.ext_value[t] = ext_value

        #  save a copy of the normalization statistics.
        if args.use_rnd:
            atari.ENV_STATE["observation_norm_state"] = self.model.obs_rms.save_state()

        # get value estimates for final state.
        model_out = self.forward()

        self.ext_final_value_estimate = model_out["ext_value"].detach().cpu().numpy()
        if "int_value" in model_out:
            self.int_final_value_estimate = model_out["int_value"].detach().cpu().numpy()

    def calculate_returns(self):

        self.ext_returns = calculate_returns(self.ext_rewards, self.terminals, self.ext_final_value_estimate,
                                             args.gamma)

        self.ext_advantage = calculate_gae(self.ext_rewards, self.ext_value, self.ext_final_value_estimate,
                                           self.terminals, args.gamma, args.gae_lambda, args.normalize_advantages)

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
                self.batch_int_rewards = self.int_rewards / self.intrinsic_reward_norm_scale * 0.4
            else:
                self.intrinsic_reward_norm_scale = 1
                self.batch_int_rewards = self.int_rewards

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

        if args.use_rnd:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        self.log.watch_mean("adv_mean", np.mean(self.advantage), display_width=0 if args.normalize_advantages else 10)
        self.log.watch_mean("adv_std", np.std(self.advantage), display_width=0 if args.normalize_advantages else 10)
        self.log.watch_mean("adv_max", np.max(self.advantage), display_width=10 if args.normalize_advantages else 0)
        self.log.watch_mean("adv_min", np.min(self.advantage), display_width=10 if args.normalize_advantages else 0)
        self.log.watch_mean("batch_reward_ext", np.mean(self.ext_rewards), display_name="rew_ext", display_width=0)
        self.log.watch_mean("batch_return_ext", np.mean(self.ext_returns), display_name="ret_ext")
        self.log.watch_mean("batch_return_ext_std", np.std(self.ext_returns), display_name="ret_ext_std",
                            display_width=0)
        self.log.watch_mean("value_est_ext", np.mean(self.ext_value), display_name="est_v_ext")
        self.log.watch_mean("value_est_ext_std", np.std(self.ext_value), display_name="est_v_ext_std", display_width=0)
        self.log.watch_mean("ev_ext", utils.explained_variance(self.ext_value.ravel(), self.ext_returns.ravel()))

        if args.use_intrinsic_rewards:
            self.log.watch_mean("batch_reward_int", np.mean(self.int_rewards), display_name="rew_int", display_width=0)
            self.log.watch_mean("batch_reward_int_std", np.std(self.int_rewards), display_name="rew_int_std",
                                display_width=0)
            self.log.watch_mean("batch_return_int", np.mean(self.int_returns), display_name="ret_int")
            self.log.watch_mean("batch_return_int_std", np.std(self.int_returns), display_name="ret_int_std")
            self.log.watch_mean("batch_return_int_raw_mean", np.mean(self.int_returns_raw),
                                display_name="ret_int_raw_mu",
                                display_width=0)
            self.log.watch_mean("batch_return_int_raw_std", np.std(self.int_returns_raw),
                                display_name="ret_int_raw_std",
                                display_width=0)

            self.log.watch_mean("value_est_int", np.mean(self.int_value), display_name="est_v_int")
            self.log.watch_mean("value_est_int_std", np.std(self.int_value), display_name="est_v_int_std")
            self.log.watch_mean("ev_int", utils.explained_variance(self.int_value.ravel(), self.int_returns.ravel()))
            if args.use_rnd:
                self.log.watch_mean("batch_reward_int_unnorm", np.mean(self.int_rewards), display_name="rew_int_unnorm",
                                    display_width=10, display_priority=-2)
                self.log.watch_mean("batch_reward_int_unnorm_std", np.std(self.int_rewards),
                                    display_name="rew_int_unnorm_std",
                                    display_width=0)

        if args.normalize_intrinsic_rewards:
            self.log.watch_mean("norm_scale_int", self.intrinsic_reward_norm_scale, display_width=12)

    def train_minibatch(self, data):

        # todo colapse down to mean only at end (and apply weights then)

        loss = torch.tensor(0, dtype=torch.float32, device=self.model.device)

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        prev_states = data["prev_state"]
        actions = data["actions"]
        policy_logprobs = data["log_policy"]
        advantages = data["advantages"]
        weights = data["weights"] if "weights" in data else 1

        mini_batch_size = len(prev_states)

        model_out = self.forward(prev_states)
        logps = model_out["log_policy"]

        ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])
        clipped_ratio = torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon)

        loss_clip = torch.min(ratio * advantages, clipped_ratio * advantages)
        loss_clip_mean = (weights*loss_clip).mean()

        self.log.watch_mean("loss_pg", loss_clip_mean, history_length=64)
        loss += loss_clip_mean

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

        # -------------------------------------------------------------------------
        # -------------------------------------------------------------------------

        # stub force this...
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
            self.log_high_grad_norm = False



        # -------------------------------------------------------------------------
        # Calculate loss_emi
        # -------------------------------------------------------------------------

        if args.use_emi:

            # todo: have this as a separate training function, that does not use the same batching. Just use
            # microbatching (instead of minibatches) and maybe just run for 2 epochs?

            micro_batch_size = 128

            next_states = data["next_state"]

            micro_batches = args.mini_batch_size // micro_batch_size

            with torch.no_grad():
                model_performance = self.model.fdm_error(self.emi_prev_state, self.emi_actions, self.emi_next_state)

            for i in range(micro_batches):
                mb_prev_states = prev_states[i * micro_batch_size:(i + 1) * micro_batch_size]
                mb_next_states = next_states[i * micro_batch_size:(i + 1) * micro_batch_size]
                mb_actions = actions[i * micro_batch_size:(i + 1) * micro_batch_size]

                # perform an update on forward dynamics model
                loss_emi_fdm = self.model.fdm_error(mb_prev_states, mb_actions, mb_next_states)
                self.log.watch_mean("loss_emi_fdm", loss_emi_fdm)
                self.optimizer.zero_grad()
                loss_emi_fdm.backward()
                self.optimizer.step()

                # update estimate of models performance on previous transitions
                with torch.no_grad():
                    new_model_performance = self.model.fdm_error(self.emi_prev_state, self.emi_actions,
                                                                 self.emi_next_state)

                fdm_improvement = (
                                              model_performance - new_model_performance) * 4 * 4 * micro_batches  # epochs*minibatches*microbatches
                self.log.watch_mean("emi_aft", new_model_performance)
                self.log.watch_mean("emi_imp", fdm_improvement)

                # train the model improvement estimator (note this could be done better all at once later on)
                loss_emi_pred = (self.model.predict_model_improvement(mb_next_states) - fdm_improvement).pow(2).mean()
                self.log.watch_mean("loss_emi_pred", loss_emi_pred)
                self.optimizer.zero_grad()
                loss_emi_pred.backward()
                self.optimizer.step()

                model_performance = new_model_performance

    def train(self):
        """ trains agent on it's own experience """

        # organise our data...
        batch_data = {}
        batch_size = self.N * self.A

        batch_data["prev_state"] = self.prev_state.reshape([batch_size, *self.state_shape])
        batch_data["next_state"] = self.next_state.reshape([batch_size, *self.state_shape])
        batch_data["actions"] = self.actions.reshape(batch_size).astype(np.long)
        batch_data["ext_returns"] = self.ext_returns.reshape(batch_size)

        batch_data["log_policy"] = self.log_policy.reshape([batch_size, *self.policy_shape])
        batch_data["advantages"] = self.advantage.reshape(batch_size)
        batch_data["ext_value"] = self.ext_value.reshape(batch_size)

        if args.use_intrinsic_rewards:
            batch_data["int_returns"] = self.int_returns.reshape(batch_size)
            batch_data["int_value"] = self.int_value.reshape(batch_size)

        for i in range(args.batch_epochs):

            if args.refresh_every and i != 0 and i % args.refresh_every == 0:
                assert not args.use_intrinsic_rewards, "Refresh not supported with intrinsic rewards yet."
                # refresh out value estimate and policy
                for t in range(self.N):
                    model_out = self.model.forward(self.prev_state[t])
                    self.log_policy[t] = model_out["log_policy"].detach().cpu().numpy()
                    self.ext_value[t] = model_out["ext_value"].detach().cpu().numpy()

                # get value of final state.
                model_out = self.model.forward(self.next_state[-1])
                self.ext_final_value_estimate = model_out["ext_value"].detach().cpu().numpy()

                # update the advantages
                self.advantage = calculate_gae(self.ext_rewards, self.ext_value, self.ext_final_value_estimate,
                                               self.terminals, args.gamma, args.normalize_advantages)

                # don't update our policy, just advantages and ext_value
                # batch_data["log_policy"] = self.log_policy.reshape([batch_size, *self.policy_shape])
                batch_data["advantages"] = self.advantage.reshape(batch_size)
                batch_data["ext_value"] = self.ext_value.reshape(batch_size)

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
                    minibatch_data[k] = torch.from_numpy(v[sample]).to(self.model.device)

                self.train_minibatch(minibatch_data)

        if args.use_emi:
            # save batch data for next EMI training step.
            ordering = list(range(batch_size))
            np.random.shuffle(ordering)
            ordering = ordering[:args.emi_test_size]
            # saves some time uploading these once here.
            self.emi_prev_state, self.emi_next_state = [
                self.model.prep_for_model(x.reshape([batch_size, *self.state_shape])[ordering])
                for x in [self.prev_state, self.next_state]
            ]
            self.emi_actions = torch.from_numpy(
                self.actions.reshape([batch_size])[ordering]).to(
                device=self.model.device, dtype=torch.int64
            )
