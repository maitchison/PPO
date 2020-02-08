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

import torch.multiprocessing

from . import utils, models, atari, hybridVecEnv, config, logger
from .config import args
from .vtrace import importance_sampling_v_trace, v_trace_trust_region

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
        self.policy_shape = [model.actions]

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)
        self.states = np.zeros([A, *self.state_shape], dtype=np.uint8)

        self.prev_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.next_state = np.zeros([N, A, *self.state_shape], dtype=np.uint8)
        self.actions = np.zeros([N, A], dtype=np.int64)
        self.ext_rewards = np.zeros([N, A], dtype=np.float32)
        self.log_policy = np.zeros([N, A, *self.policy_shape], dtype=np.float32)
        self.terminals = np.zeros([N, A], dtype=np.bool)
        self.ext_value = np.zeros([N, A], dtype=np.float32)

        # rar
        rar_state_space = 2 ** 16
        self.token_shape = (32, 16)  # 32 tokens, of length 16
        self.rar_visited = [set() for _ in range(A)]

        rnd_state = np.random.get_state()
        np.random.seed(args.rar_seed)
        self.rar_reward_states = set(np.random.choice(rar_state_space, int(rar_state_space * args.rar_frequency)))
        np.random.set_state(rnd_state)

        self.rar_reward_tokens = np.zeros([N, A, *self.token_shape], dtype=np.uint8)

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

        # path to load experience from (disables rollouts and training)
        self.experience_path = None

        # outputs tensors when clip loss is very high.
        self.log_high_grad_norm = True

    def create_envs(self):
        """ Creates environments for runner"""
        env_fns = [atari.make for _ in range(args.agents)]

        self.vec_env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=args.workers,
                                                         verbose=True) if not args.sync_envs else gym.vector.SyncVectorEnv(
            env_fns)
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

        self.rar_visited = [set() for _ in range(len(self.rar_visited))]

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

    def forward(self, states=None, tokens=None):
        """ Forward states through model, resturns output, which is a dictionary containing
            "log_policy" etc.
        """
        if args.use_rar and args.rar_use_tokens:
            return self.model.forward(
                self.states if states is None else states,
                self.model.make_tokens(self.rar_visited) if tokens is None else tokens
            )
        else:
            return self.model.forward(self.states if states is None else states)

    def export_movie(self, filename):
        """ Exports a movie of agent playing game.
            which_frames: model, real, or both
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
        video_out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

        state = env.reset()

        rar_visited = set()

        # play the game...
        while not done:
            if args.use_rar:
                model_out = self.model.forward(state[np.newaxis], self.model.make_tokens([rar_visited]))
            else:
                model_out = self.model.forward(state[np.newaxis])
            logprobs = model_out["log_policy"][0].detach().cpu().numpy()
            actions = utils.sample_action_from_logp(logprobs)

            state, reward, done, info = env.step(actions)

            channels = info.get("channels", None)
            rendered_frame = info.get("monitor_obs", state)

            frame = utils.compose_frame(state, rendered_frame, channels)
            if frame.shape[0] != width or frame.shape[1] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

            if args.use_rar:
                mapped_state = self.model.get_mapped_states(state[np.newaxis])[0]
                state_token = self.model.make_tokens([{mapped_state}])[0][0:1, :]
                utils.draw_image(frame, state_token, 0, 0, scale=4)
                if mapped_state in self.rar_reward_states and mapped_state not in rar_visited:
                    utils.draw_pixel(frame, 10, 10, [255, 0, 0], sx=10, sy=10)
                    rar_visited.add(mapped_state)
                visited_tokens = self.model.make_tokens([rar_visited])[0]
                utils.draw_image(frame, visited_tokens[:len(rar_visited), :], 150, 0, scale=4)

            # show current state
            assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
                (width, height, 3), frame.shape)

            video_out.write(frame)

        video_out.release()

    def generate_rollout(self, is_warmup=False):

        if self.experience_path is not None:
            # just load the experience from file.
            # stub:
            print("loading exp...")
            self.load_experience(self.experience_path, "iteration-{}".format(self.log["iteration"]))
            return

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

                if args.use_rar:
                    # convert pixels to a smaller state space
                    states = self.model.get_mapped_states(self.states)

                    # check if these states have aux rewards in them
                    for i, state in enumerate(states):
                        # give reward for being in a reward state.
                        if state in self.rar_reward_states and state not in self.rar_visited[i]:
                            int_rewards[i] += args.rar_scale
                            self.rar_visited[i].add(state)

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

                    if args.use_rar:
                        # random reward  visited needs to be reset to 0 on environment reset
                        self.rar_visited[i] = set()

            self.prev_state[t] = prev_states
            self.next_state[t] = self.states
            self.actions[t] = actions
            if args.use_rar:
                self.rar_reward_tokens[t] = self.model.make_tokens(self.rar_visited)

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

        if self.experience_path is not None:
            return

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

        if args.use_rar:
            model_out = self.forward(prev_states, data["rar_reward_tokens"])
        else:
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

    def load_experience(self, folder, filename):

        input_file = os.path.join(folder, filename + ".bzip")

        with bz2.open(input_file, "rb") as f:
            load_dict = pickle.load(f)

        self.prev_state = load_dict["prev_state"]
        final_state = load_dict["final_state"]
        self.actions = load_dict["actions"]
        self.rewards = load_dict["rewards"]
        self.ext_returns = load_dict["ext_returns"]
        self.log_policy = load_dict["log_policy"]
        self.ext_value = load_dict["ext_value"]
        self.ext_final_value_estimate = load_dict["ext_final_value_estimate"]

        # we don't actually save the next_states, so I recreate it here.
        self.next_state = np.concatenate([self.prev_state[1:], [final_state]])

    def save_experience(self, folder, filename):
        """ Saves all necessary experience for agent. """

        # only saves extrinsic rewards at the moment.

        save_dict = {}
        save_dict["prev_state"] = self.prev_state
        save_dict["final_state"] = self.next_state[-1]
        save_dict["actions"] = self.actions
        save_dict["rewards"] = self.ext_rewards
        save_dict["ext_returns"] = self.ext_returns
        save_dict["log_policy"] = self.log_policy
        save_dict["ext_value"] = self.ext_value
        save_dict["ext_final_value_estimate"] = self.ext_final_value_estimate

        output_file = os.path.join(folder, filename + ".bzip")

        os.makedirs(folder, exist_ok=True)

        # Atari is about 250Mb per iteration, which for 3,000 iterations would be 750GB, which is far too much
        # the video frames do compress very well though, so we apply compression.
        # using bzip gets to < 0.25 Mb, which is 0.75GB per agent trained.

        # I tried gzip, bzip, and lzma, and bzip gave the best compression.
        with bz2.open(output_file, "wb", compresslevel=5) as f:
            pickle.dump(save_dict, f)

    def train_from_off_policy_experience(self, other_agents):
        """ trains agents using experience from given other agents. """

        assert not args.use_intrinsic_rewards, "PBL does not work with intrisic rewards yet."

        # create a super-batch of data from all the agents.

        batch_data = defaultdict(list)
        batch_size = self.N * self.A
        mega_batch_size = self.N * self.A * len(other_agents)

        for agent in other_agents:

            assert agent.N == self.N and agent.A == self.A, "all agents must share the same number of sub-agents, and rollout length."

            batch_data["prev_state"].append(agent.prev_state.reshape([batch_size, *agent.state_shape]))
            batch_data["next_state"].append(agent.next_state.reshape([batch_size, *agent.state_shape]))
            batch_data["actions"].append(agent.actions.reshape(batch_size).astype(np.long))

            # generate our policy probabilities, and value estimates
            target_log_policy = np.zeros_like(agent.log_policy)
            target_value_estimates = np.zeros_like(agent.ext_value)
            for i in range(agent.N):
                # note, this could be mini-batched to make it a bit faster...
                model_out = self.model.forward(agent.prev_state[i])
                target_log_policy[i] = model_out["log_policy"].detach().cpu().numpy()
                target_value_estimates[i] = model_out["ext_value"].detach().cpu().numpy()

            batch_data["ext_value"].append(target_value_estimates.reshape(batch_size))
            batch_data["log_policy"].append(target_log_policy.reshape([batch_size, *agent.policy_shape]))

            # get estimate for last state.
            model_out = self.model.forward(agent.next_state[-1])
            target_value_final_estimate = model_out["ext_value"].detach().cpu().numpy()

            behaviour_log_policy = agent.log_policy
            dones = agent.terminals

            # apply off-policy correction (v-trace)
            actions = agent.actions
            rewards = agent.ext_rewards

            # stub show policy
            # print("-------------")
            # print("Behaviour", np.max(behaviour_policy), np.min(behaviour_policy))
            # print("Target", np.max(target_policy), np.min(target_policy))
            # print("Delta", np.max(target_policy-behaviour_policy), np.min(target_policy-behaviour_policy))

            vs, pg_adv, cs = importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions,
                                                     rewards, dones, target_value_estimates, target_value_final_estimate,
                                                     args.gamma)

            if args.pbl_trust_region:
                weights = v_trace_trust_region(behaviour_log_policy, target_log_policy)
                batch_data["weights"].append(weights.reshape(batch_size))
                for w in weights.ravel():
                    self.log.watch_full("kl_weights", w, history_length=4096)

            # stub monitor some values.
            for x in vs.ravel():
                self.log.watch_full("vs", x, history_length=10000)
            for x in pg_adv.ravel():
                self.log.watch_full("pg_adv", x, history_length=10000)
            for x in cs.ravel():
                self.log.watch_full("cs", x, history_length=10000)

            batch_data["ext_returns"].append(vs.reshape(batch_size))
            batch_data["advantages"].append(pg_adv.reshape(batch_size))

        # make one large super-batch from each agent
        for k, v in batch_data.items():
            batch_data[k] = np.concatenate(v, axis=0)

        # clip probabilities so that all actions have atleast 1e-6 probability (which close to what
        # 32bit float precision will handle anyway.
        # we're assuming here that the v-trace algorithm is more stable, it's the ppo training that seems
        # to be causing the issues.
        if args.pbl_policy_soften:
            # this will denormalize the policy, but I'm ok with that.
            batch_data["log_policy"] = np.clip(batch_data["log_policy"], -13, 0)


        # I think we probably don't want to normalize these as sometimes the advantages would be very small
        # also, if we do normalize it should be the entire mega batch.
        normalization_mode = args.pbl_normalize_advantages.lower()
        if normalization_mode in ["none"]:
            pass
        elif normalization_mode in ["clipped", "full"]:
            mu = batch_data["advantages"].mean()
            sigma = batch_data["advantages"].std()
            if normalization_mode == "clipped":
                sigma = max(0.2, sigma) # make sure we don't inflate the advantages by too much.
            batch_data["advantages"] = (batch_data["advantages"] - mu) / sigma
        else:
            raise Exception("Invalid pbl clipping parameter")


        # thinning mode:
        # hard thins out example early on, so we train multiple times on one reduced set
        # soft thins out while training so all example are used, but less often
        # none trains fully on all.
        thinning_mode = args.pbl_thinning.lower()

        master_set_ordering = list(range(mega_batch_size))
        if thinning_mode == "hard":
            # thin out examples early on
            np.random.shuffle(master_set_ordering)
            master_set_ordering = master_set_ordering[:batch_size]

        for i in range(args.batch_epochs):

            np.random.shuffle(master_set_ordering)

            # thin the data, so we don't over-train
            if thinning_mode == "soft":
                ordering = master_set_ordering[:batch_size]
            else:
                ordering = master_set_ordering[:]

            n_batches = math.ceil(len(ordering) / args.mini_batch_size)

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * args.mini_batch_size
                batch_end = (j + 1) * args.mini_batch_size
                sample = ordering[batch_start:batch_end]

                minibatch_data = {}

                for k, v in batch_data.items():
                    minibatch_data[k] = torch.tensor(v[sample]).to(self.model.device)

                self.train_minibatch(minibatch_data)

    def train(self):
        """ trains agent on it's own experience """

        if self.experience_path is not None:
            return

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

        if args.use_rar:
            batch_data["rar_reward_tokens"] = self.rar_reward_tokens.reshape(batch_size, *self.token_shape)

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
                    minibatch_data[k] = torch.tensor(v[sample]).to(self.model.device)

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


def train_population(ModelConstructor, master_log: Logger):
    """
    Trains a population of models, each with their own set of parameters, and potentially their own set of goals.

    ModelConstructor: Function without required parameters that constructs a model.
    """

    # create a population of models, each with their own parameters.
    models = []
    logs = []
    for i in range(args.pbl_population_size):
        models.append(ModelConstructor())
        log = logger.Logger()

        log.csv_path = os.path.join(args.log_folder, "training_log_{}.csv".format(i))
        log.txt_path = os.path.join(args.log_folder, "log_{}.txt".format(i))

        log.add_variable(LogVariable("ep_score", 100, "stats",
                                     display_width=16))  # these need to be added up-front as it might take some
        log.add_variable(LogVariable("ep_length", 100, "stats", display_width=16))  # time get get first score / length.

        log.add_variable(LogVariable("iteration", 0, type="int"))

        logs.append(log)

    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    # create optimizers
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon) for model in
                  models]

    names = ['agent-'+str(x) for x in range(args.pbl_population_size)]

    # create our runners
    runners = [Runner(model, optimizer, log, name) for model, optimizer, log, name in zip(models, optimizers, logs, names)]

    # todo allow for restoration from checkpoint
    start_iteration = 0
    walltime = 0
    did_restore = False

    # create environments for each agent.
    for runner in runners:
        runner.create_envs()
        runner.reset()

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"), "w") as f:
        params = {k: v for k, v in args.__dict__.items()}
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        master_log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations * batch_size / 1000 / 1000))
    else:
        master_log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(n_iterations * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    master_log.info()

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on
    checkpoints = [x // batch_size for x in range(0, n_iterations * batch_size + 1, args.checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]]  # add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    # setup agents within the population
    if args.pbl_use_experience is not None:
        master_log.important("Using prior experience")
        runners[0].experience_path = os.path.join(args.pbl_use_experience, "agent_experience-0")
        runners[1].experience_path = os.path.join(args.pbl_use_experience, "agent_experience-1")
        # turn off logging for these 'fake' agents.
        logs = []
        for i in range(2, args.pbl_population_size):
            logs.append(runners[i].log)

    # train all models together
    for iteration in range(start_iteration, n_iterations + 1):

        step_start_time = time.time()

        env_step = iteration * batch_size

        master_log.watch("iteration", iteration, display_priority=5)
        master_log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6,
                         display_postfix="M",
                         display_precision=2)
        master_log.watch("walltime", walltime,
                         display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        # move some variables from master log to the individual logs
        for log in logs:
            for var_name in ["iteration", "env_step", "walltime"]:
                log.watch(var_name, master_log[var_name])

        for optimizer in optimizers:
            adjust_learning_rate(optimizer, env_step / 1e6)

        # generate the rollout
        rollout_start_time = time.time()
        for runner in runners:
            runner.generate_rollout()
        rollout_time = (time.time() - rollout_start_time) / batch_size / len(runners)

        # calculate returns
        returns_start_time = time.time()
        for runner in runners:
            runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size / len(runners)

        # train our population...
        # for the moment agent 0, and 1 is on-policy and all others are mixed off-policy.
        train_start_time = time.time()
        assert len(runners) in [4], "Only population sizes of 4 are supported at the moment."

        # we train all these 'off-policy' just to make sure v-trace works on policy.
        runners[0].train_from_off_policy_experience([runners[0]]) # this is just to make sure off policy works as expected.
        runners[1].train_from_off_policy_experience([runners[1],runners[1]]) # this is just to make sure off policy works as expected with multiple agents
        runners[2].train_from_off_policy_experience([runners[0], runners[1]])
        runners[3].train_from_off_policy_experience([runners[0], runners[1], runners[3]])

        train_time = (time.time() - train_start_time) / batch_size / len(runners)

        step_time = (time.time() - step_start_time) / batch_size / len(runners)

        log_start_time = time.time()

        fps = 1.0 / (step_time)

        # save experience
        if args.pbl_save_experience:
            for i in range(2):
                runners[i].save_experience(
                    os.path.join(args.log_folder, "agent_experience-{}".format(i)),
                    "iteration-{}".format(iteration)
                )

        # record some training stats
        master_log.watch_mean("fps", int(fps))
        master_log.watch_mean("time_train", train_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_step", step_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        master_log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_log", log_time * 1000, type="float", display_postfix="ms", display_precision=2,
                              display_width=0)

        master_log.aggretate_logs(logs, ignore=["iteration", "env_step", "walltime"])
        master_log.record_step()

        for log in logs:
            log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_freq:
            save_progress(master_log)

            # stub: and print all agents except first one...
            if args.algo=="pbl":
                for i in range(args.pbl_population_size):
                    runners[i].log.print_variables(include_header= i == 0)
            else:
                master_log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug_log_freq:
            utils.lock_job()
            master_log.export_to_csv()
            master_log.save_log()
            for log in logs:
                log.export_to_csv()
                log.save_log()
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            master_log.info()
            master_log.important("Checkpoint: {}".format(args.log_folder))

            if args.save_checkpoints:
                for i, runner in enumerate(runners):
                    checkpoint_name = utils.get_checkpoint_path(env_step, "params_{}.pt".format(i))
                    runner.save_checkpoint(checkpoint_name, env_step)
                master_log.log("  -checkpoints saved")

            if args.export_video:

                for i, runner in enumerate(runners):
                    video_name = utils.get_checkpoint_path(env_step, "{}-{}.mp4".format(args.environment, i))
                    runner.export_movie(video_name)
                master_log.info("  -video exported")

            master_log.info()

        log_time = (time.time() - log_start_time) / batch_size

        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += (step_time * batch_size)

    # -------------------------------------
    # save final information

    save_progress(master_log)
    master_log.export_to_csv()
    master_log.save_log()

    master_log.info()
    master_log.important("Training Complete.")
    master_log.info()


def train(model: models.BaseModel, log: Logger):
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
    log.add_variable(LogVariable("ep_score", 100, "stats",
                                 display_width=16))  # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats", display_width=16))  # time get get first score / length.

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
        log.info("  (resumed from step {:.0f}M)".format(restored_step / 1000 / 1000))
        start_iteration = (restored_step // batch_size) + 1
        walltime = log["walltime"]
        did_restore = True
    else:
        start_iteration = 0
        walltime = 0
        did_restore = False

    runner.create_envs()

    if not did_restore and args.use_rnd:
        # this will get an initial estimate for the normalization constants.
        runner.run_random_agent(20)

    runner.reset()

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"), "w") as f:
        params = {k: v for k, v in args.__dict__.items()}
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations * batch_size / 1000 / 1000))
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
    checkpoints = [x // batch_size for x in range(0, n_iterations * batch_size + 1, args.checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]]  # add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    for iteration in range(start_iteration, n_iterations + 1):

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
        log.watch_mean("time_train", train_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_step", step_time * 1000, display_postfix="ms", display_precision=2, display_width=10)
        log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_log", log_time * 1000, type="float", display_postfix="ms", display_precision=2,
                       display_width=0)

        log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_freq:
            save_progress(log)
            log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

            # export debug frames
            if args.use_emi:
                try:
                    with torch.no_grad():
                        img = model.generate_debug_image(runner.emi_prev_state, runner.emi_actions,
                                                         runner.emi_next_state)
                    os.makedirs(os.path.join(args.log_folder, "emi"), exist_ok=True)
                    torchvision.utils.save_image(img, os.path.join(args.log_folder, "emi",
                                                                   "fdm-{:04d}K.png".format(env_step // 1000)))
                except Exception as e:
                    log.warn(str(e))

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
                video_name = utils.get_checkpoint_path(env_step, args.environment + ".mp4")
                runner.export_movie(video_name)
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
