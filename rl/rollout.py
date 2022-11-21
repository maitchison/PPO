import os

import rl

import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F

import time as clock
import json
import gzip
from typing import Union, Optional, List
import math

from .logger import Logger
from . import utils, hybridVecEnv, wrappers, models, compression, config, hash
from . import atari, mujoco, procgen
from .config import args
from .mutex import Mutex
from .replay import ExperienceReplayBuffer

from .returns import gae, calculate_bootstrapped_returns, td_lambda

from .utils import open_checkpoint

import collections


def add_relative_noise(X:np.ndarray, rel_error:float):
    # does not change the expectation.
    if rel_error <= 0:
        return X
    factors = np.asarray(1 - (rel_error / 2) + (np.random.rand(*X.shape) * rel_error), dtype=np.float32)
    return X * factors

def add_10x_noise(X:np.ndarray, p:float):
    # does not change the expectation, rewards are 10x with propbability p, otherwise reduced so that expectation matches.
    factors = torch.bernoulli(torch.ones(X.shape, dtype=torch.float32) * p)
    factors[factors == 1] = 10
    factors[factors == 0] = (1-(10*p))/(1-p)
    return X * factors.numpy()


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


class RunnerModule:
    """
    Used to provide extensions to rollout.
    Modules can access the runner data (e.g. the rollout)
    """

    def __init__(self, parent):
        self.runner: Runner = parent

    def on_reset(self):
        pass

    def on_before_generate_rollout(self):
        pass

    def on_train_value_minibatch(self, loss, model_out, data, **kwargs):
        pass

    def save(self):
        pass

    def load(self):
        pass



class Runner:

    def __init__(self, model: models.TVFModel, log, name="agent", action_dist='discrete'):
        """ Setup our rollout runner. """

        self.name = name
        self.model = model
        self.step = 0

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

        self.policy_optimizer = make_optimizer(model.policy_net.parameters(), args.opt_p)
        self.value_optimizer = make_optimizer(model.value_net.parameters(), args.opt_v)
        if args.opt_d.epochs > 0:
            self.distil_optimizer = make_optimizer(model.policy_net.parameters(), args.opt_d)
        else:
            self.distil_optimizer = None
        if args.opt_a.epochs > 0:
            self.aux_optimizer = make_optimizer(model.parameters(), args.opt_a)
        else:
            self.aux_optimizer = None

        if args.use_rnd:
            self.rnd_optimizer = make_optimizer(model.prediction_net.parameters(), args.opt_r)
        else:
            self.rnd_optimizer = None

        self.vec_env = None
        self.log = log

        self.N = N = args.n_steps
        self.A = A = args.agents
        self.VH = VH = len(self.value_heads)
        if args.tvf.enabled:
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

        if args.env_type == "mujoco":
            obs_type = np.float32
            obs_type_torch = torch.float32
        else:
            obs_type = np.uint8
            obs_type_torch = torch.uint8

        self.episode_score = np.zeros([A], dtype=np.float32)
        self.discounted_episode_score = np.zeros([A], dtype=np.float32)
        self.episode_len = np.zeros([A], dtype=np.int32)

        self.obs = np.zeros([A, *self.state_shape], dtype=obs_type)
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
                all_obs = torch.zeros(size=[N + 1, A, *self.state_shape], dtype=obs_type_torch)
                all_obs = all_obs.pin_memory()
                self.all_obs = all_obs.numpy()
            else:
                self.all_obs = np.zeros([N + 1, A, *self.state_shape], dtype=obs_type)

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

        self.ttt_predictions = [[] for _ in range(self.N)]  # our prediction of when a termination will occur.

        self.replay_value_estimates = np.zeros([N, A], dtype=np.float32) # what is this?

        # intrinsic rewards
        self.int_rewards = np.zeros([N, A], dtype=np.float32)

        self.intrinsic_reward_norm_scale: float = 1

        # value and returns
        self.value = np.zeros([N+1, A, VH], dtype=np.float32)
        self.returns = np.zeros([N, A, VH], dtype=np.float32)


        # hashing
        if args.hash.enabled:
            self.hash_global_counts = np.zeros([2 ** args.hash.bits], dtype=np.float)
            self.hash_recent_counts = np.zeros([2 ** args.hash.bits], dtype=np.float)
            hashers = {
                'linear': hash.LinearStateHasher,
                'conv': hash.ConvStateHasher,
            }
            assert args.hash.method in hashers.keys(), f"Invalid hashing method '{args.hash.method}' ({hashers.keys()})"

            if args.env_type == "atari":
                # for atari, because of frame stacking only use the first state.
                hash_state_shape = list(self.state_shape)
                hash_state_shape[0] = 1
            else:
                hash_state_shape = self.state_shape

            if args.hash.rescale != 1:
                C, H, W = hash_state_shape
                hash_state_shape = (C, H//args.hash.rescale, W//args.hash.rescale)

            self.hash_fn = hashers[args.hash.method](hash_state_shape, args.hash.bits, device=args.device, bias=args.hash.bias)

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
        self.ttt_error_buffer = collections.deque(maxlen=1000)

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

        # modules
        if args.tvf.enabled:
            self.tvf = rl.tvf.TVFRunnerModule(self)

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
        return self.anneal(args.opt_v.lr, mode="linear" if args.opt_v.lr_anneal else "off")

    @property
    def distil_lr(self):
        return self.anneal(args.opt_d.lr, mode="linear" if args.opt_d.lr_anneal else "off")

    @property
    def policy_lr(self):
        return self.anneal(args.opt_p.lr, mode="linear" if args.opt_p.lr_anneal else "off")

    @property
    def ppo_epsilon(self):
        return self.anneal(args.ppo_epsilon, mode="linear" if args.ppo_epsilon_anneal else "off")

    @property
    def rnd_lr(self):
        return args.opt_r.lr

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

        # ema normalization is handled externally.
        if args.reward_normalization == "rms":
            self.vec_env = wrappers.VecNormalizeRewardWrapper(
                self.vec_env,
                gamma=args.reward_normalization_gamma,
                mode="rms",
                clip=args.reward_normalization_clipping,
            )

        if args.max_repeated_actions > 0 and args.env_type != "mujoco":
            self.vec_env = wrappers.VecRepeatedActionPenalty(self.vec_env, args.max_repeated_actions, args.repeated_action_penalty)

        if verbose:
            model_total_size = self.model.model_size(trainable_only=True)/1e6
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
            'episode_score': self.episode_score,
            'discounted_episode_score': self.discounted_episode_score,
            'stats': self.stats,
            'vars': self.vars,
        }

        if args.hash.enabled:
            data['hash_global_counts'] = self.hash_global_counts
            data['hash_recent_counts'] = self.hash_recent_counts

        if not disable_optimizer:
            data['policy_optimizer_state_dict'] = self.policy_optimizer.state_dict()
            data['value_optimizer_state_dict'] = self.value_optimizer.state_dict()
            if args.use_rnd:
                data['rnd_optimizer_state_dict'] = self.rnd_optimizer.state_dict()
            if self.distil_optimizer is not None:
                data['distil_optimizer_state_dict'] = self.distil_optimizer.state_dict()
            if self.aux_optimizer is not None:
                data['aux_optimizer_state_dict'] = self.aux_optimizer.state_dict()

        if not disable_log:
            data['logs'] = self.log
        if not disable_env_state:
            data['env_state'] = utils.save_env_state(self.vec_env)

        if args.sns.enabled:
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

        checkpoint = open_checkpoint(checkpoint_path, map_location=args.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
        if args.use_rnd:
            self.rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer_state_dict'])
        if 'distil_optimizer_state_dict' in checkpoint:
            self.distil_optimizer.load_state_dict(checkpoint['distil_optimizer_state_dict'])
        if args.opt_a.epochs > 0:
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
        self.episode_score = checkpoint['episode_score']
        self.discounted_episode_score = checkpoint['discounted_episode_score']

        if args.hash.enabled:
            self.hash_global_counts = checkpoint['hash_global_counts']
            self.hash_recent_counts = checkpoint['hash_recent_counts']

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

    def get_modules(self) -> List[RunnerModule]:
        result = []
        for child in self.__dict__.values():
            if issubclass(type(child), RunnerModule):
                result.append(child)
        return result

    def reset(self):

        assert self.vec_env is not None, "Please call create_envs first."

        for module in self.get_modules():
            module.on_reset()

        # initialize agent
        self.obs = self.vec_env.reset()
        self.done = np.zeros_like(self.done)
        self.episode_score *= 0
        self.discounted_episode_score *= 0
        self.episode_len *= 0
        self.step = 0
        self.time *= 0
        if args.hash.enabled:
            self.hash_recent_counts *= 0
            self.hash_global_counts *= 0

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
    def detached_batch_forward(self, obs: Union[np.ndarray, torch.Tensor], **kwargs):
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
        assert args.tvf.enabled
        return self.model.tvf_fixed_head_horizons

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

    def get_current_actions_std(self):

        if self.action_dist == "discrete":
            return 0.0
        elif self.action_dist == "gaussian":
            # hard coded for the moment (switch to log scale)
            return torch.exp(self.model.policy_net.log_std)
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
            mu = model_out["raw_policy"].cpu().numpy()
            model_std = self.get_current_actions_std().detach().cpu().numpy()[None, :]
            if not train:
                model_std = 0.0
            return np.random.randn(*mu.shape) * model_std + mu
        else:
            raise ValueError(f"invalid action distribution {self.action_dist}")

    def generate_hashes(self, obs: np.ndarray):
        """
        Applies hash preprocessing, returns hashing for obs
        @param obs np array of dims [A, *state_shape]
        """

        A, C, H, W = obs.shape

        # note this could be much faster by only processing the channel.

        assert self.obs.dtype == np.uint8, "hashin currently requires 8-bit input"

        # give reward for action that lead to a novel state...
        if args.env_type == "atari":
            channel_filter = slice(0, 1)  # just take first channel, might not work for rgb...
        else:
            channel_filter = None  # select all channels.

        if args.hash.quantize:
            obs = ((obs // args.hash.quantize) * args.hash.quantize).astype(np.uint8)

        # process the observations
        if args.hash.input == "raw":
            hash_input = obs[:, channel_filter]
        elif args.hash.input == "raw_centered":
            hash_input = (obs[:, channel_filter].astype(np.float) - 128)
        elif args.hash.input == "normed":
            hash_input = self.model.prep_for_model(obs)
            hash_input = self.model.perform_normalization(hash_input)[:, channel_filter]
        elif args.hash.input == "normed_offset":
            # this should make the cosine distance more stable.
            hash_input = self.model.prep_for_model(obs)
            hash_input = self.model.perform_normalization(hash_input)[:, channel_filter] + 3.0
        else:
            raise ValueError("Invalid hash_input {args.hash_input}")

        # downscale
        if args.hash.rescale:
            import cv2
            if type(hash_input) == torch.Tensor:
                hash_input = hash_input.cpu().numpy()
            new_frames = []
            for a in range(A):
                new_frames.append(cv2.resize(hash_input[a, 0], (H//args.hash.rescale, W//args.hash.rescale), interpolation=cv2.INTER_AREA))
            new_frames = np.asarray(new_frames).reshape([A, 1, H//args.hash.rescale, W//args.hash.rescale])
            hash_input = new_frames

        return self.hash_fn(hash_input)


    @torch.no_grad()
    def generate_rollout(self):

        assert self.vec_env is not None, "Please call create_envs first."

        def upload_if_needed(x):
            if type(self.all_obs) is torch.Tensor:
                x = torch.from_numpy(x).to(self.all_obs.device)
            return x

        self.model.train()

        self.int_rewards *= 0
        self.ext_rewards *= 0
        self.value *= 0
        self.all_time *= 0

        for module in self.get_modules():
            module.on_before_generate_rollout()

        obs_hashes = np.zeros([self.N, self.A], dtype=np.int32)

        rollout_discounted_returns = np.zeros_like(self.ext_rewards)

        for k in self.stats.keys():
            if k.startswith("batch_"):
                self.stats[k] *= 0

        for t in range(self.N):

            prev_obs = self.obs.copy()
            prev_time = self.time.copy()

            # forward state through model, then detach the result and convert to numpy.
            model_out = self.detached_batch_forward(
                self.obs,
                output="full",
                include_rnd=args.use_rnd,
                update_normalization=True
            )

            # remap to sensible defaults
            model_out['value'] = model_out['value_value']
            model_out['log_policy'] = model_out['policy_log_policy']
            model_out['raw_policy'] = model_out['policy_raw_policy']
            if args.tvf.enabled:
                model_out['tvf_value'] = model_out['value_tvf_value']

            # sample actions and run through environment.
            actions = self.sample_actions(model_out)
            self.obs, ext_rewards, dones, infos = self.vec_env.step(actions)
            self.time = np.asarray([info["time"] for info in infos], dtype=np.int32)

            # hashing if needed...
            if args.hash.enabled:
                hashes = self.generate_hashes(self.obs)
                self.hash_recent_counts *= args.hash.decay
                for a, obs_hash in enumerate(hashes):
                    self.hash_global_counts[obs_hash] += 1
                    self.hash_recent_counts[obs_hash] += 1
                    obs_hashes[t, a] = obs_hash

            if args.use_rnd:
                # update the intrinsic rewards
                self.int_rewards[t] += model_out["rnd_error"].detach().cpu().numpy()

            # save raw rewards for monitoring the agents progress
            raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(ext_rewards, infos)],
                                     dtype=np.float32)

            if args.noisy_zero >= 0:
                ext_rewards = np.random.normal(0, args.noisy_zero, size=ext_rewards.shape).astype(np.float32)
                raw_rewards *= 0

            self.episode_score += raw_rewards
            self.discounted_episode_score = args.gamma * self.discounted_episode_score + ext_rewards
            rollout_discounted_returns[t] = self.discounted_episode_score
            self.episode_len += 1

            # per step reward noise
            # (should be after discounted return...)
            if args.noisy_reward > 0:
                ext_rewards = add_relative_noise(ext_rewards, args.noisy_reward)
            if args.noisy_reward_v2 > 0:
                ext_rewards = add_10x_noise(ext_rewards, args.noisy_reward_v2)

            # log repeated action stats
            if 'max_repeats' in infos[0]:
                self.log.watch_mean('max_repeats', infos[0]['max_repeats'], display_name="reps", display_width=7)
            if 'mean_repeats' in infos[0]:
                self.log.watch_mean('mean_repeats', infos[0]['mean_repeats'], display_width=0)

            # compress observations if needed
            if args.obs_compression:
                prev_obs = np.asarray([compression.BufferSlot(prev_obs[i]) for i in range(len(prev_obs))])

            # take advantage of the fact that V_h = V_min(h, remaining_time).
            if args.tvf.enabled:
                start_time = clock.time()
                tvf_values = model_out["tvf_value"].cpu().numpy()
                self.tvf.tvf_value[t], ttt = self.tvf.trim_horizons(
                    tvf_values,
                    prev_time,
                    method=args.tvf.trimming,
                    mode=args.tvf.trimming_mode
                )
                if ttt is not None:
                    for a in range(self.A):
                        self.ttt_predictions[a].append(ttt[a])

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
                if "room_count" in info:
                    self.log.watch_mean("av_room_count", info["room_count"], history_length=100,
                                        display_name="rooms_av")

                if done:
                    # this should be always updated, even if it's just a loss of life terminal
                    self.episode_length_buffer.append(info["ep_length"])

                    if "fake_done" in info:
                        # this is a fake reset on loss of life...
                        continue

                    predictions = self.ttt_predictions[i]

                    # check how good our ttt predictions were
                    deltas = []
                    for j, pred_ttt in enumerate(predictions):
                        true_ttt = len(predictions) - j
                        delta = pred_ttt - true_ttt
                        self.ttt_error_buffer.append(delta)
                        deltas.append(delta)

                    predictions.clear()

                    # reset is handled automatically by vectorized environments
                    # so just need to keep track of book keeping
                    self.ep_count += 1
                    self.log.watch_full("ep_score", info["ep_score"], history_length=100)
                    self.log.watch_full("ep_length", info["ep_length"], history_length=100)
                    if "room_count" in info:
                        self.log.watch_mean("ep_room_count", info["room_count"], history_length=100,
                                            display_name="rooms_ep")
                        try:
                            old_room_count = self.log['max_room_count']
                        except:
                            old_room_count = 0
                        self.log.watch("*max_room_count", max(old_room_count, info["room_count"]))
                    self.log.watch_mean("ep_count", self.ep_count, history_length=1)

                    self.episode_score[i] = 0
                    self.episode_len[i] = 0
                    self.discounted_episode_score[i] = 0

        # process the final state
        if args.obs_compression:
            last_obs = np.asarray([compression.BufferSlot(self.obs[i]) for i in range(len(self.obs))])
        else:
            last_obs = self.obs
        self.all_obs[-1] = upload_if_needed(last_obs)
        self.all_time[-1] = self.time
        final_model_out = self.detached_batch_forward(self.obs, output="default")
        self.value[-1] = final_model_out["value"].cpu().numpy()

        if args.tvf.enabled:
            self.tvf.tvf_value[-1], ttt = self.tvf.trim_horizons(
                final_model_out["tvf_value"].cpu().numpy(),
                self.time,
                method=args.tvf.trimming,
                mode=args.tvf.trimming_mode
            )
            if ttt is not None:
                for a in range(self.A):
                    self.ttt_predictions[a].append(ttt[a])

        # -----------------------------------------------
        # give hashing bonus
        # note: could make this much faster

        def get_bonus(hashes: np.ndarray, x: int, threshold=None):
            if args.hash.bonus_method == "hyperbolic":
                return 1 / hashes[x]
            elif args.hash.bonus_method == "quadratic":
                return 1 / (hashes[x] ** 2)
            elif args.hash.bonus_method == "binary":
                return 1 if hashes[x] < threshold else -1
            else:
                raise ValueError(f"Invalid hash_bonus_method {args.hash.bonus_method}")

        if args.hash.enabled:
            # calculate threshold
            def calc_threshold(counts: np.ndarray):
                x = counts.copy()
                x.sort()
                x = np.cumsum(x)
                threshold_idx = np.searchsorted(x, x[-1] / 2)
                delta = x[threshold_idx] - x[threshold_idx - 1]  # not sure if this is right...
                return delta

            hash_threshold = calc_threshold(self.hash_recent_counts)

            if args.hash.bonus != 0:
                for t in range(self.N):
                    for a in range(self.A):
                        obs_hash = obs_hashes[t, a]
                        self.int_rewards[t, a] += args.hash.bonus * get_bonus(self.hash_recent_counts, obs_hash, hash_threshold)

        # turn off train mode (so batch norm doesn't update more than once per example)
        self.model.eval()

        self.int_rewards = np.clip(self.int_rewards, -5, 5) # just in case there are extreme values here

        aux_fields = {}

        # log how well our termination predictiosn are going
        if len(self.ttt_error_buffer) > 0:
            self.log.watch_stats("teb", self.ttt_error_buffer, display_width=0, history_length=1)
            self.log.watch_stats("teba", np.abs(self.ttt_error_buffer), display_width=0, history_length=1)
            self.log.watch_stats("tebz", np.minimum(self.ttt_error_buffer, 0), display_width=0, history_length=1)

        # calculate targets for ppg
        if args.opt_a.epochs > 0:
            v_target = td_lambda(
                self.ext_rewards,
                self.ext_value[:self.N],
                self.ext_value[self.N],
                self.terminals,
                args.gamma,
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

    def log_feature_statistics(self):
        # also log feature statistics
        model_out = self.detached_batch_forward(
            self.prev_obs[0, :],  # just get the first obs from each agent
            output="full",
            include_features=True,
        )

        for key in ["policy", "value"]:
            features = model_out[f"{key}_raw_features"]
            self.log.watch_stats(f"*{key}_raw_features", features)
            features = model_out[f"{key}_features"]
            self.log.watch_stats(f"*{key}_features", features)

    @torch.no_grad()
    def log_dna_value_quality(self):
        targets = calculate_bootstrapped_returns(
            self.ext_rewards, self.terminals, self.ext_value[self.N], args.gamma
        )
        values = self.ext_value[:self.N]
        ev = utils.explained_variance(values.ravel(), targets.ravel())
        self.log.watch_mean("ev_ext", ev, history_length=1)

        # also for int
        if args.use_intrinsic_rewards:
            targets = calculate_bootstrapped_returns(
                self.int_rewards, self.terminals, self.int_value[self.N], args.gamma_int
            )
            values = self.int_value[:self.N]
            ev = utils.explained_variance(values.ravel(), targets.ravel())
            self.log.watch_mean("ev_int", ev, history_length=1)


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

            # work out ratio between average prediction at horizon and average return at horizon
            # should be close to 1.
            self.log.watch_mean(
                f"*vr_ratio_{name}",
                np.mean(value)/(abs(np.mean(target))+1e6),
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
        start_head = 1 if self.tvf_horizons[0] == 0 else 0 # skip first head if it is 0
        heads_to_log = utils.even_sample_down(range(len(self.tvf_horizons[start_head:])), args.sns.max_heads)
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
                self.ext_rewards, self.terminals, self.ext_value[self.N], args.gamma
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


    def calculate_intrinsic_returns(self):

        if not args.use_intrinsic_rewards:
            return 0

        N, A, *state_shape = self.prev_obs.shape

        if args.ir_normalize:
            # normalize returns using EMS
            # this is this how openai did it (i.e. forward rather than backwards)
            for t in range(self.N):
                terminals = (not args.ir_propagation) * self.terminals[t, :]
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

        if args.ir_center:
            self.int_rewards = self.int_rewards - self.int_rewards.mean()

        int_advantage = gae(
            self.int_rewards,
            self.int_value[:N],
            self.int_value[N],
            (not args.ir_propagation) * self.terminals,
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
        self.tvf.tvf_returns *= 0
        N, A, *state_shape = self.prev_obs.shape

        self.model.eval()

        # 1. first we calculate 'ext_value' estimate, which is the primarily value estimate
        if args.tvf.enabled:
            ext_value_estimates = self.tvf.get_tvf_ext_value_estimate(new_gamma=args.gamma)
        else:
            # in this case just use the value networks value estimate
            ext_value_estimates = self.ext_value

        # mostly interested in how noisy these are...
        self.log.watch_mean_std("*ext_value_estimates", ext_value_estimates)

        ext_advantage = gae(
            self.ext_rewards,
            ext_value_estimates[:N],
            ext_value_estimates[N],
            self.terminals,
            args.gamma,
            args.lambda_policy
        )
        # calculate ext_returns.
        self.ext_returns[:] = td_lambda(
            self.ext_rewards,
            ext_value_estimates[:N],
            ext_value_estimates[N],
            self.terminals,
            args.gamma,
            args.lambda_value,
        )

        self.advantage = ext_advantage.copy()
        if args.use_intrinsic_rewards:
            int_advantage = args.ir_scale * self.calculate_intrinsic_returns()
            self.advantage += int_advantage
            self.log.watch_mean_std("*adv_int", int_advantage, display_width=0)
            self.log.watch_mean("adv_ratio", ((ext_advantage**2).mean() / (int_advantage**2).mean())**0.5, display_width=0)
            self.log.watch_mean("*ir_scale", self.intrinsic_reward_norm_scale)

        # tvf
        if args.tvf.enabled:
            # only ext enabled at the moment...
            self.tvf.tvf_returns[..., 0] = self.tvf.calculate_tvf_returns(value_head='ext')

        # logging
        if args.observation_normalization:
            self.log.watch_mean("norm_scale_obs_mean", np.mean(self.model.obs_rms.mean), display_width=0)
            self.log.watch_mean("norm_scale_obs_var", np.mean(self.model.obs_rms.var), display_width=0)

        if args.hash.enabled:
            try:
                old_delta = self.log['hash_states']
            except:
                old_delta = 0
            self.log.watch("hash_states", int(np.count_nonzero(self.hash_global_counts)), display_width=8, display_name="h_states")
            self.log.watch("*hash_delta", int(np.count_nonzero(self.hash_global_counts) - old_delta), display_name="h_delta")
            self.log.watch("*hash_recent", int(np.count_nonzero(self.hash_recent_counts.astype(int))), display_name="h_batch")

        self.log.watch_mean_std("adv_ext", ext_advantage, display_width=0)

        for i, head in enumerate(self.value_heads):
            self.log.watch_mean_std(f"*return_{head}", self.returns[..., i], display_width=0)
            self.log.watch_mean_std(f"value_{head}", self.value[..., i], display_name="v_"+head)
        # self.log.watch_mean_std(f"*return_ext", self.ext_returns)
        # self.log.watch_mean_std(f"value_ext", ext_value_estimates, display_name="ve")

        self.log.watch_mean("reward_scale", self.reward_scale, display_width=0, history_length=1)
        self.log.watch_mean("entropy_bonus", self.current_entropy_bonus, display_width=0, history_length=1)

        for k, v in self.stats.items():
            self.log.watch(k, v, display_width=0)

        self.log.watch("gamma", args.gamma, display_width=0)
        if args.tvf.enabled:
            self.log.watch("tvf_gamma", args.tvf.gamma)
            # just want to know th max horizon std, should be about 3 I guess, but also the max.
            self.log.watch_stats("*tvf_return_ext", self.tvf.tvf_returns[:, :, -1])

        if self.batch_counter % 4 == 0:
            # this can be a little slow, ~2 seconds, compared to ~40 seconds for the rollout generation.
            # so under normal conditions we do it every other update.
            if args.replay_size > 0:
                self.replay_buffer.log_stats(self.log)

        if not args.disable_ev and self.batch_counter % 4 == 3:
            # only about 3% slower with this on.
            if args.tvf.enabled:
                self.log_feature_statistics()
                self.tvf.log_tvf_curve_quality()
            else:
                self.log_feature_statistics()
                self.log_dna_value_quality()

        if args.noisy_return > 0:
            self.returns = add_relative_noise(self.returns, args.noisy_return)
            self.tvf_returns = add_relative_noise(self.tvf.tvf_returns, args.noisy_return)
            self.tvf_returns[:, :, 0] = 0 # by definition...

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

    @property
    def tvf_weights(self):
        """ Returns (loss) weight for each tvf head """
        # these are due to duplication removal.
        base_weights = np.asarray(self.model.tvf_fixed_head_weights, dtype=np.float32).copy()
        return base_weights


    def train_distil_minibatch(self, data, loss_scale=1.0, **kwargs):

        # todo: make sure heads all line up... I think they might be offset sometimes. Perhaps make sure that
        # we always pass in all heads, and maybe just generate them all the time aswell?

        if 'context' in data:
            extra_debugging = data['context']['is_first'] and data['context']['epoch'] == 0
        else:
            extra_debugging = False

        if args.tvf.enabled and not args.distil.force_ext:
            # the following is used to only apply distil to every nth head, which can be useful as multi value head involves
            # learning a very complex function. We go backwards so that the final head is always included.
            head_sample = utils.even_sample_down(np.arange(len(self.tvf_horizons)), max_values=args.distil.max_heads)
        else:
            head_sample = None

        # weights due to duplicate head removal
        if args.tvf.enabled and not args.distil.force_ext:
            head_filter = head_sample if head_sample is not None else slice(None, None)
            weights = torch.from_numpy(self.tvf_weights[None, head_filter]).to(self.model.device)
        else:
            weights = 1.0

        model_out = self.model.forward(
            data["prev_state"],
            output="policy",
            exclude_tvf=not args.tvf.enabled or args.distil.force_ext,
            required_tvf_heads=head_sample,
        )

        targets = data["distil_targets"] # targets are [B or B, K]

        if args.tvf.enabled and not args.distil.force_ext:
            predictions = model_out["tvf_value"][:, :, 0] # [B, K, VH] -> [B, K]
            if head_sample is not None:
                targets = targets[:, head_sample]
        else:
            if args.distil.target == "value":
                predictions = model_out["value"][:, 0]
            else:
                actions = data["distil_actions"]
                predictions = model_out["advantage"][range(len(actions)), actions]

        if args.distil.value.loss == "mse":
            loss_value = 0.5 * torch.square(targets - predictions) # [B, K]
        elif args.distil.value_loss == "l1":
            # l1 will be much higher (if errors are less than 1)
            loss_value = args.distil.l1_scale * torch.abs(targets - predictions)  # [B, K]
        elif args.distil.value_loss == "clipped_mse":
            loss_value = torch.square(torch.clip(targets - predictions, -1, 1))  # [B, K]
        elif args.distil.value_loss == "huber":
            if args.distil.delta == 0:
                loss_value = torch.abs(targets - predictions)
            else:
                loss_value = torch.nn.functional.huber_loss(targets, predictions, reduction='none', delta=args.distil.delta)
        else:
            raise ValueError(f"Invalid loss distil loss {args.distil.loss}")

        if extra_debugging:
            # first distil update
            self.log.watch("*fd_mse", torch.square(targets - predictions).mean())
            self.log.watch("*fd_bias", torch.mean(targets - predictions))
            self.log.watch("*fd_max", abs(torch.max(targets - predictions)))
            self.log.watch("*fd_ratio", torch.mean(targets)/torch.mean(predictions))
            self.log.watch("*fd_loss", loss_value.mean())

        # apply discount reweighing
        loss_value = loss_value * weights

        # normalize the loss
        # this is required as return magnitude can differ by a factor of 10x or 0.1x,
        # which can happen if we apply different discounts to the environment. This makes
        # beta hard to tune.

        if len(loss_value.shape) == 2:
            loss_value = loss_value.mean(axis=-1) # mean across final dim if targets / predictions were vector.
        loss = loss_value

        # note: mse on logits is a bad idea. The reason is we might get logits of -40 for settings where a policy
        # must be determanistic. The reality is there isn't much difference between exp(-40) and exp(-30) so don't do
        # mse on it.

        if args.env_type == "mujoco":
            # we are basically calculating the KL here, ignoring the constant term.
            # note: this might get very large when std gets very small... so we add a bias term
            # see https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
            epsilon = 1e-5
            delta = torch.square(data["old_raw_policy"] - model_out["raw_policy"]) / (
                        epsilon + 2 * self.get_current_actions_std().detach() ** 2)
            loss_policy = args.distil.beta * 0.5 * delta.mean(dim=-1)
            loss = loss + loss_policy
        else:
            if args.distil.loss == "mse_logit":
                loss_policy = args.distil.beta * 0.5 * torch.square(data["old_raw_policy"] - model_out["raw_policy"]).mean(dim=-1)
            elif args.distil.loss == "mse_policy":
                loss_policy = args.distil.beta * 0.5 * torch.square(data["old_log_policy"] - model_out["log_policy"]).mean(dim=-1)
            elif args.distil.loss == "kl_policy":
                loss_policy = args.distil.beta * F.kl_div(data["old_log_policy"], model_out["log_policy"], log_target=True, reduction="none").sum(dim=-1)
            else:
                raise ValueError(f"Invalid distil_loss {args.distil.loss}")

        loss = loss + loss_policy

        pred_var = torch.var(predictions*weights)
        targ_var = torch.var(targets*weights)

        # some debugging stats
        with torch.no_grad():
            self.log.watch_mean("distil_targ_var", targ_var, history_length=64 * args.opt_d.epochs, display_width=0)
            self.log.watch_mean("distil_pred_var", pred_var, history_length=64 * args.opt_d.epochs,
                                display_width=0)
            delta = (predictions - targets) * weights
            mse = torch.square(delta).mean()
            ev = 1 - torch.var(delta) / (torch.var(targets * weights) + 1e-8)
            self.log.watch_mean("distil_ev", ev, history_length=64 * args.opt_d.epochs,
                                display_name="ev_dist",
                                display_width=8)
            self.log.watch_mean("distil_mse", mse, history_length=64 * args.opt_d.epochs,
                                display_width=0)

        # check model sparsity
        def log_model_sparsity(model, label):
            # quick check out weight sparsity is correct
            total_edges = np.prod(model.tvf_head.weight.data.shape)
            active_edges = torch.ge(model.tvf_head.weight.data.abs(), 1e-6).sum().detach().cpu().numpy()
            self.log.watch(f"*ae_{label}", active_edges / total_edges)

        if args.tvf.enabled:
            log_model_sparsity(self.model.value_net, "value")
            log_model_sparsity(self.model.policy_net, "policy")

        # -------------------------------------------------------------------------
        # Generate Gradient
        # -------------------------------------------------------------------------

        loss = loss * loss_scale
        loss.mean().backward()

        self.log.watch_mean("loss_distil_policy", loss_policy.mean(), history_length=64 * args.opt_d.epochs, display_width=0)
        self.log.watch_mean("loss_distil_value", loss_value.mean(), history_length=64 * args.opt_d.epochs, display_width=0)
        self.log.watch_mean("loss_distil", loss.mean(), history_length=64*args.opt_d.epochs, display_name="ls_distil", display_width=8)

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
        history_length = 2 * args.opt_a.epochs * args.distil.batch_size // args.opt_d.mini_batch_size

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

        loss = torch.zeros(size=[B], dtype=torch.float32, device=self.model.device, requires_grad=True)

        if "returns" in data:
            loss = loss + self.train_value_heads(model_out, data)

        for module in self.get_modules():
            context = {
                'single_value_head': single_value_head,
                'required_tvf_heads': required_tvf_heads
            }
            module.on_train_value_minibatch(loss, model_out, data, **context)

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

        result = {}

        mini_batch_size = len(data["prev_state"])

        prev_states = data["prev_state"]
        old_log_pac = data["log_pac"]
        advantages = data["advantages"]

        model_out = self.model.forward(prev_states, output="policy", exclude_tvf=True)

        gain = torch.scalar_tensor(0, dtype=torch.float32, device=prev_states.device)

        # -------------------------------------------------------------------------
        # Calculate loss_pg
        # -------------------------------------------------------------------------

        if self.action_dist == "discrete":
            actions = data["actions"].to(torch.long)
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
            actions = data["actions"].to(torch.float32)
            mu = model_out["raw_policy"]
            logpac = torch.distributions.normal.Normal(mu, self.get_current_actions_std()).log_prob(actions)
            ratio = torch.exp(logpac - old_log_pac)

            clip_frac = torch.gt(torch.abs(ratio - 1.0), self.ppo_epsilon).float().mean()
            clipped_ratio = torch.clamp(ratio, 1 - self.ppo_epsilon, 1 + self.ppo_epsilon)

            loss_clip = torch.min(ratio * advantages[:, None], clipped_ratio * advantages[:, None])
            gain = gain + loss_clip.mean(dim=-1) # mean over actions..

            # no entropy bonus... ?

            # todo kl for gaussian
            kl_approx = torch.zeros(1)
            kl_true = torch.zeros(1)

            for i, std in enumerate(self.get_current_actions_std()):
                self.log.watch_mean(f'astd_{i}', std)

        else:
            raise ValueError(f"Invalid action distribution type {self.action_dist}")

        # -------------------------------------------------------------------------
        # Global KL
        # -------------------------------------------------------------------------

        # the idea here is to get an estimate for E_{s ~ \mu, \pi} KL(\pi^{new}(s), \pi^{old}(s)

        if args.gkl.enabled:
            old_global_log_policy = data["*global_log_policy"]
            global_states = data["*global_states"]
            global_model_out = self.model.forward(global_states, output="policy", exclude_tvf=True)
            new_global_log_policy = global_model_out["log_policy"]
            global_kl = F.kl_div(
                utils.merge_down(old_global_log_policy), utils.merge_down(new_global_log_policy),
                reduction="batchmean", log_target=True
            )

            if args.gkl.penalty != 0:
                gkl_loss = global_kl * args.gkl.penalty
                gain = gain - gkl_loss
                self.log.watch_mean("*loss_gkl", gkl_loss, history_length=64 * args.opt_p.epochs, display_name=f"ls_gkl", display_width=8)

            result["global_kl"] = global_kl.detach().cpu()

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

        self.log.watch_mean("loss_pg", loss_clip.mean(), history_length=64*args.opt_p.epochs, display_name=f"ls_pg", display_width=8)
        self.log.watch_mean("clip_frac", clip_frac, display_width=8, display_name="clip")
        self.log.watch_mean("loss_policy", gain.mean(), display_name=f"ls_policy")

        result.update({
            'losses': loss.detach(),
            'kl_approx': float(kl_approx.detach()),  # make sure we don't pass the graph through.
            'kl_true': float(kl_true.detach()),
            'clip_frac': float(clip_frac.detach()),
        })

        return result

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
    def reward_scale(self):
        """ The amount rewards have been multiplied by. """
        if args.noisy_zero > 0:
            # no reward scaling for noisy zero rewards.
            return 1.0
        elif args.reward_normalization == "rms":
            norm_wrapper = wrappers.get_wrapper(self.vec_env, wrappers.VecNormalizeRewardWrapper)
            return 1.0 / norm_wrapper.std
        elif args.reward_normalization == "off":
            return 1.0
        else:
            raise ValueError(f"Invalid reward normalization {args.reward_normalization}")

    def train_rnd_minibatch(self, data, loss_scale: float = 1.0, **kwargs):

        # -------------------------------------------------------------------------
        # Random network distillation update
        # -------------------------------------------------------------------------
        # note: we include this here so that it can be used with PPO. In practice, it does not matter if the
        # policy network or the value network learns this, as the parameters for the prediction model are
        # separate anyway.

        loss_rnd = self.model.rnd_prediction_error(data["prev_state"]).mean()
        self.log.watch_mean("loss_rnd", loss_rnd)

        self.log.watch_mean("*feat_mean", self.model.rnd_features_mean)
        self.log.watch_mean("*feat_var", self.model.rnd_features_var)
        self.log.watch_mean("*feat_max", self.model.rnd_features_max, display_precision=1)

        loss = loss_rnd * loss_scale
        loss.backward()


    def train_rnd(self):

        batch_data = {}
        B = args.batch_size
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([B, *state_shape])[:round(B*args.rnd_experience_proportion)]

        for epoch in range(args.opt_r.epochs):
            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_rnd_minibatch,
                mini_batch_size=args.opt_r.mini_batch_size,
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

        if args.opt_p.epochs == 0:
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
            mu = self.raw_policy
            batch_data["actions"] = self.actions.reshape(B, self.model.actions).astype(np.float32)
            batch_data["log_pac"] = torch.distributions.normal.Normal(
                torch.from_numpy(mu),
                self.get_current_actions_std().detach().cpu()
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

        # get global kl states (if needed)
        if args.gkl.enabled:
            assert args.gkl.source == "rollout", "Only rollout source supported at the moment"

            global_states = utils.merge_down(self.prev_obs)
            global_states = global_states[np.random.choice(len(global_states), args.gkl.samples, replace=False)]
            batch_data["*global_states"] = global_states.clone()

            model_out = self.detached_batch_forward(
                obs=global_states,
                output="policy",
            )
            batch_data["*global_log_policy"] = model_out["log_policy"].detach()

        epochs = 0
        for epoch in range(args.opt_p.epochs):
            results = self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_policy_minibatch,
                mini_batch_size=args.opt_p.mini_batch_size,
                optimizer=self.policy_optimizer,
                label="policy",
                epoch=epoch,
            )
            expected_mini_batches = (args.batch_size / args.opt_p.mini_batch_size)
            epochs += results["mini_batches"] / expected_mini_batches
            if "did_break" in results:
                break

        self.log.watch(f"time_train_policy", (clock.time() - start_time),
                       display_width=6, display_name='t_pol', display_precision=3)


    def train_value(self):

        # ----------------------------------------------------
        # value phase

        start_time = clock.time()

        if args.opt_v.epochs == 0:
            return

        batch_data = {}
        N, A, *state_shape = self.prev_obs.shape

        batch_data["prev_state"] = self.prev_obs.reshape([N*A, *state_shape])

        if not args.tvf.enabled or args.tvf.include_ext:
            # these are not really needed, maybe they provide better features, I don't know.
            # one issue is that they will be the wrong scale if rediscounting is applied.
            # e.g. if gamma defaults to 0.99997, but these are calculated at 0.999 they might be extremely large
            batch_data["returns"] = self.returns.reshape(N*A, self.VH)

        if args.tvf.enabled:
            # just train ext heads for the moment
            batch_data["tvf_returns"] = self.tvf_returns[:, :, :, -1].reshape(N*A, self.K)

            # per horizon noise estimates
            # note: it's about 2x faster to generate accumulated noise all at one go, but this means
            # the generic code for noise estimation no longer works well.
            if rl.sns.wants_noise_estimate(self, 'value_heads') and args.sns.max_heads > 0:
                if args.upload_batch:
                    self.upload_batch(batch_data)
                # generate our per-horizon estimates
                rl.sns.log_accumulated_gradient_norms(self, batch_data)
                if args.sns.fake_noise:
                    rl.sns.log_fake_accumulated_gradient_norms(self, optimizer=self.value_optimizer)

        for value_epoch in range(args.opt_v.epochs):
            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_value_minibatch,
                mini_batch_size=args.opt_v.mini_batch_size,
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

    def get_distil_batch(self, samples_wanted: int):
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
            if args.tvf.enabled and not args.distil.force_ext: # tvf_value is [N, A, K, VH]
                assert args.distil.target == "value", "Only value targets supported for TVF distil"
                batch_data["distil_targets"] = utils.merge_down(self.tvf.tvf_value[:self.N, :, :, 0]) # N*A, K
            else:
                if args.distil.target == "value":
                    batch_data["distil_targets"] = utils.merge_down(self.ext_value[:self.N])
                elif args.distil.target in ["return", "advantage"]:
                    # note, we use the value estimates, which are tvf_gamma,
                    # perhaps we want to use policy gamma instead, which would mean
                    # transforming the value estimates. This can be done with rediscounting... I probably won't
                    # bother though.
                    batch_data["distil_actions"] = utils.merge_down(self.actions)
                    advantage_estimate = gae(
                        self.ext_rewards,
                        self.ext_value[:self.N],
                        self.ext_value[self.N],
                        self.terminals,
                        args.tvf.gamma,
                        args.distil.adv_lambda,
                    )
                    if args.distil.target == "return":
                        batch_data["distil_targets"] = utils.merge_down(advantage_estimate + self.ext_value[:self.N])
                    else:
                        batch_data["distil_targets"] = utils.merge_down(advantage_estimate)
                else:
                    raise ValueError(f"Invalid distil target {args.distil.target}")

            if args.distil.order == "before_policy":
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

        # slower path, for when replay is used and we need to regenerate all targets
        assert args.distil.target == "value", "Replay distil required value targets."
        obs, distil_aux = self.get_replay_sample(samples_wanted)

        batch_data = {}

        batch_data["prev_state"] = obs

        # forward through model to get targets from model
        model_out = self.detached_batch_forward(
            obs=obs,
            output="full",
        )

        if args.tvf.enabled and not args.distil.force_ext:
            # we could skip this if we trained on rollout rather then replay
            batch_data["distil_targets"] = model_out["value_tvf_value"][:, :, 0].detach().cpu().numpy()
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

        if args.opt_d.epochs == 0:
            return

        batch_data = self.get_distil_batch(args.distil.batch_size)

        for distil_epoch in range(args.opt_d.epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_distil_minibatch,
                mini_batch_size=args.opt_d.mini_batch_size,
                optimizer=self.distil_optimizer,
                label="distil",
                epoch=distil_epoch,
            )

        self.log.watch(f"time_train_distil", (clock.time() - start_time) / args.distil.period,
                       display_width=6, display_name='t_dis', display_precision=3)

    def train_aux(self):

        # ----------------------------------------------------
        # aux phase
        # borrows a lot of hyperparameters from distil

        start_time = clock.time()

        if args.opt_a.epochs == 0:
            return

        # we could train on terminals, or reward.
        # time would be policy dependant, and is aliased.

        replay_obs, replay_aux = self.get_replay_sample(args.distil.batch_size)
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

        for aux_epoch in range(args.opt_a.epochs):

            self.train_batch(
                batch_data=batch_data,
                mini_batch_func=self.train_aux_minibatch,
                mini_batch_size=args.opt_a.mini_batch_size,
                optimizer=self.aux_optimizer,
                epoch=aux_epoch,
                label="aux",
            )

        self.log.watch(f"time_train_aux", (clock.time() - start_time) * 1000,
                       display_width=8, display_name='t_aux', display_precision=1)

    def wants_distil_update(self, location=None):
        location_match = location is None or location == args.distil.order
        return \
            args.architecture == "dual" and \
            args.opt_d.epochs > 0 and \
            self.batch_counter % args.distil.period == args.distil.period - 1 and \
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

        if args.opt_a.epochs > 0 and (args.aux_period == 0 or self.batch_counter % args.aux_period == args.aux_period - 1):
            self.train_aux()

        if args.use_rnd:
            self.train_rnd()

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

        if epoch == 0 and rl.sns.wants_noise_estimate(self, label): # check noise of first update only
            start_time = clock.time()
            rl.sns.estimate_noise_scale(self, batch_data, mini_batch_func, optimizer, label)
            s = clock.time()-start_time
            self.log.watch_mean(f"sns_time_{label}", s / args.sns.period, display_width=8, display_name=f"t_s{label[:3]}")

        assert "prev_state" in batch_data, "Batches must contain 'prev_state' field of dims (B, *state_shape)"
        batch_size, *state_shape = batch_data["prev_state"].shape

        for k, v in batch_data.items():
            if k.startswith('*'):
                continue
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

                    if var_name.startswith('*'):
                        # we pass these through directly.
                        micro_batch_data[var_name] = var_value.to(self.model.device, non_blocking=True)
                        continue

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