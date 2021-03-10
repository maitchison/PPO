import uuid
import socket
import argparse
import math
import torch
from . import utils
import numpy as np
from typing import List

class Config:

    def __init__(self, **kwargs):
        # put these here just so IDE can detect common parameters...
        self.environment        = str()
        self.experiment_name    = str()
        self.run_name           = str()
        self.agents             = int()
        self.filter             = str()

        self.hash_size          = int()
        self.restore            = bool()

        self.gamma              = float()
        self.gamma_int          = float()
        self.gae_lambda         = float()
        self.ppo_epsilon        = float()
        self.vf_coef            = float()
        self.max_grad_norm      = float()

        self.input_crop         = bool()
        self.learning_rate      = float()
        self.learning_rate_decay = float()
        self.adam_epsilon       = float()
        self.workers            = int()
        self.n_steps            = int()
        self.epochs             = int()
        self.limit_epochs       = int()
        self.batch_epochs       = int()

        self.observation_normalization = bool()
        self.intrinsic_reward_scale = float()
        self.extrinsic_reward_scale = float()

        self.reward_clip        = float()
        self.reward_normalization = bool()

        self.mini_batch_size    = int()
        self.sync_envs          = bool()
        self.resolution         = str()
        self.color              = bool()
        self.entropy_bonus      = float()
        self.threads            = int()
        self.export_video       = bool()
        self.export_trajectories= bool()
        self.device             = str()
        self.save_checkpoints   = bool()
        self.output_folder      = str()
        self.hostname           = str()
        self.sticky_actions     = bool()
        self.guid               = str()

        self.use_tvf            = bool()
        self.tvf_coef           = float()
        self.tvf_max_horizon    = int()
        self.tvf_n_horizons     = int()
        self.tvf_gamma          = float()
        self.tvf_lambda         = float()
        self.tvf_advantage      = bool()
        self.tvf_epsilon        = bool()
        self.tvf_distributional = bool()

        self.time_aware = bool()
        self.ed_type = str()
        self.ed_gamma = float()

        self.use_icm            = bool()
        self.icm_eta            = str()

        self.debug_print_freq   = int()
        self.debug_log_freq     = int()
        self.noop_start         = bool()

        self.frame_stack        = int()

        self.normalize_advantages = bool()

        self.use_clipped_value_loss = bool()

        # emi
        self.use_emi            = bool()

        # population based learning
        self.pbl_population_size = int()
        self.pbl_trust_region    = bool()

        # tdb (trajectory divergece bonus)
        self.use_tdb            = bool()

        self.algo               = str()

        self.log_folder         = str()
        self.checkpoint_every   = int()

        self.arl_c_cost         = float()
        self.arl_i_cost         = float()

        self.model              = str()

        self.use_mppe           = bool()

        # RNN
        self.rnn_block_length   = int()

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def propagate_intrinsic_rewards(self):
        return not self.use_rnd or self.use_mppe

    @property
    def use_intrinsic_rewards(self):
        return self.use_rnd or self.use_emi or self.use_tdb or self.use_mppe

    @property
    def normalize_intrinsic_rewards(self):
        return self.use_rnd or self.use_emi or self.use_mppe

    @property
    def normalize_observations(self):
        return self.use_rnd or self.use_mppe

LOCK_KEY = str(uuid.uuid4().hex)

# debugging variables.
PROFILE_INFO = False
VERBOSE = True

args = Config()

def str2bool(v):
    """
        Convert from string in various formats to boolean.
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():

    parser = argparse.ArgumentParser(description="Trainer for PPO2")

    parser.add_argument("environment")

    parser.add_argument("--experiment_name", type=str, default="Run", help="Name of the experiment.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")

    parser.add_argument("--agents", type=int, default=32)
    parser.add_argument("--algo", type=str, default="ppo", help="Algorithm to use [ppo|pbl|arl]")

    parser.add_argument("--filter", type=str, default="none",
                        help="Add filter to agent observation ['none', 'hash']")
    parser.add_argument("--hash_size", type=int, default=42, help="Adjusts the hash tempalte generator size.")
    parser.add_argument("--restore", type=str2bool, default=False,
                        help="Restores previous model if it exists. If set to false and new run will be started.")

    parser.add_argument("--gamma", type=float, default=0.99, help="Discount rate for extrinsic rewards")
    parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter.")
    parser.add_argument("--ppo_epsilon", type=float, default=0.1, help="PPO epsilon parameter.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function coefficient.")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="Clip gradients during training to this.")

    parser.add_argument("--input_crop", type=str2bool, default=False, help="Enables atari input cropping.")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-5, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--learning_rate_decay", type=float, default=1.0, help="Learning rate is decayed exponentially by this amount per epoch.")
    parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
    parser.add_argument("--n_steps", type=int, default=128, help="Number of environment steps per training step.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Each epoch represents 1 million environment interactions.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
    parser.add_argument("--batch_epochs", type=int, default=4, help="Number of training epochs per training batch.")

    parser.add_argument("--use_tvf", type=str2bool, default=False, help="Use truncated value function.")
    parser.add_argument("--tvf_coef", type=float, default=0.1, help="Loss multiplier for TVF loss.")
    parser.add_argument("--tvf_gamma", type=float, default=0.99, help="Gamma for TVF.")
    parser.add_argument("--tvf_lambda", type=float, default=0.95, help="Lambda for TVF(\lambda).")
    parser.add_argument("--tvf_max_horizon", type=int, default=100, help="Max horizon for TVF.")
    parser.add_argument("--tvf_n_horizons", type=int, default=100, help="Number of horizons to sample during training.")
    parser.add_argument("--tvf_advantage", type=str2bool, default=False, help="Use truncated value function for advantages, and disable model value prediction")
    parser.add_argument("--tvf_epsilon", type=float, default=0.01, help="Smallest STD for error prediction.")
    parser.add_argument("--tvf_distributional", type=str2bool, default=False, help="Enables a gaussian model for returns.")

    parser.add_argument("--observation_normalization", type=str2bool, default=False)
    parser.add_argument("--intrinsic_reward_scale", type=float, default=1)
    parser.add_argument("--extrinsic_reward_scale", type=float, default=1)

    parser.add_argument("--reward_normalization", type=str2bool, default=True)
    parser.add_argument("--reward_clip", type=float, default=5.0)

    parser.add_argument("--mini_batch_size", type=int, default=1024)
    parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                        help="Enables synchronous environments (slower).")
    parser.add_argument("--resolution", type=str, default="standard", help="['full', 'standard', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--export_video", type=str2bool, default=True)
    parser.add_argument("--export_trajectories", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ignore_device", type=str, default="[]", help="Devices to ignore when using auto")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--hostname", type=str, default=socket.gethostname())
    parser.add_argument("--sticky_actions", type=str2bool, default=False)
    parser.add_argument("--guid", type=str, default=None)
    parser.add_argument("--noop_start", type=str2bool, default=True)

    # episodic discounting
    parser.add_argument("--time_aware", type=str2bool, default=False)
    parser.add_argument("--ed_type", type=str, default="none", help="[none|geometric|hyperbolic]")
    parser.add_argument("--ed_gamma", type=float, default=1.0)

    parser.add_argument("--frame_stack", type=int, default=4)

    # RNN
    parser.add_argument("--rnn_block_length", type=int, default=32)

    # EMI
    parser.add_argument("--use_emi", type=str2bool, default=False)
    parser.add_argument("--emi_test_size", type=float, default=256)


    parser.add_argument("--log_folder", type=str, default=None)

    parser.add_argument("--use_clipped_value_loss", type=str2bool, default=True, help="Use the improved clipped value loss.")

    # icm stuff
    parser.add_argument("--use_icm", type=str2bool, default=False, help="Enables the Intrinsic Motivation Module (IDM).")
    parser.add_argument("--icm_eta", type=float, default=0.01,
                        help="How much to weight intrinsic rewards in ICM.")

    parser.add_argument("--use_rnd", type=str2bool, default=False,
                        help="Enables the Random Network Distilation (RND) module.")

    parser.add_argument("--normalize_advantages", type=str2bool, default=True)
    parser.add_argument("--intrinsic_reward_propagation", type=str2bool, default=None,
                        help="allows intrinsic returns to propagate through end of episode."
    )

    parser.add_argument("--arl_c_cost", type=float, default=0.01, help="The cost of the concentration action for agent.")
    parser.add_argument("--arl_i_cost", type=float, default=0.01,
                        help="The cost of the interferance action for anti-agent.")

    # debuging
    parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
    parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")
    parser.add_argument("--checkpoint_every", type=int, default=int(5e6), help="Number of environment steps between checkpoints.")

    # model
    parser.add_argument("--model", type=str, default="cnn", help="['cnn']")
    parser.add_argument("--use_rnn", type=str2bool, default=False)

    #parser.add_argument("--model_hidden_units", type=int, help="Number of hidden units in model.")

    # population stuff
    parser.add_argument("--pbl_population_size", type=int, default=4, help="Number of agents in population.")

    # divergence stuff
    parser.add_argument("--use_tdb", type=str2bool, default=False, help="Trajectory divergence bonus.")

    # these are really just for testing to get v-trace working
    parser.add_argument("--pbl_policy_soften", type=str2bool, default=False)
    parser.add_argument("--pbl_normalize_advantages", type=str, default="None")
    parser.add_argument("--pbl_thinning", type=str, default="None")
    parser.add_argument("--pbl_trust_region", type=str2bool, default=False)

    parser.add_argument("--use_mppe", type=str2bool, default=False, help="Model Prediction Prediction Error")

    args.update(**parser.parse_args().__dict__)

    # set defaults
    if args.intrinsic_reward_propagation is None:
        args.intrinsic_reward_propagation = args.use_rnd or args.use_emi or args.use_mppe

    # check...
    if args.use_tdb:
        raise Exception("TDB is not implemented yet.")

    assert args.tvf_n_horizons <= args.tvf_max_horizon, "tvf_n_horizons must be <= tvf_max_horizon."

