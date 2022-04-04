import ast
import logging
import uuid
import socket
import argparse
import random
from typing import Union

class Config:

    def __init__(self, **kwargs):
        # put these here just so IDE can detect common parameters...
        self.environment: Union[str, list, None] = None
        self.experiment_name    = str()
        self.run_name           = str()
        self.filter             = str()

        self.hash_size          = int()
        self.restore            = bool()

        self.gamma              = float()
        self.gamma_int          = float()
        self.gae_lambda         = float()
        self.gae_value_multiplier = float()
        self.ppo_epsilon        = float()
        self.vf_coef            = float()
        self.max_grad_norm      = float()

        self.input_crop         = bool()
        self.adam_epsilon       = float()
        self.workers            = int()
        self.epochs             = float()
        self.limit_epochs       = int()
        self.distil_beta        = float()
        self.distil_mode        = str()
        self.distil_period      = int()
        self.distil_freq_ratio  = float()
        self.distil_batch_size_ratio = float()
        self.replay_size        = int()
        self.distil_batch_size  = int()
        self.replay_mixing      = bool()
        self.replay_thinning = float()

        # value logging
        self.log_detailed_value_quality = bool()
        self.dvq_freq = int()
        self.dvq_samples = int()
        self.dvq_rollout_length = int()

        # critical batch_size
        self.abs_mode = str()
        self.save_model_interval = bool()

        # replay constraint
        self.policy_replay_constraint = float()
        self.value_replay_constraint = float()
        self.policy_replay_constraint_anneal = str()
        self.value_replay_constraint_anneal = str()

        self.quite_mode         = bool()

        self.observation_normalization = bool()
        self.freeze_observation_normalization = bool()
        self.ir_scale = float()
        self.er_scale = float()

        self.reward_normalization = bool()
        self.reward_scale       = float()
        self.override_reward_normalization_gamma = float()

        self.sync_envs          = bool()
        self.resolution         = str()
        self.max_repeated_actions = int()
        self.repeated_action_penalty = float()
        self.color              = bool()
        self.entropy_bonus      = float()
        self.threads            = int()
        self.export_video       = bool()
        self.export_trajectories= bool()
        self.device             = str()
        self.upload_batch       = bool()
        self.disable_logging    = bool()
        self.save_checkpoints   = bool()
        self.output_folder      = str()
        self.hostname           = str()
        self.repeat_action_probability = float()
        self.guid               = str()
        self.max_micro_batch_size = int()
        self.policy_mini_batch_size = int()
        self.value_mini_batch_size = int()
        self.distil_mini_batch_size = int()
        self.network = str()
        self.layer_norm = bool()


        self.tvf_coef           = float()
        self.tvf_max_horizon    = int()
        self.auto_horizon   = bool()
        self.auto_gamma     = str()
        self.auto_strategy = str()
        self.tvf_value_samples  = int()
        self.tvf_horizon_samples= int()
        self.tvf_value_distribution = str()
        self.tvf_horizon_distribution = str()
        self.tvf_gae = bool()
        self.tvf_value_scale_fn = str()
        self.tvf_value_scale_norm = str()
        self.tvf_gamma          = float()
        self.return_estimator_mode = str()

        self.tvf_return_samples = int()
        self.tvf_return_mode = str()
        self.tvf_return_n_step = int()
        self.td_lambda = float()
        self.tvf_return_use_log_interpolation = bool()
        self.sqr_return_mode = str()
        self.sqr_return_n_step = int()

        self.hidden_units = int()
        self.tvf_activation = str()
        self.tvf_horizon_scale = str()
        self.tvf_time_scale = str()

        self.tvf_force_ext_value_distil = bool()
        self.tvf_hidden_units = int()
        self.use_tvf = bool()
        self.distil_delay = int()
        self.distil_min_var = float()
        self.distil_var_boost = float()

        # entropy bonus constants
        self.eb_alpha           = float()
        self.eb_beta            = float()
        self.eb_theta           = float()
    
        self.embed_time = bool()
        self.embed_action = bool()
        self.ed_type = str()
        self.ed_gamma = float()

        # phasic
        self.policy_epochs = int()                            
        self.value_epochs = int()
        self.value_thinning = float()
        self.value_resampling = bool()
        self.distil_epochs = int()
        self.aux_epochs = int()
        self.target_kl = float()
        self.entropy_scaling = bool()
        self.ppo_epsilon =float()
        self.agents = int()
        self.n_steps = int()
        self.value_lr = float()
        self.policy_lr = float()
        self.distil_lr = float()
        self.aux_lr = float()
        self.distil_delay = int()
        self.architecture = str()
        self.dna_shared_initialization = bool()
        self.dna_dual_constraint = float()

        self.use_icm            = bool()
        self.icm_eta            = str()

        self.debug_print_freq   = int()
        self.debug_log_freq     = int()
        self.noop_duration      = int()
        self.policy_lr_anneal = bool()
        self.value_lr_anneal = bool()
        self.distil_lr_anneal = bool()
        self.anneal_target_epoch = int()

        self.sa_mu = float()
        self.sa_sigma = float()

        self.deferred_rewards   = int()

        self.frame_stack        = int()
        self.frame_skip         = int()
        self.timeout = int()

        self.normalize_advantages = str()
        self.advantage_clipping = float()
        self.checkpoint_compression = bool()

        self.use_clipped_value_loss = bool()
        self.reward_clipping    = str()

        self.log_folder         = str()
        self.checkpoint_every   = int()
        self.disable_ev         = bool()

        self.use_rnd            = bool()
        self.use_ebd            = bool()
        self.use_erp            = bool()
        self.erp_samples        = int()
        self.erp_reduce         = str()
        self.erp_relu           = bool()
        self.erp_bias           = bool()
        self.erp_source         = str()
        self.warmup_period      = int()
        self.rnd_lr             = float()
        self.rnd_experience_proportion = float()
        self.intrinsic_reward_propagation = bool()
        self.advantage_epsilon = float()
        self.advantage_clipping = float()

        self.per_step_reward_noise = float()
        self.per_step_termination_probability = float()
        self.debug_terminal_logging = bool()
        self.debug_value_logging = bool()
        self.seed = int()
        self.atari_rom_check = bool()

        self.full_action_space = bool()
        self.terminal_on_loss_of_life = bool()
        self.force_restore = bool()
        self.reference_policy = str()

        self.learn_second_moment = bool()

        # ema frame stack
        self.ema_frame_stack_gamma = float()
        self.ema_frame_stack = bool()

        # tvf loss
        self.tvf_loss_fn = str()
        self.tvf_huber_loss_delta = float()
        self.use_tanh_clipping = bool()

        self.use_compression = bool()
        self.mutex_key = str()
        self.description = str()
        self.benchmark_mode = bool()

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

        if type(self.use_compression) is str:

            THRESHOLD = 2*128*128

            if str(self.use_compression).lower() == "auto":
                # always enable when using replay buffer (makes hashing faster, and reduces copy time).
                self.use_compression =\
                    self.batch_size >= THRESHOLD or \
                    self.replay_size >= 0
            else:
                self.use_compression = str2bool(str(self.use_compression))

    def get_env_name(self, n: int=0):
        """
        environment name for the nth environment
        """
        if type(self.environment) is str:
            return self.environment
        if type(self.environment) is list:
            return self.environment[n % len(self.environment)]
        raise ValueError(f"Invalid type for environment {type(self.environment)} expecting str or list.")


    @property
    def reward_normalization_gamma(self):
        gamma = self.tvf_gamma if self.use_tvf else self.gamma
        if self.override_reward_normalization_gamma is not None:
            gamma = self.override_reward_normalization_gamma
        return gamma

    @property
    def use_intrinsic_rewards(self):
        return self.use_rnd or self.use_ebd or self.use_erp

    @property
    def needs_dual_constraint(self):
        return args.dna_dual_constraint != 0 and args.architecture == "dual" and args.distil_epochs > 0

    @property
    def rnd_epochs(self):
        return 1

    @property
    def rnd_mini_batch_size(self):
        return self.value_mini_batch_size

    @property
    def get_mutex_key(self):
        if self.mutex_key.lower() == 'device':
            return args.device
        else:
            return self.mutex_key

    @property
    def full_curve_distil(self):
        return self.use_tvf and not self.tvf_force_ext_value_distil

    @property
    def normalize_intrinsic_rewards(self):
        return self.use_intrinsic_rewards

    @property
    def noop_start(self):
        return self.noop_duration > 0

    @property
    def batch_size(self):
        return self.n_steps * self.agents


LOCK_KEY = str(uuid.uuid4().hex)
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


def parse_args(no_env=False, args_override=None):

    REMOVED_PARAMS = [
        "tvf_soft_anchor", "tvf_exp_mode", "value_transform",
        "tvf_n_dedicated_value_heads",
    ]

    parser = argparse.ArgumentParser(description="Trainer for PPO2")

    if not no_env:
        parser.add_argument("environment", help="Name of environment (e.g. pong) or alternatively a list of environments (e.g.) ['Pong', 'Breakout']")

    parser.add_argument("--experiment_name", type=str, default="Run", help="Name of the experiment.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")

    parser.add_argument("--filter", type=str, default="none",
                        help="Add filter to agent observation ['none', 'hash']")
    parser.add_argument("--hash_size", type=int, default=42, help="Adjusts the hash template generator size.")
    parser.add_argument("--restore", type=str2bool, default=False,
                        help="Restores previous model or raises error. If set to false and new run will be started.")

    parser.add_argument("--reference_policy", type=str, default=None,
                        help="Path to checkpoint to use for a reference policy. In this case policy will not be updated.")


    parser.add_argument("--network", type=str, default="nature", help="Encoder used, [nature|impala]")
    parser.add_argument("--architecture", type=str, default="dual", help="[dual|single]")

    parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter.")
    parser.add_argument("--gae_value_multiplier", type=float, default=1.0, help="Modifies value before going into GAE. Used to see how bad value estimates affect performance.")
    parser.add_argument("--max_grad_norm", type=float, default=20.0, help="Clip gradients during training to this.")

    parser.add_argument("--input_crop", type=str2bool, default=False, help="Enables atari input cropping.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-5, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
    parser.add_argument("--epochs", type=float, default=50.0,
                        help="Each epoch represents 1 million environment interactions.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Loss multiplier for default value loss.")
    parser.add_argument("--timeout", type=int, default=60*60*30, help="Set the timeout for the environment, 0=off, (given in unskipped environment steps)")
    parser.add_argument("--hidden_units", type=int, default=512)

    parser.add_argument("--tvf_gae", type=str2bool, default=False, help="Uses TVF aware GAE (with support for alternative discounts)")
    parser.add_argument("--tvf_value_scale_fn", type=str, default="identity",
                        help="Model predicts value/f(x) instead of value. For example setting f(x) to h predicts average_reward. [identity|linear|log|sqrt]")
    parser.add_argument("--tvf_value_scale_norm", type=str, default="max",
                        help="Return prediction is normed, e.g. when using h model predicts = value/(h/max_h) [none|max|half_max] ")
    parser.add_argument("--tvf_force_ext_value_distil", type=str2bool, default=False)
    parser.add_argument("--tvf_coef", type=float, default=1.0, help="Loss multiplier for TVF loss.")
    parser.add_argument("--tvf_gamma", type=float, default=None, help="Gamma for TVF, defaults to gamma")

    parser.add_argument("--tvf_return_mode", type=str, default="exponential", help="[fixed|adaptive|exponential|geometric]")
    parser.add_argument("--tvf_return_samples", type=int, default=32, help="Number of n-step samples to use for distributional return calculation")
    parser.add_argument("--tvf_return_n_step", type=int, default=80, help="n step to use for tvf_return estimation")
    parser.add_argument("--td_lambda", type=float, default=0.95, help="lambda to use for return estimations when using PPO or DNA")
    parser.add_argument("--tvf_return_use_log_interpolation", type=str2bool, default=False, help="Interpolates in log space.")

    parser.add_argument("--sqr_return_n_step", type=int, default=80, help="n step to use for tvf_return_sqr estimation")
    parser.add_argument("--sqr_return_mode", type=str, default="exponential", help="[fixed|exponential|joint]")

    parser.add_argument("--log_detailed_value_quality", type=str2bool, default=False,
                        help="Enables recording of variance / bias for *all* return estimators durning training. (this is very slow!).")
    parser.add_argument("--dvq_samples", type=int, default=64)
    parser.add_argument("--dvq_freq", type=int, default=64)
    parser.add_argument("--dvq_rollout_length", type=int, default=1024*16)

    parser.add_argument("--tvf_max_horizon", type=int, default=1000, help="Max horizon for TVF.")
    parser.add_argument("--auto_horizon", type=str2bool, default=False, help="Automatically adjust max_horizon to clip(mean episode length + 3std, max(horizon samples, value samples), max_horizon)")
    parser.add_argument("--auto_gamma", type=str, default="off",
                        help="[off|tvf|gamma|both]")
    parser.add_argument("--auto_strategy", type=str, default="episode_length",
                        help="[episode_length|agent_age_slow|sa_return|sa_reward]")

    parser.add_argument("--tvf_value_samples", type=int, default=64, help="Number of values to sample during training.")
    parser.add_argument("--tvf_horizon_samples", type=int, default=64, help="Number of horizons to sample during training. (-1 = all)")
    parser.add_argument("--tvf_value_distribution", type=str, default="fixed_linear", help="Sampling distribution to use when generating value samples.")
    parser.add_argument("--tvf_horizon_distribution", type=str, default="fixed_linear", help="Sampling distribution to use when generating horizon samples.")

    parser.add_argument("--checkpoint_compression", type=str2bool, default=True, help="Enables checkpoint compression.")

    parser.add_argument("--tvf_activation", type=str, default="relu", help="[relu|tanh|sigmoid]")

    parser.add_argument("--abs_mode", type=str, default="off", help="Enables adaptive batch size. [off|on|shadow]")
    parser.add_argument("--save_model_interval", type=int, default=0, help="Period for which to saves model history during training (uses a lot of space!). 0 = off.")

    parser.add_argument("--tvf_horizon_scale", type=str, default="default", help="[default|centered|wide|log|zero]")
    parser.add_argument("--tvf_time_scale", type=str, default="default", help="[default|centered|wide|log|zero]")
    parser.add_argument("--tvf_hidden_units", type=int, default=512, help="units used for value prediction")

    parser.add_argument("--use_tvf", type=str2bool, default=False, help="Enabled TVF mode.")

    # simulated annealing
    parser.add_argument("--sa_mu", type=float, default=0.0)
    parser.add_argument("--sa_sigma", type=float, default=0.05)

    # second moment
    parser.add_argument("--learn_second_moment", type=str2bool, default=False, help="Learns the second moment of the return.")

    # phasic inspired stuff
    parser.add_argument("--policy_epochs", type=int, default=3, help="Number of policy training epochs per training batch.")
    parser.add_argument("--value_epochs", type=int, default=2, help="Number of value training epochs per training batch.")
    parser.add_argument("--value_thinning", type=float, default=1.0, help="Uses this proportion of the rollout for value learning. Can be used to obtain less than one epoch of value update.")
    parser.add_argument("--value_resampling", type=str2bool, default=False, help="Recalculates value estimates each epoch.")

    # optional aux phase
    parser.add_argument("--aux_epochs", type=int, default=0, help="Number of auxiliary epochs")

    # distil / replay
    parser.add_argument("--distil_epochs", type=int, default=0, help="Number of distillation epochs")
    parser.add_argument("--distil_beta", type=float, default=1.0)
    parser.add_argument("--distil_period", type=int, default=1)
    parser.add_argument("--distil_mode", type=str, default="value",
                        help="[value|features|projection]")
    parser.add_argument("--distil_batch_size", type=int, default=None, help="Size of batch to use when training distil. Defaults to rollout_size.")

    parser.add_argument("--distil_freq_ratio", type=float, default=None, help="Sets distil period to replay_size / batch_size * distil_freq_ratio")
    parser.add_argument("--distil_batch_size_ratio", type=float, default=None,
                        help="Sets distil_batch_size to rollout_size * distil_batch_size_ratio")

    parser.add_argument("--replay_mode", type=str, default="overwrite", help="[overwrite|sequential|uniform|off]")
    parser.add_argument("--replay_size", type=int, default=0, help="Size of replay buffer. 0=off.")
    parser.add_argument("--replay_mixing", type=str2bool, default=False)
    parser.add_argument("--replay_thinning", type=float, default=1.0, help="Adds this fraction of experience to replay buffer.")
    parser.add_argument("--policy_replay_constraint", type=float, default=0.0,
                        help="How much to constrain policy on historical data when making updates.")
    parser.add_argument("--value_replay_constraint", type=float, default=0.0,
                        help="How much to constrain value on historical data when making updates.")
    parser.add_argument("--value_replay_constraint_anneal", type=str, default="off",
                        help="[off|linear|cos|cos_linear]")
    parser.add_argument("--policy_replay_constraint_anneal", type=str, default="off",
                        help="[off|linear|cos|cos_linear]")

    parser.add_argument("--distil_delay", type=int, default=0, help="Number of steps to wait before starting distillation")
    parser.add_argument("--distil_min_var", type=float, default=0.0,
                        help="If the variance of the value networks value estimates are less than this distil will not run.")
    parser.add_argument("--distil_var_boost", type=float, default=0.0,
                        help="Variance based bonus for distillation.")

    parser.add_argument("--dna_shared_initialization", type=str2bool, default=False,
                        help="Policy and value network start with same weight initialization")
    parser.add_argument("--dna_dual_constraint", type=float, default=0,
                        help="Policy updates are constrained by value prediction.")

    parser.add_argument("--entropy_scaling", type=str2bool, default=False,
                        help="Scales entropy bonus by 1/|std(adv)|.")
    parser.add_argument("--target_kl", type=float, default=-1, help="Approximate divergence before early stopping on policy.")
    parser.add_argument("--policy_mini_batch_size", type=int, default=2048)
    parser.add_argument("--value_mini_batch_size", type=int, default=256)
    parser.add_argument("--distil_mini_batch_size", type=int, default=256)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon parameter.")
    parser.add_argument("--n_steps", type=int, default=256, help="Number of environment steps per training step.")
    parser.add_argument("--agents", type=int, default=256)
    parser.add_argument("--warmup_period", type=int, default=250, help="Number of random steps to take before training agent.")

    parser.add_argument("--value_lr", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--policy_lr", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--distil_lr", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--aux_lr", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--rnd_lr", type=float, default=2.5e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--advantage_epsilon", type=float, default=1e-8, help="Epsilon used when normalizing advantages.")
    parser.add_argument("--advantage_clipping", type=float, default=None,
                        help="Advantages will be clipped to this, (after normalization)")


    # experimental...
    parser.add_argument("--tvf_loss_fn", type=str, default="MSE", help="[MSE|huber|h_weighted]")
    parser.add_argument("--tvf_huber_loss_delta", type=float, default=1.0)
    parser.add_argument("--use_tanh_clipping", type=str2bool, default=False)

    parser.add_argument("--policy_lr_anneal", type=str2bool, nargs='?', const=True, default=False,
                        help="Anneals learning rate to 0 (linearly) over training")
    parser.add_argument("--value_lr_anneal", type=str2bool, nargs='?', const=True, default=False,
                        help="Anneals learning rate to 0 (linearly) over training")
    parser.add_argument("--distil_lr_anneal", type=str2bool, nargs='?', const=True, default=False,
                        help="Anneals learning rate to 0 (linearly) over training")

    parser.add_argument("--anneal_target_epoch", type=float, default=None, help="Epoch to anneal to zero by")

    # -----------------

    parser.add_argument("--gamma", type=float, default=0.999, help="Discount rate for extrinsic rewards")

    parser.add_argument("--observation_normalization", type=str2bool, default=False)
    parser.add_argument("--freeze_observation_normalization", type=str2bool, default=False, help="Disables updates to observation normalization constants.")
    parser.add_argument("--er_scale", type=float, default=1.0, help="Extrinsic reward scale.")
    parser.add_argument("--ir_scale", type=float, default=0.3, help="Intrinsic reward scale.")
    parser.add_argument("--ir_anneal", type=str, default="off",
                        help="Anneals intrinsic rewards over training. [off|linear|cos|cos_linear]")

    parser.add_argument("--max_micro_batch_size", type=int, default=512)
    parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                        help="Enables synchronous environments (slower).")
    parser.add_argument("--resolution", type=str, default="nature", help="['full', 'nature', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--max_repeated_actions", type=int, default=100, help="Agent is given a penalty if it repeats the same action more than this many times.")
    parser.add_argument("--repeated_action_penalty", type=float, default=0.0,
                        help="Penalty if agent repeats the same action more than this many times.")
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--export_video", type=str2bool, default=True)
    parser.add_argument("--export_trajectories", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--upload_batch", type=str2bool, default=False, help='Uploads an entire batch to GPU, faster, but uses more GPU RAM.')
    parser.add_argument("--disable_logging", type=str2bool, default=False,
                        help='Useful when profiling.')

    parser.add_argument("--ignore_device", type=str, default="[]", help="Devices to ignore when using auto")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--hostname", type=str, default=socket.gethostname())
    parser.add_argument("--repeat_action_probability", type=float, default=0.0)
    parser.add_argument("--guid", type=str, default=None)
    parser.add_argument("--noop_duration", type=int, default=30, help="maximum number of no-ops to add on reset")
    parser.add_argument("--per_step_reward_noise", type=float, default=0.0, help="Standard deviation of noise added to (normalized) reward each step.")
    parser.add_argument("--per_step_termination_probability", type=float, default=0.0,
                        help="Probability that each step will result in unexpected termination (used to add noise to value).")
    parser.add_argument("--reward_clipping", type=str, default="off", help="[off|[<R>]|sqrt]")
    parser.add_argument("--reward_normalization", type=str2bool, default=True)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--deferred_rewards", type=int, default=0,
                        help="If positive, all rewards accumulated so far will be given at time step deferred_rewards, then no reward afterwards.")
    parser.add_argument("--use_compression", type=str, default='auto',
                        help="Use LZ4 compression on states (around 20x smaller), but is 10% slower")
    parser.add_argument("--override_reward_normalization_gamma", type=float, default=None)

    parser.add_argument("--eb_alpha", type=float, default=0.0)
    parser.add_argument("--eb_beta", type=float, default=0.0)
    parser.add_argument("--eb_theta", type=float, default=1.0)

    parser.add_argument("--ema_frame_stack_gamma", type=float, default=2.0)
    parser.add_argument("--ema_frame_stack", type=str2bool, default=False)

    parser.add_argument("--atari_rom_check", type=str2bool, default=True,
                        help="Verifies on load, that the MD5 of atari ROM matches the ALE.")

    # episodic discounting
    parser.add_argument("--embed_time", type=str2bool, default=True, help="Encodes time into observation")
    parser.add_argument("--embed_action", type=str2bool, default=True, help="Encodes actions into observation")
    parser.add_argument("--ed_type", type=str, default="none", help="[none|finite|geometric|quadratic|power|harmonic]")
    parser.add_argument("--ed_gamma", type=float, default=0.99)
    parser.add_argument("--full_action_space", type=str2bool, default=False)
    parser.add_argument("--terminal_on_loss_of_life", type=str2bool, default=False)
    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--frame_skip", type=int, default=4)
    parser.add_argument("--log_folder", type=str, default=None)
    parser.add_argument("--use_clipped_value_loss", type=str2bool, default=False, help="Use the improved clipped value loss.")

    # icm stuff
    parser.add_argument("--use_icm", type=str2bool, default=False, help="Enables the Intrinsic Motivation Module (IDM).")
    parser.add_argument("--icm_eta", type=float, default=0.01,
                        help="How much to weight intrinsic rewards in ICM.")

    parser.add_argument("--use_rnd", type=str2bool, default=False,
                        help="Enables the Random Network Distillation (RND) module.")
    parser.add_argument("--rnd_experience_proportion", type=float, default=0.25)

    parser.add_argument("--use_ebd", type=str2bool, default=False,
                        help="Enables the exploration by disagreement reward.")
    parser.add_argument("--use_erp", type=str2bool, default=False,
                        help="Enables the exploration by replay diversity reward.")
    parser.add_argument("--erp_source", type=str, default="replay",
                        help="[replay|rollout|both]")
    parser.add_argument("--erp_reduce", type=str, default="min",
                        help="reduce function for exploration by replay diversity [mean|min|top5]")
    parser.add_argument("--erp_relu", type=str2bool, default=True)
    parser.add_argument("--erp_bias", type=str, default="internal", help="[centered|none|internal]")

    parser.add_argument("--normalize_advantages", type=str, default="norm")
    parser.add_argument("--intrinsic_reward_propagation", type=str2bool, default=None,
                        help="allows intrinsic returns to propagate through end of episode."
    )
    parser.add_argument("--erp_samples", type=int, default=512,
                        help="Number of samples to use for exploration by replay diversity density estimator")

    parser.add_argument("--disable_ev", type=str2bool, default=False,
                        help="disables explained variance calculations (faster)."
                        )

    # debugging
    parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
    parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")
    parser.add_argument("--debug_terminal_logging", type=str2bool, default=False,
                        help="Log information around terminals.")
    parser.add_argument("--debug_value_logging", type=str2bool, default=False,
                        help="Log information around terminals.")
    parser.add_argument("--checkpoint_every", type=int, default=int(5e6),
                        help="Number of environment steps between checkpoints.")
    parser.add_argument("--quiet_mode", type=str2bool, default=False)
    parser.add_argument("--return_estimator_mode", type=str, default="default",
                        help='Allows the use of the reference return estimator (very slow). [default|reference|verify]'
                        )

    # other
    parser.add_argument("--mutex_key", type=str, default='',
                        help="uses mutex locking so that only one GPU can be working on a rollout at a time. " +
                             "(use DEVICE) to set automatically to current device."
                        )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--description", type=str, default=None, help="Can be used as needed. (logged in params.txt)")
    parser.add_argument("--layer_norm", type=str2bool, default=False)
    parser.add_argument("--benchmark_mode", type=str2bool, default=False, help="Enables benchmarking mode.")

    # legacy
    # parser.add_argument("--time_aware", type=str2bool, default=None, help=argparse.SUPPRESS)
    # parser.add_argument("--sticky_actions", type=str2bool, default=None, help=argparse.SUPPRESS)
    # parser.add_argument("--tvf_exp_gamma", type=float, default=None, help=argparse.SUPPRESS)
    # parser.add_argument("--tvf_mode", type=str, default=None, help=argparse.SUPPRESS)
    # parser.add_argument("--tvf_n_step", type=int, default=None, help=argparse.SUPPRESS)

    for param in REMOVED_PARAMS:
        parser.add_argument(f"--{param}", type=str, default=None, help=argparse.SUPPRESS)

    cmd_args = parser.parse_args(args_override).__dict__
    args.update(**cmd_args)

    # conversions
    try:
        # handle environment as an array.
        args.environment = ast.literal_eval(args.environment)
    except:
        # just assume this is a normal unformatted string.
        args.environment = args.environment

    # checks
    if args.reference_policy is not None:
        assert args.architecture == "dual", "Reference policy loading requires a dual network."
    assert not (args.use_rnd and not args.observation_normalization), "RND requires observation normalization"
    assert not (args.color and args.observation_normalization), "Observation normalization averages over channels, so " \
                                                               "best to not use it with color at the moment."

    assert not (args.use_ebd and not args.architecture == "dual"), "EBD requires dual architecture"
    assert not (args.erp_source == "both" and args.replay_size == 0), "erp_source=both requires a replay buffer"

    assert args.abs_mode in ["off", "on", "shadow"]
    assert args.return_estimator_mode in ["default", "reference", "verify"]
    if args.log_detailed_value_quality:
        assert args.learn_second_moment, "Logging requires second moment to be enabled."

    # set defaults
    if args.intrinsic_reward_propagation is None:
        # this seems key to getting intrinsic motivation to work
        # without it the agent might never want to die (as it can gain int_reward forever).
        # maybe this is correct behaviour? Not sure.
        args.intrinsic_reward_propagation = (args.use_rnd or args.use_ebd or args.use_erp)
    if args.tvf_gamma is None:
        args.tvf_gamma = args.gamma
    if args.distil_batch_size is None:
        args.distil_batch_size = args.batch_size

    # legacy settings (for compatability)
    # having these here just causes bugs as the override the newer settings...
    # better to simply throw an error
    # if args.sticky_actions is not None:
    #     if args.sticky_actions:
    #         args.repeat_action_probability = 0.25
    #     else:
    #         args.repeat_action_probability = 0.0
    # if args.time_aware is not None:
    #     args.embed_time = args.time_aware
    # if args.tvf_exp_gamma is not None:
    #     args.tvf_return_rho = args.tvf_exp_gamma
    # if args.tvf_mode is not None:
    #     args.tvf_return_mode = args.tvf_mode
    # if args.tvf_n_step is not None:
    #     args.tvf_return_n_step = args.tvf_n_step

    for param in REMOVED_PARAMS:
        if param in vars(args).keys() and vars(args)[param] is not None:
            print(f"Warning, {param} has been removed, and is being ignored.")

    # smart config
    buffer_size = args.replay_size if args.replay_size > 0 else args.batch_size
    rollout_size = args.agents * args.n_steps
    if args.distil_batch_size_ratio is not None:
        args.distil_batch_size = round(rollout_size * args.distil_batch_size_ratio)

    if args.distil_freq_ratio is not None:
        # for period faster than 1 per epoch just up the epochs.
        args.distil_period = buffer_size / args.batch_size / args.distil_freq_ratio
        while args.distil_period < 1:
            args.distil_period *= 2
            args.distil_epochs *= 2
        args.distil_period = round(args.distil_period)

    if args.replay_mode == "off":
        args.replay_size = 0

