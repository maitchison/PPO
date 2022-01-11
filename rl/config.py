import logging
import uuid
import socket
import argparse
import random

class Config:

    def __init__(self, **kwargs):
        # put these here just so IDE can detect common parameters...
        self.environment        = str()
        self.experiment_name    = str()
        self.run_name           = str()
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
        self.adam_epsilon       = float()
        self.workers            = int()
        self.epochs             = float()
        self.limit_epochs       = int()
        self.distil_beta        = float()
        self.distil_period      = int()
        self.distil_freq_ratio  = float()
        self.distil_batch_size_ratio = float()
        self.replay_size        = int()
        self.distil_batch_size  = int()
        self.replay_mixing      = bool()
        self.replay_hashing     = bool()
        self.quite_mode         = bool()

        self.observation_normalization = bool()
        self.intrinsic_reward_scale = float()
        self.extrinsic_reward_scale = float()

        self.reward_normalization = bool()
        self.reward_scale       = float()
        self.override_reward_normalization_gamma = float()

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
        self.tvf_lambda         = float()
        self.tvf_lambda_samples = int()
        self.hidden_units   = int()
        self.tvf_activation     = str()
        self.tvf_soft_anchor    = float()
        self.tvf_horizon_scale  = str()
        self.tvf_time_scale = str()
        self.tvf_n_step         = int()
        self.tvf_mode           = str()
        self.tvf_n_dedicated_value_heads  = int()
        self.tvf_exp_gamma      = float()
        self.tvf_exp_mode       = str()
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
    
        self.time_aware = bool()
        self.ed_type = str()
        self.ed_gamma = float()

        # phasic
        self.policy_epochs = int()                            
        self.value_epochs = int()
        self.distil_epochs = int()
        self.target_kl = float()                            
        self.ppo_epsilon =float()
        self.agents = int()
        self.n_steps = int()
        self.value_lr = float()
        self.policy_lr = float()
        self.distil_lr = float()
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

        self.sa_mu = float()
        self.sa_sigma = float()

        self.deferred_rewards   = int()

        self.frame_stack        = int()
        self.frame_skip         = int()
        self.timeout = int()

        self.normalize_advantages = bool()
        self.checkpoint_compression = bool()

        self.use_clipped_value_loss = bool()
        self.reward_clipping    = str()

        self.log_folder         = str()
        self.checkpoint_every   = int()
        self.disable_ev         = bool()

        self.use_rnd            = bool()
        self.use_ebd            = bool()
        self.warmup_period      = int()
        self.rnd_lr             = float()
        self.rnd_experience_proportion = float()

        self.per_step_reward_noise    = float()
        self.debug_terminal_logging = bool()
        self.debug_value_logging = bool()
        self.seed = int()
        self.atari_rom_check = bool()
        self.debug_replay_shadow_buffers = bool()

        self.full_action_space = bool()
        self.terminal_on_loss_of_life = bool()
        self.value_transform = str()
        self.force_restore = bool()

        # log optimal
        self.use_log_optimal = bool()
        self.lo_alpha = float()
        self.lo_alpha_anneal = bool()

        # ema frame stack
        self.ema_frame_stack_gamma = float()
        self.ema_frame_stack = bool()

        # tvf loss
        self.tvf_loss_fn = bool()
        self.tvf_huber_loss_delta = bool()
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
                    self.replay_size >= 0 or \
                    self.debug_replay_shadow_buffers
            else:
                self.use_compression = str2bool(str(self.use_compression))

    @property
    def propagate_intrinsic_rewards(self):
        return not self.use_rnd

    @property
    def reward_normalization_gamma(self):
        return self.override_reward_normalization_gamma if self.override_reward_normalization_gamma >= 0 else self.gamma

    @property
    def use_intrinsic_rewards(self):
        return self.use_rnd or self.use_ebd

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
        return args.use_tvf and not args.tvf_force_ext_value_distil

    @property
    def normalize_intrinsic_rewards(self):
        return self.use_rnd or self.use_ebd

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

    parser = argparse.ArgumentParser(description="Trainer for PPO2")

    if not no_env:
        parser.add_argument("environment")

    parser.add_argument("--experiment_name", type=str, default="Run", help="Name of the experiment.")
    parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")

    parser.add_argument("--filter", type=str, default="none",
                        help="Add filter to agent observation ['none', 'hash']")
    parser.add_argument("--hash_size", type=int, default=42, help="Adjusts the hash tempalte generator size.")
    parser.add_argument("--restore", type=str2bool, default=False,
                        help="Restores previous model or raises error. If set to false and new run will be started.")

    parser.add_argument("--network", type=str, default="nature", help="Encoder used, [nature|impala]")
    parser.add_argument("--architecture", type=str, default="dual", help="[dual|single]")

    parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter.")
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
    parser.add_argument("--tvf_lambda", type=float, default=1.0, help="Lambda for TVF(\lambda)")
    parser.add_argument("--tvf_lambda_samples", type=int, default=16, help="Number of n-step samples to use for tvf_lambda calculation")
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

    parser.add_argument("--tvf_n_dedicated_value_heads", type=int, default=0)
    parser.add_argument("--tvf_activation", type=str, default="relu", help="[relu|tanh|sigmoid]")
    parser.add_argument("--tvf_soft_anchor", type=float, default=50.0, help="MSE loss for V(*,0) being non-zero.")

    parser.add_argument("--tvf_horizon_scale", type=str, default="default", help="[default|centered|wide|log|zero]")
    parser.add_argument("--tvf_time_scale", type=str, default="default", help="[default|centered|wide|log|zero]")
    parser.add_argument("--tvf_hidden_units", type=int, default=512, help="units used for value prediction")

    parser.add_argument("--tvf_n_step", type=int, default=16, help="n step to use")
    parser.add_argument("--tvf_mode", type=str, default="nstep", help="[nstep|adaptive|exponential|lambda]")
    parser.add_argument("--tvf_exp_gamma", type=float, default=2.0)
    parser.add_argument("--tvf_exp_mode", type=str, default="default", help="[default|masked|transformed]")
    parser.add_argument("--use_tvf", type=str2bool, default=False, help="Enabled TVF mode.")

    # simulated annealing
    parser.add_argument("--sa_mu", type=float, default=0.0)
    parser.add_argument("--sa_sigma", type=float, default=0.05)

    # log-optimal
    parser.add_argument("--use_log_optimal", type=str2bool, default=False, help="Enabled Log-Optimal mode.")
    parser.add_argument("--lo_alpha", type=float, default=1.0, help="Risk factor for log-optimal mode.")
    parser.add_argument("--lo_alpha_anneal", type=str2bool, default=False)

    # phasic inspired stuff
    parser.add_argument("--policy_epochs", type=int, default=3, help="Number of policy training epochs per training batch.")
    parser.add_argument("--value_epochs", type=int, default=2, help="Number of value training epochs per training batch.")

    # distil / replay
    parser.add_argument("--distil_epochs", type=int, default=0, help="Number of distillation epochs")
    parser.add_argument("--distil_beta", type=float, default=1.0)
    parser.add_argument("--distil_period", type=int, default=1)
    parser.add_argument("--distil_batch_size", type=int, default=None, help="Size of batch to use when training distil. Defaults to replay_size (or rollout batch size if replay is disabled).")

    parser.add_argument("--distil_freq_ratio", type=float, default=None, help="Sets distil period to replay_size / batch_size * distil_freq_ratio")
    parser.add_argument("--distil_batch_size_ratio", type=float, default=None,
                        help="Sets distil_batch_size to replay_size * distil_batch_size_ratio")

    parser.add_argument("--replay_mode", type=str, default="overwrite", help="[overwrite|sequential|uniform]")
    parser.add_argument("--replay_size", type=int, default=0, help="Size of replay buffer. 0=off.")
    parser.add_argument("--replay_mixing", type=str2bool, default=False)
    parser.add_argument("--replay_hashing", type=str2bool, default=False)
    parser.add_argument("--distil_delay", type=int, default=0, help="Number of steps to wait before starting distillation")
    parser.add_argument("--distil_min_var", type=float, default=0.0,
                        help="If the variance of the value networks value estimates are less than this distil will not run.")
    parser.add_argument("--distil_var_boost", type=float, default=0.0,
                        help="Variance based bonus for distillation.")

    parser.add_argument("--dna_shared_initialization", type=str2bool, default=False,
                        help="Policy and value network start with same weight initialization")
    parser.add_argument("--dna_dual_constraint", type=float, default=0,
                        help="Policy updates are constrained by value prediction.")

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
    parser.add_argument("--rnd_lr", type=float, default=1.0e-4, help="Learning rate for Adam optimizer")

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
    parser.add_argument("--value_transform", type=str, default="identity", help="[identity|sqrt]")

    # -----------------

    parser.add_argument("--gamma", type=float, default=0.999, help="Discount rate for extrinsic rewards")

    parser.add_argument("--observation_normalization", type=str2bool, default=False)
    parser.add_argument("--intrinsic_reward_scale", type=float, default=1)
    parser.add_argument("--extrinsic_reward_scale", type=float, default=1)

    parser.add_argument("--max_micro_batch_size", type=int, default=512)
    parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                        help="Enables synchronous environments (slower).")
    parser.add_argument("--resolution", type=str, default="standard", help="['full', 'standard', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--export_video", type=str2bool, default=True)
    parser.add_argument("--export_trajectories", type=str2bool, default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--ignore_device", type=str, default="[]", help="Devices to ignore when using auto")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--hostname", type=str, default=socket.gethostname())
    parser.add_argument("--sticky_actions", type=str2bool, default=False)
    parser.add_argument("--guid", type=str, default=None)
    parser.add_argument("--noop_duration", type=int, default=30, help="maximum number of no-ops to add on reset")
    parser.add_argument("--per_step_reward_noise", type=float, default=0.0, help="Standard deviation of noise added to (normalized) reward each step.")
    parser.add_argument("--reward_clipping", type=str, default="off", help="[off|[<R>]|sqrt]")
    parser.add_argument("--reward_normalization", type=str2bool, default=True)
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--deferred_rewards", type=int, default=0,
                        help="If positive, all rewards accumulated so far will be given at time step deferred_rewards, then no reward afterwards.")
    parser.add_argument("--use_compression", type=str, default='auto',
                        help="Use LZ4 compression on states (around 20x smaller), but is 10% slower")
    parser.add_argument("--override_reward_normalization_gamma", type=float, default=-1)

    parser.add_argument("--eb_alpha", type=float, default=0.0)
    parser.add_argument("--eb_beta", type=float, default=0.0)
    parser.add_argument("--eb_theta", type=float, default=1.0)

    parser.add_argument("--ema_frame_stack_gamma", type=float, default=2.0)
    parser.add_argument("--ema_frame_stack", type=str2bool, default=False)

    parser.add_argument("--atari_rom_check", type=str2bool, default=True,
                        help="Verifies on load, that the MD5 of atari ROM matches the ALE.")

    # episodic discounting
    parser.add_argument("--time_aware", type=str2bool, default=True)
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
                        help="Enables the Exploration by Disagreement reward.")

    parser.add_argument("--normalize_advantages", type=str2bool, default=True)
    parser.add_argument("--intrinsic_reward_propagation", type=str2bool, default=None,
                        help="allows intrinsic returns to propagate through end of episode."
    )

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
    parser.add_argument("--debug_replay_shadow_buffers", type=str2bool, default=False,
                        help="Creates shadow buffers, used only for testing, and which take a lot of memory.")
    parser.add_argument("--checkpoint_every", type=int, default=int(5e6),
                        help="Number of environment steps between checkpoints.")
    parser.add_argument("--quiet_mode", type=str2bool, default=False)

    # other
    parser.add_argument("--mutex_key", type=str, default='',
                        help="uses mutex locking so that only one GPU can be working on a rollout at a time. " +
                             "(use DEVICE) to set automatically to current device."
                        )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--description", type=str, default=None, help="Can be used as needed. (logged in params.txt)")
    parser.add_argument("--layer_norm", type=str2bool, default=False)
    parser.add_argument("--benchmark_mode", type=str2bool, default=False, help="Enables benchmarking mode.")

    # due to compatability
    parser.add_argument("--use_mutex", type=str2bool, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--distill_epochs", dest="distil_epochs", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--distill_beta", dest="distil_beta", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--tvf_force_ext_value_distill", dest="tvf_force_ext_value_distil", type=str2bool,
                        help=argparse.SUPPRESS)
    parser.add_argument("--distil_resampling", type=str2bool, help=argparse.SUPPRESS) # ignored
    parser.add_argument("--distill_lr", dest="distil_lr", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--distill_lr_anneal", dest="distil_lr_anneal", type=str2bool, help=argparse.SUPPRESS)

    cmd_args = parser.parse_args(args_override).__dict__
    args.update(**cmd_args)

    # checks
    assert not (args.use_rnd and not args.observation_normalization), "RND requires observation normalization"
    assert not (args.color and args.observation_normalization), "Observation normalization averages over channels, so " \
                                                               "best to not use it with color at the moment."

    assert not (args.use_ebd and not args.architecture == "dual"), "EBD requires dual architecture"

    # set defaults
    if args.intrinsic_reward_propagation is None:
        # this seems keen to getting intrinsic motivation to work
        args.intrinsic_reward_propagation = (args.use_rnd or args.use_ebd)
    if args.tvf_gamma is None:
        args.tvf_gamma = args.gamma
    if cmd_args.get("use_mutex", False):
        print("warning, use_mutex is deprecated, use mutex_key instead.")
        args.mutex_key = "DEVICE"
    if args.distil_batch_size is None or args.distil_batch_size < 0:
        args.distil_batch_size = args.replay_size if args.replay_size > 0 else args.batch_size

    # smart config
    buffer_size = args.replay_size if args.replay_size > 0 else args.batch_size
    if args.distil_batch_size_ratio is not None:
        args.distil_batch_size = round(buffer_size * args.distil_batch_size_ratio)
        while args.distil_batch_size > buffer_size:
            args.distil_batch_size //= 2
            args.distil_epochs *= 2

    if args.distil_freq_ratio is not None:
        # for period faster than 1 per epoch just up the epochs.
        args.distil_period = buffer_size / args.batch_size / args.distil_freq_ratio
        while args.distil_period < 1:
            args.distil_period *= 2
            args.distil_epochs *= 2
        args.distil_period = round(args.distil_period)

    try:
        args.tvf_lambda = float(args.tvf_lambda)
    except:
        pass

    assert args.tvf_value_samples <= args.tvf_max_horizon, "tvf_value_samples must be <= tvf_max_horizon."
    assert args.tvf_horizon_samples <= args.tvf_max_horizon, "tvf_horizon_samples must be <= tvf_max_horizon."