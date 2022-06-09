import ast
import uuid
import socket
import argparse

"""
    Colors class for use with terminal.
"""
HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

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


class BaseConfig:

    def __init__(self, prefix=''):
        self._prefix = prefix

    def update(self, **kwargs):

        if self._prefix == '':
            self.__dict__.update(kwargs)
        else:
            for k, v in kwargs.items():
                if k.startswith(f"{self._prefix}_"):
                    self.__dict__[k[len(self._prefix)+1:]] = v

        # children
        for k, v in vars(self).items():
            if issubclass(type(v), BaseConfig):
                v.update(**kwargs)

    def flatten(self):
        params = {}
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            if issubclass(type(v), BaseConfig):
                if v._prefix == '':
                    continue
                sub_params = v.flatten()
                prefix = v._prefix
                params.update({
                    f"{prefix}_{k}": v for k, v in sub_params.items()
                })
            else:
                params[k] = v
        return params


    def _print_vars(self):
        """
        Useful for copy and pasting into the init so we get auto-complete
        """
        for k, v in vars(self).items():
            try:
                if v is None:
                    print(f"self.{k} = object()")
                else:
                    print(f"self.{k} = {type(v).__name__}()")
            except:
                pass

class TVFConfig(BaseConfig):
    """
    Config settings for TVF
    """

    def __init__(self, parser: argparse.ArgumentParser = None):

        super().__init__()

        if parser is None:
            return

        parser.add_argument("--use_tvf", type=str2bool, default=False)

        parser.add_argument("--tvf_gamma", type=float, default=None, help="Gamma for TVF, defaults to gamma")
        parser.add_argument("--tvf_coef", type=float, default=1.0, help="Loss is multiplied by this")
        parser.add_argument("--tvf_sum_horizons", type=str2bool, default=False, help="Sum horizon errors instead of mean.")
        parser.add_argument("--tvf_horizon_trimming", type=str, default='False', help="off|interpolate|average")
        parser.add_argument("--tvf_horizon_dropout", type=float, default=0.0, help="fraction of horizons to exclude per epoch")
        parser.add_argument("--tvf_return_mode", type=str, default="exponential", help="[fixed|adaptive|exponential|geometric|advanced]")
        parser.add_argument("--tvf_return_samples", type=int, default=32, help="Number of n-step samples to use for distributional return calculation")
        parser.add_argument("--tvf_return_n_step", type=int, default=80, help="n step to use for tvf_return estimation")
        parser.add_argument("--tvf_return_use_log_interpolation", type=str2bool, default=False, help="Interpolates in log space.")
        parser.add_argument("--tvf_max_horizon", type=int, default=1000, help="Max horizon for TVF.")
        parser.add_argument("--tvf_value_heads", type=int, default=64, help="Number of value heads to use.")
        parser.add_argument("--tvf_head_spacing", type=str, default="geometric", help="[geometric|linear]")
        parser.add_argument("--tvf_head_weighting", type=str, default="off", help="[off|h_weighted]")
        parser.add_argument("--tvf_activation", type=str, default="relu", help="[relu|tanh|sigmoid]")



class OptimizerConfig(BaseConfig):
    """
    Config settings for (generic) optimizer
    """
    def __init__(self, prefix: str = '', parser=None):

        super().__init__(prefix)

        if parser is None:
            return

        if prefix in ["value", "policy", "distil"]:
            epoch_default = 2
        else:
            epoch_default = 0

        # general
        parser.add_argument(f"--{prefix}_optimizer", type=str, default="adam", help="[adam|sgd|csgo]")
        parser.add_argument(f"--{prefix}_lr_anneal", type=str2bool, nargs='?', const=True, default=False, help="Anneals learning rate to 0 (linearly) over training")
        parser.add_argument(f"--{prefix}_epochs", type=int, default=epoch_default, help=f"Number of training epochs per {prefix} batch.")
        parser.add_argument(f"--{prefix}_mini_batch_size", type=int, default=256, help="Number of examples used for each optimization step.")
        # todo: implement anneal
        parser.add_argument(f"--{prefix}_lr", type=float, default=2.5e-4, help="Learning rate for optimizer")

        # adam
        parser.add_argument(f"--{prefix}_adam_epsilon", type=float, default=1e-5, help="Epsilon parameter for Adam optimizer")
        parser.add_argument(f"--{prefix}_adam_beta1", type=float, default=0.9, help="beta1 parameter for Adam optimizer")
        parser.add_argument(f"--{prefix}_adam_beta2", type=float, default=0.999, help="beta1 parameter for Adam optimizer")

        self.optimizer = str()
        self.lr_anneal = bool()
        self.epochs = int()
        self.mini_batch_size = int()
        self.lr = float()
        self.adam_epsilon = float()
        self.adam_beta1 = float()
        self.adam_beta2 = float()


class Config(BaseConfig):

    def __init__(self):

        super().__init__()

        self._parser = parser = argparse.ArgumentParser(description="Trainer for RL")

        # --------------------------------
        # main arguments

        parser.add_argument("environment", help="Name of environment (e.g. pong) or alternatively a list of environments (e.g.) ['Pong', 'Breakout']")
        parser.add_argument("--experiment_name", type=str, default="Run", help="Name of the experiment.")
        parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")
        parser.add_argument("--restore", type=str, default='never', help="Restores previous model. 'always' will restore, or error, 'never' will not restore, 'auto' will restore if it can.")
        parser.add_argument("--reference_policy", type=str, default=None, help="Path to checkpoint to use for a reference policy. In this case policy will not be updated.")
        parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
        parser.add_argument("--threads", type=int, default=2, help="Number of numpy/torch threads. Usually does not improve performance.")
        parser.add_argument("--epochs", type=float, default=50.0, help="Each epoch represents 1 million environment interactions.")
        parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
        parser.add_argument("--checkpoint_compression", type=str2bool, default=True, help="Enables checkpoint compression.")
        parser.add_argument("--save_model_interval", type=int, default=0, help="Period for which to saves model history during training (uses a lot of space!). 0 = off.")
        parser.add_argument("--obs_compression", type=str2bool, default=False, help="Use LZ4 compression on states (around 20x smaller), but is 10% slower")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use [cpu|cuda:n|auto]")
        parser.add_argument("--upload_batch", type=str2bool, default=False, help='Uploads an entire batch to GPU, faster, but uses more GPU RAM.')
        parser.add_argument("--disable_logging", type=str2bool, default=False, help='Useful when profiling.')
        parser.add_argument("--ignore_lock", type=str2bool, default=False, help="ignores previous lock")
        parser.add_argument("--ignore_device", type=str, default="[]", help="Devices to ignore when using auto")
        parser.add_argument("--save_checkpoints", type=str2bool, default=True)
        parser.add_argument("--output_folder", type=str, default="./")
        parser.add_argument("--hostname", type=str, default=socket.gethostname())
        parser.add_argument("--guid", type=str, default=None)
        parser.add_argument("--disable_ev", type=str2bool, default=False, help="disables explained variance calculations (slightly faster).")
        parser.add_argument("--anneal_target_epoch", type=float, default=None, help="Epoch to anneal to zero by")
        parser.add_argument("--mutex_key", type=str, default='',
                            help="uses mutex locking so that only one GPU can be working on a rollout at a time. " +
                                 "(use DEVICE) to set automatically to current device."
                            )
        parser.add_argument("--seed", type=int, default=-1)
        parser.add_argument("--description", type=str, default=None, help="Can be used as needed. (logged in params.txt)")
        parser.add_argument("--quiet_mode", type=str2bool, default=False)
        parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
        parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")
        parser.add_argument("--checkpoint_every", type=int, default=int(5e6), help="Number of environment steps between checkpoints.")
        parser.add_argument("--log_folder", type=str, default=None)
        parser.add_argument("--observation_normalization", type=str2bool, default=False)
        parser.add_argument("--freeze_observation_normalization", type=str2bool, default=False,
                            help="Disables updates to observation normalization constants.")
        parser.add_argument("--max_micro_batch_size", type=int, default=512, help="Can be useful to limit GPU memory")
        parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                            help="Enables synchronous environments (slower, but helpful for debuging env errors).")
        parser.add_argument("--benchmark_mode", type=str2bool, default=False, help="Enables benchmarking mode.")

        # --------------------------------
        # Rewards
        parser.add_argument("--intrinsic_reward_scale", type=float, default=0.3, help="Intrinsic reward scale.")
        parser.add_argument("--tvf_return_estimator_mode", type=str, default="default",
                            help='Allows the use of the reference return estimator (very slow). [default|reference|verify|historic]')
        parser.add_argument("--intrinsic_reward_propagation", type=str2bool, default=None, help="allows intrinsic returns to propagate through end of episode.")
        parser.add_argument("--override_reward_normalization_gamma", type=float, default=None)

        # --------------------------------
        # Model

        parser.add_argument("--encoder", type=str, default="nature", help="Encoder used for all models, [nature|impala]")
        parser.add_argument("--encoder_args", type=str, default=None, help="Additional arguments for encoder. (encoder specific)")
        parser.add_argument("--hidden_units", type=int, default=512)
        parser.add_argument("--architecture", type=str, default="dual", help="[dual|single]")
        parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards")
        parser.add_argument("--gamma", type=float, default=0.999, help="Discount rate for extrinsic rewards")
        parser.add_argument("--lambda_policy", type=float, default=0.95, help="GAE parameter.")
        parser.add_argument("--lambda_value", type=float, default=0.95, help="lambda to use for return estimations when using PPO or DNA")
        parser.add_argument("--max_grad_norm", type=float, default=20.0, help="Clipping used when global_norm is set.")
        parser.add_argument("--grad_clip_mode", type=str, default="global_norm", help="[off|global_norm|cak]")

        # --------------------------------
        # Environment
        parser.add_argument("--env_type", type=str, default="atari", help="[atari|mujoco|procgen]")
        parser.add_argument("--warmup_period", type=int, default=250,
                            help="Number of random steps to take before training agent.")
        parser.add_argument("--timeout", type=int, default=60 * 60 * 30,
                            help="Set the timeout for the environment, 0=off, (given in unskipped environment steps)")
        parser.add_argument("--repeat_action_probability", type=float, default=0.0)
        parser.add_argument("--noop_duration", type=int, default=30, help="maximum number of no-ops to add on reset")
        parser.add_argument("--per_step_reward_noise", type=float, default=0.0,
                            help="Standard deviation of noise added to (normalized) reward each step.")
        parser.add_argument("--per_step_termination_probability", type=float, default=0.0,
                            help="Probability that each step will result in unexpected termination (used to add noise to value).")
        parser.add_argument("--reward_clipping", type=str, default="off", help="[off|[<R>]|sqrt]")
        parser.add_argument("--reward_normalization", type=str2bool, default=True)
        parser.add_argument("--deferred_rewards", type=int, default=0,
                            help="If positive, all rewards accumulated so far will be given at time step deferred_rewards, then no reward afterwards.")
        # (atari)
        parser.add_argument("--resolution", type=str, default="nature", help="['full', 'nature', 'half']")
        parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--cv2_bw", type=str2bool, default=False, help='uses cv2 to implement black and white filter.')
        parser.add_argument("--full_action_space", type=str2bool, default=False)
        parser.add_argument("--terminal_on_loss_of_life", type=str2bool, default=False)
        parser.add_argument("--frame_stack", type=int, default=4)
        parser.add_argument("--frame_skip", type=int, default=4)
        parser.add_argument("--embed_time", type=str2bool, default=True, help="Encodes time into observation")
        parser.add_argument("--embed_action", type=str2bool, default=True, help="Encodes actions into observation")
        parser.add_argument("--atari_rom_check", type=str2bool, default=True, help="Verifies on load, that the MD5 of atari ROM matches the ALE.")
        # (stuck)
        parser.add_argument("--max_repeated_actions", type=int, default=100, help="Agent is given a penalty if it repeats the same action more than this many times.")
        parser.add_argument("--repeated_action_penalty", type=float, default=0.0, help="Penalty if agent repeats the same action more than this many times.")

        # --------------------------------

        self.policy = OptimizerConfig('policy', parser)
        self.value = OptimizerConfig('value', parser)
        self.distil = OptimizerConfig('distil', parser)
        self.aux = OptimizerConfig('aux', parser)
        self.rnd = OptimizerConfig('rnd', parser)

        # --------------------------------
        # PPO
        parser.add_argument("--ppo_vf_coef", type=float, default=0.5, help="Loss multiplier for default value loss.")
        parser.add_argument("--entropy_bonus", type=float, default=0.01)
        parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon parameter.")
        parser.add_argument("--n_steps", type=int, default=256, help="Number of environment steps per training step.")
        parser.add_argument("--agents", type=int, default=256)
        parser.add_argument("--advantage_epsilon", type=float, default=1e-8, help="Epsilon used when normalizing advantages.")
        parser.add_argument("--advantage_clipping", type=float, default=None, help="Advantages will be clipped to this, (after normalization)")
        parser.add_argument("--ppo_epsilon_anneal", type=str2bool, nargs='?', const=True, default=False,
                            help="Anneals learning rate to 0 (linearly) over training") # remove

        # --------------------------------
        # TVF
        self.tvf = TVFConfig(parser)

        # --------------------------------
        # Debugging
        # ...

        # --------------------------------
        # Auto Gamma
        parser.add_argument("--use_ag", type=str2bool, default=False, help="Enables auto gamma")
        parser.add_argument("--ag_mode", type=str, default="episode_length", help="[episode_length|training|sns]")
        parser.add_argument("--ag_target", type=str, default="policy", help="[policy|value|both]")
        parser.add_argument("--ag_sns_threshold", type=float, default=5.0, help="horizon heads with noise levels below this threshold are considered low noise.")
        parser.add_argument("--ag_sns_alpha", type=float, default=0.999,
                            help="alpha value used in EMA for horizon.")
        parser.add_argument("--ag_sns_delay", type=int, default=int(5e6),
                            help="alpha value used in EMA for horizon.")
        parser.add_argument("--ag_sns_min_h", type=int, default=100,
                            help="Minimum auto gamma horizon.")

        # --------------------------------
        # Simple Noise Scale
        parser.add_argument("--use_sns", type=str2bool, default=False, help="Enables generation of simple noise scale estimates")
        parser.add_argument("--sns_labels", type=str, default="['policy', 'value', 'distil']"),
        parser.add_argument("--sns_period", type=int, default=4, help="Generate estimates every n updates.")
        parser.add_argument("--sns_max_heads", type=int, default=8, help="Limit to this number of heads when doing per head noise estimate.")
        parser.add_argument("--sns_b_big", type=int, default=8192, help="")
        parser.add_argument("--sns_b_small", type=int, default=32, help="")
        parser.add_argument("--sns_small_samples", type=int, default=32, help="")

        # --------------------------------
        # Auxiliary phase
        parser.add_argument("--aux_target", type=str, default='reward', help="[reward|vtarg]]")
        parser.add_argument("--aux_source", type=str, default='aux', help="[aux|value]]")
        parser.add_argument("--aux_period", type=int, default=0, help="")

        # --------------------------------
        # Distil phase
        parser.add_argument("--distil_beta", type=float, default=1.0)
        parser.add_argument("--distil_period", type=int, default=1)
        parser.add_argument("--distil_loss", type=str, default="mse_logit", help="[mse_logit|mse_policy|kl_policy]")
        parser.add_argument("--distil_batch_size", type=int, default=None, help="Size of batch to use when training distil. Defaults to rollout_size.")
        parser.add_argument("--distil_freq_ratio", type=float, default=None, help="Sets distil period to replay_size / batch_size * distil_freq_ratio")
        parser.add_argument("--distil_batch_size_ratio", type=float, default=None, help="Sets distil_batch_size to rollout_size * distil_batch_size_ratio")
        parser.add_argument("--distil_max_heads", type=int, default=8, help="Max number of heads to apply distillation to.")

        # --------------------------------
        # Replay
        parser.add_argument("--replay_mode", type=str, default="overwrite", help="[overwrite|sequential|uniform|off]")
        parser.add_argument("--replay_size", type=int, default=0, help="Size of replay buffer. 0=off.")
        parser.add_argument("--replay_mixing", type=str2bool, default=False)
        parser.add_argument("--replay_thinning", type=float, default=1.0, help="Adds this fraction of experience to replay buffer.")

        # -----------------
        # RND
        parser.add_argument("--use_rnd", type=str2bool, default=False, help="Enables the Random Network Distillation (RND) module.")
        parser.add_argument("--rnd_experience_proportion", type=float, default=0.25)

        # this is just so we get autocomplete, as well as IDE hints if we spell something wrong

        self.use_tvf = bool()
        self.environment = str()
        self.experiment_name = str()
        self.run_name = str()
        self.restore = str()
        self.reference_policy = object()
        self.workers = int()
        self.threads = int()
        self.epochs = float()
        self.limit_epochs = object()
        self.checkpoint_compression = bool()
        self.save_model_interval = int()
        self.obs_compression = bool()
        self.device = str()
        self.upload_batch = bool()
        self.disable_logging = bool()
        self.ignore_lock = bool()
        self.ignore_device = str()
        self.save_checkpoints = bool()
        self.output_folder = str()
        self.hostname = str()
        self.guid = object()
        self.disable_ev = bool()
        self.anneal_target_epoch = object()
        self.mutex_key = str()
        self.seed = int()
        self.description = object()
        self.quiet_mode = bool()
        self.debug_print_freq = int()
        self.debug_log_freq = int()
        self.checkpoint_every = int()
        self.log_folder = object()
        self.observation_normalization = bool()
        self.freeze_observation_normalization = bool()
        self.max_micro_batch_size = int()
        self.sync_envs = bool()
        self.benchmark_mode = bool()
        self.intrinsic_reward_scale = float()
        self.intrinsic_reward_propagation = object()
        self.override_reward_normalization_gamma = object()
        self.encoder = str()
        self.encoder_args = object()
        self.hidden_units = int()
        self.architecture = str()
        self.gamma_int = float()
        self.gamma = float()
        self.lambda_policy = float()
        self.lambda_value = float()
        self.max_grad_norm = float()
        self.grad_clip_mode = str()
        self.env_type = str()
        self.warmup_period = int()
        self.timeout = int()
        self.repeat_action_probability = float()
        self.noop_duration = int()
        self.per_step_reward_noise = float()
        self.per_step_termination_probability = float()
        self.reward_clipping = str()
        self.reward_normalization = bool()
        self.deferred_rewards = int()
        self.resolution = str()
        self.color = bool()
        self.cv2_bw = bool()
        self.max_repeated_actions = int()
        self.repeated_action_penalty = float()
        self.full_action_space = bool()
        self.terminal_on_loss_of_life = bool()
        self.frame_stack = int()
        self.frame_skip = int()
        self.embed_time = bool()
        self.embed_action = bool()
        self.atari_rom_check = bool()

        self.ppo_vf_coef = float()
        self.entropy_bonus = float()
        self.ppo_epsilon = float()
        self.n_steps = int()
        self.agents = int()
        self.advantage_epsilon = float()
        self.advantage_clipping = object()
        self.ppo_epsilon_anneal = bool()
        self.tvf_return_estimator_mode = str()
        self.tvf_gamma = object()
        self.tvf_coef = float()
        self.tvf_sum_horizons = bool()
        self.tvf_horizon_trimming = str()
        self.tvf_horizon_dropout = float()
        self.tvf_return_mode = str()
        self.tvf_return_samples = int()
        self.tvf_return_n_step = int()
        self.tvf_return_use_log_interpolation = bool()
        self.tvf_max_horizon = int()
        self.tvf_value_heads = int()
        self.tvf_head_spacing = str()
        self.tvf_head_weighting = str()
        self.use_ag = bool()
        self.ag_mode = str()
        self.ag_target = str()
        self.ag_sns_threshold = float()
        self.ag_sns_alpha = float()
        self.ag_sns_delay = int()
        self.ag_sns_min_h = int()

        self.use_sns = bool()
        self.sns_labels = str()
        self.sns_period = int()
        self.sns_max_heads = int()
        self.sns_b_big = int()
        self.sns_b_small = int()
        self.sns_small_samples = int()

        self.aux_target = str()
        self.aux_source = str()
        self.aux_period = int()
        self.distil_beta = float()
        self.distil_period = int()
        self.distil_loss = str()
        self.distil_batch_size = object()
        self.distil_freq_ratio = float()
        self.distil_batch_size_ratio = float()
        self.distil_max_heads = int()
        self.replay_mode = str()
        self.replay_size = int()
        self.replay_mixing = bool()
        self.replay_thinning = float()
        self.use_rnd = bool()
        self.rnd_experience_proportion = float()

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
        return self.use_rnd

    @property
    def get_mutex_key(self):
        if self.mutex_key.lower() == 'device':
            return args.device
        else:
            return self.mutex_key

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
args:Config = Config()

def parse_args(args_override=None):

    parser = args._parser

    REMAPPED_PARAMS = {
        'gae_lambda': 'lambda_policy',
        'td_lambda': 'lambda_value',
        'use_compression': 'obs_compression',
        'export_video': None,
        'tvf_value_distribution': None,
        'tvf_horizon_distribution': None,
        'tvf_value_samples': None,
        'tvf_horizon_samples': 'tvf_value_heads',
        'tvf_hidden_units': None,
        'tvf_trimming': "tvf_horizon_trimming",
        'tvf_force_ext_value_distil': None,
        'tvf_horizon_scale': None,
        'tvf_time_scale': None,
        'policy_network': "encoder",
        'value_network': "encoder",
        'tvf_mode': None,
    }

    for k,v in REMAPPED_PARAMS.items():
        parser.add_argument(f"--{k}", type=str, default=None, help=argparse.SUPPRESS)

    cmd_args = parser.parse_args(args_override).__dict__
    args.update(**cmd_args)

    # fix restore using legacy settings
    if args.restore is True or args.restore == "True":
        args.restore = "always"
    if args.restore is False or args.restore == "False":
        args.restore = "never"

    # mappings
    for old_name, new_name in REMAPPED_PARAMS.items():
        if vars(args).get(old_name, None) is None:
            continue

        if new_name is None:
            print(f"Warning! Using deprecated parameter {FAIL}{old_name}{ENDC} which is being ignored.")
            continue

        legacy_value = vars(args)[old_name]
        new_type = type(vars(args)[new_name])
        if new_type is bool:
            cast_legacy_value = str2bool(legacy_value)
        else:
            cast_legacy_value = new_type(legacy_value)
        if vars(args)[new_name] is None:
            print(f"Warning! Using deprecated parameter {FAIL}{old_name}{ENDC} which is being mapped to {BOLD}{new_name}{ENDC} with value {legacy_value}")
            vars(args)[new_name] = cast_legacy_value
            del vars(args)[old_name]
        else:
            non_legacy_value = vars(args)[new_name]
            print(
                f"Warning! Using deprecated parameter {FAIL}{old_name}{ENDC} was specified but clashes with value assigned to {BOLD}{new_name}{ENDC}. Using legacy value {legacy_value} overwriting {non_legacy_value}.")
            vars(args)[new_name] = cast_legacy_value

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

    assert args.tvf_return_estimator_mode in ["default", "reference", "verify", "historic"]

    # set defaults
    if args.intrinsic_reward_propagation is None:
        # this seems key to getting intrinsic motivation to work
        # without it the agent might never want to die (as it can gain int_reward forever).
        # maybe this is correct behaviour? Not sure.
        args.intrinsic_reward_propagation = args.use_rnd
    if args.tvf_gamma is None:
        args.tvf_gamma = args.gamma
    if args.distil_batch_size is None:
        args.distil_batch_size = args.batch_size

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
            args.distil.epochs *= 2
        args.distil_period = round(args.distil_period)

    if args.replay_mode == "off":
        args.replay_size = 0

    # fixup horizon trimming
    if str(args.tvf_horizon_trimming) == 'False':
        args.tvf_horizon_trimming = "off"
    if str(args.tvf_horizon_trimming) == 'True':
        args.tvf_horizon_trimming = "interpolate"


if __name__ == "__main__":
    pass
    # c = Config()
    # args = c.parser.parse_args({'environment': 'Pong'})
    # c.update(**vars(args))
    #
    # c._print_vars()