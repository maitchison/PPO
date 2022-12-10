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

resolution_map = {
    "full": (210, 160),
    "procgen": (64, 64),
    "nature": (84, 84),
    "muzero": (96, 96),
    "half": (105, 80),    # this may produce cleaner resampling
}

class BaseConfig:

    def __init__(self, prefix='', parser: argparse.ArgumentParser = None):
        self._prefix = prefix
        if parser is not None:
            self._auto_add_params(parser)

    def verify(self):
        """
        Make sure parameters are ok.
        """
        for child in self.get_children():
            child.verify()

    def auto(self):
        """
        Apply any auto logic.
        """
        for child in self.get_children():
            child.auto()

    @property
    def prefix(self):
        return self._prefix

    def _auto_add_params(self, parser):
        """
        Adds parser entry for each class variable
        """
        class_vars = vars(self.__class__)
        var_types = class_vars['__annotations__']
        var_helps = {}
        try:
            import inspect
            source_code = inspect.getsource(self.__class__).split("\n")
            for line in source_code:
                line = line.lstrip(' \t')
                if line == "" or '#' not in line:
                    continue
                comment_part = line[line.find('#')+1:].lstrip(' ')
                first_word = line.split(' ')[0]
                if first_word == "":
                    continue
                if first_word.endswith(':'):
                    first_word = first_word[:-1]

                if first_word in class_vars:
                    var_helps[first_word] = comment_part
        except:
            # getsource might fail
            pass

        for var_name, var_default in class_vars.items():
            if self._is_hidden_var(var_name):
                continue

            var_type = var_types.get(var_name, object)
            var_help = var_helps.get(var_name, None)

            if var_type is bool:
                # parse bool correctly.
                # otherwise "False" will evaluate to True.
                var_type = str2bool

            prefix_part = "" if (self._prefix == "") else self._prefix.lower() + "_"

            parser.add_argument(
                f"--{prefix_part}{var_name}",
                type=var_type,
                default=var_default,
                help=var_help,
            )

    def get_children(self):
        """
        Returns list of config children
        """
        result = []
        for k, v in vars(self).items():
            if issubclass(type(v), BaseConfig):
                result.append(v)
        return result

    def update(self, params):

        # children first...
        for child in self.get_children():
            child.update(params)

        prefix = "" if self._prefix == "" else self._prefix + "_"

        for k in list(params.keys()):
            if not k.startswith(prefix):
                continue
            k_without_prefix = k[len(prefix):]
            # add and remove if we have this variable
            if k_without_prefix in self.__dict__:
                self.__dict__[k_without_prefix] = params[k]
                del params[k]
            # have to also check class vars, eventually just use class vars.
            if k_without_prefix in vars(self.__class__):
                setattr(self.__class__, k_without_prefix, params[k])
                del params[k]


    def _is_hidden_var(self, x: str):
        return x.startswith("_")

    def get_vars(self):
        """
        Returns a dictionary of config variable names:values.
        """
        result = {}
        # object vars... remove eventually
        for k, v in self.__dict__.items():
            if self._is_hidden_var(k):
                continue
            if issubclass(type(v), BaseConfig):
                continue
            result[k] = v
        # class vars (eventually just use these)
        for k, v in vars(self.__class__).items():
            if callable(v) or issubclass(type(v), property):
                # don't want methods
                continue
            if self._is_hidden_var(k):
                continue
            if issubclass(type(v), BaseConfig):
                continue
            result[k] = v
        return result

    def flatten(self):
        params = {}
        # process our vars, output in sorted order.
        vars = self.get_vars()
        for k in sorted(vars.keys()):
            params[k] = vars[k]
        # process our children
        for child in self.get_children():
            if child._prefix == '':
                continue
            sub_params = child.flatten()
            prefix = child._prefix
            params.update({
                f"{prefix}_{k}": v for k, v in sub_params.items()
            })
        return params


class SimpleNoiseScaleConfig(BaseConfig):
    """
    Config settings for simple noise scale
    """
    enabled:bool = False                    # Enables generation of simple noise scale estimates.
    labels: str = ['policy', 'distil', 'value', 'value_heads']  # value|value_heads|distil|policy
    period: int = 3                         # Generate estimates every n updates.
    max_heads: int = 7                      # Limit to this number of heads when doing per head noise estimate.
    b_big: int = 2048
    b_small: int = 128
    fake_noise: bool = False                # Replaces value_head gradient with noise based on horizon.
    smoothing_mode: str = "ema"             # ema|avg
    smoothing_horizon_avg: int = 1e6        # how big to make averaging window
    smoothing_horizon_s: int = 0.2e6        # how much to smooth s
    smoothing_horizon_g2: int = 1.0e6       # how much to smooth g2
    smoothing_horizon_policy: int = 5e6     # how much to smooth g2 for policy (normally much higher)

    def __init__(self, parser):
        super().__init__(prefix="sns", parser=parser)


class TVFConfig(BaseConfig):
    """
    Config settings for TVF
    """

    enabled: bool = False
    gamma: float = None             # Gamma for TVF, defaults to args.gamma.
    coef: float = 1.0               # Loss is multiplied by this.
    trimming: str = "off"           # off|timelimit|est_term
    trimming_mode: str = "average"  # interpolate|average|substitute
    eta_minh: int = 128              # estimated timelimit algorithm
    eta_buffer: int = 32
    eta_percentile: float = 90
    horizon_dropout: float = 0.0    # fraction of horizons to exclude per epoch.
    return_mode: str = "advanced"   # standard|advanced|full "
    return_distribution: str = "exponential"  # fixed|exponential|uniform|hyperbolic|quadratic
    return_samples: int = 8         # Number of n-step samples to use for distributional return calculation.
    return_use_log_interpolation: bool = False # Interpolates in log space.
    max_horizon: int = 30000        # Max horizon for TVF.
    value_heads: int = 128          # Number of value heads to use.
    head_spacing: str = "geometric" # geometric|linear|even_x
    head_weighting: str = "off"     # off|h_weighted.
    feature_window: int = -1        # Limits each head to a window of this many features.
    feature_sparsity: float = 0.0   # Zeros out this proprition of features for each head.
    include_ext: bool = False       # Also learn the rediscounted value estimate that will be used for advantages.

    def __init__(self, parser):
        super().__init__(prefix="tvf", parser=parser)

    def auto(self):
        # stub:
        print("AUTO TVF")
        # set defaults
        if TVFConfig.gamma is None:
            TVFConfig.gamma = args.gamma
        if TVFConfig.max_horizon <= -1:
            TVFConfig.max_horizon = args.env.timeout
            print(f"Auto max_horizon={TVFConfig.max_horizon}")


class OptimizerConfig(BaseConfig):
    """
    Config settings for (generic) optimizer
    """
    def __init__(self, prefix: str = '', parser=None):

        super().__init__(prefix)

        self.name = prefix

        if parser is None:
            return

        epoch_defaults = {
            'policy_opt': 2,
            'value_opt': 1,
            'distil_opt': 2,
            'aux_opt': 0,
            'rnd_opt': 1,
        }

        epoch_default = epoch_defaults.get(prefix, 0)

        # general
        parser.add_argument(f"--{prefix}_optimizer", type=str, default="adam", help="[adam|sgd|csgo]")
        parser.add_argument(f"--{prefix}_per_epoch_optimizer", type=str2bool, default=False)
        parser.add_argument(f"--{prefix}_lr_anneal", type=str2bool, nargs='?', const=True, default=False, help="Anneals learning rate to 0 (linearly) over training")
        parser.add_argument(f"--{prefix}_epochs", type=int, default=epoch_default, help=f"Number of training epochs per {prefix} batch.")
        parser.add_argument(f"--{prefix}_batch_mode", type=str, default="default", help=f"Batch method.")
        parser.add_argument(f"--{prefix}_mini_batch_size", type=int, default=256, help="Number of examples used for each optimization step.")
        # todo: implement anneal
        parser.add_argument(f"--{prefix}_lr", type=float, default=2.5e-4, help="Learning rate for optimizer")
        parser.add_argument(f"--{prefix}_flood_level", type=float, default=-1, help="Used to stop before we get to a minima")
        parser.add_argument(f"--{prefix}_stop_level", type=float, default=-1,
                            help="Used to perform early stopping when loss gets below this threshold")

        # adam
        parser.add_argument(f"--{prefix}_adam_epsilon", type=float, default=1e-5, help="Epsilon parameter for Adam optimizer")
        parser.add_argument(f"--{prefix}_adam_beta1", type=float, default=0.9, help="beta1 parameter for Adam optimizer. Set to -1 for auto")
        parser.add_argument(f"--{prefix}_adam_beta2", type=float, default=0.999, help="beta2 parameter for Adam optimizer")

        # note we use instance variables instead of class variables here as we need
        # to have multiple instances of this one.

        self.optimizer = str()
        self.lr_anneal = bool()
        self.epochs = int()
        self.mini_batch_size = int()
        self.lr = float()
        self.adam_epsilon = float()
        self.adam_beta1 = float()
        self.adam_beta2 = float()
        self.flood_level = float()
        self.stop_level = float()
        self.batch_mode = bool()
        self.per_epoch_optimizer = str()

    def n_updates(self, rollout_size):
        return (rollout_size / self.mini_batch_size) * self.epochs

    def auto(self):
        if self.adam_beta1 < 0:
            self.adam_beta1 = 1 - (1 / self.n_updates(args.n_steps * args.agents))
            print(f"Set {self.name} beta1 to {self.adam_beta1}")


class DebugConfig(BaseConfig):
    """
    Config settings for debug settings
    """

    zero_obs: bool = False      # Zeros the environment observation. Useful to see if the model can learn from time only.
    checkpoint_slides: bool = False  # Generates images containing states during epoch saves.
    print_freq: int = 60        # Number of seconds between debug prints.
    log_freq: int = 300         # Number of seconds between log writes.
    compress_csv: bool = False  # Enables log compression.

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="debug", parser=parser)


class DistilConfig(BaseConfig):
    """
    Config settings for Distilation
    """

    order: str = "after_policy" # [after_policy|before_policy]
    beta: float = 1.0
    target: str = "value"       # [return|value|advantage]
    batch_size: int = -1        # Size of batch to use when training distil. Negative for rollout_size.
    adv_lambda: float = 0.6     # Used for return or advantage distil targets.
    period: int = 1
    loss: str = "kl_policy"     # [mse_logit|mse_policy|kl_policy]
    max_heads: int = -1         # Max number of heads to apply distillation to, -1 for all.
    force_ext: bool = False     # Use value_ext instead of value_tvf for distillation.
    value_loss: str = "mse"     # [mse|clipped_mse|l1|huber]
    delta: float = 0.1          # delta for huber loss
    l1_scale: float = 1 / 30    # scaling for l1 loss

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="distil", parser=parser)

    def auto(self):
        if DistilConfig.batch_size < 0:
            DistilConfig.batch_size = args.batch_size


class ReplayConfig(BaseConfig):
    """
    Config settings for replay buffer
    """

    mode: str = "off"               # [overwrite|sequential|uniform|off]
    size: int = 0                   # Size of replay buffer. 0=off.")
    mixing: bool = False
    thinning: float = 1.0           # Adds this fraction of experience to replay buffer.

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="replay", parser=parser)

    @property
    def enabled(self):
        return ReplayConfig.size > 0 and ReplayConfig.mode != "off"

    def auto(self):
        pass


class GlobalKLConfig(BaseConfig):
    """
    Config settings for Global KL constraint
    """

    # --------------------------------
    # Global KL constraint
    # Not fully tested...

    enabled: bool = False    # Use a global kl constraint.
    threshold: float = -1    # 0.004 is probably good.
    penalty: float = 0.01
    source: str = "rollout"  # help=[rollout]
    samples: int = 1024      # Number of samples to use for global sample of state distribution."

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="gkl", parser=parser)


class RNDConfig(BaseConfig):
    """
    Config for random network distilation
    """

    enabled: bool = False  # Enables the Random Network Distillation (RND) module.
    experience_proportion: float = 0.25

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="rnd", parser=parser)

    def verify(self):
        if self.enabled:
            assert args.observation_normalization, "RND requires observation normalization"

class HashConfig(BaseConfig):
    """
    Config settings for State hashing
    """

    enabled: bool = False  # Enables state hashing (used to track exploration.
    bits: int = 16  # Number of bits to hash to, requires O(2^n) memory.
    bonus: float = 0.0  # Intrinsic reward bonus for novel hashed states.
    method: str = "linear"  # [linear|conv]
    input: str = "raw"  # [raw|raw_centered|normed|normed_offset]
    bonus_method: str = "hyperbolic"  # [hyperbolic|quadratic|binary]
    rescale: int = 1
    quantize: float = 1
    bias: float = 0.0
    decay: float = 0.99

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="hash", parser=parser)

class AUXConfig(BaseConfig):
    """
    Config optional fourth auxiliary phase.
    (enable by setting aux_opt_epochs>=1)
    """

    target: str = 'reward'  # [reward|vtarg]]
    source: str = 'aux'     # [aux|value]]
    period: int = 0         #

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="aux", parser=parser)


class IRConfig(BaseConfig):
    """
    Config for Intrinsic Rewards
    """
    propagation: bool = True    # Allows intrinsic returns to propagate through end of episode.
    scale: float = 0.3          # Intrinsic reward scale.
    center: bool = False        # Per-batch centering of intrinsic rewards.
    normalize: bool = True      # Normalizes intrinsic rewards such that they have unit variance.

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="ir", parser=parser)


class ModelConfig(BaseConfig):
    """
    Config for Model
    """

    encoder: str = "nature"         # Encoder used for all models, [nature|impala]
    encoder_args: str = None        # Additional arguments for encoder. (encoder specific).
    hidden_units: int = 256
    architecture: str = "dual"      # [dual|single]
    head_scale: float = 0.1         # Scales weights for value and policy heads.
    head_bias: bool = True          # Enables bias on output heads.

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="model", parser=parser)

    def verify(self):
        # checks
        if args.reference_policy is not None:
            assert self.architecture == "dual", "Reference policy loading requires a dual network."


class SIDEConfig(BaseConfig):
    """
    Config for State-independant Exploration
    """

    enabled: bool = False           # Enable state independant exploration
    noise_std: float = 0.1          # noise to add to each action
    period: int = 10                # update noise every n rollouts

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="side", parser=parser)

    def verify(self):
        pass


class EnvConfig(BaseConfig):
    """
    Environment Config
    """

    name: str = "Pong"                  # Name of environment (e.g. pong) or alternatively a list of environments (e.g.) ['Pong', 'Breakout'].
    type: str = "atari"                 # [atari|mujoco|procgen]
    warmup_period: int = 250            # Number of random steps to take before training agent.
    timeout: str = "auto"               # "Set the timeout for the environment, 0=off, (given in unskipped environment steps)")
    repeat_action_probability: float = 0.0
    noop_duration: int = 30             # "maximum number of no-ops to add on reset")
    per_step_termination_probability: float = 0.0   #  Probability that each step will result in unexpected termination (used to add noise to value).
    reward_clipping: str = "off"        # [off|[<R>]|sqrt]
    reward_normalization: str = "rms"   # "off|rms"
    reward_normalization_clipping: float = 10  # How much to clip rewards after normalization, negative to disable.
    deferred_rewards: int = 0           #  If positive, all rewards accumulated so far will be given at time step deferred_rewards, then no reward afterwards.

    # (stuck)
    max_repeated_actions: int = 100     # "Agent is given a penalty if it repeats the same action more than this many times.
    repeated_action_penalty: float = 0.0  # Penalty if agent repeats the same action more than this many times.

    # discrete action
    full_action_space: bool = False

    # pixel based
    resolution: str = "nature"          # [full|nature|half|muzero]
    color_mode: str = "default"         # [default|bw|rgb|yuv|hsv]
    frame_stack: int = None
    frame_skip: int = None
    embed_time: bool = True             # Encodes time into observation
    embed_action: bool = True           # Encodes actions into observation
    embed_state: bool = False           # Encodes state history into observation

    # specific to atari
    atari_terminal_on_loss_of_life: bool = False
    atari_rom_check: bool = True        # Makes sure atari MD5 matches expectation.

    # specific to procgen
    procgen_difficulty: str = "hard"    # [hard|easy]

    @property
    def is_vision_env(self):
        """
        Is this environment pixel based?
        """
        return self.type in ['atari', 'procgen']

    @property
    def res_x(self):
        return resolution_map[self.resolution][0]

    @property
    def res_y(self):
        return resolution_map[self.resolution][0]

    @property
    def noop_start(self):
        return self.noop_duration > 0

    def verify(self):
        if self.type == "procgen":
            assert self.frame_stack == 1, "Frame stacking not supported on procgen yet"
            assert self.frame_skip == 1, "Frame skipping not supported on procgen yet."
        if self.type == "mujoco":
            assert self.frame_stack == 1, "Frame stacking not supported on mujoco yet"
            assert self.frame_skip == 1, "Frame skipping not supported on mujoco yet."
        assert self.full_action_space in [True, False]

    def auto(self):

        if self.frame_skip in [None, -1]:
            EnvConfig.frame_skip = 4 if self.type == "atari" else 1
        if self.frame_stack in [None, -1]:
            EnvConfig.frame_stack = 4 if self.type == "atari" else 1

        # auto  color
        assert self.color_mode in ["default", "bw", "rgb", "yuv", "hsv"]
        if self.color_mode == "default":
            EnvConfig.color_mode = {
                'atari': 'bw',
                'procgen': 'yuv',
            }.get(self.type, 'bw')

        # auto timeout
        if self.timeout == "auto":
            if self.type == "atari":
                EnvConfig.timeout = 27000  # excludes skipped frames
            elif self.type == "procgen":
                env_timeouts = {
                    'bigfish': 8000,
                    'bossfight': 8000,
                    'plunder': 2000,
                }
                EnvConfig.timeout = env_timeouts.get(args.env.name, 1000)
            else:
                EnvConfig.timeout = 0  # unlimited
        else:
            EnvConfig.timeout = int(self.timeout)

        #stub:
        print(EnvConfig.timeout)

    def __init__(self, parser: argparse.ArgumentParser):
        super().__init__(prefix="env", parser=parser)



class Config(BaseConfig):

    # list of params that have been remapped
    _REMAPPED_PARAMS = {
    }

    def __init__(self):

        super().__init__()

        # this is just so we get autocomplete, as well as IDE hints if we spell something wrong

        self.experiment_name = str()
        self.run_name = str()
        self.restore = str()
        self.initial_model = str()
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
        self.save_initial_checkpoint = bool()
        self.save_early_checkpoint = bool()
        self.output_folder = str()
        self.hostname = str()
        self.guid = object()
        self.disable_ev = bool()
        self.anneal_target_epoch = object()
        self.mutex_key = str()
        self.seed = int()
        self.description = object()
        self.quiet_mode = bool()

        self.checkpoint_every = int()
        self.log_folder = object()
        self.observation_scaling = str()
        self.observation_normalization = bool()
        self.observation_normalization_epsilon = float()
        self.freeze_observation_normalization = bool()
        self.max_micro_batch_size = int()
        self.sync_envs = bool()
        self.benchmark_mode = bool()

        self.override_reward_normalization_gamma = object()

        self.gamma = float()
        self.gamma_int = float()
        self.lambda_policy = float()
        self.lambda_value = float()
        self.max_grad_norm = float()
        self.grad_clip_mode = str()

        self.ppo_vf_coef = float()
        self.entropy_bonus = float()
        self.ppo_epsilon = float()
        self.n_steps = int()
        self.agents = int()
        self.advantage_epsilon = float()
        self.advantage_clipping = object()
        self.ppo_epsilon_anneal = bool()

        # extra
        self.precision = str()

        self._parser = parser = argparse.ArgumentParser(description="Trainer for RL")

        # modules...
        self.debug = DebugConfig(self._parser)
        self.distil = DistilConfig(self._parser)
        self.gkl = GlobalKLConfig(self._parser)
        self.hash = HashConfig(self._parser)
        self.sns = SimpleNoiseScaleConfig(self._parser)
        self.env = EnvConfig(self._parser) # before tvf
        self.tvf = TVFConfig(self._parser)
        self.replay = ReplayConfig(self._parser)
        self.rnd = RNDConfig(self._parser)
        self.aux = AUXConfig(self._parser)
        self.ir = IRConfig(self._parser)
        self.model = ModelConfig(self._parser)
        self.side = SIDEConfig(self._parser)

        # --------------------------------

        self.policy_opt = OptimizerConfig('policy_opt', parser)
        self.value_opt = OptimizerConfig('value_opt', parser)
        self.distil_opt = OptimizerConfig('distil_opt', parser)
        self.aux_opt = OptimizerConfig('aux_opt', parser)
        self.rnd_opt = OptimizerConfig('rnd_opt', parser)

    def setup(self):

        # --------------------------------
        # main arguments

        # todo: move these to class

        parser = self._parser

        parser.add_argument("--experiment_name", type=str, default="Run", help="Name of the experiment.")
        parser.add_argument("--run_name", type=str, default="run", help="Name of the run within the experiment.")
        parser.add_argument("--restore", type=str, default='auto', help="Restores previous model. 'always' will restore, or error, 'never' will not restore, 'auto' will restore if it can.")
        parser.add_argument("--reference_policy", type=str, default=None, help="Path to checkpoint to use for a reference policy. In this case policy will not be updated.")
        parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
        parser.add_argument("--threads", type=int, default=2, help="Number of numpy/torch threads. Usually does not improve performance.")
        parser.add_argument("--epochs", type=float, default=50.0, help="Each epoch represents 1 million environment interactions.")
        parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
        parser.add_argument("--checkpoint_compression", type=str2bool, default=True, help="Enables checkpoint compression.")
        parser.add_argument("--save_model_interval", type=int, default=0, help="Period for which to saves model history during training (uses a lot of space!). 0 = off.")
        parser.add_argument("--initial_model", type=str, default=None,
                            help="path to model to load for initialization")
        parser.add_argument("--obs_compression", type=str2bool, default=False, help="Use LZ4 compression on states (around 20x smaller), but is 10% slower")
        parser.add_argument("--device", type=str, default="cpu", help="Device to use [cpu|cuda:n|auto]")
        parser.add_argument("--upload_batch", type=str2bool, default=False, help='Uploads an entire batch to GPU, faster, but uses more GPU RAM.')
        parser.add_argument("--disable_logging", type=str2bool, default=False, help='Useful when profiling.')
        parser.add_argument("--ignore_lock", type=str2bool, default=False, help="ignores previous lock")
        parser.add_argument("--ignore_device", type=str, default="[]", help="Devices to ignore when using auto")
        parser.add_argument("--save_checkpoints", type=str2bool, default=True)
        parser.add_argument("--save_initial_checkpoint", type=str2bool, default=False, help="Saves a checkpoint before any training.")
        parser.add_argument("--save_early_checkpoint", type=str2bool, default=False,
                            help="Saves a checkpoint at 1M steps.")
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
        parser.add_argument("--checkpoint_every", type=int, default=int(10e6), help="Number of environment steps between checkpoints.")
        parser.add_argument("--log_folder", type=str, default=None)

        parser.add_argument("--observation_scaling", type=str, default="scaled", help="scaled|centered|unit")
        parser.add_argument("--observation_normalization", type=str2bool, default=False)
        parser.add_argument("--observation_normalization_epsilon", type=float, default=0.003)
        parser.add_argument("--freeze_observation_normalization", type=str2bool, default=False,
                            help="Disables updates to observation normalization constants.")
        parser.add_argument("--max_micro_batch_size", type=int, default=512, help="Can be useful to limit GPU memory")
        parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                            help="Enables synchronous environments (slower, but helpful for debuging env errors).")
        parser.add_argument("--benchmark_mode", type=str2bool, default=False, help="Enables benchmarking mode.")
        parser.add_argument("--precision", type=str, default="medium", help="low|medium|high")

        # --------------------------------
        # Other

        parser.add_argument("--gamma", type=float, default=0.999, help="Discount rate for extrinsic rewards")
        parser.add_argument("--gamma_int", type=float, default=0.99,
                            help="Discount rate for intrinsic rewards")  # tood: rename to int_gamma
        parser.add_argument("--lambda_policy", type=float, default=0.95, help="GAE parameter.")
        parser.add_argument("--lambda_value", type=float, default=0.95,
                            help="lambda to use for return estimations when using PPO or DNA")
        parser.add_argument("--max_grad_norm", type=float, default=20.0, help="Clipping used when global_norm is set.")
        parser.add_argument("--grad_clip_mode", type=str, default="global_norm", help="[off|global_norm]")

        # --------------------------------
        # Rewards
        parser.add_argument("--override_reward_normalization_gamma", type=float, default=None)

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

        self._add_remappings()
        self._parse()
        self.auto()
        self.verify()

    def auto(self):
        """
        Apply special config settings.
        """

        super().auto()

        # We used to use restore=True instead of restore = "always"
        if self.restore in ["True", "true", True]:
            self.restore = "always"

    def verify(self):
        super().verify()

        valid_restore = ["always", "never", "auto"]
        assert self.restore in valid_restore, f"Expecting {self.restore} to be one of {valid_restore} but was {self.restore}"


    def _add_remappings(self):
        """
        Add stubs for parameters that have been removed.
        """
        for k, v in self._REMAPPED_PARAMS.items():
            self._parser.add_argument(f"--{k}", type=str, default=None, help=argparse.SUPPRESS)

    def _parse(self, args_override=None):
        """
        Setup config values based on commandline args and (potentially) and specified overrides.
        """

        # clean this up...
        parser = self._parser
        REMAPPED_PARAMS = self._REMAPPED_PARAMS
        args = self

        cmd_args = parser.parse_args(args_override).__dict__.copy()
        args.update(cmd_args)

        # check if anything was missing...
        if len(cmd_args) > 0:
            raise ValueError(f"Found the following parameters that could not be linked. {list(cmd_args.keys())}")

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
                print(
                    f"Warning! Using deprecated parameter {FAIL}{old_name}{ENDC} which is being mapped to {BOLD}{new_name}{ENDC} with value {legacy_value}")
                vars(args)[new_name] = cast_legacy_value
                del vars(args)[old_name]
            else:
                non_legacy_value = vars(args)[new_name]
                print(
                    f"Warning! Using deprecated parameter {FAIL}{old_name}{ENDC} was specified but clashes with value assigned to {BOLD}{new_name}{ENDC}. Using legacy value {legacy_value} overwriting {non_legacy_value}.")
                vars(args)[new_name] = cast_legacy_value

        if args.hash.bonus != 0:
            assert args.hash.enabled, "use_hashing must be enabled."

    @property
    def reward_normalization_gamma(self):
        gamma = self.tvf.gamma if self.tvf.enabled else self.gamma
        if self.override_reward_normalization_gamma is not None:
            gamma = self.override_reward_normalization_gamma
        return gamma

    @property
    def use_intrinsic_rewards(self):
        return self.rnd.enabled or (self.hash.bonus != 0)

    @property
    def tvf_return_n_step(self):
        if self.lambda_value >= 1:
            return self.env.timeout
        else:
            return round(1/(1-self.lambda_value))

    @property
    def get_mutex_key(self):
        if self.mutex_key.lower() == 'device':
            return args.device
        else:
            return self.mutex_key

    @property
    def batch_size(self):
        return self.n_steps * self.agents

LOCK_KEY = str(uuid.uuid4().hex)
args:Config = Config()

if __name__ == "__main__":
    pass