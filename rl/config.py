import uuid
import socket
import argparse

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
        self.epochs             = int()
        self.limit_epochs       = int()
        self.batch_epochs       = int()

        self.observation_normalization = bool()
        self.intrinsic_reward_scale = float()
        self.extrinsic_reward_scale = float()

        self.reward_normalization = bool()

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
        self.max_micro_batch_size = float()
        self.policy_mini_batch_size = int()
        self.value_mini_batch_size = int()

        self.use_tvf            = bool()
        self.tvf_coef           = float()
        self.tvf_max_horizon    = int()
        self.tvf_value_samples  = int()
        self.tvf_horizon_samples= int()
        self.tvf_value_distribution = str()
        self.tvf_value_distribution = str()
        self.tvf_gamma          = float()
        self.tvf_lambda         = float()
        self.tvf_lambda_samples = int()
        self.tvf_epsilon        = bool()
        self.tvf_horizon_warmup = float()
        self.tvf_hidden_units   = int()
        self.tvf_model          = str()
        self.tvf_activation     = str()
        self.tvf_first_and_last = float()
        self.tvf_soft_anchor    = float()
        self.tvf_horizon_scale  = str()

    
        self.time_aware = bool()
        self.ed_type = str()
        self.ed_gamma = float()

        # phasic
        self.policy_epochs = int()                            
        self.value_epochs = int()                            
        self.target_kl = float()                            
        self.ppo_epsilon =float()
        self.agents = int()
        self.n_steps = int()
        self.value_lr = float()
        self.policy_lr = float()

        #

        self.env_desync = bool()

        self.use_icm            = bool()
        self.icm_eta            = str()

        self.debug_print_freq   = int()
        self.debug_log_freq     = int()
        self.noop_duration      = int()

        self.deferred_rewards   = bool()

        self.frame_stack        = int()
        self.timeout = int()

        self.normalize_advantages = bool()

        self.use_clipped_value_loss = bool()
        self.reward_clipping    = str()

        self.log_folder         = str()
        self.checkpoint_every   = int()

        self.model              = str()
        self.use_rnd            = bool()

        self.per_step_reward    = float()
        self.debug_terminal_logging = bool()

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    @property
    def propagate_intrinsic_rewards(self):
        return not self.use_rnd

    @property
    def use_intrinsic_rewards(self):
        return self.use_rnd

    @property
    def normalize_intrinsic_rewards(self):
        return self.use_rnd

    @property
    def normalize_observations(self):
        return self.use_rnd

    @property
    def noop_start(self):
        return self.noop_duration > 0

    @property
    def batch_size(self):
        return self.n_steps * self.agents

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
                        help="Restores previous model if it exists. If set to false and new run will be started.")

    parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter.")
    parser.add_argument("--max_grad_norm", type=float, default=20.0, help="Clip gradients during training to this.")

    parser.add_argument("--input_crop", type=str2bool, default=False, help="Enables atari input cropping.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-5, help="Epsilon parameter for Adam optimizer")
    parser.add_argument("--workers", type=int, default=-1, help="Number of CPU workers, -1 uses number of CPUs")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Each epoch represents 1 million environment interactions.")
    parser.add_argument("--limit_epochs", type=int, default=None, help="Train only up to this many epochs.")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Loss multiplier for default value loss.")
    parser.add_argument("--timeout", type=int, default=60*60*30, help="Set the timeout for the environment, 0=off, (given in unskipped environment steps)")

    parser.add_argument("--use_tvf", type=str2bool, default=True, help="Use truncated value function.")
    parser.add_argument("--tvf_coef", type=float, default=0.1, help="Loss multiplier for TVF loss.")
    parser.add_argument("--tvf_gamma", type=float, default=None, help="Gamma for TVF, defaults to gamma")
    parser.add_argument("--tvf_lambda", type=float, default=1.0, help="Lambda for TVF(\lambda), negative values use n_step(-lambda)")
    parser.add_argument("--tvf_lambda_samples", type=int, default=16, help="Number of n-step samples to use for tvf_lambda calculation")
    parser.add_argument("--tvf_max_horizon", type=int, default=1000, help="Max horizon for TVF.")
    parser.add_argument("--tvf_value_samples", type=int, default=64, help="Number of values to sample during training.")
    parser.add_argument("--tvf_horizon_samples", type=int, default=64, help="Number of horizons to sample during training. (-1 = all)")
    parser.add_argument("--tvf_value_distribution", type=str, default="uniform", help="Sampling distribution to use when generating value samples.")
    parser.add_argument("--tvf_horizon_distribution", type=str, default="uniform", help="Sampling distribution to use when generating horizon samples.")
    parser.add_argument("--tvf_horizon_warmup", type=float, default=0, help="Fraction of training before horizon reaches max_horizon (-1 = all)")
    parser.add_argument("--tvf_hidden_units", type=float, default=512)
    parser.add_argument("--tvf_activation", type=str, default="relu", help="[relu|tanh|sigmoid]")
    parser.add_argument("--tvf_loss_weighting", type=str, default="default", help="IGNORED")
    parser.add_argument("--tvf_first_and_last", type=float, default=1/32, help="Fraction of horizon samples to dedicate to first and last horizons")
    parser.add_argument("--tvf_soft_anchor", type=float, default=1.0, help="MSE loss for V(*,0) being non-zero.")
    parser.add_argument("--tvf_horizon_scale", type=str, default="default", help="[default|centered]")
    parser.add_argument("--tvf_h_scale", type=str, default="", help="IGNORED")

    # phasic inspired stuff
    parser.add_argument("--policy_epochs", type=int, default=2, help="Number of policy training epochs per training batch.")
    parser.add_argument("--value_epochs", type=int, default=6, help="Number of value training epochs per training batch.")
    parser.add_argument("--target_kl", type=float, default=0.01, help="Approximate divergence before early stopping on policy.")
    parser.add_argument("--policy_mini_batch_size", type=int, default=2048)
    parser.add_argument("--value_mini_batch_size", type=int, default=256)
    parser.add_argument("--ppo_epsilon", type=float, default=0.2, help="PPO epsilon parameter.")
    parser.add_argument("--n_steps", type=int, default=256, help="Number of environment steps per training step.")
    parser.add_argument("--agents", type=int, default=256)

    parser.add_argument("--value_lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--policy_lr", type=float, default=1e-4, help="Learning rate for Adam optimizer")
    
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
    parser.add_argument("--noop_duration", type=int, default=30, help="maximum number of no-ops to add on reset")
    parser.add_argument("--per_step_reward", type=float, default=0.0)
    parser.add_argument("--reward_clipping", type=str, default="off", help="[off|[<R>]|sqrt]")
    parser.add_argument("--reward_normalization", type=str2bool, default=True)
    parser.add_argument("--deferred_rewards", type=str2bool, default=False)

    # episodic discounting
    parser.add_argument("--time_aware", type=str2bool, default=True)
    parser.add_argument("--ed_type", type=str, default="none", help="[none|geometric|hyperbolic]")
    parser.add_argument("--ed_gamma", type=float, default=1.0)

    parser.add_argument("--frame_stack", type=int, default=4)
    parser.add_argument("--env_desync", type=str2bool, default=True, help="Runs environments without policy to desync before training.")

    parser.add_argument("--log_folder", type=str, default=None)

    parser.add_argument("--use_clipped_value_loss", type=str2bool, default=True, help="Use the improved clipped value loss.")

    # icm stuff
    parser.add_argument("--use_icm", type=str2bool, default=False, help="Enables the Intrinsic Motivation Module (IDM).")
    parser.add_argument("--icm_eta", type=float, default=0.01,
                        help="How much to weight intrinsic rewards in ICM.")

    parser.add_argument("--use_rnd", type=str2bool, default=False,
                        help="Enables the Random Network Distillation (RND) module.")

    parser.add_argument("--normalize_advantages", type=str2bool, default=True)
    parser.add_argument("--intrinsic_reward_propagation", type=str2bool, default=None,
                        help="allows intrinsic returns to propagate through end of episode."
    )

    # debuging
    parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
    parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")
    parser.add_argument("--debug_terminal_logging", type=str2bool, default=False,
                        help="Log information around terminals.")
    parser.add_argument("--checkpoint_every", type=int, default=int(5e6),
                        help="Number of environment steps between checkpoints.")

    # model
    parser.add_argument("--model", type=str, default="cnn", help="['cnn']")

    if args_override is not None:
        args.update(**parser.parse_args(args_override).__dict__)
    else:
        args.update(**parser.parse_args().__dict__)

    # set defaults
    if args.intrinsic_reward_propagation is None:
        args.intrinsic_reward_propagation = args.use_rnd
    if args.tvf_gamma is None:
        args.tvf_gamma = args.gamma

    assert args.tvf_value_samples <= args.tvf_max_horizon, "tvf_value_samples must be <= tvf_max_horizon."
    assert args.tvf_horizon_samples <= args.tvf_max_horizon, "tvf_horizon_samples must be <= tvf_max_horizon."

