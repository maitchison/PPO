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
        parser.add_argument("--tvf_trimming", type=str, default='off', help="off|timelimit|av_term|est_term")
        parser.add_argument("--tvf_trimming_mode", type=str, default='average', help="interpolate|average")
        parser.add_argument("--tvf_at_minh", type=int, default=128)
        parser.add_argument("--tvf_at_percentile", type=float, default=90)
        parser.add_argument("--tvf_horizon_dropout", type=float, default=0.0, help="fraction of horizons to exclude per epoch")
        parser.add_argument("--tvf_return_mode", type=str, default="exponential", help="[fixed|adaptive|exponential|geometric|advanced]")
        parser.add_argument("--tvf_return_samples", type=int, default=32, help="Number of n-step samples to use for distributional return calculation")
        parser.add_argument("--tvf_return_use_log_interpolation", type=str2bool, default=False, help="Interpolates in log space.")
        parser.add_argument("--tvf_max_horizon", type=int, default=1000, help="Max horizon for TVF.")
        parser.add_argument("--tvf_value_heads", type=int, default=64, help="Number of value heads to use.")
        parser.add_argument("--tvf_head_spacing", type=str, default="geometric", help="[geometric|linear|even_x]")
        parser.add_argument("--tvf_head_weighting", type=str, default="off", help="[off|h_weighted]")
        parser.add_argument("--tvf_activation", type=str, default="relu", help="[relu|tanh|sigmoid]")
        parser.add_argument("--tvf_per_head_hidden_units", type=int, default=0, help="Number of units in each heads hidden layer")
        parser.add_argument("--tvf_feature_window", type=int, default=-1,
                            help="Limits each head to a window of this many features.")
        parser.add_argument("--tvf_feature_sparsity", type=float, default=0.0, help="Zeros out this proprition of features for each head")
        parser.add_argument("--tvf_include_ext", type=str2bool, default=True, help="Also learn the rediscounted value estimate that will be used for advantages.")
        parser.add_argument("--tvf_sqrt_transform", type=str2bool, default=False,
                            help="Learns the (signed) sqrt of the return. May make weight scale more uniform across heads.")
        parser.add_argument("--tvf_boost_final_head", type=float, default=0.0,
                            help="Increases loss on final tvf head.")



class OptimizerConfig(BaseConfig):
    """
    Config settings for (generic) optimizer
    """
    def __init__(self, prefix: str = '', parser=None):

        super().__init__(prefix)

        if parser is None:
            return

        epoch_defaults = {
            'value': 2,
            'policy': 2,
            'distil': 2,
            'aux': 0,
            'rnd': 1,
        }

        epoch_default = epoch_defaults.get(prefix, 0)

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
        parser.add_argument("--procgen_difficulty", type=str, default="hard", help="[hard|]")
        parser.add_argument("--restore", type=str, default='never', help="Restores previous model. 'always' will restore, or error, 'never' will not restore, 'auto' will restore if it can.")
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
        parser.add_argument("--save_initial_checkpoint", type=str2bool, default=True, help="Saves a checkpoint before any training.")
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
        parser.add_argument("--checkpoint_every", type=int, default=int(5e6), help="Number of environment steps between checkpoints.")
        parser.add_argument("--log_folder", type=str, default=None)
        parser.add_argument("--observation_normalization", type=str2bool, default=False)
        parser.add_argument("--observation_normalization_epsilon", type=float, default=0.003)
        parser.add_argument("--freeze_observation_normalization", type=str2bool, default=False,
                            help="Disables updates to observation normalization constants.")
        parser.add_argument("--max_micro_batch_size", type=int, default=512, help="Can be useful to limit GPU memory")
        parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                            help="Enables synchronous environments (slower, but helpful for debuging env errors).")
        parser.add_argument("--benchmark_mode", type=str2bool, default=False, help="Enables benchmarking mode.")
        parser.add_argument("--precision", type=str, default="medium", help="low|medium|high")
        parser.add_argument("--head_bias", type=str2bool, default=True, help="Enables bias on output heads")

        # --------------------------------
        # Episodic Discounting
        parser.add_argument("--use_ed", type=str2bool, default=False, help="Enables episodic discounting.")
        parser.add_argument("--ed_mode", type=str, default="power", help="power|quadratic|none")
        parser.add_argument("--ed_bias", type=float, default=1, help="added to t.")

        # --------------------------------
        # Rewards
        parser.add_argument("--tvf_return_estimator_mode", type=str, default="default",
                            help='Allows the use of the reference return estimator (very slow). [default|reference|verify|historic]')
        parser.add_argument("--ir_propagation", type=str2bool, default=True, help="allows intrinsic returns to propagate through end of episode.")
        parser.add_argument("--override_reward_normalization_gamma", type=float, default=None)

        # --------------------------------
        # Model

        parser.add_argument("--encoder", type=str, default="nature", help="Encoder used for all models, [nature|impala]")
        parser.add_argument("--encoder_args", type=str, default=None, help="Additional arguments for encoder. (encoder specific)")
        parser.add_argument("--hidden_units", type=int, default=256)
        parser.add_argument("--architecture", type=str, default="dual", help="[dual|single]")
        parser.add_argument("--gamma_int", type=float, default=0.99, help="Discount rate for intrinsic rewards") # tood: rename to int_gamma
        parser.add_argument("--gamma", type=float, default=0.999, help="Discount rate for extrinsic rewards")
        parser.add_argument("--lambda_policy", type=float, default=0.95, help="GAE parameter.")
        parser.add_argument("--lambda_value", type=float, default=0.95, help="lambda to use for return estimations when using PPO or DNA")
        parser.add_argument("--max_grad_norm", type=float, default=20.0, help="Clipping used when global_norm is set.")
        parser.add_argument("--grad_clip_mode", type=str, default="global_norm", help="[off|global_norm]")
        parser.add_argument("--head_scale", type=float, default=0.1, help="Scales weights for value and policy heads.")

        # --------------------------------
        # Extra

        parser.add_argument("--use_gkl", type=str2bool, default=False, help="Use a global kl constraint.")
        parser.add_argument("--gkl_threshold", type=float, default=-1) # 0.004 is probably good.
        parser.add_argument("--gkl_penalty", type=float, default=0.01)
        parser.add_argument("--gkl_source", type=str, default="rollout", help="[rollout]")
        parser.add_argument("--gkl_samples", type=int, default=1024, help="Number of samples to use for global sample of state distrubtion.")
        # --------------------------------
        # Environment
        parser.add_argument("--env_type", type=str, default="atari", help="[atari|mujoco|procgen]")
        parser.add_argument("--warmup_period", type=int, default=250,
                            help="Number of random steps to take before training agent.")
        parser.add_argument("--timeout", type=str, default="auto",
                            help="Set the timeout for the environment, 0=off, (given in unskipped environment steps)")
        parser.add_argument("--repeat_action_probability", type=float, default=0.0)
        parser.add_argument("--noop_duration", type=int, default=30, help="maximum number of no-ops to add on reset")
        parser.add_argument("--per_step_termination_probability", type=float, default=0.0,
                            help="Probability that each step will result in unexpected termination (used to add noise to value).")
        parser.add_argument("--reward_clipping", type=str, default="off", help="[off|[<R>]|sqrt]")
        parser.add_argument("--reward_normalization", type=str, default="rms", help="off|rms|ema")
        parser.add_argument("--reward_normalization_clipping", type=float, default=10, help="how much to clip rewards after normalization, negative to disable")
        parser.add_argument("--reward_normalization_horizon", type=float, default=5e6, help="how much to smooth variance estimates for ema mode.")
        parser.add_argument("--reward_normalization_correction", type=str2bool, default=False, help="")
        parser.add_argument("--rnc_no_value", type=str2bool, default=False)
        parser.add_argument("--rnc_value_only", type=str2bool, default=False)

        parser.add_argument("--reward_curve", type=float, default=-1,
                            help="Rewards get larger over time, set to 1/1000 or so. Negative means disabled.")

        parser.add_argument("--deferred_rewards", type=int, default=0,
                            help="If positive, all rewards accumulated so far will be given at time step deferred_rewards, then no reward afterwards.")
        # (atari)
        parser.add_argument("--resolution", type=str, default="nature", help="[full|nature|half|muzero]")
        parser.add_argument("--color_mode", type=str, default="default", help="default|bw|rgb|yuv|hsv")
        parser.add_argument("--full_action_space", type=str2bool, default=False)
        parser.add_argument("--terminal_on_loss_of_life", type=str2bool, default=False)
        parser.add_argument("--frame_stack", type=int, default=4)
        parser.add_argument("--frame_skip", type=int, default=4)
        parser.add_argument("--embed_time", type=str2bool, default=True, help="Encodes time into observation")
        parser.add_argument("--embed_action", type=str2bool, default=True, help="Encodes actions into observation")
        parser.add_argument("--embed_state", type=str2bool, default=False, help="Encodes state history into observation")
        parser.add_argument("--atari_rom_check", type=str2bool, default=True, help="Verifies on load, that the MD5 of atari ROM matches the ALE.")
        # (stuck)
        parser.add_argument("--max_repeated_actions", type=int, default=100, help="Agent is given a penalty if it repeats the same action more than this many times.")
        parser.add_argument("--repeated_action_penalty", type=float, default=0.0, help="Penalty if agent repeats the same action more than this many times.")

        # -----------------
        # Noisy environments
        parser.add_argument("--noisy_return", type=float, default=0, help="Relative error applied after return calculations. Used to simulate a noisy environment.")
        parser.add_argument("--noisy_reward", type=float, default=0, help="Relative error applied to all rewards. Used to simulate a noisy environment.")
        parser.add_argument("--noisy_reward_v2", type=float, default=0,
                            help="Relative error applied to all rewards. Used to simulate a noisy environment.")
        parser.add_argument("--noisy_zero", type=float, default=-1, help="Instead of environment rewards, agent is given random rewards drawn from gausian with this std.")

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
        parser.add_argument("--debug_zero_obs", type=str2bool, default=False, help="")
        parser.add_argument("--debug_log_rediscount_curve", type=str2bool, default=False, help="")
        parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
        parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")
        parser.add_argument("--debug_checkpoint_slides", type=str2bool, default=True, help="Generates state images during epoch saves.")
        parser.add_argument("--debug_state_distort", type=int, default=-1, help="After this many (per agent) frames apply a negation filter. (< 0 for disabled)")


        # --------------------------------
        # Auto Gamma
        parser.add_argument("--use_ag", type=str2bool, default=False, help="Enables auto gamma")
        parser.add_argument("--ag_mode", type=str, default="episode_length", help="[episode_length|training|sns|shadow|h_best]")
        parser.add_argument("--ag_target", type=str, default="policy", help="[policy|value|both]")
        parser.add_argument("--ag_ratio_threshold", type=float, default=0.3, help="target variance ratio for gamma.")
        parser.add_argument("--ag_ratio_source", type=str, default="value", help="returns|value|td|advantages")
        parser.add_argument("--ag_ratio_algorithm", type=str, default="best", help="min|best|adv")
        parser.add_argument("--ag_ratio_factor", type=float, default=0.1, help="how much noise can we accept to increase horizon by 10x")
        parser.add_argument("--ag_sns_threshold", type=float, default=10.0, help="target noise level for gamma.")
        parser.add_argument("--ag_sns_ema_horizon", type=float, default=int(5e6),
                            help="horizon used in EMA for horizon.")
        parser.add_argument("--ag_delay", type=int, default=int(5e6),
                            help="alpha value used in EMA for horizon.")
        parser.add_argument("--ag_min_h", type=int, default=100,  # I'd like to make this 50
                            help="Minimum auto gamma horizon.")
        parser.add_argument("--ag_initial_h", type=int, default=1000,
                            help="Initial auto gamma horizon.")
        parser.add_argument("--ag_max_h", type=int, default=10000,
                            help="Maximum auto gamma horizon.")

        # --------------------------------
        # Simple Noise Scale
        parser.add_argument("--use_sns", type=str2bool, default=False, help="Enables generation of simple noise scale estimates")
        parser.add_argument("--sns_labels", type=str, default="['policy','distil','value', 'value_heads']", help="value|value_heads|distil|policy"),
        parser.add_argument("--sns_period", type=int, default=3, help="Generate estimates every n updates.")
        parser.add_argument("--sns_max_heads", type=int, default=7, help="Limit to this number of heads when doing per head noise estimate.")
        parser.add_argument("--sns_b_big", type=int, default=2048, help="")
        parser.add_argument("--sns_b_small", type=int, default=128, help="")
        parser.add_argument("--sns_fake_noise", type=str2bool, default=False, help="Replaces value_head gradient with noise based on horizon.")
        parser.add_argument("--sns_smoothing_mode", type=str, default="ema", help="ema|avg")
        parser.add_argument("--sns_smoothing_horizon_avg", type=int, default=1e6, help="how big to make averaging window")
        parser.add_argument("--sns_smoothing_horizon_s", type=int, default=0.2e6, help="how much to smooth s")
        parser.add_argument("--sns_smoothing_horizon_g2", type=int, default=1.0e6, help="how much to smooth g2")
        parser.add_argument("--sns_smoothing_horizon_policy", type=int, default=5e6, help="how much to smooth g2 for policy (normally much higher)")

        # --------------------------------
        # Auxiliary phase
        parser.add_argument("--aux_target", type=str, default='reward', help="[reward|vtarg]]")
        parser.add_argument("--aux_source", type=str, default='aux', help="[aux|value]]")
        parser.add_argument("--aux_period", type=int, default=0, help="")

        # --------------------------------
        # Distil phase
        parser.add_argument("--distil_order", type=str, default="after_policy", help="after_policy|before_policy")
        parser.add_argument("--distil_beta", type=float, default=10.0)
        parser.add_argument("--distil_l1_scale", type=float, default=1/30)
        parser.add_argument("--shared_distil_optimizer", type=str2bool, default=False)

        # extra
        parser.add_argument("--distil_target", type=str, default="value", help="return|value|advantage")
        parser.add_argument("--distil_lambda", type=float, default=0.6, help="Used for return or advantage distil targets")

        parser.add_argument("--distil_delta", type=float, default=0.1)
        parser.add_argument("--distil_period", type=int, default=1)
        parser.add_argument("--distil_loss", type=str, default="kl_policy", help="[mse_logit|mse_policy|kl_policy]")
        parser.add_argument("--distil_batch_size", type=int, default=None, help="Size of batch to use when training distil. Defaults to rollout_size.")
        parser.add_argument("--distil_freq_ratio", type=float, default=None, help="Sets distil period to replay_size / batch_size * distil_freq_ratio")
        parser.add_argument("--distil_batch_size_ratio", type=float, default=None, help="Sets distil_batch_size to rollout_size * distil_batch_size_ratio")
        parser.add_argument("--distil_max_heads", type=int, default=8+1, help="Max number of heads to apply distillation to.")
        parser.add_argument("--distil_force_ext", type=str2bool, default=False,
                            help="use value_ext instead of tvf heads for distilation.")
        parser.add_argument("--distil_rediscount", type=str2bool, default=False, help="uses rediscounted targets for distillation.")
        parser.add_argument("--distil_renormalize", type=str2bool, default=False)
        parser.add_argument("--distil_reweighing", type=str2bool, default=False,
                            help="Reduces loss for horizons which have been discounted away. Generally better than rediscounting")
        parser.add_argument("--distil_loss_value_target", type=float, default=None,
                            help="Normalizes loss_value_distil to approximately this level. Useful if distil_rediscount is enabled.")
        parser.add_argument("--distil_lvt_mode", type=str, default="first", help="first|mean")
        parser.add_argument("--distil_value_loss", type=str, default="mse", help="mse|clipped_mse|l1|huber")

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

        parser.add_argument("--use_ebd", type=str2bool, default=False,
                            help="Enables Exploration by Disagreement (EBD) module (a poor mans rnd).")
        parser.add_argument("--use_hashing", type=str2bool, default=False,
                            help="Enables state hashing (used to track exploration.")
        parser.add_argument("--hash_bits", type=int, default=16,
                            help="Number of bits to hash to, requires O(2^n) memory.")
        parser.add_argument("--hash_bonus", type=float, default=0.0,
                            help="Intrinsic reward bonus for novel hashed states")
        parser.add_argument("--hash_method", type=str, default="linear",
                            help="linear|conv")
        parser.add_argument("--hash_input", type=str, default="raw",
                            help="raw|raw_centered|normed|normed_offset")
        parser.add_argument("--hash_bonus_method", type=str, default="hyperbolic", help="hyperbolic|quadratic|binary")
        parser.add_argument("--hash_rescale", type=int, default=1)
        parser.add_argument("--hash_quantize", type=float, default=1)
        parser.add_argument("--hash_bias", type=float, default=0.0)
        parser.add_argument("--hash_decay", type=float, default=0.99)


        parser.add_argument("--ir_scale", type=float, default=0.3, help="Intrinsic reward scale.")
        parser.add_argument("--ir_center", type=str2bool, default=False, help="Per-batch centering of intrinsic rewards.")
        parser.add_argument("--ir_normalize", type=str2bool, default=True, help="Normalizes intrinsic rewards such that they have unit variance")



        # --------------------------------
        # Temp, remove
        parser.add_argument("--tmp_median_return", type=str2bool, default=False, help="Use median of return samples rather than mean")


        # this is just so we get autocomplete, as well as IDE hints if we spell something wrong

        self.use_tvf = bool()
        self.environment = str()
        self.experiment_name = str()
        self.run_name = str()
        self.procgen_difficulty = str()
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
        self.observation_normalization_epsilon = float()
        self.freeze_observation_normalization = bool()
        self.max_micro_batch_size = int()
        self.sync_envs = bool()
        self.benchmark_mode = bool()

        self.ir_propagation = bool()
        self.ir_normalize = bool()
        self.ir_center = bool()
        self.ir_scale = float()

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
        self.per_step_termination_probability = float()
        self.reward_clipping = str()
        self.reward_normalization = str()
        self.reward_normalization_clipping = float()
        self.reward_normalization_horizon = float()
        self.reward_normalization_correction = bool()
        self.rnc_no_value = bool()
        self.rnc_value_only = bool()
        self.reward_curve = float()
        self.deferred_rewards = int()
        self.resolution = str()
        self.color_mode = str()
        self.max_repeated_actions = int()
        self.repeated_action_penalty = float()
        self.full_action_space = bool()
        self.terminal_on_loss_of_life = bool()
        self.frame_stack = int()
        self.frame_skip = int()
        self.embed_time = bool()
        self.embed_action = bool()
        self.embed_state = bool()
        self.atari_rom_check = bool()
        self.shared_distil_optimizer = bool()

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
        self.tvf_trimming = str()
        self.tvf_trimming_mode = str()
        self.tvf_at_minh = int()
        self.tvf_at_percentile = float()
        self.tvf_horizon_dropout = float()
        self.tvf_return_mode = str()
        self.tvf_return_samples = int()
        self.tvf_return_use_log_interpolation = bool()
        self.tvf_max_horizon = int()
        self.tvf_boost_final_head = float()
        self.tvf_value_heads = int()
        self.tvf_head_spacing = str()
        self.tvf_head_weighting = str()
        self.tvf_per_head_hidden_units = int()
        self.tvf_feature_sparsity = float()
        self.tvf_feature_window = int()
        self.tvf_include_ext = bool()
        self.tvf_sqrt_transform = bool()

        self.debug_zero_obs = bool()
        self.debug_log_rediscount_curve = bool()
        self.debug_checkpoint_slides = bool()
        self.debug_state_distort = bool()

        self.use_ag = bool()
        self.ag_mode = str()
        self.ag_target = str()
        self.ag_sns_threshold = float()
        self.ag_sns_ema_horizon = float()
        self.ag_ratio_threshold = float()
        self.ag_ratio_source = str()
        self.ag_ratio_algorithm = str()
        self.ag_ratio_factor = float()
        self.ag_delay = int()
        self.ag_min_h = int()
        self.ag_initial_h = int()
        self.ag_max_h = int()

        self.use_sns = bool()
        self.sns_labels = str()
        self.sns_period = int()
        self.sns_max_heads = int()
        self.sns_b_big = int()
        self.sns_b_small = int()
        self.sns_fake_noise = int()
        self.sns_smoothing_mode = str()
        self.sns_smoothing_horizon_avg = int()
        self.sns_smoothing_horizon_s = int()
        self.sns_smoothing_horizon_g2 = int()
        self.sns_smoothing_horizon_policy = int()

        self.use_ed = bool()
        self.ed_mode = str()
        self.ed_bias = float()

        # extra
        self.use_gkl = bool()
        self.gkl_threshold = float()
        self.gkl_penalty = float()
        self.gkl_source = str()
        self.gkl_samples = int()
        self.head_scale = float()

        self.aux_target = str()
        self.aux_source = str()
        self.aux_period = int()
        self.distil_order = str()
        self.distil_target = str()
        self.distil_lambda = str()
        self.distil_beta = float()
        self.distil_l1_scale = float()
        self.distil_delta = float()
        self.distil_period = int()
        self.distil_loss = str()
        self.distil_batch_size = object()
        self.distil_freq_ratio = float()
        self.distil_batch_size_ratio = float()
        self.distil_max_heads = int()
        self.distil_force_ext = bool()
        self.distil_rediscount = bool()
        self.distil_renormalize = bool()
        self.distil_reweighing = bool()
        self.distil_loss_value_target = float()
        self.distil_lvt_mode = str()
        self.distil_value_loss = str()
        self.replay_mode = str()
        self.replay_size = int()
        self.replay_mixing = bool()
        self.replay_thinning = float()
        self.use_rnd = bool()
        self.use_ebd = bool()
        self.use_hashing = bool()
        self.hash_method = str()
        self.hash_input = str()
        self.hash_rescale = int()
        self.hash_quantize = float()
        self.hash_bits = int()
        self.hash_bonus = float()
        self.hash_decay = float()
        self.hash_bonus_method = str()
        self.hash_bias = float()
        self.rnd_experience_proportion = float()

        # noise stuff
        self.noisy_return = float()
        self.noisy_reward = float()
        self.noisy_reward_v2 = float()
        self.noisy_zero = float()
        self.precision = str()

        self.head_bias = bool()

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
        return self.use_rnd or self.use_ebd or (self.hash_bonus != 0)

    @property
    def tvf_return_n_step(self):
        if self.lambda_value >= 1:
            return self.timeout//self.frame_skip
        else:
            return round(1/(1-self.lambda_value))

    @property
    def get_mutex_key(self):
        if self.mutex_key.lower() == 'device':
            return args.device
        else:
            return self.mutex_key

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
        'tvf_head_sparsity': 'tvf_feature_sparsity',
        'export_video': None,
        'tvf_value_distribution': None,
        'tvf_horizon_distribution': None,
        'tvf_value_samples': None,
        'tvf_horizon_samples': 'tvf_value_heads',
        'tvf_hidden_units': None,
        'tvf_horizon_trimming': None,
        'tvf_force_ext_value_distil': None,
        'tvf_horizon_scale': None,
        'tvf_time_scale': None,
        'tvf_mode': None,
        'tvf_sum_horizons': None,
        'sns_small_samples': None,
        'tvf_return_n_step': None,
        'color': None,

        "ag_sns_delay": "ag_delay",
        "ag_sns_min_h": "ag_min_h",
        "ag_sns_max_h": "ag_max_h",
        "ag_sns_initial_h": "ag_initial_h",

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

    assert args.tvf_return_estimator_mode in ["default", "reference", "verify", "historic"]

    # set defaults
    if args.tvf_gamma is None:
        args.tvf_gamma = args.gamma
    if args.distil_batch_size is None:
        args.distil_batch_size = args.batch_size

    if args.hash_bonus != 0:
        assert args.use_hashing, "use_hashing must be enabled."

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

    if args.use_ebd:
        assert args.distil.epochs > 0, "EBD requires distilation to be enabled."

    # fixup horizon trimming
    if str(args.tvf_horizon_trimming) == 'False':
        args.tvf_horizon_trimming = "off"
    if str(args.tvf_horizon_trimming) == 'True':
        args.tvf_horizon_trimming = "interpolate"

    # normalization used to be a bool
    try:
        bool_value = str2bool(args.reward_normalization)
        print(f"Warning! Using deprecated version of {FAIL}reward_normalization {ENDC}. This should now be a string, not a bool.")
        if bool_value:
            args.reward_normalization = "rms"
        else:
            args.reward_normalization = "off"
    except Exception as e:
        # this just means we are not using the old bool values
        pass

    # timeout
    if args.timeout == "auto":
        if args.env_type == "atari":
            args.timeout = 108000 # includes skipped frames
        elif args.env_type == "procgen":
            # might be more fair to just set this so something like 8000 for all envs?
            # the trimming can auto adapt so it's just when we put the time frac into the obs
            # maybe we should use log time instead?
            if args.environment in ['bigfish', 'plunder', 'bossfight']:
                args.timeout = 8000
            else:
                args.timeout = 1000
        else:
            args.timeout = 0  # unlimited

    else:
        args.timeout = int(args.timeout)

    # color mode
    assert args.color_mode in ["default", "bw", "rgb", "yuv", "hsv"]
    if args.color_mode == "default":
        args.color_mode = {
            'atari': 'bw',
            'procgen': 'rgb',
        }.get(args.env_type, 'bw')

    if args.use_ag and args.ag_mode in ['sns', 'shadow']:
        assert 'value_heads' in ast.literal_eval(args.sns_labels), "sns_labels must include value_head"


if __name__ == "__main__":
    pass
    # c = Config()
    # args = c.parser.parse_args({'environment': 'Pong'})
    # c.update(**vars(args))
    #
    # c._print_vars()