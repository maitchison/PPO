import uuid
import socket
import argparse
import torch
from . import utils

class Config:

    def __init__(self, **kwargs):
        # put these here just so IDE can detect common parameters...
        self.environment = ""
        self.experiment_name = ""
        self.run_name = ""
        self.agents = 0
        self.filter = ""

        self.hash_size = 0
        self.restore = False

        self.gamma = 0.0
        self.gamma_int = 0.0
        self.gae_lambda = 0.0
        self.ppo_epsilon = 0.0
        self.vf_coef = 0.0
        self.max_grad_norm = 0.0

        self.freeze_layers = 0

        self.input_crop = False
        self.learning_rate = 0.0
        self.learning_rate_decay = 0.0
        self.adam_epsilon = 0.0
        self.workers = 0
        self.n_steps = 0
        self.epochs = 0
        self.limit_epochs = 0
        self.batch_epochs = 0

        self.observation_normalization = False
        self.intrinsic_reward_scale = 0.0

        self.tensorboard_logging = False

        self.reward_clip = 0.0
        self.reward_normalization = True

        self.mini_batch_size = 0
        self.sync_envs = False
        self.resolution = ""
        self.color = False
        self.entropy_bonus = 0.0
        self.threads = 0
        self.dtype = ""
        self.export_video = False
        self.device = ""
        self.save_checkpoints = False
        self.output_folder = ""
        self.hostname = ""
        self.sticky_actions = False
        self.guid = ""

        self.use_icm = False
        self.icm_eta = 0.0

        self.use_rnd = False

        self.model = None
        self.model_hidden_units = 0

        self.memorize_cards = 0
        self.memorize_actions = 0

        self.debug_print_frequency = 0
        self.debug_log_frequency = 0
        self.noop_start = False

        self.log_folder = ""

        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

LOCK_KEY = str(uuid.uuid4().hex)

# debugging variables.
PROFILE_INFO = False
VERBOSE = True

CHECKPOINT_EVERY_STEPS = int(5e6)

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

    parser.add_argument("--observation_normalization", type=str2bool, default=False)
    parser.add_argument("--intrinsic_reward_scale", type=float, default=0.5)

    parser.add_argument("--tensorboard_logging", type=str2bool, default=False)
    parser.add_argument("--freeze_layers", type=int, default=0, help="Freeze the nth first layers in model.")

    parser.add_argument("--reward_normalization", type=str2bool, default=True)
    parser.add_argument("--reward_clip", type=float, default=5.0)

    parser.add_argument("--mini_batch_size", type=int, default=1024)
    parser.add_argument("--sync_envs", type=str2bool, nargs='?', const=True, default=False,
                        help="Enables synchronous environments (slower).")
    parser.add_argument("--resolution", type=str, default="standard", help="['full', 'standard', 'half']")
    parser.add_argument("--color", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--entropy_bonus", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", type=str, default=torch.float)
    parser.add_argument("--export_video", type=str2bool, default=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save_checkpoints", type=str2bool, default=True)
    parser.add_argument("--output_folder", type=str, default="./")
    parser.add_argument("--hostname", type=str, default=socket.gethostname())
    parser.add_argument("--sticky_actions", type=str2bool, default=False)
    parser.add_argument("--guid", type=str, default=None)
    parser.add_argument("--noop_start", type=str2bool, default=True)

    # icm stuff
    parser.add_argument("--use_icm", type=str2bool, default=False, help="Enables the Intrinsic Motivation Module (IDM).")
    parser.add_argument("--icm_eta", type=float, default=0.01,
                        help="How much to weight intrinsic rewards in ICM.")

    parser.add_argument("--use_rnd", type=str2bool, default=False,
                        help="Enables the Random Network Distilation (RND) module.")

    # debuging
    parser.add_argument("--debug_print_freq", type=int, default=60, help="Number of seconds between debug prints.")
    parser.add_argument("--debug_log_freq", type=int, default=300, help="Number of seconds between log writes.")

    # model
    parser.add_argument("--model", type=str, default="cnn", help="['cnn']")
    parser.add_argument("--model_hidden_units", type=int, help="Number of hidden units in model.")

    # memorize game
    parser.add_argument("--memorize_cards", type=int, default=100, help="Memorize environment: Number of cards in the game.")
    parser.add_argument("--memorize_actions", type=int, default=2,
                        help="Memorize environment: Number of actions to pick from.")

    args.update(**parser.parse_args().__dict__)

