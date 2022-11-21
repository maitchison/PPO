import io
import unittest
import os
from datetime import datetime
import uuid
import rl.config
import multiprocessing

import gym.version

resolution_map = {
    "full": (210, 160),
    "nature": (84, 84),
    "muzero": (96, 96),
    "half": (105, 80)     # this may produce cleaner resampling
}

def get_previous_experiment_guid(experiment_path, run_name):
    """ Looks for previous experiment with same run_name. Returns the guid if found. """
    if not os.path.exists(experiment_path):
        return None
    for f in os.listdir(experiment_path):
        if f[:-(8+3)] == run_name:
            guid = f[-(8+1):-1]
            return guid
    return None


def get_n_actions(space):
    import gym
    if type(space) == gym.spaces.Discrete:
        return space.n
    elif type(space) == gym.spaces.Box:
        assert len(space.shape) == 1
        return space.shape[0]
    else:
        raise ValueError(f"Action space of type {type(space)} not implemented yet.")


def make_model(args, log=None):
    """
    Construct model based on env, and arguments.
    """

    from rl import models, envs
    import torch

    fake_env = envs.make_env(args.env_type, args.get_env_name())
    n_actions = get_n_actions(fake_env.action_space)
    obs_space = fake_env.observation_space.shape

    if log is not None:
        log.info("Playing {} with {} obs_space and {} actions.".format(args.environment, obs_space, n_actions))

    if args.tvf.enabled:
        tvf_fixed_head_horizons, tvf_weights = rl.tvf.get_value_head_horizons(args.tvf.value_heads, args.tvf.max_horizon, args.tvf.head_spacing, include_weight=True)
        args.tvf.value_heads = len(tvf_fixed_head_horizons) # sometimes this will not match (with even distribution for example)
    else:
        tvf_fixed_head_horizons = None
        tvf_weights = None

    value_head_names = ['ext']
    if args.use_intrinsic_rewards:
        value_head_names.append('int')

    model = models.TVFModel(
        encoder=args.encoder,
        encoder_args=args.encoder_args,
        input_dims=obs_space,
        actions=n_actions,
        device=args.device,
        dtype=torch.float32,
        use_rnd=args.use_rnd,
        encoder_activation_fn="tanh" if args.env_type == "mujoco" else "relu",
        tvf_fixed_head_horizons=tvf_fixed_head_horizons,
        tvf_fixed_head_weights=tvf_weights,
        architecture=args.architecture,
        hidden_units=args.hidden_units,
        observation_normalization=args.observation_normalization,
        freeze_observation_normalization=args.freeze_observation_normalization,
        tvf_feature_sparsity=args.tvf.feature_sparsity,
        tvf_feature_window=args.tvf.feature_window,
        head_scale=args.head_scale,
        value_head_names=tuple(value_head_names),
        norm_eps=args.observation_normalization_epsilon,
        head_bias=args.head_bias,
        observation_scaling=args.observation_scaling,
    )
    return model



def main():

    # see http://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # import here to make workers load faster / use less memory
    import torch.backends.cudnn, torch.backends.cuda
    from rl import utils, rollout
    from rl import ppo
    from rl.config import args
    import numpy as np

    if args.quiet_mode:
        log.print_level = log.WARN

    # work out device to use
    if args.device.lower() == "auto":
        args.device = utils.get_auto_device(utils.get_disallowed_devices())

    if args.device is None:
        log.important("Training preempted, no device available.")
        exit()

    log.compress_csv = args.debug.compress_csv

    log.info(f"System is host:<white>{args.hostname}<end> torch:{torch.__version__} cuda:{torch.version.cuda} gym:{gym.version.VERSION} numpy:{np.__version__} ")
    log.info(f"Using device: <white>{args.device}<end>")


    # check to see if the device we are using has been disallowed
    if args.device in utils.get_disallowed_devices():
        log.important("Training preempted, device is not allowed.")
        exit()

    # set threading
    torch.set_num_threads(int(args.threads))

    # calculate number of workers to use.
    if args.workers < 0:
        args.workers = multiprocessing.cpu_count()
        while args.agents % args.workers != 0:
            # make sure workers divides number of jobs.
            args.workers -= 1

    # set resolution
    args.resolution = args.resolution.lower()
    if args.resolution in resolution_map:
        args.res_x, args.res_y = resolution_map[args.resolution]
    else:
        raise Exception("Invalid resolution " + args.resolution)

    # check the output folder is valid...
    assert os.path.isdir(args.output_folder), "Can not find path " + args.output_folder

    # set a guid
    if args.restore in ["always", "auto"]:
        # look for a previous experiment and use it if we find one...
        guid = get_previous_experiment_guid(os.path.join(args.output_folder, args.experiment_name), args.run_name)
        if guid is None:
            if args.restore == "always":
                log.error(f"Could not restore experiment {args.experiment_name}:{args.run_name}. Previous run not found.")
            else:
                # this is fine, we are in auto mode
                args.guid = str(uuid.uuid4().hex)
        else:
            args.guid = guid
    else:
        args.guid = str(uuid.uuid4().hex)

    # if seed was defined then set the seed and enable deterministic mode.
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.run_benchmark = False
    else:
        torch.backends.cudnn.run_benchmark = True

    # sort out our precision..
    if args.precision == "low":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision == "medium":
        # these were the old default settings from PyTorch 1.7-1.11 and they make the most sense
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision == "high":
        # do not use for convolutions or for matrix multiply...
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        raise ValueError(f"Invalid precision mode {args.precision}")

    x = torch.tensor([[0.1, 0.1], [0.1, 0.1]], dtype=torch.float32, device=args.device)
    ident = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32, device=args.device)
    x = x @ ident
    delta = abs(0.1 - float(x[0, 0].detach().cpu()))
    print(f"Multiplication precision is ~{-np.log10((delta+1e-15) / 0.1):.1f} sig fig.")

    # work out the logging folder...
    args.log_folder = args.log_folder or "{} [{}]".format(os.path.join(args.output_folder, args.experiment_name, args.run_name), args.guid[-8:])
    log.info("Logging to folder " + args.log_folder)

    # population training gets a summary log, which we need to name differently as it can not be processed by
    # the analysis scripts (due to missing varaibles). The training_log_0.csv, training_log_1.csv can be read
    # just fine though.
    log.csv_path = os.path.join(args.log_folder, "training_log.csv")
    log.txt_path = os.path.join(args.log_folder, "log.txt")

    os.makedirs(args.log_folder, exist_ok=True)

    utils.lock_job(force=args.ignore_lock)
    actor_critic_model = make_model(args, log)

    if args.reference_policy is not None:
        assert args.architecture == "dual"
        # load only the policy parameters, and the normalization constants
        checkpoint = rollout.open_checkpoint(os.path.join(args.log_folder, args.reference_policy), map_location=args.device)
        policy_checkpoint = {k[len('policy_net.'):]: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("policy_net.")}
        actor_critic_model.policy_net.load_state_dict(policy_checkpoint)
        actor_critic_model.obs_rms = checkpoint['obs_rms']
        log.info(f"Loaded reference policy {args.reference_policy}.")

    ppo.train(actor_critic_model, log)

def log_code_info():
    """
    Logs information about codebase.
    """
    code_hash = code_diff.get_code_hash()
    code_date = code_diff.get_code_date()
    log.info(f"Using code {datetime.fromtimestamp(code_date).strftime('%m/%d/%Y, %H:%M:%S')} [{code_hash[:8]}]")

def run_unit_tests():
    """
    Runs the units tests.
    """
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)

    s = io.StringIO()
    runner = unittest.TextTestRunner(verbosity=2, stream=s)
    runner.run(suite)
    for line in s.getvalue().split("\n"):
        if line.strip() != "":
            log.info(line)


if __name__ == "__main__":

    # run unit tests every time we run
    # this way we get verification in the log file for each experiment.

    rl.config.args.setup()

    # install procgen... this is a bit dodgy, but it'll get things working on the cluster
    import sys
    if "procgen" in sys.argv:
        print("Installing Procgen...")
        import subprocess
        p = subprocess.Popen(["pip", "install", "procgen"], stdout=subprocess.PIPE)
        print(p.communicate())

    # special setup for mujoco
    if "--env_type=mujoco" in sys.argv:
        print("Setting up for mujoco")
        mujoco_path="/home/matthew/.mujoco/mujoco210/bin"
        # print("Old path was", os.environ["LD_LIBRARY_PATH"])
        if mujoco_path not in os.environ.get("LD_LIBRARY_PATH", ""):
            print(" - updating path.")
            os.environ["LD_LIBRARY_PATH"] = f":{mujoco_path}:/usr/lib/nvidia"

    from rl import logger, code_diff

    log = logger.Logger()
    print("=" * 80)
    log_code_info()
    run_unit_tests()
    print("=" * 90)
    try:
        main()
    except Exception as e:
        try:
            print("!" * 60)
            print(e)
            print("!" * 60)
            log.error("ERROR:"+str(e))
            import traceback
            log.error(traceback.format_exc())
            log.save_log()
        except Exception as logging_error:
            # just ignore any errors while trying to log result
            print(f"An error occurred while logging this error, {logging_error}")
        raise e