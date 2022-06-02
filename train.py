import os

import uuid
import multiprocessing

resolution_map = {
    "full": (210, 160),
    "nature": (84, 84),
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


def make_model(args):
    """
    Construct model based on env, and arguments.
    """

    from rl import rollout, models
    import torch

    fake_env = rollout.make_env(args.env_type, args.get_env_name())
    n_actions = get_n_actions(fake_env.action_space)
    obs_space = fake_env.observation_space.shape
    log.info("Playing {} with {} obs_space and {} actions.".format(args.environment, obs_space, n_actions))

    tvf_fixed_head_horizons = rollout.Runner.get_standard_horizon_sample(args.tvf_max_horizon)
    tvf_fixed_head_horizons = sorted(set(tvf_fixed_head_horizons))

    model = models.TVFModel(
        networks=(args.policy_network, args.value_network),
        network_args=(args.policy_network_args, args.value_network_args),
        input_dims=obs_space,
        actions=n_actions,
        device=args.device,
        dtype=torch.float32,

        use_rnd=args.use_rnd,
        use_rnn=False,
        tvf_mode=args.tvf_mode,
        tvf_horizon_transform=rollout.horizon_scale_function,
        tvf_time_transform=rollout.time_scale_function,
        tvf_max_horizon=args.tvf_max_horizon,
        tvf_value_scale_fn=args.tvf_value_scale_fn,
        tvf_value_scale_norm=args.tvf_value_scale_norm,
        feature_activation_fn="tanh" if args.env_type == "mujoco" else "relu",
        tvf_fixed_head_horizons=tvf_fixed_head_horizons,
        architecture=args.architecture,

        hidden_units=args.hidden_units,
        tvf_hidden_units=args.tvf_hidden_units,
        tvf_activation=args.tvf_activation,
        shared_initialization=args.dna_shared_initialization,
        observation_normalization=args.observation_normalization,
        freeze_observation_normalization=args.freeze_observation_normalization,
    )
    return model



def main():

    # see http://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # import here to make workers load faster / use less memory
    import torch
    from rl import utils, models, config, rollout
    from rl import ppo
    from rl.config import args
    import numpy as np
    import gym.spaces

    config.parse_args()

    if args.quite_mode:
        log.print_level = log.WARN

    # work out device to use
    if args.device.lower() == "auto":
        args.device = utils.get_auto_device(utils.get_disallowed_devices())

    if args.device is None:
        log.important("Training preempted, no device available.")
        exit()

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

    if args.use_icm and args.use_rnd:
        raise Exception("Can only use either ICM or RND, not both.")

    # check the output folder is valid...
    assert os.path.isdir(args.output_folder), "Can not find path " + args.output_folder

    # set a guid
    if args.restore:
        # look for a previous experiment and use it if we find one...
        guid = get_previous_experiment_guid(os.path.join(args.output_folder, args.experiment_name), args.run_name)
        if guid is None:
            if args.error_on_missing_restore:
                # exit with an error
                log.error(
                    "Could not restore experiment {}:{}. Previous run not found.".format(args.experiment_name, args.run_name))
                return
            else:
                # this is fine, we are in auto mode
                args.guid = str(uuid.uuid4().hex)
        else:
            args.guid = guid
    else:
        args.guid = str(uuid.uuid4().hex)

    # if seed was defined then set the seed and enable determanistic mode.
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.run_benchmark = False
    else:
        torch.backends.cudnn.run_benchmark = True

    # work out the logging folder...
    args.log_folder = args.log_folder or "{} [{}]".format(os.path.join(args.output_folder, args.experiment_name, args.run_name), args.guid[-8:])
    log.info("Logging to folder " + args.log_folder)

    # population training gets a summary log, which we need to name differently as it can not be processed by
    # the analysis scripts (due to missing varaibles). The training_log_0.csv, training_log_1.csv can be read
    # just fine though.
    log.csv_path = os.path.join(args.log_folder, "training_log.csv")
    log.txt_path = os.path.join(args.log_folder, "log.txt")

    os.makedirs(args.log_folder, exist_ok=True)

    utils.lock_job()
    actor_critic_model = make_model(args)

    if args.reference_policy is not None:
        assert args.architecture == "dual"
        # load only the policy parameters, and the normalization constants
        # "Run/TVF_EV1/game=Zaxxon samples=4 (1) [6bf5f217]/checkpoint-001M-params.pt.gz"
        checkpoint = rollout._open_checkpoint(os.path.join(args.log_folder, args.reference_policy), map_location=args.device)
        policy_checkpoint = {k[len('policy_net.'):]: v for k, v in checkpoint["model_state_dict"].items() if k.startswith("policy_net.")}
        actor_critic_model.policy_net.load_state_dict(policy_checkpoint)
        actor_critic_model.obs_rms = checkpoint['obs_rms']
        log.info(f"Loaded reference policy {args.reference_policy}.")

    ppo.train(actor_critic_model, log)

    utils.release_lock()

if __name__ == "__main__":
    from rl import logger
    log = logger.Logger()
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
