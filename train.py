import os
import torch
import uuid
import multiprocessing

from rl import utils, models, ppo, atari, config
from rl.config import args

resolution_map = {
    "full": (210, 160),
    "standard": (84, 84),
    "half": (42, 42)
}

def get_previous_experiment_guid(experiment_path, run_name):
    """ Looks for previous experiment with same run_name. Returns the guid if found. """
    if not os.path.exists(experiment_path):
        return None
    for f in os.listdir(experiment_path):
        if f[:-19] == run_name:
            guid = f[-17:-1]
            return guid
    return None


def run_experiment():
    """ Runs experient specifed by config.args """
    env = atari.make(args.env_name)
    n_actions = env.action_space.n
    obs_space = env.observation_space.shape
    print("Playing {} with {} obs_space and {} actions.".format(args.env_name, obs_space, n_actions))
    actor_critic_model = args.model(obs_space, n_actions, args.device, args.dtype)
    ppo.train(args.env_name, actor_critic_model)

def get_environment_name(environment, sticky_actions=False):
    environment = environment.capitalize()
    return "{}NoFrameskip-v{}".format(environment, "0" if sticky_actions else "4")


if __name__ == "__main__":

    config.parse_args()

    # set threading
    torch.set_num_threads(int(args.threads))

    # work out device to use
    if args.device.lower() == "auto":
        args.device = utils.get_auto_device()
    print("Using device: {}".format(utils.Color.BOLD + args.device + utils.Color.ENDC))

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

    # get model
    if args.model.lower() == "cnn":
        args.model = models.CNNModel
    elif args.model.lower() == "improved_cnn":
        args.model = models.ImprovedCNNModel
    else:
        raise Exception("Invalid model name '{}', please use [cnn, improved_cnn]".format(args.model))
    args.env_name = get_environment_name(args.environment, args.sticky_actions)

    # check the output folder is valid...
    assert os.path.isdir(args.output_folder), "Can not find path " + args.output_folder

    # set a guid
    if args.restore:
        # look for a previous experiment and use it if we find one...
        guid = get_previous_experiment_guid(os.path.join(args.output_folder, args.experiment_name), args.run_name)
        if guid is None:
            print(
                utils.Color.FAIL + "Could not restore experiment {}:{}. Previous run not found.".format(args.experiment_name,
                                                                                                        args.run_name) + utils.Color.ENDC)
        else:
            args.guid = guid
    else:
        args.guid = str(uuid.uuid4().hex)

    # work out the logging folder...
    args.log_folder = "{} [{}]".format(os.path.join(args.output_folder, args.experiment_name, args.run_name), args.guid[-16:])
    print("Logging to folder", args.log_folder)
    os.makedirs(args.log_folder, exist_ok=True)

    run_experiment()