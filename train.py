import os
import torch
import uuid
import multiprocessing

import ast

from rl import utils, models, atari, config, logger
from rl import ppo, arl, pbl
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
        if f[:-(8+3)] == run_name:
            guid = f[-(8+1):-1]
            return guid
    return None

def get_environment_name(environment, sticky_actions=False):
    return "{}NoFrameskip-v{}".format(environment, "0" if sticky_actions else "4")

if __name__ == "__main__":

    log = logger.Logger()

    config.parse_args()

    # set threading
    torch.set_num_threads(int(args.threads))

    # work out device to use
    if args.device.lower() == "auto":
        args.device = utils.get_auto_device(ast.literal_eval(args.ignore_device))
    log.info("Using device: <white>{}<end>".format(args.device))

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

    # get model
    args.env_name = get_environment_name(args.environment, args.sticky_actions)

    # check the output folder is valid...
    assert os.path.isdir(args.output_folder), "Can not find path " + args.output_folder

    # set a guid
    if args.restore:
        # look for a previous experiment and use it if we find one...
        guid = get_previous_experiment_guid(os.path.join(args.output_folder, args.experiment_name), args.run_name)
        if guid is None:
            log.error("Could not restore experiment {}:{}. Previous run not found.".format(args.experiment_name, args.run_name))
        else:
            args.guid = guid
    else:
        args.guid = str(uuid.uuid4().hex)

    # work out the logging folder...
    args.log_folder = args.log_folder or "{} [{}]".format(os.path.join(args.output_folder, args.experiment_name, args.run_name), args.guid[-8:])
    log.info("Logging to folder " + args.log_folder)

    # population training gets a summary log, which we need to name differently as it can not be processed by
    # the analysis scripts (due to missing varaibles). The training_log_0.csv, training_log_1.csv can be read
    # just fine though.
    log.csv_path = os.path.join(args.log_folder, "training_log.csv" if args.algo == "ppo" else "master_log.csv")
    log.txt_path = os.path.join(args.log_folder, "log.txt")

    os.makedirs(args.log_folder, exist_ok=True)

    """ Runs experient specifed by config.args """
    env = atari.make()
    n_actions = env.action_space.n
    obs_space = env.observation_space.shape
    log.info("Playing {} with {} obs_space and {} actions.".format(args.env_name, obs_space, n_actions))

    if (args.use_rnd + args.use_emi) > 1:
        # todo: this should be a setting, not a set of bools
        raise Exception("EMI, RND, are not compatible.")

    model_args = {}

    if args.use_rnd:
        ACModel = models.RNDModel
    elif args.use_mppe:
        ACModel = models.MPPEModel
    elif args.use_emi:
        ACModel = models.EMIModel
    elif args.use_mvh:
        ACModel = models.MVHModel
        model_args["value_heads"] = args.mvh_heads
    else:
        ACModel = models.ActorCriticModel

    try:

        utils.lock_job()

        if args.model == "cnn":
            head_name = "Nature"
        else:
            raise Exception("invalid model name {}.".format(args.model))

        if args.algo.lower() == "arl":
            assert not args.use_rnn
            actor_critic_model = ACModel(head=head_name, input_dims=obs_space, actions=n_actions*2,
                                         device=args.device, dtype=torch.float32, **model_args)

            arl_model = ACModel(head=head_name, input_dims=obs_space, actions=n_actions+1,
                                         device=args.device, dtype=torch.float32, **model_args)

            arl.train_arl(actor_critic_model, arl_model, log)
        elif args.algo.lower() == "ppo":

            # reduce default hidden units from 512 to 64
            # otherwise we have a 512x512 array, which will be 0.25M parameters
            # actually... this is probably ok...
            # if args.use_rnn and "hidden_units" not in model_args:
            #     model_args["hidden_units"] = 64

            actor_critic_model = ACModel(
                head=head_name,
                input_dims=obs_space,
                actions=n_actions,
                device=args.device,
                dtype=torch.float32,
                use_rnn=args.use_rnn,
                **model_args
            )
            ppo.train(actor_critic_model, log)
        elif args.algo.lower() == "pbl":
            assert not args.use_rnn
            model_constructor = lambda : ACModel(head=head_name, input_dims=obs_space, actions=n_actions,
                                         device=args.device, use_rnn=args.use_rnn, dtype=torch.float32, **model_args)
            pbl.train_population(model_constructor, log)
        else:
            raise Exception("Invalid algorithm", args.algo)

        utils.release_lock()

    except Exception as e:
        print("!" * 60)
        print(e)
        print("!" * 60)
        try:
            log.ERROR(str(e))
            log.save_log()
        except:
            # just ignore any errors while trying to log result
            pass
        raise e
