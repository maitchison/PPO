import torch
import os
import math
import json
import time
import torchvision

import numpy as np

from . import models, utils, atari, logger
from .config import args
from .rollout import Runner, save_progress, adjust_learning_rate
from .logger import Logger, LogVariable
from .rollout import Runner

def train_population(ModelConstructor, master_log: Logger):
    """
    Trains a population of models, each with their own set of parameters, and potentially their own set of goals.

    ModelConstructor: Function without required parameters that constructs a model.
    """

    # create a population of models, each with their own parameters.
    models = []
    logs = []
    for i in range(args.pbl_population_size):
        models.append(ModelConstructor())
        log = logger.Logger()

        log.csv_path = os.path.join(args.log_folder, "training_log_{}.csv".format(i))
        log.txt_path = os.path.join(args.log_folder, "log_{}.txt".format(i))

        log.add_variable(LogVariable("ep_score", 100, "stats",
                                     display_width=16))  # these need to be added up-front as it might take some
        log.add_variable(LogVariable("ep_length", 100, "stats", display_width=16))  # time get get first score / length.

        log.add_variable(LogVariable("iteration", 0, type="int"))

        logs.append(log)

    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    # create optimizers
    optimizers = [torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon) for model in
                  models]

    names = ['agent-'+str(x) for x in range(args.pbl_population_size)]

    # create our runners
    runners = [Runner(model, optimizer, log, name) for model, optimizer, log, name in zip(models, optimizers, logs, names)]

    # todo allow for restoration from checkpoint
    start_iteration = 0
    walltime = 0
    did_restore = False

    # create environments for each agent.
    for runner in runners:
        runner.create_envs()
        runner.reset()

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"), "w") as f:
        params = {k: v for k, v in args.__dict__.items()}
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        master_log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations * batch_size / 1000 / 1000))
    else:
        master_log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(n_iterations * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    master_log.info()

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on
    checkpoints = [x // batch_size for x in range(0, n_iterations * batch_size + 1, args.checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]]  # add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    # train all models together
    for iteration in range(start_iteration, n_iterations + 1):

        step_start_time = time.time()

        env_step = iteration * batch_size

        master_log.watch("iteration", iteration, display_priority=5)
        master_log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6,
                         display_postfix="M",
                         display_precision=2)
        master_log.watch("walltime", walltime,
                         display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        # move some variables from master log to the individual logs
        for log in logs:
            for var_name in ["iteration", "env_step", "walltime"]:
                log.watch(var_name, master_log[var_name])

        for optimizer in optimizers:
            adjust_learning_rate(optimizer, env_step / 1e6)

        # generate the rollout
        rollout_start_time = time.time()
        for runner in runners:
            runner.generate_rollout()
        rollout_time = (time.time() - rollout_start_time) / batch_size

        # calculate returns
        returns_start_time = time.time()
        for runner in runners:
            runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size

        # train our population...
        # for the moment agent 0, and 1 is on-policy and all others are mixed off-policy.
        train_start_time = time.time()

        # we train all these 'off-policy' just to make sure v-trace works on policy.
        for runner in runners:
            runner.train_from_off_policy_experience(
                runners
            )

        train_time = (time.time() - train_start_time) / batch_size

        step_time = (time.time() - step_start_time) / batch_size

        log_start_time = time.time()

        fps = 1.0 / (step_time)

        # record some training stats
        master_log.watch_mean("fps", int(fps))
        master_log.watch_mean("time_train", train_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_step", step_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        master_log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2,
                              display_width=0)
        master_log.watch_mean("time_log", log_time * 1000, type="float", display_postfix="ms", display_precision=2,
                              display_width=0)

        master_log.aggretate_logs(logs, ignore=["iteration", "env_step", "walltime"])
        master_log.record_step()

        for log in logs:
            log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_freq:
            save_progress(master_log)

            if args.algo=="pbl":
                for i in range(args.pbl_population_size):
                    runners[i].log.print_variables(include_header= i == 0)
            else:
                master_log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug_log_freq:
            utils.lock_job()
            master_log.export_to_csv()
            master_log.save_log()
            for log in logs:
                log.export_to_csv()
                log.save_log()
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            master_log.info()
            master_log.important("Checkpoint: {}".format(args.log_folder))

            if args.save_checkpoints:
                for i, runner in enumerate(runners):
                    checkpoint_name = utils.get_checkpoint_path(env_step, "params_{}.pt".format(i))
                    runner.save_checkpoint(checkpoint_name, env_step)
                master_log.log("  -checkpoints saved")

            if args.export_video:

                for i, runner in enumerate(runners):
                    video_name = utils.get_checkpoint_path(env_step, "{}-{}.mp4".format(args.environment, i))
                    runner.export_movie(video_name)
                master_log.info("  -video exported")

            master_log.info()

        log_time = (time.time() - log_start_time) / batch_size

        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += (step_time * batch_size)

    # -------------------------------------
    # save final information

    save_progress(master_log)
    master_log.export_to_csv()
    master_log.save_log()

    master_log.info()
    master_log.important("Training Complete.")
    master_log.info()


