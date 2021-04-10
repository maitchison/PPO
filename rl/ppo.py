import os
import numpy as np
import torch
import time
import json
import math
import sys


from .logger import Logger, LogVariable
from .rollout import Runner, save_progress

import torch.multiprocessing

from . import utils, models, keyboard
from .config import args

class DualOptimizer:

    """
    Two optimizers on one set of parameters
    """


    def __init__(self, **kwargs):
        self.opt1 = torch.optim.Adam(**kwargs)
        self.opt2 = torch.optim.Adam(**kwargs)

    def zero_grad(self):
        self.opt1.zero_grad()
        self.opt2.zero_grad()

    def step(self):
        self.opt1.step()
        self.opt2.step()

    # to get this to work we will need to...
    # zero grad
    # backwards_1
    # step_1
    # zero grad
    # bacwards_2
    # step_2

    # which is a bit fidly, and does not work with micro_batching


def desync_envs(runner, warmup_duration:int):
    print(f"Warming up environments for {warmup_duration} steps:", end='', flush=True)

    max_steps = np.random.randint(1, warmup_duration, [args.agents])

    for t in range(max(max_steps)):

        masks = t < max_steps

        with torch.no_grad():
            model_out = runner.forward(runner.obs, output="policy")
            log_policy = model_out["log_policy"].cpu().numpy()

        actions = np.asarray([
            utils.sample_action_from_logp(prob) if mask else -1 for prob, mask in zip(log_policy, masks)
        ], dtype=np.int32)
        runner.obs, ext_rewards, dones, infos = runner.vec_env.step(actions)

        if t % 100 == 0:
            print(".", end='', flush=True)
    print()


def train(model: models.TVFModel, log: Logger):
    """
    Default parameters from stable baselines

    https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

    gamma             0.99
    n_steps            128
    ent_coef          0.01
    learning_rate   2.5e-4
    vf_coef            0.5
    max_grad_norm      0.5
    lam               0.95
    nminibatches         4
    noptepoch            4
    cliprange          0.1

    atari usually requires ~10M steps

    """

    # setup logging
    log.add_variable(LogVariable("ep_score", 100, "stats",
                                 display_width=12))  # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats", display_width=12))  # time get get first score / length.

    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    runner = Runner(model, log)

    # detect a previous experiment
    checkpoints = runner.get_checkpoints(args.log_folder)
    if len(checkpoints) > 0:
        log.info("Previous checkpoint detected.")
        checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
        restored_step = runner.load_checkpoint(checkpoint_path)
        log = runner.log
        log.info("  (resumed from step {:.0f}M)".format(restored_step / 1000 / 1000))
        start_iteration = (restored_step // batch_size) + 1
        walltime = log["walltime"]
        did_restore = True
    else:
        start_iteration = 0
        walltime = 0
        did_restore = False

    if not did_restore:
        log.log("To run experiment again use:")
        log.log("python train.py "+ " ".join(sys.argv[1:]))

    runner.create_envs()

    if not did_restore and args.normalize_observations:
        # this will get an initial estimate for the normalization constants.
        runner.run_random_agent(20)

    runner.reset()

    # if we restored from a checkpoint the environments will all be in sync
    # we run the environments for a number of steps sampled uniformly from [0...5000] without any training
    # note: this is a good idea even if we didn't restore, so as to make sure we are out of sync at the start.


    if args.env_desync:

        if not did_restore:
            warmup_duration = 500
        else:
            if "ep_length" in log._vars:
                ep_mean, ep_std, ep_min, ep_max = log._vars["ep_length"].value
                warmup_duration = math.ceil((ep_mean + ep_std * 2) / 100) * 100  # this should be enough...
                warmup_duration = min(ep_max, warmup_duration)  # no need to go above maximum episode length
                warmup_duration = int(warmup_duration)
            else:
                warmup_duration = 2000

        desync_envs(runner, warmup_duration)

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"), "w") as f:
        params = {k: v for k, v in args.__dict__.items()}
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations * batch_size / 1000 / 1000))
    else:
        log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(n_iterations * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    log.info()

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on
    checkpoints = [x // batch_size for x in range(0, n_iterations * batch_size + 1, args.checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]]  # add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    for iteration in range(start_iteration, n_iterations + 1):

        step_start_time = time.time()

        env_step = iteration * batch_size

        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precision=2)
        log.watch("walltime", walltime,
                  display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        # generate the rollout
        rollout_start_time = time.time()
        runner.generate_rollout()
        rollout_time = (time.time() - rollout_start_time) / batch_size

        # calculate returns
        returns_start_time = time.time()
        runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size

        train_start_time = time.time()
        runner.train(env_step)
        train_time = (time.time() - train_start_time) / batch_size

        step_time = (time.time() - step_start_time) / batch_size

        log_start_time = time.time()

        fps = 1.0 / (step_time)

        # record some training stats
        log.watch_mean("fps", int(fps))
        log.watch_mean("time_train", train_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_step", step_time * 1000, display_postfix="ms", display_precision=2, display_width=10)
        log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_log", log_time * 1000, type="float", display_postfix="ms", display_precision=2,
                       display_width=0)

        log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_freq:
            save_progress(log)
            log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug_log_freq:
            utils.lock_job()
            log.export_to_csv()
            log.save_log()
            last_log_time = time.time()

        # hotkeys
        if keyboard.kb.kbhit():
            c = keyboard.kb.getch()
            if c == "v":
                print("Exporting video...")
                video_name = utils.get_checkpoint_path(env_step, args.environment)
                runner.export_movie(video_name)
                log.info("  -video exported")

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            log.info()
            log.important("Checkpoint: {}".format(args.log_folder))

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                runner.save_checkpoint(checkpoint_name, env_step)
                log.log("  -checkpoint saved")

            if args.export_video:
                video_name = utils.get_checkpoint_path(env_step, args.environment)
                runner.export_movie(video_name)
                log.info("  -video exported")

            if args.export_trajectories:
                video_name = utils.get_trajectory_path(env_step, args.environment)
                os.makedirs(os.path.split(video_name)[0], exist_ok=True)
                for i in range(16):
                    runner.export_movie(video_name+"-{:02}".format(i), include_rollout=True, include_video=False)
                log.info("  -trajectories exported")

            log.info()

        log_time = (time.time() - log_start_time) / batch_size

        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += (step_time * batch_size)

        # check to see if the device we are using has been disallowed
        if args.device in utils.get_disallowed_devices():
            # notify user, release lock and hard exit
            # we could save a checkpoint but it's cleaner not do, otherwise graphs that generate datapoints at
            # each checkpoint will get confused (unless we save it as most recent...? Actually that would work?
            log.important("Training interrupted, as device was disallowed.")
            utils.release_lock()

            return


    # -------------------------------------
    # save final information

    save_progress(log)
    log.export_to_csv()
    log.save_log()

    log.info()
    log.important("Training Complete.")
    log.info()
