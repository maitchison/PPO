import os
import numpy as np
import torch
import time
import json
import math
import sys
import shlex

from . import compression
from .logger import Logger, LogVariable
from .rollout import Runner, save_progress
from .utils import Color

import torch.multiprocessing

from . import utils, models, keyboard, envs
from .config import args


def desync_envs(runner, min_duration:int, max_duration:int, verbose=True):
    if verbose:
        print(f"Warming up environments for {min_duration} to {max_duration} steps:", end='', flush=True)
    max_steps = np.random.randint(min_duration, max_duration+1, [args.agents])

    start_time = time.time()

    for t in range(max(max_steps)):
        masks = t < max_steps
        with torch.no_grad():
            model_out = runner.detached_batch_forward(runner.obs, output="policy", update_normalization=True)
            log_policy = model_out["log_policy"].cpu().numpy()
        actions = np.asarray([
            utils.sample_action_from_logp(prob) if mask else -1 for prob, mask in zip(log_policy, masks)
        ], dtype=np.int32)
        runner.obs, ext_rewards, dones, infos = runner.vec_env.step(actions)
        runner.time = np.asarray([info.get("time", 0.0) for info in infos])
        if t % 100 == 0 and verbose:
            print(".", end='', flush=True)

    if verbose:
        steps = np.sum(max_steps)
        time_taken = time.time() - start_time
        print(f" ({utils.comma(steps/time_taken)} steps per second).")


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

    start_time = time.time()

    # setup logging
    log.add_variable(LogVariable("ep_score", 100, "stats",
                                 display_width=12))  # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats", display_width=12))  # time to get first score / length.

    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    end_iteration = math.ceil((final_epoch * 1e6) / batch_size)

    runner = Runner(model, log, action_dist="gaussian" if args.env.type == "mujoco" else "discrete")
    runner.vec_env = envs.create_envs_classic()
    runner.reset()

    # logging
    model_total_size = runner.model.model_size(trainable_only=True) / 1e6
    log.important("Generated {} agents ({}) using {} ({:.2f}M params) {} model.".
                       format(args.agents, "async" if not args.sync_envs else "sync", runner.model.name,
                              model_total_size, runner.model.dtype))


    # detect a previous experiment
    checkpoints = runner.get_checkpoints(args.log_folder)
    has_checkpoint = len(checkpoints) > 0

    if args.restore == "always" and not has_checkpoint:
        raise Exception(f"Error: no restore point at {args.log_folder} found.")

    if args.initial_model is not None:
        # load the model but start from step 0. with a new log and in a new folder
        runner.load_checkpoint(os.path.join(args.log_folder, args.initial_model))
        runner.log = log
        start_iteration = 0
        walltime = 0
        did_restore = False
        log.info(f"Initialized with reference policy {Color.OKGREEN}{args.initial_model}{Color.ENDC}.")
    else:
        if has_checkpoint and args.restore in ['auto', 'always']:
            log.info("Previous checkpoint detected.")
            checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
            restored_step = runner.load_checkpoint(checkpoint_path)
            log = runner.log
            log.info(" -resumed from step {:.0f}M".format(restored_step / 1000 / 1000))
            start_iteration = (restored_step // batch_size) + 1
            walltime = log["walltime"]
            did_restore = True
            # fixup log path (incase the folder was renamed changed between saves)
            log.csv_path = os.path.join(args.log_folder, "training_log.csv")
            log.txt_path = os.path.join(args.log_folder, "log.txt")
        else:
            # do not restore.
            start_iteration = 0
            walltime = 0
            did_restore = False

    if not did_restore:
        log.log("To rerun experiment use:")
        log.log("python train.py " + " ".join(shlex.quote(x) for x in sys.argv[1:] if not x.startswith("description")))
        if args.env.warmup_period <= 0:
            # some wrappers really want a 1-frame warmup after restoring.
            log.warn("Extending warmup to 1 frame.")
        desync_envs(runner, 1, max(args.env.warmup_period, 1))
    else:
        # this is really just the throw a few new frames through the wrappers
        desync_envs(runner, 1, 4, verbose=False)

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"), "wt") as t:
        params = args.flatten()
        t.write(json.dumps(params, indent=4))


    # make a copy of training files for reference
    try:
        root_folder, train_script = os.path.split(sys.argv[0])
        utils.copy_source_files(root_folder, args.log_folder)
    except:
        # it's fine if this fails
        pass

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        log.info("Training for <yellow>{:.1f}M<end> steps".format(end_iteration * batch_size / 1000 / 1000))
    else:
        log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(end_iteration * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on
    if args.checkpoint_every != 0:
        checkpoints = [x // batch_size for x in range(0, end_iteration * batch_size + 1, args.checkpoint_every)]
        if args.save_early_checkpoint:
            checkpoints += [x // batch_size for x in [1e6]]  # add a checkpoint early on (1m steps)
        checkpoints.append(end_iteration)
        checkpoints = sorted(set(checkpoints))
    else:
        checkpoints = []

    log_time = 0
    pause_at_end = False

    log.info(f"Training started. (init took {time.time()-start_time:.1f} seconds)")
    log.info()

    start_train_time = time.time()

    def save_checkpoint_image():

        if args.env.type not in ["atari", "mujoco"]:
            # not supported yet.
            return

        image_name = utils.get_checkpoint_path(env_step, "slides.png")
        from PIL import Image
        assert not args.obs_compression

        A, C, H, W = runner.obs.shape

        if args.env.color_mode == "bw":
            obs = np.zeros((A, 3, H, W), dtype=np.uint8)
            obs[:, 0] = runner.obs[:, 0]
            obs[:, 1] = runner.obs[:, 0]
            obs[:, 2] = runner.obs[:, 0]
        else:
            obs = runner.obs[:, :3]

        C = 3

        wide = 16
        assert A % wide == 0, "For image export agents must be divisible by 16"
        high = A // wide
        output_image = obs.transpose((0, 2, 3, 1))  # N H W C
        output_image = output_image.reshape((high, wide * H, W, C))
        output_image = output_image.transpose(1, 0, 2, 3)
        output_image = output_image.reshape(high * H, wide * W, C)
        im = Image.fromarray(output_image)
        im.save(image_name)

    def save_checkpoint(runner, log):

        checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")

        log.info()
        log.important("Checkpoint: {}".format(args.log_folder))

        if not utils.log_folder_exists():
            raise Exception("Folder was removed, exiting.")

        if args.debug.checkpoint_slides:
            save_checkpoint_image()

        if args.save_checkpoints:
            runner.save_checkpoint(checkpoint_name, env_step)
            log.log("  -checkpoint saved")

        log.info()

    def log_iteration():
        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precision=2)
        log.watch("walltime", walltime,
                  display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)
        log.watch("time", time.time(), display_width=0)

    # save early progress
    iteration = start_iteration
    env_step = start_iteration * batch_size
    log_iteration()
    save_progress(log)

    old_header = None

    if args.save_initial_checkpoint:
        save_checkpoint(runner, log)

    for _ in range(start_iteration, end_iteration):

        runner.step = iteration*batch_size

        step_start_time = time.time()

        rollout_start_time = time.time()
        runner.generate_rollout()
        rollout_time = (time.time() - rollout_start_time) / batch_size

        # calculate returns
        returns_start_time = time.time()
        runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size

        train_start_time = time.time()
        runner.train()
        train_time = (time.time() - train_start_time) / batch_size

        step_time = (time.time() - step_start_time) / batch_size

        log_start_time = time.time()

        fps = 1.0 / step_time

        # record some training stats
        log.watch_mean("fps", int(fps))
        log.watch_mean("time_train", train_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_step", step_time * 1000, display_precision=2, display_width=0, display_name="step_ms")
        log.watch_mean("time_rollout", rollout_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_returns", returns_time * 1000, display_postfix="ms", display_precision=2, display_width=0)
        log.watch_mean("time_log", log_time * 1000, type="float", display_postfix="ms", display_precision=2,
                       display_width=0)

        log.record_step()

        # move to next iteration
        iteration += 1
        env_step += batch_size
        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += step_time * batch_size
        log_iteration()

        # make sure we still have lock
        if not utils.have_lock():
            log.important("Lock was lost, aborting...")
            return

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug.print_freq:
            if not utils.log_folder_exists():
                raise Exception("Folder was removed, exiting.")
            save_progress(log)
            header_changed = old_header is None or log.header != old_header
            log.print_variables(include_header=print_counter % 10 == 0 or header_changed)
            old_header = log.header
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug.log_freq:
            if not utils.log_folder_exists():
                raise Exception("Folder was removed, exiting.")
            utils.lock_job()
            log.export_to_csv()
            log.save_log()
            last_log_time = time.time()

        # hotkeys
        wants_manual_save=False
        if keyboard.kb.kbhit():
            c = keyboard.kb.getch()
            if c == "q":
                pause_at_end = not pause_at_end
                log.info(f"Pausing at end of chunk [<warn>{pause_at_end}<end>]")
            if c == "s":
                wants_manual_save = True
                log.info(f"Manual checkpoint saving")

        # periodically save checkpoints
        if (
                (iteration in checkpoints) and
                (not iteration == end_iteration-1) and
                (not did_restore or iteration != start_iteration)) or \
                wants_manual_save:
            save_checkpoint(runner, log)

        log_time = (time.time() - log_start_time) / batch_size

        # check to see if the device we are using has been disallowed
        if args.device in utils.get_disallowed_devices():
            # notify user, release lock and hard exit
            # we could save a checkpoint but it's cleaner not do, otherwise graphs that generate datapoints at
            # each checkpoint will get confused (unless we save it as most recent...? Actually that would work?
            log.important("Training interrupted, as device was disallowed.")
            utils.release_lock()
            return

    # -------------------------------------
    # benchmark information
    if args.benchmark_mode:
        # this is a bit more accurate than the IPS counter during training
        time_to_complete = time.time() - start_train_time
        steps = end_iteration * batch_size
        print(f"Completed {steps:,} steps in {time_to_complete:.1f}s")
        if args.obs_compression:
            print(f"Compression stats: "
                  f"{1000*compression.av_compression_time():.4f}ms / "
                  f"{1000*compression.av_decompression_time():.4f}ms, "
                  f"{compression.ratio():.1f}x ratio"
                  )
        print(f"IPS: {round(steps/time_to_complete):,}")

    # -------------------------------------
    # save final information

    save_progress(log)
    log.export_to_csv()
    log.save_log()

    log.info()
    log.important("Training Complete.")
    log.info()

    utils.release_lock()

    if pause_at_end:

        while True:
            time.sleep(1)