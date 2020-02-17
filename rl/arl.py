# todo make a PBL script, and also generalize the training so we don't have so much duplicated code.
# the way to do this is have a function for rollout, train, and returns?
import torch
import os
import math
import json
import time
import torchvision

import numpy as np

from . import models, utils, atari
from .config import args
from .rollout import Runner, save_progress, adjust_learning_rate, logger
from .logger import Logger, LogVariable


def generate_adversarial_rollout(main_agent, other_agent, warm_up = False):
    """
    Generates a rollout where another agent is able to interfer with the actions (at a price)
    :param other_agent:
    :return:
    """

    assert not args.use_intrinsic_rewards, "not supported with arl."

    for t in range(main_agent.N):

        prev_states = main_agent.states.copy()

        # forward state through both models, then detach the result and convert to numpy.
        model_out = main_agent.forward()
        log_policy = model_out["log_policy"].detach().cpu().numpy()
        ext_value = model_out["ext_value"].detach().cpu().numpy()

        model_out = other_agent.forward()
        arl_log_policy = model_out["log_policy"].detach().cpu().numpy()
        arl_ext_value = model_out["ext_value"].detach().cpu().numpy()

        # sample actions and run through environment.

        # the rules are, if max agent 'concentrates' we pick their action with cost
        # otherwise if min agent performs anything other than no-op we pick their action
        actions = np.asarray([utils.sample_action_from_logp(prob) for prob in log_policy], dtype=np.int32)
        arl_actions = np.asarray([utils.sample_action_from_logp(prob) for prob in arl_log_policy], dtype=np.int32)

        # decode the actions
        concentrations = np.asarray(actions % 2, dtype=np.bool)
        noops = np.asarray(arl_actions == 0, dtype=np.bool)

        true_actions = [action // 2 if (concentration or warm_up) or noop else arl_action - 1
                        for action, arl_action, concentration, noop in zip(actions, arl_actions, concentrations, noops)]

        main_agent.states, ext_rewards, dones, infos = main_agent.vec_env.step(true_actions)

        # modify ext rewards based on cost
        reward_modifier = [-0.01 if concentrate else +0.01 if not noop else 0 for concentrate, noop in zip(concentrations, noops)]
        ext_rewards += reward_modifier

        # it's a bit silly to have this here...
        if "returns_norm_state" in infos[0]:
            atari.ENV_STATE["returns_norm_state"] = infos[0]["returns_norm_state"]
            norm_mean, norm_var, norm_count = infos[0]["returns_norm_state"]
            main_agent.log.watch("returns_norm_mu", norm_mean)
            main_agent.log.watch("returns_norm_std", norm_var ** 0.5)

        # save raw rewards for monitoring the agents progress
        raw_rewards = np.asarray([info.get("raw_reward", ext_rewards) for reward, info in zip(ext_rewards, infos)],
                                 dtype=np.float32)

        main_agent.episode_score += raw_rewards
        main_agent.episode_len += 1

        for i, done in enumerate(dones):
            if done:
                # reset is handled automatically by vectorized environments
                # so just need to keep track of book-keeping
                main_agent.log.watch_full("ep_score", main_agent.episode_score[i])
                main_agent.log.watch_full("ep_length", main_agent.episode_len[i])
                main_agent.episode_score[i] = 0
                main_agent.episode_len[i] = 0

        for agent in [main_agent, other_agent]:
            agent.prev_state[t] = prev_states
            agent.next_state[t] = main_agent.states
            agent.terminals[t] = dones

        main_agent.actions[t] = actions
        main_agent.ext_rewards[t] = ext_rewards
        main_agent.log_policy[t] = log_policy
        main_agent.ext_value[t] = ext_value

        other_agent.actions[t] = arl_actions
        other_agent.ext_rewards[t] = -ext_rewards
        other_agent.log_policy[t] = arl_log_policy
        other_agent.ext_value[t] = arl_ext_value

    # get value estimates for final state.
    model_out = main_agent.forward()

    main_agent.ext_final_value_estimate = model_out["ext_value"].detach().cpu().numpy()


def train_arl(model: models.BaseModel, arl_model: models.BaseModel, log: Logger):
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
    log = logger.Logger()
    log.csv_path = os.path.join(args.log_folder, "training_log.csv")
    log.txt_path = os.path.join(args.log_folder, "log.txt")
    log.add_variable(LogVariable("ep_score", 100, "stats",
                                 display_width=16))  # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats", display_width=16))  # time get get first score / length.
    log.add_variable(LogVariable("iteration", 0, type="int"))

    arl_log = logger.Logger()
    arl_log.csv_path = os.path.join(args.log_folder, "training_log_arl.csv")
    arl_log.txt_path = os.path.join(args.log_folder, "log_arl.txt")
    arl_log.add_variable(LogVariable("ep_score", 100, "stats",
                                 display_width=16))  # these need to be added up-front as it might take some
    arl_log.add_variable(LogVariable("ep_length", 100, "stats", display_width=16))  # time get get first score / length.
    arl_log.add_variable(LogVariable("iteration", 0, type="int"))


    # calculate some variables
    batch_size = (args.n_steps * args.agents)
    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    arl_optimizer = torch.optim.Adam(arl_model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    runner = Runner(model, optimizer, log)
    arl_runner = Runner(arl_model, arl_optimizer, arl_log)

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

    runner.create_envs()

    if not did_restore and args.use_rnd:
        # this will get an initial estimate for the normalization constants.
        runner.run_random_agent(20)

    runner.reset()

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

        # alternate between conflict and not... (note, this will kill the value estimate...)
        # also, could work if we alternate at 5m or 10m steps...
        enable_conflict = bool(int(env_step // 1e-6) % 2)

        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precision=2)
        log.watch("walltime", walltime,
                  display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        adjust_learning_rate(optimizer, env_step / 1e6)

        # generate the rollout
        rollout_start_time = time.time()
        generate_adversarial_rollout(runner, arl_runner, warm_up = not enable_conflict)
        rollout_time = (time.time() - rollout_start_time) / batch_size

        # calculate returns
        returns_start_time = time.time()
        if enable_conflict:
            runner.calculate_returns()
            arl_runner.calculate_returns()
        else:
            runner.calculate_returns()
        returns_time = (time.time() - returns_start_time) / batch_size

        train_start_time = time.time()
        if enable_conflict:
            runner.train()
            arl_runner.train()
        else:
            runner.train()
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

            # export debug frames
            if args.use_emi:
                try:
                    with torch.no_grad():
                        img = model.generate_debug_image(runner.emi_prev_state, runner.emi_actions,
                                                         runner.emi_next_state)
                    os.makedirs(os.path.join(args.log_folder, "emi"), exist_ok=True)
                    torchvision.utils.save_image(img, os.path.join(args.log_folder, "emi",
                                                                   "fdm-{:04d}K.png".format(env_step // 1000)))
                except Exception as e:
                    log.warn(str(e))

        # save log and refresh lock
        if time.time() - last_log_time >= args.debug_log_freq:
            utils.lock_job()
            log.export_to_csv()
            log.save_log()
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            log.info()
            log.important("Checkpoint: {}".format(args.log_folder))

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                runner.save_checkpoint(checkpoint_name, env_step)
                log.log("  -checkpoint saved")

            if args.export_video:
                video_name = utils.get_checkpoint_path(env_step, args.environment + ".mp4")
                runner.export_movie(video_name)
                log.info("  -video exported")

            log.info()

        log_time = (time.time() - log_start_time) / batch_size

        # update walltime
        # this is not technically wall time, as I pause time when the job is not processing, and do not include
        # any of the logging time.
        walltime += (step_time * batch_size)

    # -------------------------------------
    # save final information

    save_progress(log)
    log.export_to_csv()
    log.save_log()

    log.info()
    log.important("Training Complete.")
    log.info()
