import os
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import math
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from .logger import Logger, LogVariable

from . import utils, models, atari, hybridVecEnv, config
from .config import args

def train_minibatch(model: models.PolicyModel, optimizer, log, prev_states, next_states, actions,
                    returns_ext, returns_int, policy_logprobs, advantages, values_ext, values_int):
    """
    :param model:           The model for this agent.
    :param optimizer:       The optimizer to use.
    :param prev_states:     tensor of dims [N, C, H, W]
    :param actions:         tensor of dims [N]
    :param returns:         tensor of dims [N]
    :param policy_logprobs: tensor of dims [N, Actions]
    :param advantages:      tensor of dims [N]
    :param values_ext:      tensor of dims [N]
    :param values_int:      tensor of dims [N]
    :return:
    """

    # prepare the tensors for the model (moves to GPU and converts to float.
    # note, we don't do this with the states as these are expected to be in uint8 format.
    policy_logprobs = model.prep_for_model(policy_logprobs)
    advantages = model.prep_for_model(advantages)
    returns_ext = model.prep_for_model(returns_ext)
    returns_int = model.prep_for_model(returns_int)
    pred_values_ext = model.prep_for_model(values_ext)
    pred_values_int = model.prep_for_model(values_int)

    mini_batch_size = len(prev_states)

    logps, value_ext_prediction, value_int_prediction = model.forward(prev_states)

    ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])

    loss_clip = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon) * advantages))

    # this is the trust region clipped value estimator, which reduces variance. taken from the PPO2 OpenAi baseline.
    loss_value = 0
    for value_prediction, returns, pred_values in [(value_ext_prediction, returns_ext, pred_values_ext), (value_int_prediction, returns_int, pred_values_int)]:
        value_prediction_clipped = pred_values + torch.clamp(value_prediction - pred_values, -args.ppo_epsilon, +args.ppo_epsilon)
        vf_losses1 = (value_prediction - returns).pow(2)
        vf_losses2 = (value_prediction_clipped - returns).pow(2)
        loss_value = loss_value - args.vf_coef * 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    loss_entropy = args.entropy_bonus * utils.log_entropy(logps) / mini_batch_size

    loss = -(loss_clip + loss_value + loss_entropy)  # gradient ascent.

    # calculate RND gradient
    if args.use_rnd:

        # note: this could be much much more efficent, I really should just be passing in mu and sigma, and then
        # letting the model do the transformation... (i.e. keep everything as uint8)

        # train on only 25% of minibatch to slow down the predictor network when used with large number of agents.

        predictor_proportion = np.clip(32 / args.agents, 0.01, 1)
        n = int(len(prev_states) * predictor_proportion)
        loss_rnd = model.prediction_error(prev_states[:n]).mean()
        loss += loss_rnd
        log.watch_mean("loss_rnd", loss_rnd)
        log.watch_mean("rnd_feat_mean", model.features_mean)
        log.watch_mean("rnd_feat_std", model.features_std)

    # calculate ICM gradient
    # this should be done with rewards, but since I'm on PPO I'll do it with gradient rather than reward...
    # be interesting to compare and see which one is better?
    if args.use_icm:

        assert type(model) == models.ICMModel, "ICM requires using the ICMModel Network."

        # todo: track how well IDM and FDM is working..
        # note: the ICM should be a totally seperate module... no reason to build it into the model
        # this allows other models to be used which is nice.

        # step 1, try to learn IDM model
        # from https://github.com/pathak22/noreward-rl/blob/master/src/model.py we have
        # IDM_loss = sparse_softmax_cross_entropy_with_logits(logits, aindex)

        beta = 0.2

        nlog_probs = model.idm(prev_states, next_states)
        targets = torch.tensor(actions).to(model.device).long()
        loss_idm = F.nll_loss(nlog_probs, targets) * (1-beta) * 0.1 # loss_idm end up being way to big, so we reduce it here.

        # step 2 learn the FDM
        # from https://github.com/pathak22/noreward-rl/blob/master/src/model.py we have
        # FDM_loss = 0.5*MSE * 288 (which is 0.5 * SSE as in the paper)
        pred_embedding = model.fdm(prev_states, torch.tensor(actions).to(model.device).long())
        next_frames = model.extract_down_sampled_frame(next_states)
        next_embedding = model.encode(next_frames)

        loss_fdm = F.mse_loss(pred_embedding, next_embedding) * 288 * beta
        loss_fdm = 0

        # step 2.5 learn a little of of a reconstruction loss (to stop collapse)
        # disable this, it's not a good idea...
        # next_decoding = model.decode(next_embedding)
        # loss_ae = F.mse_loss(next_decoding, next_frames)
        loss_ae = 0

        # addition: encourage the representation to be unit norm (this is sometimes done with batch-norm, but this
        # seems easyer.
        embedding_mean = next_embedding.mean()
        embedding_std = next_embedding.std()
        loss_norm = (embedding_mean - 0.0)**2 + (embedding_std - 1.0)**2

        accuracy = (torch.argmax(nlog_probs, dim=1) == targets).float().mean()

        # todo, get ICM working again...
        # if my_counter % 100 == 0:
        #     print("loss_idm {:.3f} accuracy {:.3f} loss_fdm {:.3f} loss_ae {:.3f} loss_norm {:.3f} embd_std {:.3f}".format(
        #         float(loss_idm), float(accuracy), float(loss_fdm), float(loss_ae), float(loss_norm),
        #         float(torch.std(next_embedding[0]))))

        loss += (loss_idm + loss_fdm + loss_ae + loss_norm)

    optimizer.zero_grad()
    loss.backward()

    if args.max_grad_norm is not None and args.max_grad_norm != 0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    else:
        # even if we don't clip the gradient we should atleast log the norm. This is probably a bit slow though.
        # we could do this every 10th step, but it's important that a large grad_norm doesn't get missed.
        grad_norm = 0
        parameters = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in parameters:
            param_norm = p.grad.data.norm(2)
            grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** (0.5)

    log.watch_mean("loss", loss)
    log.watch_mean("loss_clip", loss_clip)
    log.watch_mean("loss_value", loss_value)
    log.watch_mean("loss_entropy", loss_entropy)
    log.watch_mean("opt_grad_norm", grad_norm)

    optimizer.step()

def run_agents_vec(n_steps, model, vec_envs, states, episode_score, episode_len, log,
               state_shape, state_dtype, policy_shape):
    """
    Runs agents given number of steps, using a single thread, but batching the updates
    :return:
        N is number of steps per run
        A is number of agents

        batch_prev_state [N, A, (obs dims)]
        ...
    """

    N = n_steps
    A = args.agents

    batch_prev_state = np.zeros([N, A, *state_shape], dtype=state_dtype)
    batch_next_state = np.zeros([N, A, *state_shape], dtype=state_dtype) # this doubles memory, but makes life easier.
    batch_action = np.zeros([N, A], dtype=np.int32)
    batch_reward_ext = np.zeros([N, A], dtype=np.float32)
    batch_reward_int = np.zeros([N, A], dtype=np.float32)
    batch_logpolicy = np.zeros([N, A, *policy_shape], dtype=np.float32)
    batch_terminal = np.zeros([N, A], dtype=np.bool)
    batch_value_ext = np.zeros([N, A], dtype=np.float32)
    batch_value_int = np.zeros([N, A], dtype=np.float32)

    for t in range(N):

        logprobs, value_ext, value_int = (x.detach().cpu().numpy() for x in model.forward(states))

        actions = np.asarray([utils.sample_action_from_logp(prob) for prob in logprobs], dtype=np.int32)
        prev_states = states.copy()

        states, rewards, dones, infos = vec_envs.step(actions)

        intrinsic_rewards = np.zeros_like(rewards)

        # generate prediction error bonus
        if args.use_icm and args.icm_eta != 0:
            pred_embedding = model.fdm(prev_states, torch.tensor(actions).to(model.device).long())
            next_embedding = model.encode(model.extract_down_sampled_frame(states))
            loss_fdm = 0.5 * F.mse_loss(pred_embedding, next_embedding, reduction='none').sum(dim=1).detach().cpu().numpy() * config.args.icm_eta
            intrinsic_rewards += loss_fdm
        # generate rnd bonus
        if args.use_rnd:
            loss_rnd = model.prediction_error(states).detach().cpu().numpy()
            intrinsic_rewards += loss_rnd

        raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(rewards, infos)], dtype=np.float32)

        # save a copy of the normalization statistics.
        for key in ["returns_norm_state"]:
            if key in infos[0]:
                atari.ENV_STATE[key] = infos[0][key]
        if args.use_rnd:
            atari.ENV_STATE["observation_norm_state"] = model.obs_rms.save_state()

        episode_score += raw_rewards
        episode_len += 1

        for i, done in enumerate(dones):
            if done:
                # reset is handled automatically by vectorized environments
                # so just need to keep track of book-keeping
                log.watch_full("ep_score", episode_score[i])
                log.watch_full("ep_length", episode_len[i])
                episode_score[i] = 0
                episode_len[i] = 0

        batch_prev_state[t] = prev_states
        batch_next_state[t] = states
        batch_action[t] = actions
        batch_reward_ext[t] = rewards
        batch_reward_int[t] = intrinsic_rewards
        batch_logpolicy[t] = logprobs
        batch_terminal[t] = dones
        batch_value_ext[t] = value_ext
        batch_value_int[t] = value_int

    return (batch_prev_state, batch_next_state, batch_action, batch_reward_ext, batch_reward_int, batch_logpolicy,
            batch_terminal, batch_value_ext, batch_value_int, states)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.learning_rate_decay == 1.0:
        return args.learning_rate
    else:
        lr = args.learning_rate * (args.learning_rate_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def save_progress(log: Logger):
    """ Saves some useful information to progress.txt. """

    details = {}
    details["max_epochs"] = args.epochs
    details["completed_epochs"] = log["env_step"] / 1e6  # include the current completed step.
    details["score"] = log["ep_score"][0]
    details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
    details["fps"] = log["fps"]
    frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
    details["eta"] = frames_remaining / details["fps"]
    details["host"] = args.hostname
    details["last_modified"] = time.time()
    with open(os.path.join(args.log_folder, "progress.txt"), "w") as f:
        json.dump(details, f, indent=4)


def calculate_returns(rewards, dones, final_value_estimate, gamma):
    """
    Calculates returns given a batch of rewards, dones, and a final value estimate.
    Input is vectorized so it can calculate returns for multiple agents at once.
    :param rewards: nd array of dims [N,A]
    :param dones:   nd array of dims [N,A] where 1 = done and 0 = not done.
    :param final_value_estimate: nd array [A] containing value estimate of final state after last action.
    :param gamma:   discount rate.
    :return:
    """

    N,A = rewards.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_value_estimate

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma * (1.0 - dones[i])

    return returns

def train(env_name, model: models.PolicyModel, log:Logger):
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
    log.add_variable(LogVariable("ep_score", 100, "stats"))   # these need to be added up-front as it might take some
    log.add_variable(LogVariable("ep_length", 100, "stats"))  # time get get first score / length.

    utils.lock_job()

    # get shapes and dtypes
    _env = atari.make(env_name)
    obs = _env.reset()
    state_shape = obs.shape
    state_dtype = obs.dtype
    policy_shape = model.policy(obs[np.newaxis])[0].shape

    # Just export the model for the moment.
    if args.tensorboard_logging:
        writer = SummaryWriter()
        writer.add_graph(model, torch.tensor(obs))
        writer.close()

    _env.close()

    # epsilon = 1e-5 is required for stability.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    batch_size = (args.n_steps * args.agents)

    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    # detect a previous experiment
    checkpoints = utils.get_checkpoints(args.log_folder)
    if len(checkpoints) > 0:
        log.info("Previous checkpoint detected.")
        checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
        restored_step, log, norm_state = utils.load_checkpoint(checkpoint_path, model, optimizer)

        for k, v in norm_state.items():
            atari.ENV_STATE[k] = v

        log.info("  (resumed from step {:.0f}M)".format(restored_step/1000/1000))
        start_iteration = (restored_step // batch_size) + 1
        walltime = log["walltime"]
        did_restore = True
    else:
        start_iteration = 0
        walltime = 0
        did_restore = False

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"),"w") as f:
        params = {k:v for k,v in args.__dict__.items()}
        # fixup some of the non-serializable types.
        params["dtype"] = str(params["dtype"])
        params["model"] = params["model"].name
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    env_fns = [lambda : atari.make(env_name) for _ in range(args.agents)]
    vec_env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=args.workers, verbose=True) if not args.sync_envs else gym.vector.SyncVectorEnv(env_fns)

    log.important("Generated {} agents ({}) using {} ({}) model.".format(args.agents, "async" if not args.sync_envs else "sync", model.name, model.dtype))

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        log.info("Training for <yellow>{:.1f}M<end> steps".format(n_iterations*batch_size/1000/1000))
    else:
        log.info("Training block from <yellow>{}M<end> to (<yellow>{}M<end> / <white>{}M<end>) steps".format(
            str(round(start_iteration * batch_size / 1000 / 1000)),
            str(round(n_iterations * batch_size / 1000 / 1000)),
            str(round(args.epochs))
        ))

    log.info()

    # initialize agent
    states = vec_env.reset()

    episode_score = np.zeros([args.agents], dtype = np.float32)
    episode_len = np.zeros([args.agents], dtype = np.int32)

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on

    checkpoints = [x // batch_size for x in range(0, n_iterations*batch_size+1, config.CHECKPOINT_EVERY_STEPS)]
    checkpoints += [x // batch_size for x in [1e6]] #add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    log_time = 0

    for iteration in range(start_iteration, n_iterations+1):

        step_start_time = time.time()

        env_step = iteration * batch_size

        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precision=2)
        log.watch("walltime", walltime,
                  display_priority=3, display_scale=1 / (60 * 60), display_postfix="h", display_precision=1)

        adjust_learning_rate(optimizer, env_step / 1e6)

        # the idea here is that all our batch arrays are of dims
        # N, A, ...,
        # Where n is the number of steps, and A is the number of agents.
        # this means we can process each step as a vector

        # collect experience
        batch_prev_state, batch_next_state, batch_action, batch_rewards_ext, batch_rewards_int, \
        batch_logpolicy, batch_terminal, batch_value_ext, batch_value_int, states = \
            run_agents_vec(args.n_steps, model, vec_env, states, episode_score, episode_len, log,
                state_shape, state_dtype, policy_shape)

        # scale internal rewards...
        batch_rewards_int *= args.intrinsic_reward_scale

        # ----------------------------------------------------
        # calculate returns for intrinsic and extrinsic rewards
        # ----------------------------------------------------

        _, final_value_estimate_ext, final_value_estimate_int = (x.detach().cpu().numpy() for x in model.forward(states))

        batch_returns_ext = calculate_returns(batch_rewards_ext, batch_terminal, final_value_estimate_ext, args.gamma)
        # note: we zero all the terminals here so that intrinsic rewards propagate through episodes as per
        # the RND paper.
        batch_returns_int = calculate_returns(batch_rewards_int, 0*batch_terminal, final_value_estimate_int, args.gamma_int)

        log.watch_mean("batch_reward_int", np.mean(batch_rewards_int), display_name="rew_int")
        log.watch_mean("batch_reward_ext", np.mean(batch_rewards_ext), display_name="rew_ext")
        log.watch_mean("batch_return_int", np.mean(batch_returns_int), display_name="ret_int")
        log.watch_mean("batch_return_ext", np.mean(batch_returns_ext), display_name="ret_ext")

        # normalize intrinsic reward across rollout such that batch returns have 1 std.
        # seems like this makes training less stable...
        # just disable this for the moment..
        # norm_scale = (np.std(batch_returns_int) + 1e-5)
        # norm_scale = 1.0
        #batch_rewards_int = batch_rewards_int / norm_scale * args.intrinsic_reward_scale
        #batch_returns_int = batch_returns_int / norm_scale * args.intrinsic_reward_scale
        #log.watch_mean("batch_reward_int_norm", np.mean(batch_rewards_int), display_name="rew_int_n")
        #log.watch_mean("batch_return_int_norm", np.mean(batch_returns_int), display_name="ret_int_n")

        log.watch_mean("value_est_ext", np.mean(batch_value_ext), display_name="ve_ext")
        log.watch_mean("value_est_int", np.mean(batch_value_int), display_name="ve_int")

        # calculate summaries
        batch_value = batch_value_ext + batch_value_int
        batch_reward = batch_rewards_ext + batch_rewards_int

        # ----------------------------------------------------
        # estimate advantages
        # ----------------------------------------------------

        # we calculate the advantages by going backwards..
        # estimated return is the estimated return being in state i
        # this is largely based off https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py

        batch_advantage = np.zeros([args.n_steps, args.agents], dtype=np.float32)
        terminal_next_i = np.asarray([False] * args.agents) # this should be final dones...
        value_next_i = final_value_estimate_ext + final_value_estimate_int
        prev_adv = np.zeros([args.agents], dtype=np.float32)

        # second note... this is not correct as we do not separate out the int and ext gammas

        # note: I think the terminal next_i is off here, need to look into this
        for i in reversed(range(args.n_steps)):
            delta = batch_reward[i] + args.gamma * value_next_i * (1.0-terminal_next_i) - batch_value[i]

            batch_advantage[i] = prev_adv = delta + args.gamma * args.gae_lambda * (1.0-terminal_next_i) * prev_adv

            value_next_i = batch_value[i]
            terminal_next_i = batch_terminal[i]

        rollout_time = (time.time() - step_start_time) / batch_size

        train_start_time = time.time()

        # normalize batch advantages
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

        batch_arrays = [
            batch_prev_state.reshape([batch_size, *state_shape]),
            batch_next_state.reshape([batch_size, *state_shape]),
            batch_action.reshape(batch_size),
            batch_returns_ext.reshape(batch_size),
            batch_returns_int.reshape(batch_size),
            batch_logpolicy.reshape([batch_size, *policy_shape]),
            batch_advantage.reshape(batch_size),
            batch_value_ext.reshape(batch_size),
            batch_value_int.reshape(batch_size)
        ]

        for i in range(args.batch_epochs):

            ordering = list(range(batch_size))
            np.random.shuffle(ordering)

            n_batches = math.ceil(batch_size / args.mini_batch_size)

            for j in range(n_batches):

                # put together a minibatch.
                batch_start = j * args.mini_batch_size
                batch_end = (j + 1) * args.mini_batch_size
                sample = ordering[batch_start:batch_end]

                slices = (x[sample] for x in batch_arrays)

                train_minibatch(model, optimizer, log, *slices)

        train_time = (time.time() - train_start_time) / batch_size

        step_time = (time.time() - step_start_time) / batch_size

        log_start_time = time.time()

        fps = 1.0 / (step_time)

        # record some training stats
        log.watch_mean("fps", int(fps))
        log.watch_mean("time_train", train_time*1000, display_postfix="ms", display_precision=1)
        log.watch_mean("time_step", step_time*1000, display_width=0)
        log.watch_mean("time_rollout", rollout_time*1000, display_postfix="ms", display_precision=1)
        log.watch_mean("time_log", log_time*1000, display_postfix="ms", display_precision=1, type="float")

        log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= args.debug_print_frequency:
            save_progress(log)
            log.print_variables(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= config.LOG_EVERY_SEC:
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
                utils.save_checkpoint(checkpoint_name, env_step, model, log, optimizer, atari.ENV_STATE)
                log.log("  -checkpoint saved")

            if args.export_video:
                video_name  = utils.get_checkpoint_path(env_step, env_name+".mp4")
                utils.export_movie(video_name, model, env_name)
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

    # ------------------------------------
    # release the lock

    utils.release_lock()

    log.info()
    log.important("Training Complete.")
    log.info()




