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

my_counter = 0

def train_minibatch(model: models.PolicyModel, optimizer, prev_states, next_states, actions, returns, policy_logprobs, advantages, values):
    """
    :param model:           The model for this agent.
    :param optimizer:       The optimizer to use.
    :param prev_states:     tensor of dims [N, C, H, W]
    :param actions:         tensor of dims [N]
    :param returns:         tensor of dims [N]
    :param policy_logprobs: tensor of dims [N, Actions]
    :param advantages:      tensor of dims [N]
    :param values:          tensor of dims [N]
    :return:
    """

    global my_counter
    my_counter += 1

    # prepare the tensors for the model (moves to GPU and converts to float.
    # note, we don't do this with the states as these are expected to be in uint8 format.
    policy_logprobs = model.prep_for_model(policy_logprobs)
    advantages = model.prep_for_model(advantages)
    returns = model.prep_for_model(returns)
    old_pred_values = model.prep_for_model(values)

    mini_batch_size = len(prev_states)

    logps, value_prediction = model.forward(prev_states)

    ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])

    loss_clip = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - args.ppo_epsilon, 1 + args.ppo_epsilon) * advantages))

    # this one is taken from PPO2 baseline, reduces variance but not sure why? does it stop the values from moving
    # too much?
    value_prediction_clipped = old_pred_values + torch.clamp(value_prediction - old_pred_values, -args.ppo_epsilon, +args.ppo_epsilon)

    vf_losses1 = (value_prediction - returns).pow(2)
    vf_losses2 = (value_prediction_clipped - returns).pow(2)
    loss_value = - args.vf_coef * 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    loss_entropy = args.entropy_bonus * utils.log_entropy(logps) / mini_batch_size

    loss = -(loss_clip + loss_value + loss_entropy)  # gradient ascent.

    # calculate RND gradient
    if args.use_rnd:

        # note: this could be much much more efficent, I really should just be passing in mu and sigma, and then
        # letting the model do the transformation... (i.e. keep everything as uint8)

        # train on only 25% of minibatch to slow down the predictor network when used with large number of agents.

        predictor_proportion = np.clip(32 / args.agents, 0.01, 1)

        mu = atari.ENV_STATE["observation_norm_state"][0]
        sigma = np.sqrt(atari.ENV_STATE["observation_norm_state"][1])

        normed_states = np.asarray(np.clip((prev_states - mu) / (sigma + 0.0001), -5, 5), dtype=np.float32)

        n = int(len(prev_states) * predictor_proportion)
        loss_rnd = model.prediction_error(normed_states[:n]).mean()
        loss += loss_rnd
        if my_counter % 100 == 0:
            print("loss_rnd {:.4f}".format(float(loss_rnd)))

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

        if my_counter % 100 == 0:
            print("loss_idm {:.3f} accuracy {:.3f} loss_fdm {:.3f} loss_ae {:.3f} loss_norm {:.3f} embd_std {:.3f}".format(
                float(loss_idm), float(accuracy), float(loss_fdm), float(loss_ae), float(loss_norm),
                float(torch.std(next_embedding[0]))))

        loss += (loss_idm + loss_fdm + loss_ae + loss_norm)


    optimizer.zero_grad()
    loss.backward()

    if args.max_grad_norm is not None and args.max_grad_norm != 0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    else:
        grad_norm = 0

    optimizer.step()

    return (float(x) for x in [-loss, loss_clip, loss_value, loss_entropy, grad_norm])


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
    batch_reward = np.zeros([N, A], dtype=np.float32)
    batch_intrinsic_reward = np.zeros([N, A], dtype=np.float32)
    batch_logpolicy = np.zeros([N, A, *policy_shape], dtype=np.float32)
    batch_terminal = np.zeros([N, A], dtype=np.bool)
    batch_value = np.zeros([N, A], dtype=np.float32)

    for t in range(N):

        logprobs, value = model.forward(states)

        logprobs = logprobs.detach().cpu().numpy()
        value = value.detach().cpu().numpy()

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

            # todo: norming should be handled by model... and I should update the model's constants..
            if "observation_norm_state" in atari.ENV_STATE:
                mu = atari.ENV_STATE["observation_norm_state"][0]
                sigma = np.sqrt(atari.ENV_STATE["observation_norm_state"][1])

                normed_states = np.asarray(np.clip((states - mu) / (sigma + 0.0001), -5, 5), dtype=np.float32)

                loss_rnd = model.prediction_error(normed_states).detach().cpu().numpy()
                intrinsic_rewards += loss_rnd

        raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(rewards, infos)], dtype=np.float32)

        # save a copy of the normalization statistics.
        for key in ["returns_norm_state", "observation_norm_state"]:
            if key in infos[0]:
                atari.ENV_STATE[key] = infos[0][key]

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
        batch_reward[t] = rewards
        batch_intrinsic_reward[t] = intrinsic_rewards
        batch_logpolicy[t] = logprobs
        batch_terminal[t] = dones
        batch_value[t] = value

    return (batch_prev_state, batch_next_state, batch_action, batch_reward, batch_intrinsic_reward, batch_logpolicy, batch_terminal, batch_value,
            states)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
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


def train(env_name, model: models.PolicyModel):
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
    log = Logger()
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
        print("Previous checkpoint detected.")
        checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
        restored_step, log, norm_state = utils.load_checkpoint(checkpoint_path, model, optimizer)

        for k, v in norm_state.items():
            atari.ENV_STATE[k] = v

        print("  (resumed from step {:.0f}M)".format(restored_step/1000/1000))
        start_iteration = (restored_step // batch_size) + 1
        did_restore = True
    else:
        start_iteration = 0
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

    print("Generated {} agents ({}) using {} ({}) model.".format(args.agents, "async" if not args.sync_envs else "sync", model.name, model.dtype))

    print_counter = 0

    if start_iteration == 0 and (args.limit_epochs is None):
        print("Training for {:.1f}M steps".format(n_iterations*batch_size/1000/1000))
    else:
        print("Training block from " +
              utils.Color.WARNING + str(round(start_iteration * batch_size / 1000 / 1000)) + "M" + utils.Color.ENDC + " to (" +
              utils.Color.WARNING + str(round(n_iterations * batch_size / 1000 / 1000)) + "M" + utils.Color.ENDC +
              " / " + str(round(args.epochs)) +"M) steps")

    print()

    # initialize agent
    states = vec_env.reset()

    episode_score = np.zeros([args.agents], dtype = np.float32)
    episode_len = np.zeros([args.agents], dtype = np.int32)

    initial_start_time = time.time()

    last_print_time = -1
    last_log_time = -1

    # add a few checkpoints early on

    checkpoints = [x // batch_size for x in range(0, n_iterations*batch_size+1, config.CHECKPOINT_EVERY_STEPS)]
    checkpoints += [x // batch_size for x in [1e6]] #add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    for iteration in range(start_iteration, n_iterations+1):

        env_step = iteration * batch_size

        adjust_learning_rate(optimizer, env_step / 1e6)

        # the idea here is that all our batch arrays are of dims
        # N, A, ...,
        # Where n is the number of steps, and A is the number of agents.
        # this means we can process each step as a vector

        start_time = time.time()

        # collect experience
        batch_prev_state, batch_next_state, batch_action, batch_reward, batch_intrinsic_reward, \
        batch_logpolicy, batch_terminal, batch_value, states = \
            run_agents_vec(args.n_steps, model, vec_env, states, episode_score, episode_len, log,
                state_shape, state_dtype, policy_shape)

        # normalize intrinsic reward across rollout.
        batch_intrinsic_reward = (batch_intrinsic_reward - np.mean(batch_intrinsic_reward)) / (np.std(batch_intrinsic_reward) + 1e-5)

        # stub:
        # just add rewards together, should be processed separately with different gamma
        batch_reward += batch_intrinsic_reward * 0.05

        # ----------------------------------------------------
        # estimate advantages

        # we calculate the advantages by going backwards..
        # estimated return is the estimated return being in state i
        # this is largely based off https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/ppo2/ppo2.py
        batch_advantage = np.zeros([args.n_steps, args.agents], dtype=np.float32)
        value_next_i = model.value(states).detach().cpu().numpy()
        terminal_next_i = np.asarray([False] * args.agents)
        prev_adv = np.zeros([args.agents], dtype=np.float32)

        for i in reversed(range(args.n_steps)):
            delta = batch_reward[i] + args.gamma * value_next_i * (1.0-terminal_next_i) - batch_value[i]

            batch_advantage[i] = prev_adv = delta + args.gamma * args.gae_lambda * (1.0-terminal_next_i) * prev_adv

            value_next_i = batch_value[i]
            terminal_next_i = batch_terminal[i]

        batch_returns = batch_advantage + batch_value

        rollout_time = (time.time() - start_time) / batch_size

        start_train_time = time.time()

        # normalize batch advantages
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)

        batch_arrays = [
            batch_prev_state.reshape([batch_size, *state_shape]),
            batch_next_state.reshape([batch_size, *state_shape]),
            batch_action.reshape(batch_size),
            batch_returns.reshape(batch_size),
            batch_logpolicy.reshape([batch_size, *policy_shape]),
            batch_advantage.reshape(batch_size),
            batch_value.reshape(batch_size)
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

                loss, loss_clip, loss_value, loss_entropy, grad_norm = train_minibatch(model, optimizer, *slices)

                log.watch_mean("loss", loss)
                log.watch_mean("loss_clip", loss_clip)
                log.watch_mean("loss_value", loss_value)
                log.watch_mean("loss_entropy", loss_entropy)
                log.watch_mean("opt_grad_norm", grad_norm)

        train_time = (time.time() - start_train_time) / batch_size

        step_time = (time.time() - start_time) / batch_size

        fps = 1.0 / (step_time)

        # record some training stats
        log.watch("iteration", iteration, display_priority=5)
        log.watch("env_step", env_step, display_priority=4, display_width=12, display_scale=1e-6, display_postfix="M",
                  display_precison=2)
        log.watch("walltime", time.time()-initial_start_time,
                  display_priority=3, display_scale=1/(60*60), display_postfix="h", display_precision = 1)
        log.watch_mean("fps", int(fps))
        log.watch_mean("time_train", train_time*1000, display_postfix="ms", display_precision=1)
        log.watch_mean("time_step", step_time*1000, display_width=0)
        log.watch_mean("time_rollout", rollout_time*1000, display_postfix="ms", display_precision=1)

        log.record_step()

        # periodically print and save progress
        if time.time() - last_print_time >= config.PRINT_EVERY_SEC:
            save_progress(log)
            log.print(include_header=print_counter % 10 == 0)
            last_print_time = time.time()
            print_counter += 1

        # save log and refresh lock
        if time.time() - last_log_time >= config.LOG_EVERY_SEC:
            utils.lock_job()
            start_time = time.time()
            log.export_to_csv(os.path.join(args.log_folder, "training_log.csv"))
            log.watch("_export_log_time", time.time()-start_time * 1000, display_width=0)
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            print()
            print(utils.Color.OKGREEN + "Checkpoint: {}".format(args.log_folder) + utils.Color.ENDC)

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                utils.save_checkpoint(checkpoint_name, env_step, model, log, optimizer, atari.ENV_STATE)
                print("  -checkpoint saved")

            if args.export_video:
                video_name  = utils.get_checkpoint_path(env_step, env_name+".mp4")
                utils.export_movie(video_name, model, env_name)
                print("  -video exported")

            print()

    # -------------------------------------
    # save final information

    save_progress(log)
    log.export_to_csv(os.path.join(args.log_folder, "training_log.csv"))

    # ------------------------------------
    # release the lock

    utils.release_lock()

    print()
    print(utils.Color.OKGREEN+"Training Complete."+utils.Color.ENDC)
    print()




