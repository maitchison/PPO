import os
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import json
import math
from collections import deque

from . import utils, models, atari, hybridVecEnv, config
from .config import args

print_counter = 0
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
        # train on only 25% of minibatch to slow down the predictor network when used with large number of agents.
        predictor_proportion = np.clip(32 / args.agents, 0.01, 1)
        n = int(len(prev_states) * predictor_proportion)
        loss_rnd = model.prediction_error(prev_states[:n]).mean()
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


def run_agents_vec(n_steps, model, vec_envs, states, episode_score, episode_len, score_history, len_history,
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
            loss_rnd = model.prediction_error(prev_states).detach().cpu().numpy()
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
                score_history.append(episode_score[i])
                len_history.append(episode_len[i])
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


def save_training_log(training_log, include_graphs=True):
    filename = os.path.join(args.log_folder, "training_log.csv")
    with open(filename, mode='w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["Loss", "Loss_Clip", "Loss_Value", "Loss_Entropy",
                             "Ep_Score (100)", "Ep_Len (100)",
                             "Ep_Score (10)", "Ep_Len (10)",
                             "Elapsed", "Iteration", "Step", "FPS", "Gradient_Norm", "History"])

        for row in training_log:
            # convert values lower precision
            row = [utils.sig_fig(x,sf=4) for x in row]
            csv_writer.writerow(row)

    if include_graphs:
        save_training_graphs(training_log)

def save_profile_log(filename, timing_log):
    with open(filename, "w") as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerow(["Step", "Rollout_Time", "Train_Time", "Step_Time", "Batch_Size", "FPS", "CUDA_Memory"])
        for row in timing_log:
            csv_writer.writerow(row)

def print_profile_info(timing_log, title="Performance results:"):

    timing_log = np.asarray(timing_log)

    rollout_time = timing_log[:, 1].mean()
    train_time = timing_log[:, 2].mean()
    step_time = timing_log[:, 3].mean()
    fps = timing_log[:, 5].mean()
    fps_std_error = timing_log[:, 5].std(ddof=1) / math.sqrt(len(timing_log))

    print(title+": {:.2f}ms / {:.2f}ms / {:.2f}ms  [{:.0f} FPS +- {:.1f}]".format(
        step_time, rollout_time, train_time, fps, fps_std_error))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (args.learning_rate_decay ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_progress(env_step, score_history, fps_history):
    if True:
        # save current step information.
        details = {}
        details["max_epochs"] = args.epochs
        details["completed_epochs"] = env_step / 1e6  # include the current completed step.
        details["score"] = np.percentile(utils.smooth(score_history, 0.9), 95) if len(score_history) > 0 else None
        details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
        details["fps"] = int(np.mean(fps_history))
        frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
        details["eta"] = frames_remaining / details["fps"]
        details["host"] = args.hostname
        details["last_modified"] = time.time()
        with open(os.path.join(args.log_folder, "progress.txt"), "w") as f:
            json.dump(details, f)


def save_training_graphs(training_log):
    clean_training_log = training_log[10:] if len(
        training_log) >= 10 else training_log  # first sample is usually extreme.

    xs = [x[10] for x in clean_training_log]
    plt.figure(figsize=(8, 8))
    plt.grid()

    labels = ["loss", "loss_clip", "loss_value", "loss_entropy"]
    ys = [[x[i] for x in clean_training_log] for i in range(4)]
    colors = ["red", "green", "blue", "black"]

    for label, y, c in zip(labels, ys, colors):
        plt.plot(xs, y, alpha=0.2, c=c)
        plt.plot(xs, utils.smooth(y), label=label, c=c)

    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Env Step")
    plt.savefig(os.path.join(args.log_folder, "losses.png"))
    plt.close()

    xs = []
    rewards = []
    lengths = []
    rewards10 = []
    lengths10 = []
    for i, x in enumerate(clean_training_log):
        if x[4] is None:
            continue
        xs.append(x[10])
        rewards.append(x[4])
        lengths.append(x[5])
        rewards10.append(x[6])
        lengths10.append(x[7])

    if len(rewards) > 10:
        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.plot(xs, rewards10, alpha=0.2)
        plt.plot(xs, rewards)
        plt.ylabel("Reward")
        plt.xlabel("Env Step")
        plt.savefig(os.path.join(args.log_folder, "ep_reward.png"))
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.grid()
        plt.plot(xs, lengths10, alpha=0.2)
        plt.plot(xs, lengths)
        plt.ylabel("Episode Length")
        plt.xlabel("Env Step")
        plt.savefig(os.path.join(args.log_folder, "ep_length.png"))
        plt.close()


def print_progress(iteration, env_step, training_log):
    global print_counter
    if print_counter % 10 == 0:
        print(
            "{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>8}".format("iter", "step", "loss", "l_clip", "l_value",
                                                                               "l_ent", "ep_score", "ep_len", "elapsed",
                                                                               "fps"))
        print("-" * 120)

    print("{:>8}{:>8}{:>10.3f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.2f}{:>10.0f}{:>10}{:>8.0f} {:<10}".format(
        str(iteration),
        "{:.2f}M".format(env_step / 1000 / 1000),
        training_log[-1][0],
        training_log[-1][1],
        training_log[-1][2],
        training_log[-1][3],
        utils.with_default(training_log[-1][4], 0),
        utils.with_default(training_log[-1][5], 0),
        "{:.0f} min".format(training_log[-1][8] / 60),
        training_log[-1][11],
        utils.with_default(training_log[-1][13], 0)
    ))
    print_counter += 1

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

    utils.lock_job()

    # get shapes and dtypes
    _env = atari.make(env_name)
    obs = _env.reset()
    state_shape = obs.shape
    state_dtype = obs.dtype
    policy_shape = model.policy(obs[np.newaxis])[0].shape
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
        restored_step, logs, norm_state = utils.load_checkpoint(checkpoint_path, model, optimizer)

        for k, v in norm_state:
            atari.ENV_STATE[k] = v

        training_log, timing_log, score_history, len_history = logs
        print("  (resumed from step {:.0f}M)".format(restored_step/1000/1000))
        start_iteration = (restored_step // batch_size) + 1
        did_restore = True
    else:
        start_iteration = 0
        did_restore = False

        training_log = []
        timing_log = []
        score_history = []
        len_history = []

    # make a copy of params
    with open(os.path.join(args.log_folder, "params.txt"),"w") as f:
        params = {k:v for k,v in args.__dict__.items()}
        # fix up some of the types...
        params["dtype"] = str(params["dtype"])
        params["model"] = params["model"].name
        f.write(json.dumps(params, indent=4))

    # make a copy of training files for reference
    utils.copy_source_files("./", args.log_folder)

    env_fns = [lambda : atari.make(env_name) for _ in range(args.agents)]
    vec_env = hybridVecEnv.HybridAsyncVectorEnv(env_fns, max_cpus=args.workers, verbose=True) if not args.sync_envs else gym.vector.SyncVectorEnv(env_fns)

    print("Generated {} agents ({}) using {} ({}) model.".format(args.agents, "async" if not args.sync_envs else "sync", model.name, model.dtype))

    if start_iteration == 0 and (args.limit_epochs is None):
        print("Training for {:.1f}M steps".format(n_iterations*batch_size/1000/1000))
    else:
        print("Training block from " +
              utils.Color.WARNING + str(round(start_iteration * batch_size / 1000 / 1000)) + "M" + utils.Color.ENDC + " to (" +
              utils.Color.WARNING + str(round(n_iterations * batch_size / 1000 / 1000)) + "M" + utils.Color.ENDC +
              " / " + str(round(args.epochs)) +"M) steps")

    print()
    print("-" * 120)

    # initialize agent
    states = vec_env.reset()

    episode_score = np.zeros([args.agents], dtype = np.float32)
    episode_len = np.zeros([args.agents], dtype = np.int32)

    initial_start_time = time.time()

    last_print_time = -1
    last_log_time = -1

    fps_history = deque(maxlen=10 if not config.PROFILE_INFO else None)


    env_step = 0

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
            run_agents_vec(args.n_steps, model, vec_env, states, episode_score, episode_len, score_history, len_history,
                state_shape, state_dtype, policy_shape)

        # normalize intrinsic reward across rollout.
        batch_intrinsic_reward = (batch_intrinsic_reward - np.mean(batch_intrinsic_reward)) / (np.std(batch_intrinsic_reward) + 1e-5)

        # stub:
        # display some info on rewards when a true reward is found.
        if args.use_rnd and np.sum(batch_reward) != 0:
            print((utils.Color.HEADER+"Rollout rewards: Scaled: {:.2f}/{:.2f} Intrinsic {:.2f}/{:.2f}"+utils.Color.ENDC).format(
                np.mean(batch_reward), np.std(batch_reward),
                np.mean(batch_intrinsic_reward), np.std(batch_intrinsic_reward),
            ))

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

        total_loss_clip = 0
        total_loss_value = 0
        total_loss_entropy = 0
        total_loss = 0
        total_grad_norm = 0

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

                total_loss_clip += loss_clip / (args.batch_epochs*n_batches)
                total_loss_value += loss_value / (args.batch_epochs*n_batches)
                total_loss_entropy += loss_entropy / (args.batch_epochs*n_batches)
                total_loss += loss / (args.batch_epochs*n_batches)
                total_grad_norm += grad_norm / (args.batch_epochs * n_batches)

        train_time = (time.time() - start_train_time) / batch_size

        step_time = (time.time() - start_time) / batch_size

        fps = 1.0 / (step_time)

        if config.PROFILE_INFO:
            if "cuda" in args.device:
                cuda_memory = torch.cuda.max_memory_allocated()
            else:
                cuda_memory = 0

            timing_log.append((iteration, rollout_time * 1000, train_time * 1000, step_time * 1000, batch_size, fps, cuda_memory/1024/1024))

            # print early timing information from second iteration.
            if iteration == 1:
                print_profile_info(timing_log, "Early timing results")

        fps_history.append(fps)

        history_string = "({:.1f} - {:.1f}) +- {:.2f}".format(
            min(score_history[-100:]), max(score_history[-100:]), np.std(score_history[-100:])
        ) if len(score_history) > 0 else ""

        training_log.append(
            (total_loss,
             total_loss_clip,
             total_loss_value,
             total_loss_entropy,
             utils.safe_mean(score_history[-100:], 2),
             utils.safe_mean(len_history[-100:], 2),
             utils.safe_mean(score_history[-10:], 2),
             utils.safe_mean(len_history[-10:], 2),
             time.time()-initial_start_time,
             iteration,
             env_step,
             int(np.mean(fps_history)),
             total_grad_norm,
             history_string
             )
        )

        # periodically print and save progress
        if time.time() - last_print_time >= config.PRINT_EVERY_SEC:
            save_progress(env_step + batch_size, score_history, fps_history)
            print_progress(iteration, env_step, training_log)
            last_print_time = time.time()

        # save log and refresh lock
        if time.time() - last_log_time >= config.LOG_EVERY_SEC:
            utils.lock_job()
            save_training_log(training_log)
            last_log_time = time.time()

        # periodically save checkpoints
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            print()
            print(utils.Color.OKGREEN + "Checkpoint: {}".format(args.log_folder) + utils.Color.ENDC)

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                logs = (training_log, timing_log, score_history, len_history)
                utils.save_checkpoint(checkpoint_name, env_step, model, optimizer, atari.ENV_STATE, logs)
                print("  -checkpoint saved")

            if args.export_video:
                video_name  = utils.get_checkpoint_path(env_step, env_name+".mp4")
                utils.export_movie(video_name, model, env_name)
                print("  -video exported")

            print()

    # -------------------------------------
    # save final information

    if config.PROFILE_INFO:
        print_profile_info(timing_log, "Final timing results")
        save_profile_log(os.path.join(args.log_folder, "timing_info.csv"), timing_log)

    save_progress(env_step + batch_size, score_history, fps_history)
    save_training_log(training_log)

    # ------------------------------------
    # release the lock

    utils.release_lock()

    print()
    print(utils.Color.OKGREEN+"Training Complete."+utils.Color.ENDC)
    print()

    return training_log




