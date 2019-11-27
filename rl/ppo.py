import os
import numpy as np
import matplotlib.pyplot as plt
import gym
import torch
import torch.nn as nn
import time
import csv
import json
import math
from collections import deque

from . import utils, models, atari, hybridVecEnv, config
from .config import args

def train_minibatch(model: models.PolicyModel, optimizer, ppo_epsilon, vf_coef, entropy_bonus, max_grad_norm,
                    prev_states, actions, returns, policy_logprobs, advantages, values):

    policy_logprobs = model.prep_for_model(policy_logprobs)
    advantages = model.prep_for_model(advantages)
    returns = model.prep_for_model(returns)
    old_pred_values = model.prep_for_model(values)

    mini_batch_size = len(prev_states)

    logps, value_prediction = model.forward(prev_states)

    ratio = torch.exp(logps[range(mini_batch_size), actions] - policy_logprobs[range(mini_batch_size), actions])

    loss_clip = torch.mean(torch.min(ratio * advantages, torch.clamp(ratio, 1 - ppo_epsilon, 1 + ppo_epsilon) * advantages))

    # this one is taken from PPO2 baseline, reduces variance but not sure why? does it stop the values from moving
    # too much?
    value_prediction_clipped = old_pred_values + torch.clamp(value_prediction - old_pred_values, -ppo_epsilon, +ppo_epsilon)

    vf_losses1 = (value_prediction - returns).pow(2)
    vf_losses2 = (value_prediction_clipped - returns).pow(2)
    loss_value = - vf_coef * 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

    loss_entropy = entropy_bonus * utils.log_entropy(logps) / mini_batch_size

    loss = -(loss_clip + loss_value + loss_entropy)  # gradient ascent.

    optimizer.zero_grad()
    loss.backward()

    if max_grad_norm is not None and max_grad_norm != 0:
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    batch_action = np.zeros([N, A], dtype=np.int32)
    batch_reward = np.zeros([N, A], dtype=np.float32)
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

        raw_rewards = np.asarray([info.get("raw_reward", reward) for reward, info in zip(rewards, infos)], dtype=np.float32)

        # save a copy of the normalization statistics.
        norm_state = infos[0].get("returns_norm_state", None)
        if norm_state is not None:
            atari.set_env_norm_state(norm_state)

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
        batch_action[t] = actions
        batch_reward[t] = rewards
        batch_logpolicy[t] = logprobs
        batch_terminal[t] = dones
        batch_value[t] = value

    return (batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal, batch_value)


def save_training_log(training_log):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)

    batch_size = (args.n_steps * args.agents)

    final_epoch = min(args.epochs, args.limit_epochs) if args.limit_epochs is not None else args.epochs
    n_iterations = math.ceil((final_epoch * 1e6) / batch_size)

    # detect a previous experiment
    checkpoints = utils.get_checkpoints(args.log_folder)
    if len(checkpoints) > 0:
        print("Previous checkpoint detected.")
        checkpoint_path = os.path.join(args.log_folder, checkpoints[0][1])
        restored_step, logs, norm_state = utils.load_checkpoint(checkpoint_path, model, optimizer)
        atari.set_env_norm_state(norm_state)
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

    fps_history = deque(maxlen=10 if not config.PROFILE_INFO else None)

    checkpoint_every = int(5e6)

    # add a few checkpoints early on

    checkpoints = [x // batch_size for x in range(0, n_iterations*batch_size+1, checkpoint_every)]
    checkpoints += [x // batch_size for x in [1e6]] #add a checkpoint early on (1m steps)
    checkpoints.append(n_iterations)
    checkpoints = sorted(set(checkpoints))

    for iteration in range(start_iteration, n_iterations+1):

        env_step = iteration * batch_size

        # the idea here is that all our batch arrays are of dims
        # N, A, ...,
        # Where n is the number of steps, and A is the number of agents.
        # this means we can process each step as a vector

        start_time = time.time()

        # collect experience
        batch_prev_state, batch_action, batch_reward, batch_logpolicy, batch_terminal, batch_value = run_agents_vec(
            args.n_steps, model, vec_env, states, episode_score, episode_len, score_history, len_history,
            state_shape, state_dtype, policy_shape)

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
            np.asarray(batch_prev_state.reshape([batch_size, *state_shape])),
            np.asarray(batch_action.reshape(batch_size)),
            np.asarray(batch_returns.reshape(batch_size)),
            np.asarray(batch_logpolicy.reshape([batch_size, *policy_shape])),
            np.asarray(batch_advantage.reshape(batch_size)),
            np.asarray(batch_value.reshape(batch_size))
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

                loss, loss_clip, loss_value, loss_entropy, grad_norm = train_minibatch(
                    model, optimizer, args.ppo_epsilon, args.vf_coef, args.entropy_bonus, args.max_grad_norm, *slices)

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

        if True:
            # save current step information.
            details = {}
            details["max_epochs"] = args.epochs
            details["completed_epochs"] = (env_step + batch_size) / 1e6 # include the current completed step.
            details["score"] = np.percentile(utils.smooth(score_history, 0.9), 95) if len(score_history) > 0 else None
            details["fraction_complete"] = details["completed_epochs"] / details["max_epochs"]
            details["fps"] = int(np.mean(fps_history))
            frames_remaining = (details["max_epochs"] - details["completed_epochs"]) * 1e6
            details["eta"] = frames_remaining / details["fps"]
            details["host"] = args.hostname
            details["last_modified"] = time.time()
            with open(os.path.join(args.log_folder, "progress.txt"),"w") as f:
                json.dump(details, f)

        if config.PRINT_EVERY:
            if iteration % (config.PRINT_EVERY * 10) == 0:
                print("{:>8}{:>8}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>8}".format("iter", "step", "loss", "l_clip", "l_value",
                                                                      "l_ent", "ep_score", "ep_len", "elapsed", "fps"))
                print("-"*120)
            if iteration % config.PRINT_EVERY == 0 or iteration == n_iterations:
                print("{:>8}{:>8}{:>10.3f}{:>10.4f}{:>10.4f}{:>10.4f}{:>10.2f}{:>10.0f}{:>10}{:>8.0f} {:<10}".format(
                    str(iteration),
                    "{:.2f}M".format(env_step / 1000 / 1000),
                    training_log[-1][0],
                    training_log[-1][1],
                    training_log[-1][2],
                    training_log[-1][3],
                    utils.with_default(training_log[-1][4], 0),
                    utils.with_default(training_log[-1][5], 0),
                    "{:.0f} min".format(training_log[-1][8]/60),
                    training_log[-1][11],
                    utils.with_default(training_log[-1][13], 0)
                ))

        # make sure we don't save the checkpoint we just restored from.
        if (iteration in checkpoints) and (not did_restore or iteration != start_iteration):

            print()
            print(utils.Color.OKGREEN + "Checkpoint: {}".format(args.log_folder) + utils.Color.ENDC)

            if args.save_checkpoints:
                checkpoint_name = utils.get_checkpoint_path(env_step, "params.pt")
                logs = (training_log, timing_log, score_history, len_history)
                utils.save_checkpoint(checkpoint_name, env_step, model, optimizer, atari.ENV_NORM_STATE, logs)
                print("  -checkpoint saved")

            if args.export_video:
                video_name  = utils.get_checkpoint_path(env_step, env_name+".mp4")
                utils.export_movie(video_name, model, env_name)
                print("  -video exported")


            print()

        if iteration in [5, 10, 20, 30, 40] or iteration % config.SAVE_LOG_EVERY == 0 or iteration == n_iterations:

            save_training_log(training_log)

            clean_training_log = training_log[10:] if len(training_log) >= 10 else training_log  # first sample is usually extreme.

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

    if config.PROFILE_INFO:
        print_profile_info(timing_log, "Final timing results")
        save_profile_log(os.path.join(args.log_folder, "timing_info.csv"), timing_log)

    # save a final score.
    if args.limit_epochs is None:
        # only write final score once we finish the last epoch.
        with open(os.path.join(args.log_folder, "final_score.txt"), "w") as f:
            f.write(str(np.percentile(score_history,95)))

    utils.release_lock()

    print()
    print(utils.Color.OKGREEN+"Training Complete."+utils.Color.ENDC)
    print()

    return training_log




