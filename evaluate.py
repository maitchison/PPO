from rl import models, atari, config, utils
from rl.config import args
from rl import hybridVecEnv

import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json
import pickle
import time
import math

# my PC, 355 FPS on GPU, 281 on CPU

from os import listdir
from os.path import isfile, join

DEVICE = "cpu"
REWARD_SCALE = float()
MAX_HORIZON = 500 # this gets changed depending on model's max_horizon
SAMPLES = 64 # 16 is too few, might need 256...

# required for V4 experiments
FORCE_GAMMA_ON = True

# run 100 evaluations
# I want error(k) for... (and for different target gamma as well)
#   * model estimate (k=max)
#   * E(V(s,k)) for each k
#   * E(V_uncertanty_discount(s,k)) for each k

# this will help me figure out if the model is doing ok, because right now it looks like
# error_trunc is 3x error_model
#
# I might also want to figure out at what horizon we should take a reading

# then plot this in a notebook

def load_args(checkpoint_path):
    # get args
    args_path = os.path.join(os.path.split(checkpoint_path)[0], "params.txt")
    with open(args_path, 'r') as f:
        checkpoint_args = json.load(f)
        for k, v in checkpoint_args.items():
            vars(args)[k] = v
        args.log_folder = ''

# load a model and evaluate performance
def load_checkpoint(checkpoint_path, device=None):
    """ Restores model from checkpoint. Returns current env_step"""

    load_args(checkpoint_path)
    model = make_model(env)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    atari.ENV_STATE = checkpoint['env_state']
    global REWARD_SCALE
    global MAX_HORIZON
    REWARD_SCALE = checkpoint['env_state']['returns_norm_state'][1] ** 0.5
    MAX_HORIZON = min(args.tvf_max_horizon + 200, 1000)
    return model

def make_model(env):
    return models.TVFModel(
        head="Nature",
        input_dims=env.observation_space.shape,
        actions=env.action_space.n,
        device=DEVICE,
        dtype=torch.float32,
        use_rnn=False,
        epsilon=args.tvf_epsilon,
        log_horizon=args.tvf_log_horizon,
        needs_gamma=FORCE_GAMMA_ON,
    )

def discount_rewards(rewards, gamma):
    """
    Returns discounted sum of rewards
    """
    reward_sum = 0
    for k, reward in enumerate(rewards):
        reward_sum += reward * (gamma**k)
    return reward_sum

def rediscount_TVF(values, new_gamma):
    """
    Uses truncated value function to estimate value for given states.
    Rewards will be undiscounted, then rediscounted to the correct gamma.
    An alternative to this is to learn multiple discounts at once and get the MLP to generalize

    values: np array of shape [K]
    returns: value estimate (float)
    """
    K = len(values)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1
    for k in range(K):
        reward = (values[k] - prev) / old_discount
        prev = values[k]
        discounted_reward_sum += reward * discount
        old_discount *= args.tvf_gamma
        discount *= new_gamma
    return discounted_reward_sum


def rediscount_TVF_minimize_error(value_mu, value_std, new_gamma):
    """
    Uses truncated value function to get the 'best' horizon in terms of estimated error.
    Error is estimated as MSE error against cummulative rewards to an infinite horizon with the new_gamma

    values: np array of shape [K]
    returns: value estimate (float), horizon used
    """
    assert new_gamma < 1, "Finite sum requires gamma < 1"

    # first we rediscount the returns, and get error estimates
    K = len(value_mu)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1
    new_values = np.zeros_like(value_mu)
    bias_error = np.zeros_like(value_mu)
    var_error = np.zeros_like(value_mu)
    for k in range(K):
        reward = (value_mu[k] - prev) / old_discount
        prev = value_mu[k]
        discounted_reward_sum += reward * discount
        old_discount *= args.tvf_gamma
        discount *= new_gamma
        new_values[k] = discounted_reward_sum

        bias_error[k] = new_gamma ** k
        var_error[k] = value_std[k] ** 2

    return_estimate = new_values[-1] # not sure if this is the best way to get a return estimate...
    new_error = (return_estimate*bias_error)**2 + var_error
    best_k = np.argmin(new_error)
    return new_values[best_k]

def rediscount_TVF_dcyc(value_mu, value_std, new_gamma, alpha=10):
    """
    Uses truncated value function to get a "don't count your chickens" estimate.
    This works by discounting rewards until the uncertanty associated with them is retired.

    values: np array of shape [K]
    returns: value estimates of shape [k], and best estimate
    """

    # the idea here is to discount rewards based on the minimum future uncertanty

    # first we rediscount the returns, and get error estimates
    K = len(value_mu)
    prev = 0
    discounted_reward_sum = 0
    old_discount = 1
    discount = 1
    new_values = np.zeros_like(value_mu)
    for k in range(K):
        reward = (value_mu[k] - prev) / old_discount
        future_risk = min(value_std[k:])
        # note value_mu k probably isn't right here..
        rho = (future_risk**2) / (0.0001+value_mu[k]**2)
        discount_factor = np.exp(-0.5*alpha*rho)
        prev = value_mu[k]
        discounted_reward_sum += reward * discount * discount_factor
        old_discount *= args.tvf_gamma
        discount *= new_gamma
        new_values[k] = discounted_reward_sum

    return new_values[-1]

def evaluate_model(model, filename, samples=16, max_frames = 30*60*15):

    # we play the games one at a time so as to not take up too much memory
    # this is required as the need to know the future rewards at the same time as the states.
    # running twice using fixed seeds would work too, but would require simulating twice

    episode_scores = []
    episode_lengths = []

    print("Evaluating:",end='')

    remaining_samples = samples

    return_estimates = {}

    while remaining_samples > 0:

        batch_samples = min(1, remaining_samples) # 1 at a time is slower, but better on the memory for long runs...
        buffers = generate_rollouts(model, max_frames, num_rollouts=batch_samples)

        # always use these horizons, but add additional up to and including the final horizon
        horizons_to_use = [1, 10, 30, 100, 300]
        h = 500
        while h <= args.tvf_max_horizon:
            horizons_to_use.append(h)
            h *= 2

        for i, buffer in enumerate(buffers):
            # get game score and length
            raw_rewards = buffer["raw_rewards"]
            episode_score = sum(raw_rewards)
            episode_length = len(raw_rewards)
            episode_scores.append(episode_score)
            episode_lengths.append(episode_length)

            for discount in [0.99, 0.997, 0.999]:

                if discount not in return_estimates:
                    return_estimates[discount] = {
                        "trunc_err": [],
                        "model_err": [],
                        #"bestk_err": [],
                        #"count_err": [],
                        "trunc_err_k": [],
                        "true_return": [],
                    }

                trunc_err = return_estimates[discount]["trunc_err"]
                model_err = return_estimates[discount]["model_err"]
                #bestk_err = return_estimates[discount]["bestk_err"]
                #count_err = return_estimates[discount]["count_err"]
                true_return = return_estimates[discount]["true_return"]
                trunc_err_k = return_estimates[discount]["trunc_err_k"]

                for t in range(episode_length):
                    # evaluate value estimate using different methods
                    model_value = buffer["model_values"][t]
                    tvf_value = rediscount_TVF(buffer["values"][t][:args.tvf_max_horizon], discount)
                    true_value = discount_rewards(buffer["rewards"][t:], discount)

                    trunc_err.append(tvf_value - true_value)
                    model_err.append(model_value - true_value)
                    true_return.append(true_value)

                    # evaluate error over horizon
                    true_value_discount = discount_rewards(buffer["rewards"][t:], discount)
                    true_value_no_discount = discount_rewards(buffer["rewards"][t:], 1)
                    for k in horizons_to_use:
                        if k > len(buffer["values"][t]):
                            continue
                        tvf_value = rediscount_TVF(buffer["values"][t][:k], discount)
                        # true value of V(s,k) discounted
                        true_value = discount_rewards(buffer["rewards"][t:t + k], discount)

                        trunc_err_k.append((k, tvf_value, true_value, true_value_discount, true_value_no_discount, i, t))

                print(".", end='')

        remaining_samples -= batch_samples

    def print_it(label, x):
        print(f"{label:<20} {np.mean(x):.2f} +- {np.std(x)/(len(x)**0.5):.2f} [{np.min(x):.1f} to {np.max(x):.1f}]")

    print()
    print()
    print_it("Ep Score:", episode_scores)
    print_it("Ep Length:", episode_lengths)


    print()

    data = {
        'episode_lengths': episode_lengths,
        'episode_scores': episode_scores,
        'return_estimates': return_estimates,
    }

    with open(filename+".dat", "wb") as f:
        pickle.dump(data, f)

def generate_rollout(model, max_frames = 30*60*15, include_video=False):
    return generate_rollouts(model, max_frames, include_video, num_rollouts=1)[0]


def generate_rollouts(model, max_frames = 30*60*15, include_video=False, num_rollouts=1):
    """
    Generates a rollout
    todo: vectorize this so we can generate multiple rollouts at once (will take extra memory though)
    """

    env = hybridVecEnv.HybridAsyncVectorEnv([atari.make for _ in range(num_rollouts)])
    _ = env.reset()
    states = env.reset()

    is_running = [True] * num_rollouts

    frame_count = 0

    buffers = []

    for i in range(num_rollouts):
        buffers.append({
            'values': [],   # values for each horizon of dims [K]
            'errors': [],   # std error estimates for each horizon of dims [K]
            'model_values': [], # models predicted value (float)
            'rewards': [],   # normalized reward (which value predicts)
            'raw_rewards': [], # raw unscaled reward from the atari environment
        })

        if include_video:
            buffers[-1]['frames'] = []  # video frames

    horizons = np.repeat(np.arange(MAX_HORIZON)[None, :], repeats=num_rollouts, axis=0)

    while any(is_running) and frame_count < max_frames:

        model_out = model.forward(states, horizons=horizons)
        log_probs = model_out["log_policy"].detach().cpu().numpy()
        action = np.asarray([utils.sample_action_from_logp(prob) for prob in log_probs], dtype=np.int32)

        states, rewards, dones, infos = env.step(action)

        for i in range(len(states)):
            if dones[i]:
                is_running[i] = False
            if not is_running[i]:
                continue

            channels = infos[i].get("channels", None)
            rendered_frame = infos[i].get("monitor_obs", states[i])
            agent_layers = states[i]

            values = model_out["tvf_value"][i, :].detach().cpu().numpy()
            errors = model_out["tvf_std"][i, :].detach().cpu().numpy() * 1.96
            model_value = model_out["ext_value"][i].detach().cpu().numpy()
            raw_reward = infos[i].get("raw_reward", rewards[i])

            if 'frames' in buffers[i]:
                frame = utils.compose_frame(agent_layers, rendered_frame, channels)
                buffers[i]['frames'].append(frame)

            buffers[i]['values'].append(values)
            buffers[i]['errors'].append(errors)
            buffers[i]['model_values'].append(model_value)
            buffers[i]['rewards'].append(rewards[i])
            buffers[i]['raw_rewards'].append(raw_reward)

        frame_count += 1

    for buffer in buffers:
        for k, v in buffer.items():
            buffer[k] = np.asarray(v)

    return buffers


def export_movie(model, filename, max_frames = 30*60*15):
    """
    Modified version of export movie that supports display of truncated value functions
    In order to show the true discounted returns we write all observations to a buffer, which may take
    a lot of memory.
    """

    scale = 4

    env = atari.make()
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("monitor_obs", state)

    # work out our height
    first_frame = utils.compose_frame(state, rendered_frame)
    height, width, channels = first_frame.shape
    width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    video_out = cv2.VideoWriter(filename+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    start_time = time.time()
    buffer = generate_rollout(model, max_frames, include_video=True)
    rewards = buffer["rewards"]
    end_time = time.time()
    print(f"Rollout generated at {len(rewards)/(end_time-start_time):.0f} FPS")

    for t in range(len(rewards)):

        frame = buffer["frames"][t]
        values = buffer["values"][t] * REWARD_SCALE
        errors = buffer["errors"][t] * REWARD_SCALE
        model_value = buffer["model_values"][t] * REWARD_SCALE

        if not args.tvf_distributional:
            errors *= 0

        if args.vf_coef == 0:
            model_value = 0

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
            (width, height, 3), frame.shape)

        # calculate actual truncated values using real future rewards
        true_values = np.zeros(MAX_HORIZON, dtype=np.float32)
        for k in range(len(true_values)):
            this_reward = (args.tvf_gamma ** k) * rewards[t+k] * REWARD_SCALE if (t+k) < len(rewards) else 0
            prev_rewards = true_values[k-1] if k > 0 else 0
            true_values[k] = this_reward + prev_rewards

        fig = plt.figure(figsize=(7, 4), dpi=100)

        # plot predicted values...
        ys = values
        xs = list(range(len(ys)))
        plt.fill_between(xs, ys-errors, ys+errors, facecolor="blue", alpha=0.2)
        plt.plot(xs, ys, label="Predicted", c="blue")

        # plot true value
        xs = list(range(len(true_values)))
        ys = true_values
        plt.plot(xs, ys, label="True", c="red")

        # plot model value prediction
        #y_true = np.clip(true_values[-1], 0, 200)
        plt.hlines(model_value, 0, len(xs), label="Model", colors=["black"])

        plt.xlabel("k")
        plt.ylabel("Score")

        limit = max(
            100,
            10+max(true_values),
            10+model_value,
            10+max(values+errors),
        )

        plt.ylim(-10, (int(limit)//20) * 20)

        plt.grid(True)

        plt.legend(loc="upper left")

        # from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plot_height, plot_width, _ = data.shape
        frame[0:plot_height,-plot_width:] = data[:, :, ::-1]
        plt.close(fig)

        video_out.write(frame)

    video_out.release()


def run_eval(path):
    for epoch in range(200):
        checkpoint_eval_file = os.path.join(path, f"checkpoint-{epoch:03d}M-eval.dat")
        checkpoint_name = os.path.join(path, f"checkpoint-{epoch:03d}M-params.pt")
        if os.path.exists(checkpoint_name) and not os.path.exists(checkpoint_eval_file):
            print()
            print(checkpoint_name)

            # create an empty file to mark that we are working on this...
            with open(checkpoint_eval_file, "wb"):
                pass

            model = load_checkpoint(checkpoint_name, device=DEVICE)
            export_movie(model, os.path.join(path, f"checkpoint-{epoch:03d}M-eval"))
            evaluate_model(model, os.path.join(path, f"checkpoint-{epoch:03d}M-eval"), samples=SAMPLES)

def monitor(path):
    folders = [x[0] for x in os.walk(path)]
    for folder in folders:
        run_eval(folder)

if __name__ == "__main__":

    # set args by hand
    config.parse_args()

    # get model
    args.env_name = utils.get_environment_name(args.environment, args.sticky_actions)
    args.sync_envs = True

    # set resolution
    args.res_x, args.res_y = (84, 84)

    env = atari.make()

    for c in "ABCDEFGHIJKLMNOP":
        monitor(f"./Run/TVF_4{c}")

    #for c in "ABCDEFGHIJKLMNOP":
    #    monitor(f"./Run/TVF_5{c}")


