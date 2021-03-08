from rl import models, atari, config, utils
from rl.config import args

import cv2
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import json
import pickle

from os import listdir
from os.path import isfile, join

DEVICE = "cpu"
REWARD_SCALE = float()
MAX_HORIZON = 500

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

# load a model and evaluate performance
def load_checkpoint(model, checkpoint_path, device=None):
    """ Restores model from checkpoint. Returns current env_step"""

    # get args
    args_path = os.path.join(os.path.split(checkpoint_path)[0], "params.txt")
    with open(args_path, 'r') as f:
        checkpoint_args = json.load(f)
        for k, v in checkpoint_args.items():
            vars(args)[k] = v
        args.log_folder = ''

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    step = checkpoint['step']
    atari.ENV_STATE = checkpoint['env_state']
    global REWARD_SCALE
    REWARD_SCALE = checkpoint['env_state']['returns_norm_state'][1] ** 0.5
    return step

def make_model(env):
    return models.TVFModel(
        head="Nature",
        input_dims=env.observation_space.shape,
        actions=env.action_space.n,
        device=DEVICE,
        dtype=torch.float32,
        use_rnn=False,
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


def evaluate_model(model, filename, samples=16, max_frames = 30*60*15):

    # we play the games one at a time so as to not take up too much memory
    # this is required as the need to know the future rewards at the same time as the states.
    # running twice using fixed seeds would work too, but would require simulating twice

    episode_scores = []
    episode_lengths = []

    model_err = []
    trunc_err = []
    true_returns = []


    trunc_err_k = [[] for _ in range(MAX_HORIZON)]

    print("Evaluating:",end='')

    for sample in range(samples):
        buffer = generate_rollout(model, max_frames)

        # get game score and length
        raw_rewards = buffer["raw_rewards"]
        episode_score = sum(raw_rewards)
        episode_length = len(raw_rewards)
        episode_scores.append(episode_score)
        episode_lengths.append(episode_length)

        for t in range(episode_length):
            # evaluate MSE for max horizon value predictions under different algorithms
            model_value = buffer["model_values"][t]
            tvf_value = rediscount_TVF(buffer["values"][t][:args.tvf_max_horizon], 0.99)
            true_value = discount_rewards(buffer["rewards"][t:], 0.99)

            trunc_err.append(tvf_value - true_value)
            model_err.append(model_value - true_value)
            true_returns.append(true_value)

            # evaluate error over horizon
            for k in range(MAX_HORIZON):

                if t+k >= len(buffer["rewards"]):
                    continue

                tvf_value = rediscount_TVF(buffer["values"][t][:k], 0.99)
                true_value = discount_rewards(buffer["rewards"][t:t+k], 0.99)

                trunc_err_k[k].append((tvf_value, true_value, sample, t))

        print(".", end='')

    def print_it(label, x):
        print(f"{label:<20} {np.mean(x):.2f} +- {np.std(x)/(len(x)**0.5):.2f} [{np.min(x)} to {np.max(x)}]")

    print()
    print()
    print_it("Model MSE:", np.asarray(model_err) ** 2)
    print_it("Trunc MSE:", np.asarray(trunc_err) ** 2)
    print_it("Ep Score:", episode_scores)
    print_it("Ep Length:", episode_lengths)


    print()

    # show error over horizon
    # for k in range(0, 500, 10):
    #     if len(trunc_err_k[k]) < 10:
    #         continue
    #     av_err = np.mean([(x[0]-x[1])**2 for x in trunc_err_k[k]])
    #     av_reward = np.mean([x[1] for x in trunc_err_k[k]])
    #     print(f"{k:<6} {av_err:<6.2f} {av_reward:<6.24}")
    #
    # print()

    data = {
        'returns_99': true_returns,
        'trunc_err': trunc_err,
        'model_err': model_err,
        'episode_lengths': episode_lengths,
        'episode_scores': episode_scores,
        'trunc_err_k': trunc_err_k
    }

    with open(filename+".dat", "wb") as f:
        pickle.dump(data, f)

    # just for easy reading... maybe skip pickle?
    with open(filename+".txt", "w") as f:
        json.dump(data, f)



def generate_rollout(model, max_frames = 30*60*15, include_video=False):
    """
    Generates a rollout
    todo: vectorize this so we can generate multiple rollouts at once (will take extra memory though)
    """


    env = atari.make()
    _ = env.reset()
    state, reward, done, info = env.step(0)

    state = env.reset()

    frame_count = 0
    buffer = {
        'values': [],   # values for each horizon of dims [K]
        'errors': [],   # std error estimates for each horizon of dims [K]
        'model_values': [], # models predicted value (float)
        'rewards': [],   # normalized reward (which value predicts)
        'raw_rewards': [], # raw unscaled reward from the atari environment
    }

    if include_video:
        buffer['frames'] = []  # video frames

    horizons = np.repeat(np.arange(MAX_HORIZON)[None, :], repeats=1, axis=0)
    gammas = np.ones_like(horizons) * args.tvf_gamma

    while not done:

        model_out = model.forward(state[np.newaxis], horizons=horizons, gammas=gammas)

        log_probs = model_out["log_policy"][0].detach().cpu().numpy()
        action = utils.sample_action_from_logp(log_probs)

        state, reward, done, info = env.step(action)

        channels = info.get("channels", None)
        rendered_frame = info.get("monitor_obs", state)

        agent_layers = state.copy()

        values = model_out["tvf_value"][0, :].detach().cpu().numpy() * REWARD_SCALE
        errors = model_out["tvf_std"][0, :].detach().cpu().numpy() * REWARD_SCALE * 1.96
        model_value = model_out["ext_value"][0].detach().cpu().numpy() * REWARD_SCALE
        raw_reward = info.get("raw_reward", reward)

        if 'frames' in buffer:
            frame = utils.compose_frame(agent_layers, rendered_frame, channels)
            buffer['frames'].append(frame)

        buffer['values'].append(values)
        buffer['errors'].append(errors)
        buffer['model_values'].append(model_value)
        buffer['rewards'].append(reward)
        buffer['raw_rewards'].append(raw_reward)

        frame_count += 1

        if frame_count >= max_frames:
            break

    for k, v in buffer.items():
        buffer[k] = np.asarray(v)

    return buffer


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

    buffer = generate_rollout(model, max_frames, include_video=True)
    rewards = buffer["rewards"]

    for t in range(len(rewards)):

        frame = buffer["frames"][t]
        values = buffer["values"][t]
        errors = buffer["errors"][t]
        model_value = buffer["model_values"][t]

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

        limit = true_values + [model_value] + (values+errors)

        plt.ylim(0,20+ int(max(limit))//20 * 20)

        plt.grid(True)

        plt.legend()

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
        checkpoint_name = os.path.join(path, f"checkpoint-{epoch:03d}M-params.pt")
        if os.path.exists(checkpoint_name) and not os.path.exists(os.path.join(path, f"checkpoint-{epoch:03d}M.txt")):
            print()
            print(checkpoint_name)
            print()
            load_checkpoint(model, checkpoint_name, device=DEVICE)
            export_movie(model, os.path.join(path, f"checkpoint-{epoch:03d}M"))
            evaluate_model(model, os.path.join(path, f"checkpoint-{epoch:03d}M"), samples=4)

if __name__ == "__main__":

    # set args by hand
    config.parse_args()

    # get model
    args.env_name = utils.get_environment_name(args.environment, args.sticky_actions)
    args.sync_envs = True

    # set resolution
    args.res_x, args.res_y = (84, 84)

    env = atari.make()
    model = make_model(env)

    #run_eval("./Run/TVF_2G/gamma=0.99 [1c6f2ca7]/")
    run_eval("./Run/TVF_3A/gamma=0.99 [7e79a9c9]")



