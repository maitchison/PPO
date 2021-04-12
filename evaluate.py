import ast
import sys

import matplotlib.pyplot as plt

from rl import atari, config, utils
from rl.config import args
from rl import hybridVecEnv

from pathlib import Path

import cv2
import numpy as np
import torch
import os

import json
import pickle
import time
import math

DEVICE = "cuda:0"
REWARD_SCALE = float()
MAX_HORIZON = 500 # this gets changed depending on model's max_horizon
SAMPLES = 64 # 16 is too few, might need 256...
PARALLEL_ENVS = 64 # number of environments to run in parallel

GENERATE_EVAL = False
GENERATE_MOVIES = True

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

    args.env_name = utils.get_environment_name(args.environment, args.sticky_actions)
    args.res_x, args.res_y = (84, 84)

    args.experiment_name = Path(checkpoint_path).parts[-3]
    env = atari.make(monitor_video=True)

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


    import importlib
    model_module_path = f"Run.{args.experiment_name}.rl.models"
    models = importlib.import_module(model_module_path, 'models')

    import inspect
    allowed_args = set(inspect.signature(models.TVFModel.__init__).parameters.keys())

    additional_args = {}

    additional_args['use_rnd'] = args.use_rnd
    additional_args['use_rnn'] = False
    additional_args['tvf_horizon_transform'] = lambda x: x / args.tvf_max_horizon
    additional_args['tvf_hidden_units'] = args.tvf_hidden_units
    additional_args['tvf_activation'] = args.tvf_activation
    additional_args['tvf_h_scale'] = args.tvf_h_scale

    additional_args['head'] = "Nature"
    additional_args['network'] = "Nature"

    return models.TVFModel(
            input_dims=env.observation_space.shape,
            actions=env.action_space.n,
            device=DEVICE,
            dtype=torch.float32,
            **{k:v for k,v in additional_args.items() if k in allowed_args},
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

def evaluate_model(model, filename, samples=16, max_frames = 30*60*15, temperature=1.0):

    # we play the games one at a time so as to not take up too much memory
    # this is required as the need to know the future rewards at the same time as the states.
    # running twice using fixed seeds would work too, but would require simulating twice

    episode_scores = []
    episode_lengths = []

    print(f"Evaluating {filename}:", end='', flush=True)

    remaining_samples = samples

    return_estimates = {}

    while remaining_samples > 0:

        batch_samples = min(PARALLEL_ENVS, remaining_samples) # 1 at a time is slower, but better on the memory for long runs...
        buffers = generate_rollouts(model, max_frames, num_rollouts=batch_samples, temperature=temperature, include_horizons=False)

        for i, buffer in enumerate(buffers):
            # get game score and length
            raw_rewards = buffer["raw_rewards"]
            episode_score = sum(raw_rewards)
            episode_length = len(raw_rewards)
            episode_scores.append(episode_score)
            episode_lengths.append(episode_length)

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

    with open(filename, "wb") as f:
        pickle.dump(data, f)

def generate_rollout(model, max_frames = 30*60*15, include_video=False, temperature=1.0):
    return generate_rollouts(model, max_frames, include_video, num_rollouts=1, temperature=temperature)[0]

def generate_fake_rollout(num_frames = 30*60):
    """
    Generate a fake rollout for testing
    """
    return {
        'values': np.zeros([num_frames, args.tvf_n_horizons], dtype=np.float32),
        'model_values': np.zeros([num_frames], dtype=np.float32),
        'rewards': np.zeros([num_frames], dtype=np.float32),
        'raw_rewards': np.zeros([num_frames], dtype=np.float32),
        'frames': np.zeros([num_frames, 210, 334, 3], dtype=np.uint8)
    }


def generate_rollouts(
        model,
        max_frames = 30*60*15,
        include_video=False,
        num_rollouts=1,
        temperature=1.0,
        include_horizons=True,
    ):
    """
    Generates rollouts
    """

    env = hybridVecEnv.HybridAsyncVectorEnv([lambda : atari.make(monitor_video=include_video) for _ in range(num_rollouts)])

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
            'game_was_reset': False,
        })

        if include_video:
            buffers[-1]['frames'] = []  # video frames

    if include_horizons and args.use_tvf:
        horizons = np.repeat(np.arange(MAX_HORIZON)[None, :], repeats=num_rollouts, axis=0)
    else:
        horizons = None

    while any(is_running) and frame_count < max_frames:

        model_out = model.forward(
            states,
            horizons=horizons,
            **({'policy_temperature':temperature} if temperature is not None else {})
        )

        log_probs = model_out["log_policy"].detach().cpu().numpy()

        if np.isnan(log_probs).any():
            raise Exception(f"NaN found in policy ({args.experiment_name}, {args.run_name}).")

        action = np.asarray([utils.sample_action_from_logp(prob) for prob in log_probs], dtype=np.int32)

        states, rewards, dones, infos = env.step(action)

        for i in range(len(states)):
            if dones[i]:
                is_running[i] = False
            if not is_running[i]:
                continue

            model_value = model_out["ext_value"][i].detach().cpu().numpy()
            raw_reward = infos[i].get("raw_reward", rewards[i])

            if 'frames' in buffers[i]:
                agent_layers = states[i]
                channels = infos[i].get("channels", None)
                rendered_frame = infos[i].get("monitor_obs", states[i])
                frame = utils.compose_frame(agent_layers, rendered_frame, channels)
                buffers[i]['frames'].append(frame)

            game_reset = "game_freeze" in infos[i]

            if horizons is not None:
                values = model_out["tvf_value"][i, :].detach().cpu().numpy()
                buffers[i]['values'].append(values)

            buffers[i]['model_values'].append(model_value)
            buffers[i]['rewards'].append(rewards[i])
            buffers[i]['raw_rewards'].append(raw_reward)
            buffers[i]['game_was_reset'] = buffers[i]['game_was_reset'] or game_reset

        frame_count += 1

    for buffer in buffers:
        for k, v in buffer.items():
            buffer[k] = np.asarray(v)

    return buffers


def export_movie(model, filename_base, max_frames = 30*60*15, include_score_in_filename=False, temperature=1.0):
    """
    Modified version of export movie that supports display of truncated value functions
    In order to show the true discounted returns we write all observations to a buffer, which may take
    a lot of memory.
    """

    scale = 4

    env = atari.make(monitor_video=True)
    _ = env.reset()
    state, reward, done, info = env.step(0)
    rendered_frame = info.get("monitor_obs", state)

    # work out our height
    first_frame = utils.compose_frame(state, rendered_frame)
    height, width, channels = first_frame.shape
    width = (width * scale) // 4 * 4  # make sure these are multiples of 4
    height = (height * scale) // 4 * 4

    print(f"Video {filename_base} ", end='', flush=True)

    buffer = generate_rollout(model, max_frames, include_video=True, temperature=temperature)
    rewards = buffer["rewards"]

    # create video recorder, note that this ends up being 2x speed when frameskip=4 is used.
    final_score = sum(rewards)
    postfix = f" [{int(final_score):,}].mp4" if include_score_in_filename else ".mp4"
    video_filename = filename_base + postfix
    video_out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=True)

    print(f"score:{int(sum(rewards)):,} ", end='', flush=True)

    prev_limits = (0.0, 0.0)

    # work out discounting values to make plotting a little faster
    # discount_values = np.zeros([MAX_HORIZON], dtype=np.float32)
    # for k in range(len(discount_values)):
    #     discount_values[k] = args.tvf_gamma ** k
    # cumulative_sum = np.zeros([len(rewards)], dtype=np.float32)
    # reward_sum = 0
    # for k in range(len(cumulative_sum)):
    #     reward_sum += reward[k]
    #     cumulative_sum[k] = reward_sum

    max_fps = float('-inf')
    min_fps = float('inf')

    marker_time = time.time()
    very_start_time = time.time()
    fps_list = []

    for t in range(len(rewards)):

        # todo: make this extremely fast for large horizon...
        start_time = time.time()

        #1: get frames

        frame = buffer["frames"][t]
        model_value = buffer["model_values"][t] * REWARD_SCALE

        if frame.shape[0] != width or frame.shape[1] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)

        assert frame.shape[1] == width and frame.shape[0] == height, "Frame should be {} but is {}".format(
            (width, height, 3), frame.shape)

        # calculate actual truncated values using real future rewards
        true_values = np.zeros(MAX_HORIZON, dtype=np.float32)


        for k in range(len(true_values)):
            this_reward = (args.tvf_gamma ** k) * rewards[t+k] if (t+k) < len(rewards) else 0
            prev_rewards = true_values[k-1] if k > 0 else 0
            true_values[k] = this_reward + prev_rewards

        fig = plt.figure(figsize=(7, 4), dpi=100)

        # plot true value
        xs = list(range(len(true_values)))
        ys = true_values
        plt.plot(xs, ys, label="True", c="red")
        min_height = 0
        max_height = max(true_values)

        # plot predicted values...
        if args.use_tvf:
            values = buffer["values"][t] * REWARD_SCALE  # model learned scaled rewards
            ys = values
            xs = list(range(len(ys)))
            plt.plot(xs, ys, label="Predicted", c="blue", alpha=0.75)
            max_height = max(max_height, max(ys))
            min_height = min(ys)

        # plot model value prediction
        if args.vf_coef > 0:
            plt.hlines(model_value, 0, len(xs), label="Model", colors=["black"])
            max_height = max(max_height, model_value)

        plt.xlabel("k")
        plt.ylabel("Score")

        max_limit = math.ceil(max_height+5)
        min_limit = math.floor(min_height-1)
        max_limit = max(prev_limits[0], max_limit)
        min_limit = min(prev_limits[1], min_limit)

        plt.ylim(min_limit, max_limit)
        prev_limits = (min_limit, max_limit)

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

        total_time = time.time() - start_time

        fps = 1 / total_time
        fps_list.append(fps)
        max_fps = max(fps, max_fps)
        min_fps = min(fps, min_fps)

        # every 60 second print time
        if (time.time() - marker_time) > 60:
            print(f"<{t}/{len(rewards)} - {np.mean(fps_list[-10:]):.1f} FPS> ", end='')
            marker_time = time.time()

    video_out.release()

    end_time = time.time()
    print(f"completed at {len(rewards) / (end_time - very_start_time):.1f} FPS")
    return video_filename


def run_eval(path, temperature=None, max_epoch=200):

    files_in_dir = [os.path.join(path, x) for x in os.listdir(path)]

    for epoch in range(max_epoch+1):

        temp_postfix = f"_t={temperature}" if temperature is not None else ""

        checkpoint_eval_file = os.path.join(path, f"checkpoint-{epoch:03d}M-eval{temp_postfix}.dat")
        checkpoint_name = os.path.join(path, f"checkpoint-{epoch:03d}M-params.pt")

        checkpoint_movie_base = f"checkpoint-{epoch:03d}M-eval{temp_postfix}"

        if os.path.exists(checkpoint_name):

            if GENERATE_MOVIES:

                matching_files = [x for x in files_in_dir if checkpoint_movie_base+'.mp4' in x]

                if len(matching_files) >= 2:
                    print(f"Multiple matches for file {path}/{checkpoint_movie_base+'.mp4'}.")
                    continue

                if len(matching_files) == 1:
                    last_modifed = os.path.getmtime(matching_files[0])
                    file_size = os.path.getsize(matching_files[0])
                    minutes_since_modified = (time.time()-last_modifed)/60
                    needs_redo = file_size < 1024 and minutes_since_modified > 30
                    if needs_redo:
                        print(f" - found stale video {matching_files[0]}, regenerating.")
                        os.remove(matching_files[0])
                else:
                    needs_redo = False

                if len(matching_files) == 0 or needs_redo:
                    # create an empty file to mark that we are working on this...
                    model = load_checkpoint(checkpoint_name, device=DEVICE)
                    video_filename = export_movie(
                        model,
                        os.path.join(path, "_"+checkpoint_movie_base),
                        include_score_in_filename=True,
                        temperature = temperature,
                    )
                    try:
                        path, file = os.path.split(video_filename)
                        # remove _ from start of file
                        os.rename(video_filename, path+"/"+file[1:])
                    except:
                        print("failed to rename video file")

            if GENERATE_EVAL and not os.path.exists(checkpoint_eval_file):
                model = load_checkpoint(checkpoint_name, device=DEVICE)
                evaluate_model(
                    model,
                    checkpoint_eval_file,
                    samples=SAMPLES,
                    temperature=temperature
                )

def monitor(path, run_filter=None):

    folders = [x[0] for x in os.walk(path)]
    for folder in folders:
        if run_filter is not None and not run_filter(folder):
            continue
        if os.path.split(folder)[-1] == "rl":
            continue
        print(folder)
        try:
            for max_epoch in range(0, 201, 5):
                for temperature in temperatures:
                    run_eval(folder, temperature=temperature, max_epoch=max_epoch)
        except Exception as e:
            print("Error:"+str(e))

if __name__ == "__main__":

    # usage
    # python evaluate.py TVF_10_Eval bundle_0 [-0.01 , -0.1, -0.5, -1]

    config.parse_args(no_env=True, args_override=[])
    assert len(sys.argv) == 4

    filter_1 = sys.argv[1]
    filter_2 = sys.argv[2]

    if sys.argv[3] == "video":
        GENERATE_EVAL = False
        GENERATE_MOVIES = True
        temperatures = [None]
    else:
        temperatures = ast.literal_eval(sys.argv[3])
        GENERATE_EVAL = True
        GENERATE_MOVIES = False

        if type(temperatures) in [float, int]:
            temperatures = [temperatures]

    folders = [name for name in os.listdir("./Run/") if os.path.isdir(os.path.join('./Run',name))]
    for folder in folders:
        if filter_1 in folder:
            monitor(
                os.path.join('./Run', folder),
                run_filter=lambda x: filter_2 in x,
            )