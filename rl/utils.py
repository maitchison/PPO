import numpy as np
import torch
import os
import math
import cv2
import pickle
import json
import time

import torchvision

from . import hybridVecEnv, atari, config
from .config import args

NATS_TO_BITS = 1.0/math.log(2)

class Color:
    """
        Colors class for use with terminal.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------

def check_for_exteme_or_nan(X, name="array", extreme_limit=10000):
    """ Makes sure elements in array are non NaN and are within reasonable limits. """

    mean = np.mean(X)
    std = np.std(X)
    max = np.max(X)
    min = np.min(X)

    if np.any(np.isnan(X)):
        print("!!! Found nan in "+name+" with stats:", mean, std, max, min)
    if extreme_limit and max > extreme_limit:
        print("!!! Max too high in "+name+" with stats:", mean, std, max, min)
    if extreme_limit and min < -extreme_limit:
        print("!!! Max too high in "+name+" with stats:", mean, std, max, min)

def mse(a,b):
    """ returns the mean square error between a and b. """
    return (np.square(a - b, dtype=np.float32)).mean(dtype=np.float32)


def prod(X):
    """ Returns the product of X, where X is a vector."""
    y = 1
    for x in X:
        y *= x
    return y


def trace(s):
    """ Prints output. """
    print(s)


def entropy(p):
    """ Returns the entropy of a distribution (in nats). """
    if isinstance(p, torch.Tensor):
        return -(p * torch.log(p)).sum()
    else:
        p = np.asarray(p).astype(np.float32)
        return -(p * np.log(p)).sum()

def log_entropy(logp):
    """ Returns the entropy of a distribution where input are log probabilties and output is in NATS
        Note: this used to be in bits, but to standardize with openAI baselines we have switched to NATS.
    """
    if isinstance(logp, torch.Tensor):
        return -(logp.exp() * logp).sum()
    else:
        logp = np.asarray(logp).astype(np.float32)
        return -(np.exp(logp) * logp).sum()


def kl(p, q):
    """
        Returns the KL(P||Q) w
        Note: result is returned in nats (not bits)
    """
    if isinstance(p, torch.Tensor):
        assert p.dtype == q.dtype
        assert p.dtype in [torch.float32, torch.float64]
        return (torch.log(p / q) * p).sum()
    else:
        p = np.asarray(p).astype(np.float32)
        q = np.asarray(q).astype(np.float32)

        return (np.log(p/q) * p).sum()



def log_kl(logp, logq):
    """
        Returns the KL(P||Q) where p and q are log probabilities
        Note: logs are assumed to be natural log, and result is returned in nats (not bits)
    """
    if isinstance(logp, torch.Tensor):
        assert logp.dtype == logq.dtype
        assert logp.dtype in [torch.float32, torch.float64]
        return ((logp - logq) * logp.exp()).sum()
    else:
        logp = np.asarray(logp).astype(np.float32)
        logq = np.asarray(logq).astype(np.float32)
        return ((logp - logq) * np.exp(logp)).sum()


def sample_action_from_logp(logp):
    """
        Returns integer [0..len(probs)-1] based on log probabilities.
        Log probabilities will be normalized.
    """
    # taken from https://github.com/openai/baselines/blob/master/baselines/common/distributions.py
    # this is trick to sample directly from log probabilties without exponentiation.
    u = np.random.uniform(size=np.shape(logp))
    return np.argmax(logp - np.log(-np.log(u)), axis=-1)


def smooth(X, alpha=0.98):
    """
    Smooths input using a Exponential Moving Average.
    """
    y = X[0]
    results = []
    for x in X:
        y = (1 - alpha) * x + (alpha) * y
        results.append(y)
    return results


def safe_mean(X, rounding=None):
    """
    Returns the mean of X, or 0 if X has no elements.
    :param X: input
    :param rounding: if given round to this many decimal places.
    """
    result = float(np.mean(X)) if len(X) > 0 else None
    if rounding is not None:
        return safe_round(result, rounding)
    else:
        return result


def safe_round(x, digits):
    """
    Rounds x to given number of decimal places.
    If input is none will return none.
    """
    return round(x, digits) if x is not None else x


def inspect(x):
    """
    Prints the type and shape of x.
    :param x: input, an integer, float, ndarray etc.
    """
    if isinstance(x, int):
        print("Python interger")
    elif isinstance(x, float):
        print("Python float")
    elif isinstance(x, np.ndarray):
        print("Numpy", x.shape, x.dtype)
    elif isinstance(x, torch.Tensor):
        print("{:<10}{:<25}{:<18}{:<14}".format("torch", str(x.shape), str(x.dtype), str(x.device)))
    else:
        print(type(x))


def nice_display(X, title):
    """
    Prints first 5 elements of array with values rounded to 2dp.
    """
    print("{:<20}{}".format(title, [round(float(x),2) for x in X[:5]]))


def zero_format_number(x):
    return "{:03.0f}M".format(round(x/1e6))

def sig_fig(x, sf=6):
    """ returns x to 6 significant figures if x is a float and small, otherwise returns the input unchanged."""
    if type(x) is float or type(x) is np.float:
        digits = int(math.log10(abs(x)+0.00000000001))
        rounding = sf - digits
        if rounding < 0:
            rounding = 0
        return round(x, rounding)
    else:
        return x


def default(x, default):
    """ Returns x if x is not none, otherwise default. """
    return x if x is not None else default


def copy_source_files(source, destination, force=False):
    """ Copies all source files from source path to destination. Returns path to destination training script. """
    try:

        destination_train_script = os.path.join(destination, "train.py")

        if not force and os.path.exists(destination_train_script):
            return destination_train_script
        # we need to copy across train.py and then all the files under rl...
        os.makedirs(os.path.join(destination, "rl"), exist_ok=True)
        os.system("cp '{}train.py' '{}/train.py'".format(source, destination))
        os.system("cp '{}/'*.py '{}'".format(os.path.join(source, "rl"), os.path.join(destination, "rl")))
        return destination_train_script
    except Exception as e:
        print("Failed to copy training file to log folder.", e)
        return None

def comma(x):
    return "{:,.1f}".format(x) if x < 100 else "{:,.0f}".format(x)

def dump_data(X, name):
    """dumps a np / torch tensor to a file."""

    if type(X) is torch.Tensor:
        X = X.detach().cpu().numpy()

    with open(name+".txt", "wt") as f:
        f.write("min," + str(np.min(X))+"\n")
        f.write("max," + str(np.max(X))+"\n")
        f.write("mean," + str(np.mean(X))+"\n")
        f.write("std," + str(np.std(X))+"\n")
    np.savetxt(name+".csv", X, delimiter=",")

# -------------------------------------------------------------
# RMS
# -------------------------------------------------------------

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """
    Calculate and return running mean and variance.
    """

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def explained_variance(ypred, y):
    """
    # from https://github.com/openai/random-network-distillation/blob/436c47b7d42ffe904373c7b8ab6b2d0cff9c80d8/utils.py
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]
    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero
    """

    assert y.ndim == 1 and ypred.ndim == 1

    vary = np.var(y)

    return -1 if vary == 0 else np.clip(1 - np.var(y-ypred)/vary, -1, 1)

class RunningMeanStd(object):
    """
    Class to handle running mean and standard deviation book-keeping.
    From https://github.com/openai/baselines/blob/1b092434fc51efcb25d6650e287f07634ada1e08/baselines/common/running_mean_std.py
    See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """

    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):

        if type(x) in [float, int]:
            batch_mean = x
            batch_var = 0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


    def save_state(self):
        """
        Saves running statistics.
        """
        return tuple([self.mean, self.var, self.count])

    def restore_state(self, state):
        """
        Restores running statistics.
        """
        self.mean, self.var, self.count = state


# -------------------------------------------------------------
# Rollouts
# -------------------------------------------------------------

def generate_rollouts(num_rollouts, model, env_name, resolution=0.5, max_length=2000, deterministic=False):
    """ Generates roll out with given model and environment name.
        returns observations.
            num_rollouts: Number of rollouts to generate
            model: The model to use
            env_name: Name of the environment
            resolution: Resolution of returned frames
            max_length: Maximum number of environment interactions before rollouts are automatically terminated.
            deterministic: Force a deterministic environment (but not policy)
        :returns
            observations as a list np arrays of dims [c,w,h] in uint8 format.
    """

    # todo get non_determinism working again...
    #env_fns = [lambda : atari.make(env_name, non_determinism="none" if deterministic else "noop") for _ in range(num_rollouts)]
    env = hybridVecEnv.HybridAsyncVectorEnv([atari.make for _ in range(num_rollouts)])

    _ = env.reset()
    state, reward, done, info = env.step([0]*num_rollouts)
    rendered_frame = info[0].get("monitor_obs", state)
    w,h,c = rendered_frame.shape
    state = env.reset()

    frames = [[] for _ in range(num_rollouts)]

    is_running = [True] * num_rollouts

    counter = 0

    while any(is_running) and counter < max_length:

        logprobs = model.policy(state).detach().cpu().numpy()
        actions = np.asarray([sample_action_from_logp(prob) for prob in logprobs], dtype=np.int32)

        state, reward, done, info = env.step(actions)

        # append only frames for runs that are still running.
        for i in range(num_rollouts):
            if done[i]:
                is_running[i] = False
            if is_running[i]:
                rendered_frame = info[i].get("monitor_obs", state)
                rendered_frame = rendered_frame.mean(axis=2, dtype=np.float32).astype(np.uint8)  # get a black and white frame.
                if resolution != 1.0:
                    rendered_frame = cv2.resize(rendered_frame, (int(h * resolution), int(w * resolution)),
                                                interpolation=cv2.INTER_AREA)
                frames[i].append(rendered_frame)

        counter += 1

    env.close()

    return [np.asarray(frame_sequence) for frame_sequence in frames]


def evaluate_diversity(step, model, env_name, num_rollouts=8, save_rollouts=True, resolution=0.5):
    """ Generates multiple rollouts of agent, and evaluates the diversity of the rollouts.

    """

    results = []

    # we generate rollouts with the additional determanism turned on. This just removes the no-op starts
    # and gives us a better idea of how similar the runs are.
    rollouts = generate_rollouts(num_rollouts, model, env_name, resolution=resolution, deterministic=True)

    # get all distances between rollouts.
    for i in range(num_rollouts):
        for j in range(i+1, num_rollouts):
            a = rollouts[i][::5] # do comparision at around 3 fps.
            b = rollouts[j][::5]
            difference = dtw(a, b)

            results.append(difference)

    # save the rollouts for later.
    if save_rollouts:
        rollouts_name = get_checkpoint_path(step,"rollouts.dat")
        with open(rollouts_name, 'wb') as f:
            package = {"step":step, "rollouts": rollouts, "distances": results}
            pickle.dump(package, f)

    return results


# -------------------------------------------------------------
# Movies
# -------------------------------------------------------------

def expand_gray_frame(x, tint=(1,1,1)):
    dtype = x.dtype
    result = []
    result.append(x * tint[2])
    result.append(x * tint[1])
    result.append(x * tint[0])
    return np.asarray(result, dtype=dtype)

def draw_pixel(frame, dx, dy, c, sx=1, sy=1):
    frame[dy:dy+sy, dx:dx+sx,:] = c[::-1]

def draw_image(frame, im, dx, dy, scale=4):
    """ paints image (2d nd array) onto a frame. """
    height, width = im.shape
    for x in range(width):
        for y in range(height):
            c = (im[y,x]*255,im[y,x]*255,im[y,x]*255)
            draw_pixel(frame, dx+x*scale, dy+y*scale, c, sx=scale, sy=scale)

def compose_frame(state_frame, rendered_frame, channels=None):
    """ Puts together a composite frame containing rendered frame and state. """

    # note: untested on non-stacked states.

    # assume state is C, W, H
    # assume rendered frame is  is W, H, C
    assert state_frame.shape[0] < max(state_frame.shape), "Channels should be first on state {}".format(state_frame.shape)
    assert rendered_frame.shape[2] < max(state_frame.shape), "Channels should be last on rendered {}".format(
        rendered_frame.shape)

    # ---------------------------------------
    # preprocess frames

    # rendered frame is BGR but should be RGB
    rendered_frame = rendered_frame[...,::-1] # get colors around the right way...

    assert rendered_frame.dtype == np.uint8
    assert state_frame.dtype == np.uint8
    assert len(state_frame.shape) == 3
    assert len(rendered_frame.shape) == 3
    assert rendered_frame.shape[2] == 3, "Invalid rendered shape " + str(rendered_frame.shape)

    s_c, s_h, s_w = state_frame.shape
    r_h, r_w, r_c = rendered_frame.shape

    if channels is None:
        # default channels (not very smart, but hopefuly channels is given so we don't need to guess)
        channels = ["Gray-" + str(i) for i in range(s_c)]

    # combine color frames etc
    frames = []
    for i in range(s_c):
        if channels[i].startswith("Gray"):
            new_frame = expand_gray_frame(state_frame[i])
        elif channels[i].startswith("Mask"):
            new_frame = expand_gray_frame(state_frame[i], (0.25, 0.25, 1.0))
        elif channels[i].startswith("ColorR"):
            new_frame = state_frame[i:i+3, :, :]
        else:
            continue

        frames.append(torch.tensor(new_frame))

    cols = 2
    if len(frames) > 4:
        cols = 3
    if len(frames) > 9:
        cols = 4

    state_grid = torchvision.utils.make_grid(frames, nrow=cols, padding=2)
    state_grid = state_grid.numpy()
    state_grid = np.swapaxes(state_grid, 0, 2)
    state_grid = np.swapaxes(state_grid, 0, 1)

    grid_height, grid_width, _ = state_grid.shape

    full_width = r_w + grid_width
    full_height = max(r_h, grid_height)

    frame = np.zeros([full_height, full_width, 3], dtype=np.uint8)
    frame[:, :, :] += 30  # dark gray background.

    # place the rendered frame
    ofs_y = (full_height - r_h) // 2
    frame[ofs_y:ofs_y + r_h, 0:r_w] = rendered_frame
    ofs_y = (full_height - grid_height) // 2
    frame[ofs_y:ofs_y + grid_height, r_w: r_w + grid_width] = state_grid

    """

    is_stacked = s_c % 4 == 0
    is_color = s_c % 3 == 0

    
    full_width = r_w + s_w * (2 if is_stacked else 1)
    full_height = max(r_h, s_h * (2 if is_stacked else 1))

    frame = np.zeros([full_height, full_width, 3], dtype=np.uint8)
    frame[:, :, :] += 30 # dark gray background.

    # place the rendered frame
    ofs_y = (full_height - r_h) // 2
    frame[ofs_y:ofs_y+r_h, 0:r_w] = rendered_frame

    # place state frames
    y_pad = (full_height - (s_h * 2)) // 2
    if is_stacked:
        i = 0
        for x in range(2):
            for y in range(2):
                dx = x * s_w + r_w
                dy = y * s_h + y_pad
                factor = 1 if x ==0  and y == 0 else 2 # darken all but first state for clarity
                if is_color:
                    frame[dy:dy+s_h, dx:dx+s_w, :] = state_frame[:, :, i*3:(i+1)*3] // factor
                else:
                    for c in range(3):
                        frame[dy:dy+s_h, dx:dx+s_w, c] = state_frame[:, :, i] // factor
                i += 1
    else:
        dx = r_w
        dy = y_pad
        if is_color:
            frame[dy:dy+s_h, dx:dx+s_w, :] = state_frame[:, :, :]
        else:
            for c in range(3):
                frame[dy:dy+s_h, dx:dx+s_w, c] = state_frame[:, :, :]

    """
    return frame


# -------------------------------------------------------------
# Checkpointing
# -------------------------------------------------------------

def get_checkpoint_path(step, postfix):
    """ Returns the full path to a checkpoint file with given step count and postfix. """
    return os.path.join(args.log_folder, "checkpoint-{}-{}".format(zero_format_number(step), postfix))

def get_trajectory_path(step, postfix):
    """ Returns the full path to a checkpoint file with given step count and postfix. """
    return os.path.join(args.log_folder, "trajectories", "rollout-{}-{}".format(zero_format_number(step), postfix))

def generate_hash_image(key, hash_size, obs_size):

    rng =  np.random.RandomState(key % (2 ** 32))  # rng requires 32bit number...

    # seed the random generator and create an random 42x42 observation.
    # note: I'm not sure how many bits the numpy random generate will use, it's posiable it's a lot less than
    # 1024. One option is then to break up the observation into parts. Another would be to just assume that the
    # number of reachable states is much much less than this, and that the chance of a collision (alaising) is
    # very low.
    
    h,w,c = obs_size
    
    obs = rng.randint(0, 1 + 1, hash_size, dtype=np.uint8) * 255
    obs = cv2.resize(obs, (h, w), interpolation=cv2.INTER_NEAREST)
    obs = obs[:, :, np.newaxis]

    obs = np.concatenate([obs] * c, axis=2)
    return obs

# -------------------------------------------------------------
# Locking
# -------------------------------------------------------------


def lock_job():

    # make sure there isn't another lock
    previous_lock = get_lock_info()
    if previous_lock is not None and previous_lock["key"] != config.LOCK_KEY:
        raise Exception("Could not get lock for job, another worker has a lock open.")

    lock = {
        'host': str(args.hostname),
        'time': str(time.time()),
        'status': "started",
        'key': str(config.LOCK_KEY)
    }

    lock_path = os.path.join(args.log_folder, "lock.txt")
    with open(lock_path,"w") as f:
        json.dump(lock, f)


def release_lock():

    assert have_lock(), "Worker does not have lock."

    lock_path = os.path.join(args.log_folder, "lock.txt")
    os.remove(lock_path)


def get_lock_info():
    """ Gets lock information for this job. """
    lock_path = os.path.join(args.log_folder, "lock.txt")
    if os.path.exists(lock_path):
        return json.load(open(lock_path, "r"))
    else:
        return None

def have_lock():
    """ Returns if we currently have the lock."""
    lock = get_lock_info()
    return lock is not None and lock["key"] == config.LOCK_KEY


# -------------------------------------------------------------
# Algorithms
# -------------------------------------------------------------


def dtw(obs1, obs2):
    """
    Returns the distances between two observation sequences using dynamic time warping.
    obs1, obs2
        np array [N, C, W, H], where N is number of frames (they don't need to mathc), and C is channels which
                               should be 1.
    ref: https://en.wikipedia.org/wiki/Dynamic_time_warping
    """

    n = obs1.shape[0]
    m = obs2.shape[0]

    DTW = np.zeros((n+1,m+1), dtype=np.float32) + float("inf")
    DTW[0,0] = 0

    obs1 = np.float32(obs1)
    obs2 = np.float32(obs2)

    for i in range(1,n+1):
        for j in range(1,m+1):
            cost = mse(obs1[i-1], obs2[j-1])
            DTW[i,j] = cost + min(
                DTW[i - 1, j],
                DTW[i, j - 1],
                DTW[i - 1, j - 1]
            )

    return DTW[n, m]

# -------------------------------------------------------------
# CUDA
# -------------------------------------------------------------

def get_auto_device():
    """
    Returns the best device to use for training,
    Will be the GPU with most free memory if CUDA is available, otherwise CPU.
    """
    if not torch.cuda.is_available():
        return "cpu"

    if torch.cuda.device_count() == 1:
        return "cuda"
    else:
        # use the device with the most free memory.
        try:
            os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
            memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
            return "cuda:"+str(np.argmax(memory_available))
        except:
            print("Warning: Failed to auto detect best GPU.")
            return "cuda"

