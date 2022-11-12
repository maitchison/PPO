import gzip
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt

DEBUG_HORIZONS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
CACHE = {}

cm = plt.cm.get_cmap('tab10')
cm20 = plt.cm.get_cmap('tab20')
plasma = plt.cm.get_cmap('plasma')


def setup_default_plot():
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.grid(True, alpha=0.25)


class ReturnSample:

    def __init__(self, returns: np.ndarray, t: int, rollout_index: int, state_hash: int):
        self.returns = np.asarray(returns)
        self.t = t
        self.rollout_index = rollout_index
        self.state_hash = state_hash

    def __repr__(self):
        return f"rollout_{self.rollout_index} t={self.t}: {self.returns.mean():.1f} +- {self.returns.std():.1f} [{hex(self.state_hash % 2 ** 16)}]"


class ReturnDataset:

    def __init__(self, source_file: str):

        with gzip.open(source_file) as f:
            data = pickle.load(f)

        self.n_rollouts = len(data)

        self.returns = {h: [] for h in DEBUG_HORIZONS}

        for i, rollout in enumerate(data):
            reward_scale = rollout['reward_scale']
            for t, return_sample in enumerate(rollout['mv_return_sample']):
                if return_sample is None:
                    # most return samples will be None
                    continue
                state_hash = rollout['prev_state_hash'][t]
                for h_index, h in enumerate(DEBUG_HORIZONS):
                    true_return_distribution = [returns[min(h, len(returns) - 1)] for returns in return_sample]
                    rs = ReturnSample(
                        np.asarray(true_return_distribution, dtype=np.float32),
                        t=t,
                        rollout_index=i,
                        state_hash=int(state_hash)
                    )
                    self.returns[h].append(rs)


def get_data(source_file: str):
    key = ('get_data', source_file)
    if key in CACHE:
        return CACHE[key]
    if not os.path.exists(source_file):
        print(f"Warning, no data for {source_file}")
        CACHE[key] = None
        return None
    with gzip.open(source_file) as f:
        data = pickle.load(f)
    CACHE[key] = data
    return data


def get_metric(metric: str, experiment_path: str, epoch: int, h: int, bootstrap=False):

    pred, targ = get_return_estimates(experiment_path, epoch, h)

    if pred is None:
        return None

    if bootstrap:
        sample = np.random.choice(len(pred), size=len(pred), replace=True)
        pred = pred[sample]
        targ = targ[sample]


    if metric == "ev":
        error_var = np.var(pred - targ)
        true_var = np.var(targ)
        return np.clip(1 - error_var / true_var, 0, 1)
    elif metric == "mse":
        return (np.square(pred - targ)).mean()
    elif metric == "rms":
        return (np.square(pred - targ)).mean() ** 0.5
    elif metric == "nl_rms":
        return -np.log10((np.square(pred - targ)).mean() ** 0.5)
    elif metric == "nl_mse":
        return -np.log10((np.square(pred - targ)).mean())
    else:
        raise ValueError(f"Invalid metric {metric}")


def get_int_param(sentance: str, param: str, default=0):
    for word in sentance.split():
        if word.startswith(f"{param}="):
            a, b = word.split("=")
            return int(b)
    return default


run_colors = {}


def get_n_step_color(x, default):
    n_steps = get_int_param(x, 'n_step')
    if n_steps == 0:
        if x in run_colors:
            return run_colors[x]
        else:
            col = cm(len(run_colors))
            run_colors[x] = col
            return col
    else:
        i = int(np.log2(n_steps)) + 1

    return plasma(i / 10)


def plot_by_epoch(experiment_path: str, h: int, hold=False, label='estimator', metric='ev', color=None, style="-"):
    """ Evaluate the performance of a model by comparing it's return estimation to that of a reference. """

    xs = range(1, 11)
    ys = []

    for epoch in xs:
        y = get_metric(metric, experiment_path, epoch, h)
        ys.append(y)

    if not hold:
        plt.figure(figsize=(4, 3))
    setup_default_plot()
    plt.plot(xs, ys, label=label, color=color, ls=style)
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.legend()
    if not hold:
        plt.show()


def plot_by_horizon(experiment_path: str, epoch: int, hold=False, label='estimator', metric='ev', color=None,
                    style="-"):
    """ Evaluate the performance of a model by comparing it's return estimation to that of a reference. """

    xs = DEBUG_HORIZONS
    ys = []

    for h in xs:
        y = get_metric(metric, experiment_path, epoch, h)
        ys.append(y)

    if not hold:
        plt.figure(figsize=(4, 3))
    setup_default_plot()
    plt.plot(xs, ys, label=label, color=color, ls=style)
    plt.xlabel("horizon")
    plt.ylabel(metric)
    if metric in ["mse", "rms"]:
        plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    if not hold:
        plt.show()


def plot_estimates(experiment_path: str, epoch: int, h: int):
    """ Evaluate the performance of a model by comparing it's return estimation to that of a reference. """

    pred, targ = get_return_estimates(experiment_path, epoch, h)

    plt.figure(figsize=(4, 3))
    xs = np.asarray(pred)
    ys = np.asarray(targ)
    plt.scatter(xs, ys)
    plt.grid(True, alpha=0.25)
    plt.xlabel("prediction")
    plt.ylabel("target")
    plt.show()

    unexplained_variance = np.var(xs - ys)
    target_variance = np.var(ys)
    ev = 1 - (unexplained_variance / (target_variance + 1e-6))
    print(f"H = {h}")
    print(f"ev:    {ev:.4f}")
    print(f"rmse:  {np.mean(np.square(xs - ys)) ** 0.5:<5.1f}")


def get_return_estimates(experiment_path: str, epoch: int, h: int):
    """
    Returns predicted, true return estimates.
    Return estimates are in normalized space.
    """

    key = ('gre', experiment_path, epoch, h)

    if key in CACHE:
        return CACHE[key]

    if REFERENCE is None:
        raise Exception("Reference not set, please call set_reference.")

    data = get_data(os.path.join(experiment_path, f"checkpoint-{epoch:03d}M-eval_1.eval.gz"))

    if data is None:
        return None, None

    assert len(data) == REFERENCE.n_rollouts, "Source must match reference in rollouts."

    h_index = DEBUG_HORIZONS.index(h)

    horizon_predictions = []
    horizon_targets = []

    reward_scale = data[0]['reward_scale']

    for rs in REFERENCE.returns[h]:
        # check we match on state hash
        eval_state_hash = int(data[rs.rollout_index]['prev_state_hash'][rs.t])
        if eval_state_hash != rs.state_hash:
            print(f"!!! State hash missmatch on i={rs.rollout_index} t={rs.t}")
            return None

        # find this location in our data
        predicted_value = data[rs.rollout_index]['values'][rs.t][h_index]
        true_value = rs.returns.mean()

        horizon_predictions.append(predicted_value)
        horizon_targets.append(true_value * reward_scale)

    CACHE[key] = (np.asarray(horizon_predictions), np.asarray(horizon_targets))

    return CACHE[key]


def set_reference(path):
    global REFERENCE
    REFERENCE = ReturnDataset(path)

REFERENCE: ReturnDataset

# def process_env(env):
#     with gzip.open("./Run/TVF_EV/reference_10.eval.gz") as f:
#         reference = pickle.load(f)
#     print(f"Found {len(reference)} samples.")
#     print(list(reference[0].keys()))
#     print(len(reference[0]['mv_return_sample']))
#     # make sure everything lines up...
#     reference = ReturnDataset(f"./Run/TVF_EV3/{env}_10.eval.gz")
#     print(f"Reference has {reference.n_rollouts} rollouts and return samples for {len(reference.returns[1])} locations.")
