import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import json
import numpy as np
import math

from IPython.display import display, Math

import pickle
from os import listdir
from os.path import isfile, join

cmap = plt.cm.get_cmap('tab10')

cache = {}


def safe_float(x):
    try:
        return float(x)
    except:
        return None


def read_log(file_path):
    if not os.path.exists(file_path + "/params.txt"):
        return None

    params = read_params(file_path + "/params.txt")
    if "n_mini_batches" in params:
        params["mini_batch_size"] = int(params["agents"] * params["n_steps"] / params["n_mini_batches"])

    params["policy_updates"] = (params["agents"] * params["n_steps"]) / params["policy_mini_batch_size"] * params[
        "policy_epochs"]
    params["value_updates"] = (params["agents"] * params["n_steps"]) / params["value_mini_batch_size"] * params[
        "value_epochs"]

    reader = csv.reader(open(file_path + "/training_log.csv", 'r'))
    col_names = next(reader, None)
    result = {}
    data = [row for row in reader]

    for col in col_names:
        result[col] = []

    for row in data:
        for col_name, value in zip(col_names, row):
            result[col_name].append(safe_float(value))

    good_rows = len(result["env_step"])
    if good_rows <= 1:
        print(f"not enough data for {file_path} found only {good_rows} good rows from {len(data)}.")
        return None

    batch_size = int(result['env_step'][1] / result['iteration'][1])
    result["batch_size"] = batch_size
    result["params"] = params
    params["batch_size"] = params["agents"] * params["n_steps"]
    for v in ["tvf_max_horizon", "distill_beta", "tvf_n_step", "tvf_horizon_samples", "tvf_value_samples",
              "tvf_soft_anchor", "tvf_coef"]:
        if v in params:
            params["quant_" + v] = 2 ** int(math.log2(params[v]))

    for k in list(result.keys()):
        v = result[k]
        # this is a bit dodgy it's because the old version used to sometimes not calculate this until step 2.
        if k in ["err_trunc", "err_model"]:
            if v[0] is None:
                v[0] = v[1]
        if v is not None and type(v) is list:

            if k == "value_loss":
                v = [-x for x in v]

            if v is not None and len(v) > 0 and all(x is not None for x in v):
                if min(v) > 0:
                    result["log_" + k] = np.log2(v)
                if min(v) > -1e-6:
                    result["elog10_" + k] = np.log10(np.asarray(v) + 1e-6)
                if max(v) < 0:
                    result["logn_" + k] = np.log2([-x for x in v])
                if max(v) < 10:
                    result["exp_" + k] = np.exp(v)

    for k, v in params.items():
        result[k] = v
        if type(v) in [int, float] and v > 0:
            result["log2_" + k] = np.log2(v)
            result["rlog2_" + k] = np.round(np.log2(v))
            result["log10_" + k] = np.log10(v)

    result["score"] = compute_score(result)
    result["score_alt"] = compute_score_alt(result)
    result["epoch"] = np.asarray(result["env_step"], dtype=np.float32) / 1e6
    result["final_epoch"] = round(result["epoch"][-1], 1)

    game = params["environment"]

    result["ep_score_norm"] = np.asarray(
        [asn.normalize(game, score, count) for score, count in zip(result["ep_score_mean"], result["ep_count"])])

    if "tvf_horizon_transform" in result:
        result["tvf_horizon_transform"] = "log" if result["tvf_horizon_transform"] == "log" else "off"
    return result


def read_params(file_path):
    with open(file_path, 'r') as f:
        result = json.load(f)
    return result


def compute_score(result, x_lim=None):
    if x_lim is None:
        data = result["ep_score_mean"]
    else:
        data = [score for step, score in zip(result["iteration"], result["ep_score_mean"]) if
                step / 1000 / 1000 < x_lim]
    return np.percentile(data, 95)


def compute_score_alt(result, x_lim=None):
    if x_lim is None:
        data = result["ep_score_mean"]
    else:
        data = [score for step, score in zip(result["iteration"], result["ep_score_mean"]) if
                step / 1000 / 1000 < x_lim]
    return smooth(data, 0.95)[-1]


# def get_score_alt(results, team, n_episodes=100):
#    # get the score, which is a combination of the time taken to win and the score acheived
#    return np.mean((results[f"score_{team}"] * 0.99 ** np.asarray(results["game_length"]))[-n_episodes:])


def smooth(X, alpha=0.95):

    y = X[0]
    i = 0
    while y is None:
        y = X[i]
        i += 1
        if i == len(X):
            return [0] * len(X)

    result = []
    for x in X:
        if x is None:
            x = y
        y = alpha * y + (1 - alpha) * x
        result.append(y)
    return result


def short_run_name(s):
    result = []
    for x in s.split():
        if '=' in x:
            result.append(x.split("=")[0][:] + "=" + x.split("=")[-1])
        elif x[0] == '[':
            result.append(x[1:9])
        else:
            pass
    return " ".join(result)


def get_sort_key(x):
    """ Returns smart sorting key"""
    if type(x) is str:
        if "=" in x:
            first_part = x.split("=")[0]
            second_part = x.split("=")[1]
            second_part = second_part.split(" ")[0]
            try:
                value = f"{float(second_part):012.6f}"
            except:
                value = second_part

            return first_part + "=" + value
    return x

def get_runs(path, run_filter=None, skip_rows=1):
    if type(path) is list:
        runs = []
        for p in path:
            runs.extend(get_runs(p, run_filter=run_filter))
        return runs

    runs = []
    for subdir, dirs, files in os.walk(path):
        if run_filter is not None and not run_filter(subdir):
            continue
        for file in files:
            name = os.path.split(subdir)[-1]
            if file == "training_log.csv":
                data = RunLog(os.path.join(subdir, file), skip_rows=skip_rows)
                if data is None:
                    continue

                if len(data) <= 1:
                    continue

                params = read_params(os.path.join(subdir, "params.txt"))
                params.update({"path": subdir})
                runs.append([name, data, params])
    runs.sort(key=lambda x: get_sort_key(x[0]), reverse=True)
    return runs

def comma(x):
    return "{:,.1f}".format(x) if x < 100 else "{:,.0f}".format(x)


class RunLog():
    """
    Results from a training run
    """

    def __init__(self, filename, skip_rows=0):

        self._patterns = {
            'log_': lambda x: np.log10(x),
            'neg_': lambda x: -x,
            'exp_': lambda x: np.exp(x),
            '1m_': lambda x: 1 - x,
        }

        self._fields = {}
        self.load(filename, skip_rows=skip_rows)

    def __getitem__(self, key: str):

        for pattern, func in self._patterns.items():
            if key.startswith(pattern):
                return func(self[key[len(pattern):]])

        if key not in self._fields:
            raise Exception(f"{key} not in {list(self._fields.keys())}")

        return self._fields[key]

    def __contains__(self, key: str):

        for pattern, func in self._patterns.items():
            if key.startswith(pattern):
                return key[len(pattern):] in self

        return key in self._fields

    def __len__(self):
        if "env_step" not in self._fields:
            return 0
        else:
            return len(self._fields["env_step"])

    def load(self, file_path, skip_rows=0):
        reader = csv.reader(open(file_path, 'r'))
        col_names = next(reader, None)
        result = {}
        data = [row for row in reader]

        if len(data) <= 1:
            return None

        for col in col_names:
            result[col] = []

        for row in data[skip_rows:]:
            for col_name, value in zip(col_names, row):
                result[col_name].append(safe_float(value))

        if "value_loss" in result:
            result["value_loss_alt"] = np.asarray(result["value_loss"]) - 1e-8

        for k, v in result.items():
            if type(v) is list:
                result[k] = np.asarray(v)

        # calculate average ev as ev_av seems to not work well right now...
        evs = [k for k in result.keys() if "ev_" in k and "ev_av" not in k]
        evs.sort()

        for ev in evs:
            if None in result[ev]:
                print(f"Warning, None found in {ev} on file {file_path}")
                result[ev] = np.asarray([x if x is not None else 0 for x in result[ev]])

        if len(evs) > 0:
            result["ev_average"] = np.mean(np.stack([result[ev] for ev in evs]), axis=0)
            result["ev_max"] = result[evs[-1]]

        self._fields = result


def compare_runs(path,
                 x_lim=None,
                 x_axis="env_step",
                 y_axis="ep_score-mean",
                 show_legend=True,
                 title=None,
                 highlight=None,
                 run_filter=None,
                 color_filter=None,
                 smooth_factor=None,
                 reference_run=None
                 ):
    """ Compare runs stored in given path. """

    if highlight is not None and isinstance(highlight, str):
        highlight = [highlight]

    if title is None and highlight is not None: title = "".join(highlight) + " by " + y_axis
    if title is None: title = "Training Graph"

    runs = get_runs(path)

    if len(runs) == 0:
        return

    plt.figure(figsize=(16, 4))
    plt.grid()

    plt.title(title)

    cmap = plt.cm.get_cmap('tab10')
    counter = 0
    i = 0

    if reference_run is not None:
        runs.append(["reference", reference_run, {}])

    for run_name, run_data, run_params in runs:

        if run_filter is not None and run_data is not reference_run:
            if not run_filter(run_name):
                continue

        i += 1

        run_label = "[{}]".format(run_name[:20])

        xs = run_data[x_axis]
        if x_axis == "env_step":
            xs = np.asarray(xs, dtype=np.float) / 1e6  # in millions
        if x_axis == "walltime":
            xs = np.asarray(xs) / (60 * 60)  # in hours

        if y_axis not in run_data:
            continue

        try:
            ys = np.asarray(run_data[y_axis])
        except:
            # problem with data...
            print(f"Warning, skipping {run_name}.")
            continue

        # filter
        if x_lim:
            xs, ys = zip(*([x, y] for x, y in zip(xs, ys) if x < x_lim))

        if smooth_factor is None:
            smooth_factor = 0.995 ** (run_data["batch_size"] / 2048)

        small = (i - 1) % len(cmap.colors)
        big = (i - 1) // len(cmap.colors)
        auto_color = cmap.colors[small]
        auto_style = ["-", "--"][big % 2]

        ls = "-"

        if highlight:
            if any([x in run_name for x in highlight]):
                color = cmap.colors[counter]
                counter += 1
                alpha = 1.0
            else:
                color = "gray"
                alpha = 0.33

        else:
            color = auto_color
            ls = auto_style
            alpha = 1.0

        if run_data is reference_run:
            color = "gray"

        if len(xs) != len(ys):
            ys = xs[:len(xs)]
            print(f"Warning, missmatched data on {path}:{run_name}")

        if color_filter is not None:
            color = color_filter(run_name, color)

        plt.plot(xs, ys, alpha=0.2 * alpha, c=color)
        plt.plot(xs, smooth(ys, smooth_factor), label=run_label if alpha == 1.0 else None, alpha=alpha, c=color,
                 linestyle=ls)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)

    plt.grid(True, alpha=0.2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if show_legend:
        plt.legend()
    plt.show()


def eval_runs(path, y_axes=("ep_score_mean", "ep_length_mean"), include_table=False, **kwargs):
    title_args = {}

    for y_axis in y_axes:
        if 'title' not in kwargs:
            title_args["title"] = y_axis
        compare_runs(path, y_axis=y_axis, **{**kwargs, **title_args})

    if include_table:
        table_runs(path, run_filter=kwargs.get("run_filter", None))


def table_runs(path, run_filter=None):
    runs = get_runs(path)

    print("|{:<50}|{:>16}|{:>16}|{:>16}|".format(
        " run", "score ", "steps ", "id "))
    print("|" + '-' * 50 + "|" + '-' * 16 + "|" + '-' * 16 + "|" + '-' * 16 + "|")

    runs = sorted(runs, key=lambda x: compute_score(x[1]), reverse=True)

    for run_name, run_data, run_params in runs:

        if len(run_data) < 5:
            continue

        if run_filter is not None:
            if not run_filter(run_name):
                continue

        score_50 = compute_score(run_data, 50)
        score_200 = compute_score(run_data, 200)
        steps = run_data["env_step"][-1] / 1000 / 1000

        run_params["color"] = bool(run_params.get("color", False))
        if "ent_bouns" in run_params:
            # ops.. spelled entropy bonus wrong...
            run_params["ent_bonus"] = run_params["ent_bouns"]

        if "model" not in run_params:
            run_params["model"] = "cnn"

        run_id = run_name[-17:-1]

        print("| {:<49}|{:>15} |{:>15} |{:>16}|".format(
            run_name[:-(8 + 3)], comma(score_200), "{:.0f}M".format(steps), run_id))


def load_eval_results(path):
    """
    Load evaluation results for each epoch.
    """

    if path in cache:
        return cache[path]

    results = {
        'epoch': [],
        'lengths': [],
        'scores': []
    }
    for epoch in range(100):
        data_filename = os.path.join(path, f"checkpoint-{epoch:03d}M-eval.dat")

        try:
            with open(data_filename, "rb") as f:
                data = pickle.load(f)
        except:
            continue

        results['epoch'].append(epoch)
        results['lengths'].append(data["episode_lengths"])
        results['scores'].append(data["episode_scores"])

        # errors at horizons
        errors = data["return_estimates"][0.99]["trunc_err_k"]
        ks = set(x[0] for x in errors)
        for k in ks:
            estimated_values = np.asarray([x[1] for x in errors if x[0] == k])
            true_values_truncated = np.asarray([x[2] for x in errors if x[0] == k])
            true_values_discounted = np.asarray([x[3] for x in errors if x[0] == k])
            true_values = np.asarray([x[4] for x in errors if x[0] == k])

            key = f'h_{k}_error'
            if key not in results:
                results[key] = []
            results[key].append((estimated_values - true_values_truncated) ** 2)

            key = f'd_{k}_error'
            if key not in results:
                results[key] = []
            results[key].append((estimated_values - true_values_discounted) ** 2)

            key = f't_{k}_error'
            if key not in results:
                results[key] = []
            results[key].append((estimated_values - true_values) ** 2)

    cache[path] = results

    return results


def plot_eval_results(path, y_axis="scores", label=None):
    results = load_eval_results(path)

    if len(results["epoch"]) == 0:
        return

    xs = np.asarray(results["epoch"])
    ys = np.asarray([np.mean(x) for x in results[y_axis]])
    y_err = np.asarray([np.std(x) / len(x) ** 0.5 for x in results[y_axis]])
    plt.plot(xs, ys, label=label)
    plt.fill_between(xs, ys - y_err * 1.96, ys + y_err * 1.96, alpha=0.1)


def plot_eval_experiment(path, y_axis="scores", y_label="Score"):
    plt.figure(figsize=(12, 4))
    plt.title(f"{path} - {y_axis}")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)
    for folder in [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]:
        plot_eval_results(os.path.join(path, folder), y_axis=y_axis, label=folder)
    plt.legend()
    plt.show()


def plot_eval_error(run_path, error_code='h'):
    """
    Error codes are
        'h': predicted V(s,h) vs true V(s,h) with discount=args.tvf_gamma
        'd': predicted V(s,h) vs true V(s,inf) with discount=args.tvf_gamma
        't': predicted V(s,h) vs true V(s,inf) with no discounting (i.e. the true value of the state under policy)
    """

    results = load_eval_results(run_path)
    if len(results["epoch"]) == 0:
        return

    plt.figure(figsize=(12, 4))
    plt.title(f"{run_path}")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Error")

    xs = np.asarray(results["epoch"])

    hs = [1, 10, 30, 100, 300, 500]
    for i, y_axis in enumerate(f"{error_code}_{h}_error" for h in hs):
        if y_axis not in results:
            continue
        ys = np.asarray([np.mean(x) for x in results[y_axis]])
        y_err = np.asarray([np.std(x) / len(x) ** 0.5 for x in results[y_axis]])
        if len(ys) != len(xs):
            print("missing value in ", y_axis)
            continue

        c = mpl.cm.get_cmap("plasma")(i / len(hs))
        plt.plot(xs, ys, label=y_axis, color=c)
        plt.fill_between(xs, ys - y_err * 1.96, ys + y_err * 1.96, alpha=0.1, color=c)

    plt.yscale("log")

    plt.legend()
    plt.show()


def plot_experiment(
        path,
        y_axes=("ep_score_mean", "err_trunc", "ev_ext", "opt_grad"),
        run_filter=None,
        smooth_factor=0.95,
        **kwargs
):
    global cache
    cache = {}

    eval_runs(
        path,
        y_axes=y_axes,
        include_table=False,
        smooth_factor=smooth_factor,
        run_filter=run_filter,
        **kwargs
    )


class AtariScoreNormalizer:

    def __init__(self):
        self._normalization_scores = self._load_scores("./Atari-Human.csv")

    def _load_scores(self, filename):

        def pascal_case(s):
            return "".join(x.capitalize() for x in s.split(" "))

        import pandas as pd
        data = pd.read_csv(filename)
        result = {}
        for index, row in data.iterrows():
            key = pascal_case(row["Game"]).lower()
            result[key] = row["Random"], row["Human"]
        return result

    def normalize(self, game, score, count=1):
        if count is None or count == "" or count == 0:
            return 0.0
        if type(score) is list:
            score = np.asarray(score)
        key = game.lower()
        if key not in self._normalization_scores:
            print(f"Game not found {game}")
            return score * 0
        random, human = self._normalization_scores[key]
        return 100 * (score - random) / (human - random)


asn = AtariScoreNormalizer()

# multi game processing
from collections import defaultdict


def read_combined_log(path: str, key: str, subset='default'):
    """
    Load multiple games and average their scores
    """

    subset = subset.lower()

    if subset == 'default':
        # this was the old validation set from before
        game_list = ['Amidar', 'BattleZone', 'DemonAttack']
        game_weights = [0.35144866, 0.55116459, 0.01343885]
        c = 20.78141750170289
    elif subset == "atari-val":
        # game_list = ['Amidar', 'BankHeist', 'Centipede']
        # game_weights = [0.6795, 0.0780, 0.0711]
        # c = 68.17
        # game_list = ['DemonAttack', 'IceHockey', 'Krull']
        # game_weights = [0.0174, 0.6230, 0.0625]
        # c = 0.00
        # c = 0.00
        # game_list = ['BattleZone', 'CrazyClimber', 'TimePilot']
        # game_weights = [0.38186622, 0.19303045, 0.02880996]
        game_list = ['Krull', 'KungFuMaster', 'Seaquest']
        game_weights = [0.04573467, 0.61623311, 0.14444]
        c = 1.7093517175190982

    else:
        raise Exception("invalid subset")

    epoch_scores = defaultdict(lambda: {game: [] for game in game_list})

    folders = [x for x in os.listdir(path) if key in x]
    for folder in folders:
        game_log = read_log(os.path.join(path, folder))
        if game_log is None:
            print(f"no log for {path} {folder}")
            return None
        game = game_log["params"]["environment"]
        if game not in game_list:
            print(f"Skipping {game}")
            continue
        for env_step, ep_score in zip(game_log["env_step"], game_log["ep_score_norm"]):
            epoch_scores[env_step // 1e6][game].append(ep_score)

    if len(epoch_scores) == 0:
        return None

    result = defaultdict(list)
    epochs = sorted(epoch_scores.keys())
    for k, v in game_log["params"].items():  # last one will do...
        result[k] = v

    if len(epochs) == 0:
        return None

    # normalize the results

    for epoch in epochs:
        es = epoch_scores[epoch]
        # make sure we have data for all 3 games
        if not all(len(es[game]) > 0 for game in game_list):
            break
        # work out game scores for each game
        weighted_score = c
        for game, weight in zip(game_list, game_weights):
            norm_score = np.mean(es[game])
            #result[f"{game.lower()}_score"].append(score)
            result[f"{game.lower()}_norm"].append(norm_score)
            weighted_score += weight * norm_score
        result["score"].append(weighted_score)
        result["env_step"].append(epoch * 1e6)
        result["epoch"].append(epoch)

    if len(result["score"]) == 0:
        return

    result["score_alt"] = np.mean(result["score"][-5:])  # last 5 epochs

    score_list = [
        weight * np.mean(result[f"{game.lower()}_norm"][-5:]) for game, weight in zip(game_list, game_weights)
    ]

    result["score_min"] = round(min(score_list))
    result["score_list"] = tuple(round(x) for x in score_list)

    result["final_epoch"] = result["epoch"][-1]
    result["run_name"] = key

    keys = list(result.keys())
    for k in keys:

        v = result[k]

        if v is None:
            continue

        if type(v) in [int, float]:
            if v > 0:
                result["log2_" + k] = np.log2(v)
            if v > 0:
                result["log10_" + k] = np.log10(v)
            if v < 10:
                result["exp_" + k] = np.exp(v)

        if v is not None and type(v) is list:
            if len(v) == 0:
                continue
            if min(v) > 0:
                result["log_" + k] = np.log2(v)
            if min(v) > -1e-6:
                result["elog10_" + k] = np.log10(np.asarray(v) + 1e-6)
            if max(v) < 0:
                result["logn_" + k] = np.log2([-x for x in v])
            if max(v) < 10:
                result["exp_" + k] = np.exp(v)

    return result
