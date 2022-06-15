import bisect
import types

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import csv
import json
import math
import typing
from collections import defaultdict
from glob import glob

import pickle
from os import listdir
from os.path import isfile, join
from typing import Union

LOG_CAP = 1e-6 # when computing logs values <= 0 will be replaced with this


def safe_float(x):
    try:
        return float(x)
    except:
        return None

def safe_mean(x):
    if x is None:
        return None
    x = [z for z in x if z is not None]
    if len(x) == 0:
        return None
    return np.mean(x)

def safe_std(x):
    if x is None:
        return None
    x = [z for z in x if z is not None]
    if len(x) == 0:
        return None
    return np.std(x)

def safe_len(x):
    if x is None:
        return 0
    x = [z for z in x if z is not None]
    if len(x) == 0:
        return 0
    return len(x)

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
    # for v in ["tvf_max_horizon", "distil_beta", "tvf_n_step", "tvf_horizon_samples", "tvf_value_samples",
    #           "tvf_soft_anchor", "tvf_coef"]:
    #     if v in params:
    #         params["quant_" + v] = 2 ** int(math.log2(params[v]))

    # this is a fixup for double dunk, where score is 0 at the start before any real scores have arrived.
    if params["environment"] == "DoubleDunk":
        ep_scores = result["ep_score_mean"]
        min_score = np.min(ep_scores)
        for i in range(len(ep_scores)):
            if ep_scores[i] == 0:
                ep_scores[i] = min_score
            else:
                break

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
                    result["log_" + k] = np.log2(np.clip(v, LOG_CAP, float('inf')))
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
    epochs_done = (result["batch_size"]/1e6) + result["epoch"][-1] # include the last batch
    result["final_epoch"] = round(epochs_done, 1)

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


class RunMetric:

    def __init__(self, name):
        self.name = name

    def score(self, run, x_lim=None):
        return 0


class ScoreMetric(RunMetric):

    def __init__(self):
        super().__init__("score")

    def score(self, run, x_lim=None):
        return compute_score(run, x_lim)


class MeanMetric(RunMetric):

    def __init__(self, name: str, field: str):
        super().__init__(name)
        self.field = field

    def score(self, run, x_lim=None):
        return run[self.field].mean()

def compute_score(result, x_lim=None):
    if x_lim is None:
        data = result["ep_score_mean"]
    else:
        data = [score for step, score in zip(result["env_step"], result["ep_score_mean"]) if
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
#    # get the score, which is a combination of the time taken to win and the score achieved
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


def get_sort_key(s):
    """ Returns smart sorting key"""

    def convert_part(x):
        if "=" not in x:
            return x
        first_part = x.split("=")[0]
        second_part = x.split("=")[1]
        second_part = second_part.split(" ")[0]
        try:
            value = f"{float(second_part):012.6f}"
        except:
            value = second_part
        return first_part + "=" + value

    if type(s) is str:
        return " ".join(convert_part(part) for part in s.split(" "))
    else:
        return s

def get_runs(path: Union[str, list], run_filter=None, skip_rows=1):
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
    runs.sort(key=lambda x: get_sort_key(x[0]), reverse=False)
    return runs


def comma(x):
    return "{:,.1f}".format(x) if x < 100 else "{:,.0f}".format(x)


class RunLog():
    """
    Results from a training run
    """

    generators = {}

    def __init__(self, filename, skip_rows=0):

        self._patterns = {
            'log_': lambda x: np.log10(LOG_CAP+np.clip(x, 0, float('inf'))),
            'log2_': lambda x: np.log2(np.clip(x, LOG_CAP, float('inf'))),
            'neg_': lambda x: -x,
            'exp_': lambda x: np.exp(x),
            '1m_': lambda x: 1 - x,
            '1p_': lambda x: 1 + x,
            'abs_': lambda x: np.abs(x),
            'nz_': lambda x: np.nan_to_num(np.asarray(x, dtype=np.float)) # convert none to 0
        }

        self._fields = {}
        self.load(filename, skip_rows=skip_rows)

    def __getitem__(self, key: str):

        # stub: support var+bias
        if key in self.generators.keys():
            return self.generators[key](self)

        for pattern, func in self._patterns.items():
            if key.startswith(pattern):
                return func(self[key[len(pattern):]])

        if key not in self._fields:
            raise Exception(f"{key} not in {list(self._fields.keys())}")

        return self._fields[key]

    def __contains__(self, key: str):

        if key in self.generators.keys():
            return True


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
        evs = [k for k in result.keys() if k.startswith("ev_") and "ev_av" not in k]
        evs.sort()

        for ev in evs:
            if None in result[ev]:
                #print(f"Warning, None found in {ev} on file {file_path}")
                result[ev] = np.asarray([x if x is not None else 0 for x in result[ev]])

        if len(evs) > 0:
            result["ev_auto_average"] = np.mean(np.stack([result[ev] for ev in evs]), axis=0)
            result["ev_auto_max"] = result[evs[-1]]

        try:
            params = read_params(os.path.split(file_path)[0]+"/params.txt")
            result[f"ep_score_norm"] = asn.normalize(params["environment"], result["ep_score_mean"])
        except:
            params = {}

        try:
            # include unscaled value estimates
            keys = list(result.keys())
            for k in keys:
                if k.startswith("v_mu_") or k.startswith("v_std_"):
                    result[k+"_rs"] = result[k] * result["reward_scale"]
        except:
            pass

        # this is a fixup for double dunk, where score is 0 at the start before any real scores have arrived.
        if params.get("environment", '') == "DoubleDunk":
            ep_scores = result["ep_score_mean"]
            min_score = np.min(ep_scores)
            for i in range(len(ep_scores)):
                if ep_scores[i] == 0:
                    ep_scores[i] = min_score
                else:
                    break

        self._fields = result


def compare_runs(
        runs: list,
        x_lim=None,
        x_axis="env_step",
        y_axis="ep_score_mean",
        x_start=None,
        show_legend=True,
        title=None,
        label_filter=None,
        color_filter=None,
        style_filter=None,
        group_filter=None,
        zorder_filter=None,
        smooth_factor=None,
        reference_run=None,
        x_axis_name=None,
        y_axis_name=None,
        ref_level=None,
        hold=False,
        x_transform=None,
        ghost_alpha:float=0.1,
        jitter=0.0,
        figsize=(16, 4),
        show_group_runs=False,
        run_filter=None, # not used
):
    """
        Compare runs stored in given path.
        group_filter: If defined, all runs with same group will be plotted together.
    """

    if title is None:
        title = "Training Graph"

    if len(runs) == 0:
        return

    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.grid()

    plt.title(title)

    cmap = plt.cm.get_cmap('tab10')
    counter = 0
    i = 0

    if reference_run is not None:
        runs.append(["reference", reference_run, {}])

    run_labels_so_far = set()

    group_data = {}
    group_style_names = defaultdict(set)

    i_per_style = defaultdict(int)

    for run_name, run_data, run_params in runs:

        i += 1

        if label_filter is not None:
            run_label = label_filter(run_name)
        else:
            run_label = "[{}]".format(run_name[:20])

        xs = run_data[x_axis]
        if x_axis == "env_step":
            xs = np.asarray(xs, dtype=np.float) / 1e6  # in millions
        if x_axis == "walltime":
            xs = np.asarray(xs) / (60 * 60)  # in hours

        if x_transform is not None:
            xs = [x_transform(x) for x in xs]

        if y_axis not in run_data:
            continue

        try:
            ys = np.asarray(run_data[y_axis])
        except:
            # problem with data...
            print(f"Warning, skipping {run_name}.")
            continue

        if ref_level is not None:
            plt.hlines(ref_level, xs[0], xs[-1], color='gray')

        # filter
        if x_lim:
            xs, ys = zip(*([x, y] for x, y in zip(xs, ys) if x < x_lim))

        if smooth_factor is None:
            smooth_factor = 0.995 ** (run_data["batch_size"] / 2048)

        # figure out style and color
        if style_filter is not None:
            line_style = style_filter(run_name)
        else:
            line_style = "-"

        color_index = i_per_style[line_style]
        i_per_style[line_style] += 1
        color = cmap.colors[color_index % len(cmap.colors)]
        if color_filter is not None:
            color = color_filter(run_name, color)

        # -----

        alpha = 1.0
        zorder = None if zorder_filter is None else zorder_filter(run_name, run_params)

        if run_data is reference_run:
            color = "gray"

        if len(xs) != len(ys):
            ys = xs[:len(xs)]
            print(f"Warning, missmatched data on {run_name}")

        # make sure we don't duplicate labels
        if run_label in run_labels_so_far:
            run_label = None
        run_labels_so_far.add(run_label)

        if jitter != 0:
            ys = ys * (1 + np.random.randn(len(ys)) * jitter)

        if group_filter is not None:
            group = group_filter(run_name, run_data)
            if group != None:
                if group not in group_data:
                    # redo color...
                    if color_filter is None:
                        color = cmap.colors[len(group_style_names[line_style]) % len(cmap.colors)]
                    group_data[group] = ([xs], [ys], run_label, alpha, color, zorder, line_style)
                else:
                    group_data[group][0].append(xs[x_start:])
                    group_data[group][1].append(ys[x_start:])
                group_style_names[line_style].add(group)
                continue

        plt.plot(xs[x_start:], ys[x_start:], alpha=ghost_alpha * alpha, c=color, linestyle=line_style)
        plt.plot(xs[x_start:], smooth(ys[x_start:], smooth_factor), label=run_label if alpha == 1.0 else None, alpha=alpha, c=color,
                 linestyle=line_style, zorder=zorder)

    if x_axis_name != '':
        plt.xlabel(x_axis_name or x_axis)
    if y_axis_name != '':
        plt.ylabel(y_axis_name or y_axis)

    # show groups
    default_color_index = 0
    for k, (group_xs, group_ys, run_label, alpha, color, zorder, ls) in group_data.items():

        def get_y(index):
            return [this_ys[index] for this_ys in group_ys if index < len(this_ys)]

        xs = group_xs[0]
        all_ys = [get_y(i) for i in range(len(xs))]

        ys = []
        ys_std_err = []
        for y_sample in all_ys:
            y_sample = [x for x in y_sample if x is not None]
            if len(y_sample) == 0:
                ys.append(0)
                ys_std_err.append(0)
            else:
                ys.append(np.mean(y_sample))
                ys_std_err.append(np.std(y_sample) / (len(y_sample)**0.5))
        ys = np.asarray(ys)
        ys_std_err = np.asarray(ys_std_err)

        #ys_low = [np.max(y_sample) for y_sample in all_ys]
        #ys_high = [np.min(y_sample) for y_sample in all_ys]
        ys_low = ys - ys_std_err
        ys_high = ys + ys_std_err

        if color is None:
            color = cmap.colors[default_color_index]

        if show_group_runs:
            for x_raw, y_raw in zip(group_xs, group_ys):
                plt.plot(x_raw, y_raw, alpha=0.10, c=color, linestyle="--", zorder=-10)

        plt.fill_between(xs, smooth(ys_low, smooth_factor), smooth(ys_high, smooth_factor), alpha=0.15 * alpha, color=color)
        plt.plot(xs, smooth(ys, smooth_factor), label=run_label if alpha == 1.0 else None, alpha=alpha, c=color,
                 linestyle=ls, zorder=zorder)

        default_color_index += 1

    standard_grid()

    if show_legend is not False:
        if show_legend is True:
            loc = "best"
        else:
            loc = show_legend
        plt.legend(loc=loc)

    if not hold:
        handles, labels = plt.gca().get_legend_handles_labels()
        if len(labels) == 0:
            # nothing to draw...
            plt.close(plt.gcf())
        else:
            plt.show()


def standard_grid():
    plt.grid(True, alpha=0.2)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def eval_runs(runs, y_axes=("ep_score_mean", "ep_length_mean"), include_table=False, table_epochs=None, table_metric=None, **kwargs):

    title_args = {}

    if include_table:
        if table_metric is not None:
            args = {
                'metric': table_metric
            }
        else:
            args = {}
        table_runs(runs, run_filter=kwargs.get("run_filter", None), epochs=table_epochs, **args)

    if type(y_axes) is str:
        y_axes = (y_axes,)

    for y_axis in y_axes:
        if 'title' not in kwargs:
            title_args["title"] = y_axis
        compare_runs(runs, y_axis=y_axis, **{**kwargs, **title_args})


def table_runs(runs, run_filter=None, epochs=None, metric:RunMetric=ScoreMetric()):

    if epochs is None:
        epochs = [50]


    print(("|{:<50}|{:>16}|"+"{:>16}|"*len(epochs)+"{:>16}|").format(
            " run", *[metric.name+" @"+str(epoch)+"M " for epoch in epochs], "steps ", "id "))
    print("|"+'-'*50+("|"+'-'*16)*len(epochs)+"|"+'-'*16+"|"+'-'*16+"|")

    runs = sorted(runs, key=lambda x: metric.score(x[1]), reverse=True)

    for run_name, run_data, run_params in runs:

        if len(run_data) < 5:
            continue

        if run_filter is not None:
            if not run_filter(run_name):
                continue

        scores = {}
        for epoch in epochs:
            scores[epoch] = metric.score(run_data, epoch)

        steps = min(run_data["env_step"][-1] / 1000 / 1000, max(epochs))

        run_params["color"] = bool(run_params.get("color", False))
        if "ent_bouns" in run_params:
            # ops.. spelled entropy bonus wrong...
            run_params["ent_bonus"] = run_params["ent_bouns"]

        if "model" not in run_params:
            run_params["model"] = "cnn"

        run_id = run_name[-9:-1]+" "

        print(("| {:<49}" + "|{:>15} " * len(epochs) + "|{:>15} |{:>16}|").format(
            run_name[:-10], *[round(x,2) for x in scores.values()], "{:.0f}M".format(steps), run_id))

def get_eval_filename(epoch, temperature=None, seed=None):
    postfix = f"_t={float(temperature)}" if temperature is not None else ""
    if seed is not None:
        postfix += f"_{seed}"
    return f"checkpoint-{epoch:03d}M-eval{postfix}.dat"


def load_eval_epoch(path, epoch, temperature=None, seed=None):
    fname = os.path.join(path, get_eval_filename(epoch, temperature, seed))
    if not os.path.exists(fname) and seed is None:
        # old versions did not incorporate the seed into the filename, so if seed is none, also try 1.
        fname = os.path.join(path, get_eval_filename(epoch, temperature, 1))
    if os.path.exists(fname):
        with open(fname, "rb") as f:
            return pickle.load(f)
    else:
        return None


def load_eval_results(path, temperature=None, seed=None):
    """
    Load evaluation results for each epoch.
    """

    key = (path, temperature)

    if key in eval_cache:
        return eval_cache[key]

    results = {
        'epoch': [],
        'lengths': [],
        'scores': []
    }
    for epoch in range(201):
        data = load_eval_epoch(path, epoch, temperature, seed)
        if data is None:
            continue

        results['epoch'].append(epoch)
        results['lengths'].append(data["episode_lengths"])
        results['scores'].append(data["episode_scores"])

    eval_cache[key] = results

    return results


def plot_eval_results(path, y_axis="scores", label=None, temperature=None, quantile=None):
    results = load_eval_results(path, temperature=temperature)

    if len(results["epoch"]) == 0:
        return

    xs = np.asarray(results["epoch"])
    ys = np.asarray([np.mean(x) for x in results[y_axis]])

    if quantile is not None:
        ys = np.asarray([np.quantile(x, quantile) for x in results[y_axis]])

    y_err = np.asarray([np.std(x) / len(x) ** 0.5 for x in results[y_axis]])
    p = plt.plot(xs, ys, label=label)
    c = p[0].get_color()
    if quantile is None:
        plt.fill_between(xs, ys - y_err * 1.96, ys + y_err * 1.96, alpha=0.1, color=c)

    #print(f"{path}:{ys[-1]:<10.1f} at {np.max(xs)}")
    return ys[-1]


def plot_eval_temperature(path, temps=(0.5, 1, 2), y_axis="scores", y_label="Score", **kwargs):
    plt.figure(figsize=(12, 4))
    plt.title(f"{path} - {y_axis}")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel(y_label)

    results = {}
    for temperature in temps:
        results[temperature ] = plot_eval_results(path, y_axis=y_axis, label=f"t={temperature}", temperature=temperature, **kwargs)

    plt.legend()
    plt.show()

    return results

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
        runs:list,
        y_axes=("ep_score_mean", "err_trunc", "ev_ext", "opt_grad"),
        run_filter=None,
        smooth_factor=0.95,
        include_table=False,
        **kwargs
):
    # support old style path as input
    if type(runs) is str or (type(runs) is list and len(runs) > 0 and type(runs[0]) is str):
        runs = get_runs(runs, skip_rows=1, run_filter=run_filter)

    eval_runs(
        runs=runs,
        y_axes=y_axes,
        include_table=include_table,
        smooth_factor=smooth_factor,
        run_filter=run_filter,
        **kwargs
    )


class AtariScoreNormalizer:
    SUBSETS = {
        'Atari_Single': (
            ['Zaxxon'],
            [0.7364257371683655],
            24.1),
        'Atari_5': (
            ['BattleZone', 'DoubleDunk', 'NameThisGame', 'Phoenix', 'Qbert'],
            [0.3819751237831753, 0.0679099488887574, 0.310804148620847, 0.12412619402774641, 0.08048518286466755],
            10.4),
        'Atari_3': (
            ['Battlezone', 'Namethisgame', 'Phoenix'],
            [0.3705645071598444, 0.513268255229986, 0.10151691314166367],
            13.7
        ),
        'Atari_1': (
            ['NameThisGame'],
            [0.9975883300775931],
            27.4
        ),
        'Atari_3_Val': (
            ['Assault', 'MsPacman', 'YarsRevenge'],
            [0.3352529798243203, 0.4236379909065204, 0.1916184632131803],
            17.1
        ),
        'Atari_5_Val': (
            ['BankHeist', 'VideoPinball', 'Assault', 'MsPacman', 'YarsRevenge'],
            [0.10718453170488736, 0.09591728001257466, 0.2233635194763938, 0.294255864978837, 0.22392551763974122],
            14.3
        ),
        'Atari_1_Val': (
            ['Assault'],
            [0.7195420422814323],
            43.4
        ),
        'Atari_10': (
            ['Amidar', 'Bowling', 'Frostbite', 'KungFuMaster', 'RiverRaid', 'BattleZone', 'DoubleDunk', 'NameThisGame',
            'Phoenix', 'Qbert'],
            [0.08246357439570691, 0.0559032813915904, 0.06912572845204429, 0.09861731598935165, 0.048594292965417056,
            0.18882008828001498, 0.08519154019954558, 0.12869695652405946, 0.16432707026384483, 0.05922863791062832],
            7.2
        ),

    }

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
            result[key] = (row["Random"], row["Human"])
        return result

    @property
    def games(self):
        return list(self._normalization_scores.keys())

    @property
    def random(self):
        return {k: v[0] for k, v in self._normalization_scores.items()}

    @property
    def human(self):
        return {k: v[1] for k, v in self._normalization_scores.items()}

    def normalize(self, game, score, count=1):
        if count is None or count == "" or count == 0:
            return 0.0
        if type(score) is list:
            score = np.asarray(score)
        key = game.lower()
        if key not in self._normalization_scores:
            print(f"Warning: Game not found {game}")
            return score * 0
        random, human = self._normalization_scores[key]
        return 100 * (score - random) / (human - random)

    def subset(self, subset_name: str):
        return self.SUBSETS[subset_name][0].copy()

    def subset_score(self, subset_name: str, scores: dict):
        """
        Calculates the subset score, which is an estimate of the Median atari-57 score from some subset of log scores.
        scores: dictionary mapping from env_name to raw (unnormalized) score on that game.
        returns: float, estimate of median score.
        """
        assert subset_name in self.SUBSETS
        games, weights, rel_error = self.SUBSETS[subset_name]

        # because montezuma's revenge often gets 0, and has such low weighting, I often don't test on it and just
        # assume we would have gotten 0.
        if 'montezumarevenge' not in scores:
            scores['montezumarevenge'] = 0.0

        def transform(x):
            return np.log10(1 + np.clip(x, 0, float('inf')))

        def inv_transform(x):
            return (10 ** x) - 1

        total = 0
        for game_name, weight in zip(games, weights):
            norm_score = self.normalize(game_name, scores[game_name])
            total += transform(norm_score) * weight
        return inv_transform(total)


def read_combined_log(path: str, key: str, subset: typing.Union[list, str] = 'Atari_3', subset_weights=None, c=None, seed=None):
    """
    Load multiple games and averages their scores
    """

    if type(path) is str:
        paths = [path]
    else:
        paths = path

    if type(subset) is str:
        game_list, game_weights, _ = AtariScoreNormalizer.SUBSETS[subset]
        c = 0.0 # no intercept for these
    else:
        game_list = subset
        game_weights = subset_weights if subset_weights is not None else [1.0] * len(game_list)
        c = c or 0.0

    game_list = [x.lower() for x in game_list]
    epoch_scores = defaultdict(lambda: {x: [] for x in game_list})

    folders = []
    for path in paths:
        folders.extend([os.path.join(path, x) for x in os.listdir(path) if key in x and os.path.isdir(os.path.join(path, x))])

    if seed is not None:
        folders = [x for x in folders if f"({seed})" in x]

    game_log = None

    for folder in folders:
        if folder in ["rl", "roms"]:
            continue
        game_log = read_log(folder)
        if game_log is None:
            print(f"no log for {path} {folder}")
            return None
        game = game_log["params"]["environment"].lower()
        if game not in game_list:
            #print(f"Skipping {game} as not in {game_list}")
            continue
        for env_step, ep_score in zip(game_log["env_step"], game_log["ep_score_norm"]):
            epoch_scores[round(env_step / 1e6)][game].append(ep_score)

    if len(epoch_scores) == 0 or game_log is None:
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
        # make sure we have data for all games
        if not all(len(es[game]) > 0 for game in game_list):
            break
        # work out game scores for each game
        weighted_score = c
        for game, weight in zip(game_list, game_weights):
            norm_score = np.mean(es[game])
            #result[f"{game.lower()}_score"].append(score)
            result[f"{game}_norm"].append(norm_score)
            weighted_score += weight * np.log10(1 + max(norm_score, 0))
        weighted_score = 10 ** weighted_score - 1
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

    result["final_epoch"] = result["epoch"][-1] + 1 # if we processed epoch 2.x then say we went up to epoch 3.
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

def plot_validation(path, keys, hold=False, color=None, label=None, subset="Atari_3_Val"):

    if not hold:
        plt.figure(figsize=(18, 6))

    for key in keys:
        result = read_combined_log(path, key, subset=subset)
        if result is None:
            print(f"No run matching {path} {key}")
            continue
        xs = result["env_step"]
        ys = result["score"]

        plt.grid(True, alpha=0.2)
        ax=plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if label is None:
            _label = key
        else:
            _label = label
        plt.plot(xs, ys, label=_label, color=color)
    plt.xlim(0, 50e6)
    plt.ylim(0, 300)
    if not hold:
        plt.legend()
        plt.show()


def plot_mode(path):
    ref_result = None
    results = get_runs(path, skip_rows=1)
    for name, data, params in results:
        if "ppg" in name:
            ref_result = data

    for run_filter in ["nstep", "adaptive", "exponential"]:
        plot_experiment(
            path,
            smooth_factor=0.5,
            title=run_filter,
            run_filter = lambda x : ((run_filter in x) or ("ppg" in x)) and "Krull" in x,
            y_axes=(
                "ep_score_mean",
                "log_1m_ev_max",
            ))

    # show each run
    plt.figure(figsize=(16, 4))
    for run_filter in ["nstep", "adaptive"]:
        xs = []
        ys = []
        for nsteps in [10, 20, 30, 40, 80, 160, 320, 480, 640]:
            run_code = f"{run_filter}_{nsteps}"
            for name, data, params in results:
                if run_code in name:
                    score = np.mean(data["neg_log_1m_ev_max"][-10:])
                    xs.append(math.log2(nsteps))
                    ys.append(score)

        if run_filter == "nstep":
            ref_score = np.mean(ref_result["neg_log_1m_ev_max"][-10:])
            plt.hlines(ref_score, 1, 10, label="PPO", color="black", ls='--')

        plt.plot(xs, ys, label=run_filter)
    plt.ylim(0,1.8)
    standard_grid()
    plt.xlabel("$\log_2(n\_step)$")
    plt.legend()
    plt.show()

    plt.figure(figsize=(16, 4))
    for run_filter in ["exponential"]:
        xs = []
        ys = []
        for gamma in [1.5, 2.0, 2.5, 3.0]:
            run_code = f"{run_filter}_{gamma}"
            for name, data, params in results:
                if run_code in name:
                    score = np.mean(data["neg_log_1m_ev_max"][-10:])
                    xs.append(math.log2(gamma))
                    ys.append(score)

        if run_filter == "exponential":
            ref_score = np.mean(ref_result["neg_log_1m_ev_max"][-10:])
            plt.hlines(ref_score, 0, 2, label="PPG", color="black", ls='--')

        plt.plot(xs, ys, label=run_filter)
    plt.ylim(0,1.8)
    standard_grid()
    plt.xlabel("$\log_2(\gamma)$")
    plt.ylabel("$-\log(1-EV_{max})$")
    plt.legend()
    plt.show()


def mean_of_top(X):
    data = sorted(X)
    return np.mean(data[-5:])


def min_of_top(X):
    data = sorted(X)
    return np.min(data[-5:])


def lerp(c1, c2, factor: float):
    from matplotlib import colors
    if type(c1) is str:
        c1 = colors.to_rgb(c1)
    if type(c2) is str:
        c2 = colors.to_rgb(c2)
    return [x1 * factor + x2 * (1 - factor) for x1, x2 in zip(c1, c2)]


def adv_plot(xs, ys, samples, **kwargs):
    """
    Plot a line where alpha fades if not many samples are present.
    """
    base_color = kwargs.get('color', kwargs.get('c', "white"))
    from matplotlib import colors
    base_color = colors.to_rgb(base_color)
    if "color" in kwargs:
        del kwargs["color"]
    if "c" in kwargs:
        del kwargs["c"]

    for x1, x2, y1, y2, s1, s2 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:], samples[:-1], samples[1:]):
        factor = 1 - (1 / (((4 + s1 + s2) * 0.25) ** 0.5))
        kwargs["color"] = lerp(base_color, "black", factor)
        plt.plot([x1, x2], [y1, y2], **kwargs)
        if "label" in kwargs:
            del kwargs["label"]


def marginalize_categorical(results, var_name, sample_filter=None, setup_figure=True, hold=False, label=None, color="white"):
    """
    Show score for given hyperparameter with all other hyperparameters set optimally.
    """

    values = set(row[var_name] for row in results)
    xs = sorted(values)
    ys_mean = []
    ys_max = []
    ys_median = []

    for value in xs:
        # make sure we only look at data that is from (mostly) finished rows
        data = [float(row['score_alt']) for row in results if
                row[var_name] == value and
                ((sample_filter is None) or sample_filter(row))]

        if len(data) == 0:
            print(f"No data for value: {value}")
            data = [0]

        ys_mean.append(mean_of_top(data))
        ys_max.append(np.max(data))
        ys_median.append(np.mean(data))

        if len(data) < 6:
            print(f"only {len(data)} values for {value}")

    if setup_figure:
        plt.figure(figsize=(5, 3))
        plt.grid(True, color='gray', alpha=0.2)
        plt.ylim(0)
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.xlabel(label or var_name)
        plt.ylabel("Score")

    xs = [str(x) for x in xs]
    plt.plot(xs, ys_median, color=color, alpha=0.3)
    plt.plot(xs, ys_max, color=color, alpha=0.3)
    plt.plot(xs, ys_mean, color=color, label=label)

    # show best result
    best_index = np.argmax(ys_mean)
    best_x = xs[best_index]
    best_y = ys_mean[best_index]
    plt.scatter([best_x], [best_y], marker='.', s=75, color=color)

    if not hold:
        plt.show()


def marginalize_(
        results, search_params,
        var_name,
        x_min=None, x_max=None, x_transform=lambda x: x, x_transform_inv=None,
        bandwidth=0.10, sample_filter=None, prefix="", postfix="", divisions=10,
        hold=False, label=None, color="white", include_scatter=False, include_min_max=False,
        setup_figure=True
):
    """
    Show score for given hyperparameter with all other hyperparameters set optimally.
    Designed to work when primary hyperparameter has been randomly sampled from a continious distribution.
    """

    def sample_function(data, weight, func):
        buffer = []
        # 1000 is better, but slower, for linear functions we could just weight the end result
        samples = []
        for _ in range(1000):
            sample = [
                x for x, w in zip(data, weight) if np.random.random() < w
            ]
            samples.append(len(sample))
            if len(sample) == 0:
                continue
            buffer.append(func(sample))
        return (np.mean(buffer), np.mean(samples)) if len(buffer) > 0 else (0.0, 0.0)

    if setup_figure:
        plt.figure(figsize=(5, 3))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.grid(color='gray', alpha=0.2)
        plt.ylim(0)
        plt.xlim(0, len(set(row[var_name] for row in results)))

    v = search_params[var_name]
    is_categorical = type(v) == Categorical

    # show datapoints
    if include_scatter:
        c = lerp(color, 'darkgrey', 0.50)
        if is_categorical:
            xs = ([row[var_name] for row in results if ((sample_filter is None) or sample_filter(row))])
            ys = [row['score_alt'] for row in results if ((sample_filter is None) or sample_filter(row))]
            data = sorted([(x, y) for x, y in zip(xs, ys)])
            xs = [x for x, y in data]
            ys = [y for x, y in data]
            plt.scatter(xs, ys, marker='.', alpha=0.25, color=c, linewidth=0)
        else:
            xs = [x_transform(row[var_name]) for row in results if ((sample_filter is None) or sample_filter(row))]
            if x_transform_inv is not None:
                xs = [x_transform_inv(x) for x in xs]
            ys = [row['score_alt'] for row in results if ((sample_filter is None) or sample_filter(row))]
            plt.scatter(xs, ys, marker='.', alpha=0.25, color=c, linewidth=0)

    ys_mean = []
    ys_max = []
    ys_median = []
    ys_count = []

    if is_categorical:
        xs = sorted(set(row[var_name] for row in results))
        for value in xs:
            # make sure we only look at data that is from (mostly) finished rows
            data = [row['score_alt'] for row in results if
                    row[var_name] == value and
                    ((sample_filter is None) or sample_filter(row))]

            if len(data) == 0:
                print(f"No data for value: {value}")
                data = [0]

            ys_mean.append(mean_of_top(data))
            ys_max.append(np.max(data))
            ys_median.append(np.mean(data))
            ys_count.append(len(data))
            if len(data) < 6:
                print(f"only {len(data)} values for {value}")
    else:

        x_min = x_transform(x_min or np.min([row[var_name] for row in results]))
        x_max = x_transform(x_max or np.max([row[var_name] for row in results]))
        xs = np.linspace(x_min, x_max, divisions)

        bandwidth = (x_max - x_min) * bandwidth

        for x in xs:

            window_start = x - bandwidth
            window_end = x + bandwidth

            # window_start <= x_transform(row[var_name]) <= window_end and

            weight_func = lambda x: math.exp(-0.5 * ((x / bandwidth) ** 2))

            # make sure we only look at data that is from (mostly) finished rows
            data = [float(row['score_alt']) for row in results if
                    ((sample_filter is None) or sample_filter(row))]
            weight = [weight_func(x_transform(row[var_name]) - x) for row in results if
                      ((sample_filter is None) or sample_filter(row))]

            value, sample_count = sample_function(data, weight, mean_of_top)
            ys_mean.append(value)
            ys_count.append(sample_count)
            if include_min_max:
                ys_max.append(sample_function(data, weight, np.max)[0])
                ys_median.append(sample_function(data, weight, np.mean)[0])

                # return ouf or log scale if wanted
        if x_transform_inv is not None:
            xs = [x_transform_inv(x) for x in xs]

    ys_mean = np.asarray(ys_mean)
    ys_count = np.asarray(ys_count)

    plt.xlabel(prefix + var_name + postfix)
    plt.ylabel("Score")
    if include_min_max:
        c = lerp(color, 'black', 0.25)
        adv_plot(xs, ys_max, ys_count, c=c)
        adv_plot(xs, ys_median, ys_count, c=c)

    adv_plot(xs, ys_mean, ys_count, c=color, label=label)

    if not hold:
        plt.show()


class Categorical():
    def __init__(self, *args):
        self._values = args
        self._min = None
        self._max = None


class GammaFunc():
    def __init__(self, *args):
        self._values = args
        self._min = None
        self._max = None


class LogCategorical():
    def __init__(self, *args):
        self._values = args
        self._min = None
        self._max = None


class Uniform():
    def __init__(self, min, max, force_int=False):
        self._min = min
        self._max = max
        self._force_int = force_int


class LogUniform():
    def __init__(self, min, max, force_int=False):
        self._min = min
        self._max = max
        self._force_int = force_int


def marginalize(results, search_params, k: str, secondary: str = None, **kwargs):
    assert k in results[0], f"{k} is not a valid variable name"
    assert secondary is None or secondary in results[0], f"{secondary} is not a valid variable name"

    if secondary is not None:
        factors = set(row[secondary] for row in results)
    else:
        factors = [None]

    v = search_params[k]

    epoch_filter = lambda x: x['final_epoch'] >= 5.0

    base_filter = lambda x: x

    if k == "tvf_n_step":
        base_filter = lambda x: x["tvf_mode"] not in ["exponential", "lambda"]
    if k == "distil_beta":
        base_filter = lambda x: x["distil_epochs"] > 0

    prefix = ""
    x_transform = lambda x: x

    if type(v) is GammaFunc:
        prefix = "log(1/(1-x)) "
        x_transform = lambda x: np.log10(1 / (1 - x)) if x != 1 else -1

    if type(v) in [LogUniform, LogCategorical]:
        prefix = "log_10 "
        x_transform = lambda x: np.log10(x)

    for i, factor in enumerate(factors):

        if factor is not None:
            sample_filter = lambda x: x[secondary] == factor and base_filter(x) and epoch_filter(x)
            color = cmap.colors[i % 10]
        else:
            sample_filter = lambda x: base_filter(x) and epoch_filter(x)
            color = "white"

        marginalize_categorical(
            results,
            k,
            # x_min=v._min, x_max=v._max,
            # x_transform=x_transform, sample_filter=sample_filter, prefix=prefix,
            # include_scatter=True,
            hold=True,
            sample_filter=sample_filter,
            label=factor,
            color=color,
            setup_figure=i==0,
            #**kwargs,
        )

    y_max = np.max([row["score_alt"] for row in results if base_filter(row)])
    plt.ylim(0, math.ceil(y_max / 10) * 10)

    postfix = "" if secondary is None else f" by {secondary}"

    plt.title(f"{prefix}{k}{postfix}")

    if len(factors) > 1:
        plt.legend()

    plt.show()

def marginalize_(
        var_name,
        x_min=None, x_max=None, x_transform=lambda x: x, x_transform_inv=None,
        bandwidth=0.10, sample_filter=None, prefix="", postfix="", divisions=10,
        hold=False, label=None, color="white", include_scatter=False, include_min_max=False,
        setup_figure=True
):
    """
    Show score for given hyperparameter with all other hyperparameters set optimally.
    Designed to work when primary hyperparameter has been randomly sampled from a continious distribution.
    """

    def sample_function(data, weight, func):
        buffer = []
        # 1000 is better, but slower, for linear functions we could just weight the end result
        samples = []
        for _ in range(1000):
            sample = [
                x for x, w in zip(data, weight) if np.random.random() < w
            ]
            samples.append(len(sample))
            if len(sample) == 0:
                continue
            buffer.append(func(sample))
        return (np.mean(buffer), np.mean(samples)) if len(buffer) > 0 else (0.0, 0.0)

    if setup_figure:
        plt.figure(figsize=(5, 3))
        ax = plt.gca()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.grid(color='gray', alpha=0.2)
        plt.ylim(0)

    v = search_params[var_name]
    is_categorical = type(v) == Categorical

    # show datapoints
    if include_scatter:
        c = lerp(color, 'darkgrey', 0.50)
        if is_categorical:
            xs = ([row[var_name] for row in results if ((sample_filter is None) or sample_filter(row))])
            ys = [row['score_alt'] for row in results if ((sample_filter is None) or sample_filter(row))]
            data = sorted([(x, y) for x, y in zip(xs, ys)])
            xs = [x for x, y in data]
            ys = [y for x, y in data]
            plt.scatter(xs, ys, marker='.', alpha=0.25, color=c, linewidth=0)
        else:
            xs = [x_transform(row[var_name]) for row in results if ((sample_filter is None) or sample_filter(row))]
            if x_transform_inv is not None:
                xs = [x_transform_inv(x) for x in xs]
            ys = [row['score_alt'] for row in results if ((sample_filter is None) or sample_filter(row))]
            plt.scatter(xs, ys, marker='.', alpha=0.25, color=c, linewidth=0)

    ys_mean = []
    ys_max = []
    ys_median = []
    ys_count = []

    if is_categorical:
        xs = sorted(set(row[var_name] for row in results))
        for value in xs:
            # make sure we only look at data that is from (mostly) finished rows
            data = [row['score_alt'] for row in results if
                    row[var_name] == value and
                    ((sample_filter is None) or sample_filter(row))]

            if len(data) == 0:
                print(f"No data for value: {value}")
                data = [0]

            ys_mean.append(mean_of_top(data))
            ys_max.append(np.max(data))
            ys_median.append(np.mean(data))
            ys_count.append(len(data))
            if len(data) < 6:
                print(f"only {len(data)} values for {value}")
    else:

        x_min = x_transform(x_min or np.min([row[var_name] for row in results]))
        x_max = x_transform(x_max or np.max([row[var_name] for row in results]))
        xs = np.linspace(x_min, x_max, divisions)

        bandwidth = (x_max - x_min) * bandwidth

        for x in xs:

            window_start = x - bandwidth
            window_end = x + bandwidth

            # window_start <= x_transform(row[var_name]) <= window_end and

            weight_func = lambda x: math.exp(-0.5 * ((x / bandwidth) ** 2))

            # make sure we only look at data that is from (mostly) finished rows
            data = [float(row['score_alt']) for row in results if
                    ((sample_filter is None) or sample_filter(row))]
            weight = [weight_func(x_transform(row[var_name]) - x) for row in results if
                      ((sample_filter is None) or sample_filter(row))]

            value, sample_count = sample_function(data, weight, mean_of_top)
            ys_mean.append(value)
            ys_count.append(sample_count)
            if include_min_max:
                ys_max.append(sample_function(data, weight, np.max)[0])
                ys_median.append(sample_function(data, weight, np.mean)[0])

                # return ouf or log scale if wanted
        if x_transform_inv is not None:
            xs = [x_transform_inv(x) for x in xs]

    ys_mean = np.asarray(ys_mean)
    ys_count = np.asarray(ys_count)

    plt.xlabel(prefix + var_name + postfix)
    plt.ylabel("Score")
    if include_min_max:
        c = lerp(color, 'black', 0.25)
        adv_plot(xs, ys_max, ys_count, c=c)
        adv_plot(xs, ys_median, ys_count, c=c)

    adv_plot(xs, ys_mean, ys_count, c=color, label=label)

    if not hold:
        plt.show()


def load_hyper_parameter_search(path, table_cols: list = None, max_col_width: int = 20):

    results = []
    error_runs = []

    for key in range(300):
        try:
            # use rough weights to get game scores on a similar scale.
            result = read_combined_log(
                path,
                f"{key:04d}",
                subset=["Breakout", "SpaceInvaders", "CrazyClimber"],
                subset_weights=[0.1, 2.0, 0.5]
            )
            if result is None:
                continue
            results.append(result)
            print('.', end='', flush=True)
        except Exception as e:
            print('x', end='', flush=True)
            error_runs.append((None, e))
            pass

    print()

    if len(error_runs) > 0:
        print()
        print("Had issues loading the following runs:")
        for path, error in error_runs:
            print(" - ", path, error)
        print()

    print(f"Loaded {len(results)} results.")

    results.sort(key=lambda x: -x['score_alt'])

    def pad(s, width=max_col_width):
        s = s[:(width-1)]
        return " " * (width - len(s)) + s

    if table_cols is not None:

        col_widths = [min(len(col), max_col_width) for col in table_cols]

        print()
        print("-" * ((1 + max_col_width) * len(table_cols)))
        for col_name, col_width in zip(table_cols, col_widths):
            print(f"{pad(col_name, col_width)}", end='')
        print()
        print("-" * ((1 + max_col_width) * len(table_cols)))

        for row in results:
            for col_name, col_width in zip(table_cols, col_widths):
                value = row[col_name]
                if type(value) in [float, np.float64, np.float32]:
                    if col_name in ['score_alt', "final_epoch"]:
                        rounding = 0
                    else:
                        rounding = 6
                    value = f"{round(float(str(row[col_name])), rounding)}"
                else:
                    value = str(value)
                print(f"{pad(value, col_width)}", end='')
            print(" "+str(row["score_list"]), end='')
            print()

    print(f"Found {len(results)} results.")
    return results


import bisect


def plot_seeded_validation(path, key, seeds=3, color=None, style="-", label=None, subset="Atari_3_Val", ghost_alpha=0.15, check_seeds=False, print_results: bool=False):
    xs = range(51)  # epochs
    y_list = [[] for _ in xs]
    max_x = 0

    if label is None:
        label = key

    for seed in range(1, seeds + 1):
        result = read_combined_log(path, key, subset=subset, seed=seed)
        if result is None:
            if check_seeds:
                print(f"Missing seed {seed} for {key}")
            continue
        steps = np.asarray(result["env_step"], dtype=np.float32) / 1e6
        if check_seeds:
            if steps[-1] < 49.0:
                print(f"Seed {seed} not complete for {key} - {steps[-1]}")

        for i, y in zip(steps, result['score']):
            y_list[round(i)].append(y)
            max_x = max(round(i), max_x)
            # plot individual runs so we can check...
        plt.plot(steps * 4, result['score'], color=color, alpha=ghost_alpha, ls='--')

    y_list = y_list[:max_x + 1]
    xs = np.asarray(xs[:max_x + 1])

    def std_err(x):
        if len(x) == 0:
            return 0.0
        return np.std(x) / (len(x) ** 0.5)

    y_mean = np.asarray([np.mean(y) if len(y) > 0 else 0 for y in y_list])
    y_err = 1.0 * np.asarray([std_err(y) for y in y_list])
    plt.plot(xs * 4, y_mean, label=label, color=color, alpha=1.0, ls=style)
    plt.fill_between(xs * 4, y_mean - y_err, y_mean + y_err, color=color, alpha=0.15)

    if print_results:
        print(f"{key}: {y_mean[-1]:.2f} += {y_err[-1]:.2f}")


def setup_plot(title=None):
    plt.figure(figsize=(12, 4))
    plt.grid(True, alpha=0.25)
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #plt.title(title)


def experiment(title, path, keys: list="auto", subset='Atari_3_Val', seeds=5, hold=False, check_seeds=False, labels=None, ghost_alpha=0.0, key_filter=None):

    if type(key_filter) is str:
        mask = key_filter
        key_filter = lambda x: mask in x

    if keys == "auto":
        paths = glob(f"{path}/*/", recursive=False)
        keys = []
        for x in paths:
            try:
                run = " ".join(x.split("/")[-2].split(" ")[1:-2])
                if run in [""]:
                    continue
                if key_filter is not None and not key_filter(run):
                    continue
                keys.append(run+' ')
            except:
                pass
        keys = sorted(set(keys))

    setup_plot(title)
    for i, key in enumerate(keys):
        plot_seeded_validation(
            path,
            key,
            color=cm(i),
            subset=subset,
            seeds=seeds,
            check_seeds=check_seeds,
            label=labels[i] if labels is not None else None,
            ghost_alpha=ghost_alpha
        )
    plt.legend()
    plt.xlabel('Frame (M)')
    plt.ylabel('Score')
    ax = plt.gca()

    plt.xlim(0, 200)
    plt.title('')
    if not hold:
        plt.title(title)
        plt.plot()


asn = AtariScoreNormalizer()
cmap = plt.cm.get_cmap('tab10')
eval_cache = {}

cm = plt.cm.get_cmap('tab10')
cm20 = plt.cm.get_cmap('tab20')

ATARI_3_VAL = AtariScoreNormalizer.SUBSETS['Atari_3_Val'][0]
ATARI_5_VAL = AtariScoreNormalizer.SUBSETS['Atari_5_Val'][0]
ATARI_5 = AtariScoreNormalizer.SUBSETS['Atari_5'][0]
