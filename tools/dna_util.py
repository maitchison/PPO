"""
Helper functions for the DNA paper

"""

from .ps_util import *
from .plot_util import *
from . import plot_util
from tools.runner_tools import ATARI_57
from bisect import bisect_left

A5_SCORES = {
    # 'DQN': 89.9,
    # 'A3C': 92.7,
    # 'A3C (LSTM)': 129.6,
    'Rainbow': 225.8,
}

ref_results = {
    ("BattleZone", "Human"): 37187,
    ("BattleZone", "DQN"): 29900,
    ("BattleZone", "Rainbow"): 62010,
    ("BattleZone", "A3C FF"): 12950,
    ("DoubleDunk", "Human"): -16.4,
    ("DoubleDunk", "DQN"): -6.6,
    ("DoubleDunk", "Rainbow"): -0.3,
    ("DoubleDunk", "A3C FF"): -0.1,
    ("NameThisGame", "Human"): 8049,
    ("NameThisGame", "DQN"): 8207,
    ("NameThisGame", "Rainbow"): 13136,
    ("NameThisGame", "A3C FF"): 10476,
    ("Phoenix", "Human"): 7242,
    ("Phoenix", "DQN"): 8485,
    ("Phoenix", "Rainbow"): 108528,
    ("Phoenix", "A3C FF"): 52894,
    ("Qbert", "Human"): 13455,
    ("Qbert", "DQN"): 13117,
    ("Qbert", "Rainbow"): 33817,
    ("Qbert", "A3C FF"): 15148
}

# add in all results for Rainbow DQN
ref_results.update({
    ("Alien", "Rainbow"): 9491.7,
    ("Amidar", "Rainbow"): 5131.2,
    ("Assault", "Rainbow"): 14198.5,
    ("Asterix", "Rainbow"): 428200.3,
    ("Asteroids", "Rainbow"): 2712.8,
    ("Atlantis", "Rainbow"): 826659.5,
    ("BankHeist", "Rainbow"): 1358,
    ("BattleZone", "Rainbow"): 62010,
    ("BeamRider", "Rainbow"): 16850.2,
    ("Berzerk", "Rainbow"): 2545.6,
    ("Bowling", "Rainbow"):	30,
    ("Boxing", "Rainbow"): 99.6,
    ("Breakout", "Rainbow"): 417.5,
    ("Centipede", "Rainbow"): 8167.3,
    ("ChopperCommand", "Rainbow"):	16654,
    ("CrazyClimber", "Rainbow"): 168788.5,
    ("Defender", "Rainbow"): 55105,
    ("DemonAttack", "Rainbow"): 111185.2,
    ("Double", "Rainbow"): -0.3,
    ("Enduro", "Rainbow"): 2125.9,
    ("FishingDerby", "Rainbow"): 31.3,
    ("Freeway", "Rainbow"): 34,
    ("Frostbite", "Rainbow"): 9590.5,
    ("Gopher", "Rainbow"): 70354.6,
    ("Gravitar", "Rainbow"): 1419.3,
    ("Hero", "Rainbow"): 55887.4,
    ("IceHockey", "Rainbow"): 1.1,
    ("Jamesbond", "Rainbow"): 19480, # read off plot
    ("Kangaroo", "Rainbow"): 14637.5,
    ("Krull", "Rainbow"): 8741.5,
    ("KungFuMaster", "Rainbow"): 52181,
    ("MontezumaRevenge", "Rainbow"): 384,
    ("MsPacman", "Rainbow"): 5380.4,
    ("NameThisGame", "Rainbow"): 13136,
    ("Phoenix", "Rainbow"): 108528.6,
    ("Pitfall", "Rainbow"): 0,
    ("Pong", "Rainbow"): 20.9,
    ("PrivateEye", "Rainbow"): 4234,
    ("QBert", "Rainbow"): 33817.5,
    ("RoadRunner", "Rainbow"): 62041,
    ("Riverraid", "Rainbow"): 22500,
    ("Robotank", "Rainbow"): 61.4,
    ("Seaquest", "Rainbow"): 15898.9,
    ("Skiing", "Rainbow"): -12957.8,
    ("Solaris", "Rainbow"): 3560.3,
    ("SpaceInvaders", "Rainbow"): 18789,
    ("StarGunner", "Rainbow"): 127029,
    ("Surround", "Rainbow"): 9.7,
    ("Tennis", "Rainbow"): 0,
    ("TimePilot", "Rainbow"): 12926,
    ("Tutankham", "Rainbow"): 241,
    ("Venture", "Rainbow"):	5.5,
    ("VideoPinball", "Rainbow"): 533936.5,
    ("WizardOfWor", "Rainbow"): 17862.5,
    ("YarsRevenge", "Rainbow"): 102557,
    ("Zaxxon", "Rainbow"): 22209.5,
    ("UpNDown", "Rainbow"): 103600,

})


def plot_main_result(path, keys, labels, print_results=False):
    plt.figure(figsize=(6, 3.5))
    for i in range(len(keys)):
        key = keys[i]
        label = labels[i]
        color = cm(i)
        plot_seeded_validation(
            path=path,
            key=key,
            label=label,
            color=color,
            subset="Atari_5",
            seeds=3,
            check_seeds=True,
            ghost_alpha=0,  # cleaner
            print_results=print_results,
        )
    plt.grid(alpha=0.25)
    plt.legend()
    plt.ylabel("Score (Atari-5)")
    plt.xlabel("Step (M)")


def per_game_plots_atari_5(path, keys, ref_scores=None):
    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False, squeeze=True, figsize=(3 * 2.25, 2 * 1.5))

    fig.delaxes(axs[-1][-1])

    for env, ax in zip(ATARI_5, axs.ravel()):
        plt.sca(ax)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for i, key in enumerate(keys):
            color = cm(i)
            plot_experiment(
                path,
                show_legend=False,
                smooth_factor=0.99,
                run_filter=lambda x: env in x and key in x,
                y_axes=['ep_score_mean'],
                title=env,
                figsize=None,
                hold=True,
                group_filter=lambda x, _: " ".join(x.split(' ')[:-2]),  # remove seed and code
                x_axis_name='',
                y_axis_name='',
                x_transform=lambda x: x * 4,
                color_filter=lambda x, default: color
            )
        if ref_scores is not None:
            # random_score, human_score = asn._normalization_scores[env.lower()]
            ref_score = ref_scores[env]

            ax.hlines(ref_score, 0, 200, color="black", ls="--")
    fig.tight_layout(pad=0.15)


def table_results(path: str, runs: list, labels:list=None, subset=ATARI_5, reference_results=None):
    print()

    if labels is None:
        labels = [None for _ in range(len(runs))]

    results = {}
    if reference_results is not None:
        results.update(ref_results)

    logs = plot_util.get_runs(path)
    asn = AtariScoreNormalizer()

    for env in subset:

        # add human and random
        results[(env, "Random")] = asn.random[env.lower()]
        results[(env, "Human")] = asn.human[env.lower()]

        # add our results
        for log_name, log, params in logs:
            if env not in log_name:
                continue
            run_name = " ".join(log_name.split(" ")[1:-2])  # remove seed and code.
            if run_name not in runs:
                continue
            # print(env, run_name)
            epochs = log['env_step'][-1] / 1e6
            if epochs < 49.9:
                print(f"Warning {log_name} has not finished training {epochs:.1f}")
            # print(log['ep_score_mean'][-10:])
            samples = max(1, math.ceil(len(log['ep_score_mean']) * 0.05))
            final_score = np.mean(log['ep_score_mean'][-samples:])
            key = (env, run_name)
            if key not in results:
                results[key] = []
            results[key].append(final_score)

    print("\\toprule")

    print(f"{'Game':<20}", end=' ')
    for run, label in zip(runs, labels):
        print(f"& {(label or run):<10}", end=' ')
    print("\\\\")
    print()

    print("\\midrule")

    for env in subset:
        print(f"{env:<20}", end=' ')
        max_score = max(np.mean(v) for k, v in results.items() if k[0] == env and k[1] in runs)
        for run in runs:
            key = (env, run)
            if key in results:
                score = np.mean(results[key])
                is_best = abs(score - max_score) < 1e-6
                if score < 100:
                    score = f"{score:.1f}"
                else:
                    score = f"{round(score):,d}"
                if is_best:
                    score = "\\textbf{" + score + "}"
            else:
                score = "?"
            print(f"& {score:<10}", end=' ')
        print('\\\\')

    print("\\bottomrule")


def returns_plot(path: str, td_lambda: float, color, label=None, seeds=5, subset='Atari_3_Val', x_offset=0):
    # show scores for td vs gae returns
    keys = []
    for gae_lambda in [0.6, 0.8, 0.9, 0.95, 0.975]:
        for td_lambda in [td_lambda]:
            keys.append(f'td_lambda={td_lambda} gae_lambda={gae_lambda}')

    xs = range(50)  # epochs
    y_list = [[] for _ in xs]
    max_x = 0

    if label == None:
        label = f"TD_LAMBDA={td_lambda}"

    if label is None:
        label = key

    plot_xs = []
    xs = [str(x) for x in [0.6, 0.8, 0.9, 0.95, 0.975]]
    ys = []
    errs = []

    for i, gae_lambda in enumerate(xs):
        key = f'td_lambda={td_lambda} gae_lambda={gae_lambda}'
        scores = []
        for seed in range(1, seeds + 1):
            result = read_combined_log(path, key, subset=subset, seed=seed)
            if result is None:
                print(f"no data for {key} {seed}")
                continue
            epoch_up_to = result["env_step"][-1] / 1e6 + 1.0
            if epoch_up_to < 49.9:
                print(f"Warning, training not complete for {key} ({epoch_up_to:.1f})")
            scores.append(result['score'][-1])
        if len(scores) < seeds:
            print(f"Warning, not enough data for {key}")
        if len(scores) == 0:
            ys.append(0)
            continue

        x = str(gae_lambda)
        y = np.mean(scores)
        err = np.std(scores) / (len(scores) ** 0.5)
        errs.append(err)
        x_jitter = (np.random.rand(len(scores)) - 0.5) * 0.1
        # plt.scatter(i + x_jitter, scores, color=color, marker='x', alpha=0.33)
        # plt.errorbar(i+x_offset, y, err, color=color, marker='.', capsize=3.0)
        plot_xs.append(i + x_offset)
        ys.append(y)

    ys = np.asarray(ys)
    errs = np.asarray(errs)

    # show highest point
    #plt.scatter(np.argmax(ys), max(ys), marker='.', color=color, size=4)

    plt.xticks(range(len(xs)), xs)
    plt.plot(plot_xs, ys, label=label, color=color)
    plt.fill_between(plot_xs, ys - errs, ys + errs, color=color, alpha=0.15)


def get_game_score(data, game, epoch=50, normed=False, smoothing_epochs=0):
    game = game.lower()
    if game not in data:
        return None
    epoch_index = bisect_left(data[game]["env_step"], epoch * 1e6)
    normed_scores = data[game]["ep_score_norm" if normed else "ep_score_mean"]
    entries = len(normed_scores)
    if smoothing_epochs > 0:
        samples = max(1, math.ceil(entries * (1 / 50)))  # last epochs.
    else:
        samples = 1

    if epoch_index == len(normed_scores):
        score = normed_scores[-samples:].mean()
    else:
        score = normed_scores[max(0, epoch_index - samples):epoch_index].mean()
    return score


# Plot median score during training
def get_median_score(data, epoch=50, smoothing_epochs=0):
    scores = []
    for game in ATARI_57:
        scores.append(get_game_score(data, game, epoch, normed=True, smoothing_epochs=smoothing_epochs))
    scores = np.asarray(scores)
    scores = scores[scores != None]

    if len(scores) < len(ATARI_57):
        raise Exception("Missing scores.")
    return np.median(scores)

# Plot median score during training
def get_hlp_score(data, epoch=50, smoothing_epochs=0):
    scores = []
    for game in ATARI_57:
        scores.append(get_game_score(data, game, epoch, normed=True, smoothing_epochs=smoothing_epochs))
    scores = np.asarray(scores)
    scores = scores[scores != None]
    if len(scores) < len(ATARI_57):
        raise Exception("Missing scores.")
    return np.sum(scores >= 100)


# Plot median score during training
def get_atari5_score(data, epoch=50):
    scores = {}
    for game in asn.subset('Atari_5'):
        score = get_game_score(data, game, epoch)
        if score is None:
            print(f"Warning, missing score for {game}")
            score = 0
        scores[game] = score
    if epoch == 50:
        print(scores)
    return asn.subset_score(scores, 'Atari_5')

def median_plot(data, colors=None, labels=None):

    if colors is None:
        colors = [cmap(i) for i in range(len(data))]
    if labels is None:
        labels = list(data.keys())

    plt.figure(figsize=(8, 6))

    xs = np.linspace(0, 50, 1000)[1:]  # skip epoch 0, as no data.

    for i, (mode, mode_data) in enumerate(data.items()):
        ys = [get_median_score(mode_data, x) for x in xs]
        plt.plot(xs*4, ys, alpha=0.25, color=colors[i])
        plt.plot(xs*4, smooth(ys, 0.9), label=labels[i], color=colors[i])
        final_score = np.mean(ys[-math.ceil(len(ys)*0.05):])
        print(f"{mode} - score: {final_score:.1f}")

    plt.xlabel("Frames (M)")
    plt.ylabel("Atari-57 Human-Normalized Median Score")
    plt.legend()
    plt.grid(True, alpha=0.25)

def load_a57_data(path, modes, verbose=True):
    # load evaluation runs
    data = {}
    for mode in modes:
        data[mode] = {
            params["environment"].lower(): data
                for name, data, params
                in pu.get_runs(path, run_filter=lambda x: f" {mode} " in x)
        }
        if verbose:
            print(f"{path}: Found {len(data[mode])} runs for {mode}.")
    return data


def per_game_plots_atari_57(path, keys):
    fig, axs = plt.subplots(nrows=12, ncols=5, sharex=True, sharey=False, squeeze=False, figsize=(8 * 1.5, 12 * 1.5))

    for j, (env, ax) in enumerate(zip(sorted(ATARI_57), axs.ravel())):
        plt.sca(ax)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        for i, key in enumerate(keys):
            color = cm(i)
            plot_experiment(
                path,
                show_legend=False,
                smooth_factor=0.99,
                run_filter=lambda x: env in x and key in x and "(1)" in x,  # only 1 seed.
                y_axes=['ep_score_mean'],
                title=env,
                figsize=None,
                hold=True,
                # group_filter=lambda x, _ : " ".join(x.split(' ')[:-2]),  # remove seed and code
                x_start=5,  # remove 0 score at begining of some games
                x_axis_name='',
                y_axis_name='',
                x_transform=lambda x: x * 4,
                color_filter=lambda x, default: color
            )
            plt.title(env)
            # random_score, human_score = asn._normalization_scores[env.lower()]
            # ax.hlines(human_score, 0, 200, color="black", ls="-.-")

    fig.tight_layout(pad=0.20)

    # add legend and position correctly.
    from matplotlib.lines import Line2D
    plt.sca(axs[0][0])  # must use an axis that will not be deleted
    labels = ["DNA", "PPO (2x)"]
    lines = [
        Line2D([0], [0], color=cmap(0)),
        Line2D([0], [0], color=cmap(2)),
        #   Line2D([0], [0], color='black', ls='-.-'),
    ]
    fig.legend(lines, labels, bbox_to_anchor=(0.97, 0.05))

    # remove unused axes
    fig.delaxes(axs[11][4])
    fig.delaxes(axs[11][3])
    fig.delaxes(axs[11][2])
