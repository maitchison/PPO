from tools.plot_util import *


def gamma_plot(path: str, color, key:str, label=None, seeds=5, subset='Atari_3_Val', x_offset=0):
    # show scores for td vs gae returns
    gammas = ["0.99", "0.999", "0.9999", "0.99997", "1.0"]
    keys = []
    for gamma in gammas:
        keys.append(f'{key} {gamma} ')

    xs = range(50)  # epochs
    y_list = [[] for _ in xs]
    max_x = 0

    if label is None:
        label = key

    plot_xs = []
    xs = [str(x) for x in gammas]
    ys = []
    errs = []

    for i, gae_lambda in enumerate(xs):
        key = keys[i]
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
            print(f"Warning, not enough data for {key}, found {len(scores)} seeds but wanted {seeds}.")
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
