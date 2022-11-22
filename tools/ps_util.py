import numpy as np
import matplotlib.pyplot as plt
from tools import plot_util as pu

PS_TEMPS = [1.0, 0.5, 0.25, 0.125, 0.0]
EPOCHS = np.arange(0, 50+1, 5)
VERBOSE = True

cache = {}
cm = plt.cm.get_cmap('tab10')
eval_paths = {}

def safe(f, X):
    """
    returns f(X), if X contains any Nones, retuns None
    """
    if X is None:
        return None
    try:
        X = np.asarray(X, dtype=np.float64)
        if np.any(np.isnan(X)):
            return None
        else:
            return f(X)
    except Exception as e:
        print(f"warning, error {e} on {X}")
        return None


def get_scores(game:str, mode:str, epoch:int, temperature:float=1.0, seed:int=1, norm:bool=False):
    """
    Get the score for given game at given temperature and given epoch
    If results do not exist returns None
    """

    assert type(temperature) in [int, float]

    if seed is None:
        seed = 1

    key = (game, mode, epoch, temperature, seed, norm)
    if key in cache:
        return cache[key]
    path = eval_paths[(game, mode)]
    results = pu.load_eval_epoch(path, epoch, temperature, seed)
    if results is None:
        if VERBOSE:
            print(f"Warning, missing data for {game} {epoch} {temperature} {seed}")
        return None
    else:
        cache[key] = pu.asn.normalize(game, results["episode_scores"]) if norm else results["episode_scores"]
        return cache[key]

def get_lengths(game:str, mode:str, epoch:int, temperature:float=1.0, seed:int=1):
    """
    Get the score for given game at given temperature and given epoch
    If results do not exist returns None
    """

    assert type(temperature) in [int, float]

    if seed is None:
        seed = 1

    key = (game, mode, epoch, temperature, seed, "length")
    if key in cache:
        return cache[key]
    path = eval_paths[(game, mode)]
    results = pu.load_eval_epoch(path, epoch, temperature, seed)
    if results is None:
        if VERBOSE:
            print(f"Warning, missing data for {game} {epoch} {temperature} {seed}")
        return None
    else:
        cache[key] = results["episode_lengths"]
        return cache[key]


def get_score_mean(game:str, mode:str, epoch:int, temperature:float=1.0, seed:int=1, norm:bool=False):
    """
    Get the score for given game at given temperature and given epoch
    If results do not exist returns None
    """
    return safe(np.mean, get_scores(game, mode, epoch, temperature, seed, norm))


def get_score_error(game:str, mode: str, epoch:int, temperature=1.0, seed:int=1, norm:bool=False):
    """
    Get the score for given game at given temperature and given epoch
    If results do not exist returns None
    """
    scores = get_scores(game, mode, epoch, temperature, seed, norm)
    if scores is None:
        return None
    else:
        return np.std(scores) / (len(scores) ** 0.5)

def get_result(game:str, mode: str, epoch:int, temperature_code:str, seed=None, norm=False):
    """
    Gets temperature score, but using temperature codes.
    If result does not exist returns None

    ps: policy sharpening
    max_*: Maximum result
    argmax_*: Temperature of max result
    best_*: Reevaluation of best temperature

    """

    # t=0 is argmax, t=1 is normal, negative temps are policy blended.
    if type(temperature_code) is str:

        parts = temperature_code.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid temperature code {temperature_code}")

        if parts[1] == "ps":
            temps = PS_TEMPS
        else:
            raise ValueError(f"Invalid temperature code {temperature_code}")

        if parts[0] == "max":
            return safe(np.max, [get_score_mean(game, mode, epoch, t, norm=norm) for t in temps])
        elif parts[0] == "argmax":
            argmax = safe(np.argmax, [get_score_mean(game, mode, epoch, t, norm=norm) for t in temps])
            return None if argmax is None else temps[argmax]
        elif parts[0] == "best":
            argmax = safe(np.argmax, [get_score_mean(game, mode, epoch, t, norm=norm) for t in temps])
            if argmax is None:
                return None
            return get_score_mean(game, mode, epoch, temps[argmax], seed=1337, norm=norm)
        else:
            raise ValueError(f"Invalid temperature code {temperature_code}")

    if type(temperature_code) not in [int, float]:
        raise ValueError(f"Invalid temperature code {temperature_code}")

    return get_score_mean(game, mode, epoch, float(temperature_code), seed=seed, norm=norm)