import numpy as np
import time as clock
import bisect
import math
from multiprocessing import Pool
from typing import Union, Optional
from collections import defaultdict
from .logger import Logger

# memory sharing between workers for efficent return estimation
GLOBAL_CACHE = None

def get_return_estimate(
    mode: str,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    value_samples_m2: np.ndarray = None,
    value_samples_m3: np.ndarray = None,
    n_step: int = 40,
    max_samples: int = 40,
    max_h: Optional[int] = None,
    estimator_mode:str = "default",
    log:Logger = None,
    use_log_interpolation: bool=False,
):
    """
    Very slow reference version of return calculation. Calculates a weighted average of multiple n_step returns

    @param mode: [fixed|exponential]
    @param gamma: discount factor
    @param rewards: float32 ndarray of dims [N, A] containing rewards
    @param dones: bool ndarray of dims [N, A] containing true iff terminal state.
    @param required_horizons: int32 ndarray of dims [K] containing list of horizons to calculate returns for
    @param value_sample_horizons: int32 ndarray of dims [V] containing list of horizons for which a value estimate was calculated
    @param value_samples: float32 ndarray of dims [N+1, A, V] bootstrap first moment estimates
    @param n_step: horizon to use for fixed / exponential estimator
    @param max_samples: maximum number of samples to use for the weighted average estimators
    @param estimator_mode: default|reference|verify
    @param log: logger for when verify is used

    returns
        E(r),                       (if only value_samples are provided)
        (E(r), E(r^2)),             (if value_samples, value_samples_m2 are provided)
        (E(r), E(r^2), E(r^3))      (if value_samples, value_samples_m2, value_samples_m3 are provided)

    Where return estimates are a float32 numpy array of dims [N, A, K]

    """

    N, A = rewards.shape
    K = len(required_horizons)
    samples = None
    weights = None

    args = {
        'gamma': gamma,
        'rewards': rewards,
        'dones': dones,
        'required_horizons': required_horizons,
        'value_sample_horizons': value_sample_horizons,
        'value_samples': value_samples,
        'use_log_interpolation': use_log_interpolation,
    }

    # fixed is a special case
    if mode == "fixed":
        n_step_samples = np.zeros([K, 1], dtype=np.int32) + n_step
        return _calculate_sampled_return_multi_threaded(n_step_samples=n_step_samples, **args)
    elif mode == "advanced":
        # the idea here is that each agent and horizon gets a different set of n-step estimates

        # get our distribution
        lamb = 1-(1/n_step)
        weights = np.asarray([lamb ** x for x in range(N)], dtype=np.float32)
        weights /= np.sum(weights)

        n_step_samples = np.random.choice(range(1, N + 1), size=(K, max_samples), replace=True, p=weights)
        return _calculate_sampled_return_multi_threaded(n_step_samples=n_step_samples, **args)
    elif mode == "advanced_uniform":
        # the idea here is that each agent and horizon gets a different set of n-step estimates

        # get our distribution
        weights = np.asarray([1 for x in range(N)], dtype=np.float32)
        weights /= np.sum(weights)

        n_step_samples = np.random.choice(range(1, N + 1), size=(K, max_samples), replace=True, p=weights)

        return _calculate_sampled_return_multi_threaded(n_step_samples=n_step_samples, **args)
    elif mode == "advanced_hyperbolic":
        # the idea here is that each agent and horizon gets a different set of n-step estimates

        # get our distribution
        weights = np.asarray([1 / x for x in range(1, N + 1)], dtype=np.float32)
        weights /= np.sum(weights)

        n_step_samples = np.random.choice(range(1, N + 1), size=(K, max_samples), replace=True, p=weights)

        return _calculate_sampled_return_multi_threaded(n_step_samples=n_step_samples, **args)

    elif mode == "adaptive":
        # we do this by repeated calling exponential, which can be a bit slow...
        # note: we cut the curve off after h steps, otherwise we end up repeating the same
        # value estimate. That is R_{>h}=R_{h}
        result = np.zeros([N, A, K], dtype=np.float32)
        target_n_step = n_step
        for h_index, h in enumerate(required_horizons):
            args_copy = args.copy()
            args_copy['required_horizons'] = [h]
            args_copy['max_samples'] = max_samples
            args_copy['estimator_mode'] = estimator_mode
            args_copy['log'] = log
            n_step = int(np.clip(h, 1, 0.5*target_n_step))  # the magic heuristic...
            result[:, :, h_index] = get_return_estimate(
                mode="exponential",
                n_step=n_step,
                max_h=max(h, 1), **args_copy
            )[:, :, 0]
        return result
    elif mode == "exponential":
        # this just makes things a bit faster for small n_step horizons
        # essentially we ignore the very rare, very long n_steps.
        if max_h is None:
            max_h = min(n_step * 3, N)
        lamb = 1-(1/n_step)
        weights = np.asarray([lamb ** x for x in range(max_h)], dtype=np.float32)
    elif mode == "hyperbolic":
        max_h = N
        k = n_step / 10
        weights = np.asarray([1 / (1 + k * (x / max_h)) for x in range(max_h)], dtype=np.float32)
    elif mode == "quadratic":
        max_h = N
        k = n_step / 10
        weights = np.asarray([1 / (1 + k * (x / max_h)**2) for x in range(max_h)], dtype=np.float32)
    elif mode == "uniform":
        max_h = N
        weights = np.asarray([1.0 for _ in range(max_h)], dtype=np.float32)
    elif mode == "exponential_cap":
        # this is exponential where the weight not used all falls on the final n_step estimate.
        # this can improve performance by demphasising the short n-step return estimators.
        max_h = min(n_step * 3, N)
        lamb = 1-(1/n_step)
        weights = np.asarray([lamb ** x for x in range(max_h)], dtype=np.float32)
        remaining_weight = (lamb ** max_h) / (1 - lamb)
        weights[-1] += remaining_weight
    else:
        raise ValueError(f"Invalid returns mode {mode}")

    raise Exception("This return mode is not supported yet.")




def _get_adaptive_args(mode: str, h:int, c:float):
    """
    Returns the adaptive settings for horizon of length h, and parameter c
    """
    if mode == "fixed":
        return {'n_step': math.ceil(h / c)}
    elif mode == "uniform":
        return {'n_step': math.ceil(2 * h / c)}
    elif mode == "linear":
        return {'n_step': math.ceil(4 * h / c)}
    elif mode == "exponential":
        return {'lamb': 1-(1/math.ceil(h / c))}


def _interpolate(horizons, values, target_horizon: float):
    """
    Returns linearly interpolated value from source_values

    horizons: sorted ndarray of shape[K] of horizons, must be in *strictly* ascending order
    values: ndarray of shape [*shape, K] where values[...,h] corresponds to horizon horizons[h]
    target_horizon: the horizon we would like to know the interpolated value of

    """

    if target_horizon <= 0:
        # by definition value of a 0 horizon is 0.
        return values[..., 0] * 0

    index = horizons.searchsorted(target_horizon)

    if horizons[index] == target_horizon:
        # easy, exact match
        return values[..., index].copy()

    if index == 0:
        return values[..., 0].copy()
    if index == len(horizons):
        return values[..., -1].copy()
    value_pre = values[..., index - 1]
    value_post = values[..., index]
    dx = (horizons[index] - horizons[index - 1])
    if dx == 0:
        # this happens if there are repeated values, in this case just take leftmost result
        return value_pre.copy()
    factor = (target_horizon - horizons[index - 1]) / dx
    assert factor >= 0
    assert factor <= 1.0
    return value_pre * (1 - factor) + value_post * factor


def _calculate_sampled_return_reference(n_step_list: list, **kwargs):
    """
    Slow version of multi, using a loop. Used to check correctness.
    """
    returns = _calculate_sampled_return(n_step_list[0], **kwargs)
    for n in n_step_list[1:]:
        returns += _calculate_sampled_return(n, **kwargs)
    returns /= len(n_step_list)
    return returns


def _calculate_sampled_return_multi(
    n_step_list: list,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    value_samples_m2: np.ndarray, # ignored
    n_step_weights: list = None,
    masked=False,
    use_log_interpolation: bool=False # ignored
):
    """
    Calculate returns for a list of n_steps values and weightings

    @masked: handles behaviour when an n_step is required that is larger than given the number of remaining
        rewards. If true these will be ignored, if false, the largest n_step possible will be substituted.
        Having masked enabled seems to cause the agent to underperform, and may contain bugs.
    """

    # the idea here is to progressively update a running total of the n-step rewards, then apply bootstrapping
    # quickly in a vectorized way. Unlike the previous algorithm, this is O(n) rather then O(n^2) where n is n_steps
    # I also take advantage of the fact that nstep(n`,h) = nstep(n,h) for all n`>=n. This allows me to eliminate
    # many of the horizon updates once we get into the high n_steps.

    # if we have n_step requests that exceed the longest horizon we can cap these to the longest horizon.
    max_h = max(required_horizons)
    max_n = len(rewards)
    n_step_list = np.clip(n_step_list, 1, max_h)
    n_step_list = np.clip(n_step_list, 1, max_n)

    # calculate the weight for each n_step
    if n_step_weights is None:
        n_step_weights = [1.0 for _ in n_step_list]
    n_step_weight = defaultdict(float)
    for n, weight in zip(n_step_list, n_step_weights):
        n_step_weight[n] += weight

    # remove any duplicates (these will be handled by the weight calculation)
    n_step_list = list(set(n_step_list))
    n_step_list.sort()

    N, A = rewards.shape
    K = len(required_horizons)

    total_weight = sum(n_step_weight.values())
    remaining_weight = total_weight

    # this allows us to map to our 'sparse' returns table
    h_lookup = {}
    for index, h in enumerate(required_horizons):
        if h not in h_lookup:
            h_lookup[h] = [index]
        else:
            h_lookup[h].append(index)

    returns = np.zeros([N, A, K], dtype=np.float32)
    cumulative_rewards = np.zeros_like(rewards)
    discount = np.ones_like(rewards)

    if masked:
        h_weights = []
        n_step_copy = n_step_list.copy()
        n_step_weight_copy = n_step_weights.copy()
        weight = total_weight
        for n in range(N):
            while len(n_step_copy) > 0 and n_step_copy[-1]+n > N:
                n_step_copy.pop()
                weight -= n_step_weight_copy.pop()
            h_weights.append(weight)
        h_weights = np.asarray(h_weights, dtype=np.float32)[:, None, None]
    else:
        h_weights = np.asarray([total_weight for _ in range(N)], dtype=np.float32)[:, None, None]

    current_n_step = 0

    for n_step in n_step_list:
        weight = n_step_weight[n_step]
        # step 1, update our cumulative reward count
        while current_n_step < n_step:
            cumulative_rewards[:(N-current_n_step)] += rewards[current_n_step:] * discount[:(N-current_n_step)]
            discount[:(N-current_n_step)] *= gamma
            discount[:(N-current_n_step)] *= (1-dones[current_n_step:])

            current_n_step += 1

            # set short horizons as we go...
            for h in h_lookup.keys():
                # this is a bit dumb, but one horizon might be in the list twice, so we need to update
                # both indexes to it. This could happen with random sampling I guess?
                for i in h_lookup[h]:
                    if current_n_step < h and current_n_step == n_step:
                        returns[:, :, i] += cumulative_rewards * weight # this could be done in bulk...
                    elif current_n_step == h:
                        # this is the final one, give it all the remaining weight...
                        if masked:
                            weight_so_far = total_weight - remaining_weight
                            weight_to_go = np.clip(h_weights[:, :, 0] - weight_so_far, 0, float('inf'))
                            returns[:, :, i] += cumulative_rewards * weight_to_go
                        else:
                            returns[:, :, i] += cumulative_rewards * remaining_weight

        # we can do most of this with one big update, however near the end of the rollout we need to account
        # for the fact that we are actually using a shorter n-step
        steps_made = current_n_step
        block_size = 1 + N - current_n_step
        for h_index, h in enumerate(required_horizons):
            if h - steps_made <= 0:
                continue
            interpolated_value = _interpolate(
                value_sample_horizons,
                value_samples[steps_made:block_size + steps_made],
                h - steps_made
            )
            returns[:block_size, :, h_index] += interpolated_value * discount[:block_size] * weight

        # next do the remaining few steps
        # this is a bit slow for large n_step, but most of the estimates are low n_step anyway
        # (this could be improved by simply ignoring these n-step estimates
        # the problem with this is that it places *a lot* of emphasis on the final value estimate
        if not masked:
            for t in range(1 + N - current_n_step, N):
                steps_made = min(current_n_step, N - t)
                if t + steps_made > N:
                    continue
                for h_index, h in enumerate(required_horizons):
                    if h - steps_made > 0:
                        interpolated_value = _interpolate(value_sample_horizons, value_samples[t + steps_made], h - steps_made)
                        returns[t, :, h_index] += interpolated_value * discount[t] * weight

        remaining_weight -= weight

    return returns / h_weights


def _calculate_sampled_return(
        n_step: int,
        gamma: float,
        rewards: np.ndarray,
        dones: np.ndarray,
        required_horizons: np.ndarray,
        value_sample_horizons: np.ndarray,
        value_samples: np.ndarray,
        sqr_value_samples: np.ndarray, # not used
        masked:bool = False,
    ):
    """
    This is a fancy n-step sampled returns calculation

    gamma: discount to use
    reward: nd array of dims [N, A]
    dones: nd array of dims [N, A]
    required_horizons: nd array of dims [K]
    value_samples: nd array of dims [N, A, K], where value_samples[n, a, v] is the value of the nth timestep ath agent
        for horizon value_sample_horizons[v]
    n_step: n-step to use in calculation
    masked: ignored (used in calculate_sampled_return_multi)

    If n_step td_lambda is negative it is taken as
    """

    N, A = rewards.shape
    K = len(required_horizons)
    V = len(value_sample_horizons)

    assert value_samples.shape == (N+1, A, V)

    # this allows us to map to our 'sparse' returns table
    h_multi_lookup = {}
    for index, h in enumerate(required_horizons):
        if h not in h_multi_lookup:
            h_multi_lookup[h] = [index]
        else:
            h_multi_lookup[h].append(index)

    # maps from h, to the first instance of h in our array
    h_lookup = {k:v[0] for k,v in h_multi_lookup.items()}

    returns = np.zeros([N, A, K], dtype=np.float32)

    # generate return estimates using n-step returns
    for t in range(N):

        # first collect the rewards
        discount = np.ones([A], dtype=np.float32)
        reward_sum = np.zeros([A], dtype=np.float32)
        steps_made = 0

        for n in range(1, n_step + 1):
            if (t + n - 1) >= N:
                break
            # n_step is longer than horizon required
            # if n >= current_horizon:
            #     break
            this_reward = rewards[t + n - 1, :]
            reward_sum += discount * this_reward
            discount *= gamma * (1 - dones[t + n - 1, :])
            steps_made += 1

            # the first n_step returns are just the discounted rewards, no bootstrap estimates...
            if n in h_lookup:
                returns[t, :, h_lookup[n]] = reward_sum

        for h_index, h in enumerate(required_horizons):
            if h-steps_made <= 0:
                # these are just the accumulated sums and don't need horizon bootstrapping
                continue
            interpolated_value = _interpolate(value_sample_horizons, value_samples[t + steps_made, :], h - steps_made)
            returns[t, :, h_index] = reward_sum + interpolated_value * discount

    # this is required if there are duplicates in the h_lookup, i.e. if we want to calculate
    # returns for horizons [1,2,3,4,4,4]
    for k, v in h_multi_lookup.items():
        base = v[0]
        for duplicate in v[1:]:
            returns[:, :, duplicate] = returns[:, :, base]

    return returns

def _get_geometric_n_steps(rho:float, max_n: int):
    """
    Returns a list of horizons spaced out exponentially for given horizon.
    In some cases horizons might be duplicated (in which case they should have extra weighting)
    """
    results = []
    current_h = 1
    while True:
        if current_h > max_n:
            break
        results.append(round(current_h))
        current_h *= rho
    return results

# ------------------------------------------------------------------------------
# Second moment estimation
# ------------------------------------------------------------------------------

def calculate_second_moment_estimate_td(
        rewards:np.ndarray,
        first_moment_estimates: np.ndarray,
        second_moment_estimates: np.ndarray,
        gamma: float
):
    """
    rewards: ndarray of dims [N, A]
    first_moment_estimates: ndarray of dims [N+1, A]
    second_moment_estimates: ndarray of dims [N+1, A]
    gamma: float

    returns second moment return estimates as ndarray of dims [N, A]
    """

    # based on https://jmlr.org/papers/volume17/14-335/14-335.pdf

    # note: the second term assumes E(r*R_{t+1}) = E(r*E(R_{t+1})), which is clearly not true
    # if 1-step rewards are often 0, this term will have minimal effect, and it seems to work despite the
    # inaccuracy.

    # note: one way to get around this would be to use a MC estimate instead for the middle term.

    return rewards**2 + 2*gamma*rewards * first_moment_estimates[1:] + (gamma**2)*second_moment_estimates[1:]


def reweigh_samples(n_step_list:list, weights=None, max_n: Optional[int] = None):
    """
    Takes a list of n_step lengths, and (optinally) their weights, and returns a new list of samples / weights
    such that duplicate n_steps have had their weight added together.
    I.e. reweigh_samples = [1,1,1,2] -> ([1,2], [1:3,2:1])
    """

    n_step_list = np.clip(n_step_list, 1, max_n)
    n_step_list = list(n_step_list)

    if weights is None:
        weights = [1.0 for _ in n_step_list]

    n_step_weight = defaultdict(float)
    for n, weight in zip(n_step_list, weights):
        n_step_weight[n] += weight

    # remove any duplicates (these will be handled by the weight calculation)
    n_step_list = list(set(n_step_list))
    n_step_list.sort()
    return n_step_list, n_step_weight


def _calculate_sampled_return_multi_reference(
    n_step_list: list,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    value_samples_m2: np.ndarray=None,
    value_samples_m3: np.ndarray=None,
    n_step_weights: list = None,
    use_log_interpolation: bool=False # ignored
):
    """
    Very slow reference version of return calculation. Calculates a weighted average of multiple n_step returns

    @param n_step_list: a list of n_step estimates to use in weighted average
    @param gamma: discount factor
    @param rewards: float32 ndarray of dims [N, A] containing rewards
    @param dones: bool ndarray of dims [N, A] containing true iff terminal state.
    @param required_horizons: int32 ndarray of dims [K] containing list of horizons to calculate returns for
    @param value_sample_horizons: int32 ndarray of dims [V] containing list of horizons for which a value estimate was calculated
    @param value_samples: float32 ndarray of dims [N+1, A, V] bootstrap first moment estimates
    @param value_samples_m2: float32 ndarray of dims [N+1, A, V] bootstrap second moment estimates
    @param value_samples_m3: float32 ndarray of dims [N+1, A, V] bootstrap third moment estimates
    @param n_step_weights: list of weights corresponding to n_step_list, if not given defaults to a uniform weighting

    returns
        E(r),                       (if only value_samples are provided)
        (E(r), E(r^2)),             (if value_samples, value_samples_m2 are provided)
        (E(r), E(r^2), E(r^3))      (if value_samples, value_samples_m2, value_samples_m3 are provided)

    Where return estimates are a float32 numpy array of dims [N, A, K]

    """

    n_step_list, n_step_weights = reweigh_samples(n_step_list, n_step_weights)

    N, A = rewards.shape
    K = len(required_horizons)

    returns_m1 = np.zeros([N, A, K], dtype=np.float32)
    returns_m2 = np.zeros([N, A, K], dtype=np.float32)
    returns_m3 = np.zeros([N, A, K], dtype=np.float32)

    total_weight = sum(n_step_weights.values())

    moment = 1
    if value_samples_m2 is not None:
        moment = 2
    if value_samples_m3 is not None:
        moment = 3

    if moment == 3:
        assert value_samples_m2 is not None, "Third moment requires second moment values."

    for h_index, h in enumerate(required_horizons):

        for target_n_step in n_step_list:
            weight = n_step_weights[target_n_step] / total_weight

            # calculate the n_step squared return estimate
            for t in range(N):

                reward_sum = np.zeros([A], dtype=np.float32)
                discount = np.ones([A], dtype=np.float32)

                n_step = min(target_n_step, N-t, h)

                # calculate s part
                for i in range(n_step):
                    reward_sum += rewards[t+i] * discount
                    discount *= gamma
                    discount *= (1 - dones[t+i])

                s = reward_sum
                m = 0
                m2 = 0
                m3 = 0

                if h-n_step > 0:
                    m = _interpolate(value_sample_horizons, value_samples[t+n_step], h-n_step) * discount
                    if moment in [2, 3]:
                        m2 = _interpolate(value_sample_horizons, value_samples_m2[t+n_step], h-n_step) * (discount**2)
                    if moment == 3:
                        m3 = _interpolate(value_sample_horizons, value_samples_m3[t+n_step], h-n_step) * (discount**3)
                else:
                    # essentially a MC estimate
                    pass

                returns_m1[t, :, h_index] += (s + m) * weight
                if moment in [2, 3]:
                    returns_m2[t, :, h_index] += (s**2 + 2*s*m + m2) * weight
                if moment == 3:
                    returns_m3[t, :, h_index] += (s**3 + 3*(s**2)*m + 3*s*m2 + m3) * weight

    if moment == 1:
        return returns_m1
    elif moment == 2:
        return returns_m1, returns_m2
    else:
        return returns_m1, returns_m2, returns_m3

def _batch_n_step_estimate(params: list):
    result = None
    for param in params:
        if result is None:
            result = _n_step_estimate(param)
        else:
            result += _n_step_estimate(param)
    return result

def _n_step_estimate(params):
    """
    Processes rewards [N, A] and value_estimates [N+1, A, K] into returns [N, A]
    """

    (rewards, gamma, value_sample_horizons, value_samples, dones, discount_cache, reward_cache, log_interpolation) = GLOBAL_CACHE

    target_n_steps, idx, target_h = params

    log_value_sample_horizons = np.log10(10+value_sample_horizons)-1

    def interpolate_linear(value_estimates: np.ndarray, horizon: int):
        return _interpolate(value_sample_horizons, value_estimates, horizon)
        # not faster, and matches identically with my interpolate.
        # from scipy import interpolate as sp_interpolate
        # return sp_interpolate.interp1d(value_sample_horizons, value_estimates)(horizon)

    def interpolate_log(value_estimates: np.ndarray, horizon: int):
        return _interpolate(log_value_sample_horizons, value_estimates, np.log10(10+horizon)-1)

    interpolate = interpolate_log if log_interpolation else interpolate_linear

    N, A = rewards.shape

    if target_h == 0:
        # happens sometimes, always return 0.
        zeros = np.zeros([N, A], dtype=np.float32)
        return idx, target_h, zeros

    returns = np.zeros_like(rewards)
    m = np.zeros([N, A], dtype=np.float32)

    for target_n_step in target_n_steps:

        if target_n_step > target_h:
            target_n_step = target_h

        if target_n_step == 0:
            # return is 0 by definition
            continue

        # much quicker to use the cached version of these
        s = reward_cache[target_n_step]
        discount = discount_cache[target_n_step]

        # add the bootstrap estimate
        h_remaining = target_h - target_n_step

        if h_remaining > 0:
            m[:N-target_n_step] = interpolate(value_samples[target_n_step:-1], h_remaining)
        else:
            m *= 0

        # process the final n_step estimates, this can be slow for large target_n_step
        for i in range(target_n_step):
            # small chance we have a bug here, need to test this...
            m[N-i-1] = interpolate(value_samples[-1], target_h - i - 1)

        returns += s + m * discount

    returns *= 1/len(target_n_steps)

    return idx, target_h, returns


def _n_step_estimate_median(params):
    """
    Processes rewards [N, A] and value_estimates [N+1, A, K] into returns [N, A]
    Uses median over samples instead of mean.
    """

    (rewards, gamma, value_sample_horizons, value_samples, dones, discount_cache, reward_cache, log_interpolation) = GLOBAL_CACHE

    target_n_steps, idx, target_h = params

    log_value_sample_horizons = np.log10(10+value_sample_horizons)-1

    def interpolate_linear(value_estimates: np.ndarray, horizon: int):
        return _interpolate(value_sample_horizons, value_estimates, horizon)
        # not faster, and matches identically with my interpolate.
        # from scipy import interpolate as sp_interpolate
        # return sp_interpolate.interp1d(value_sample_horizons, value_estimates)(horizon)

    def interpolate_log(value_estimates: np.ndarray, horizon: int):
        return _interpolate(log_value_sample_horizons, value_estimates, np.log10(10+horizon)-1)

    interpolate = interpolate_log if log_interpolation else interpolate_linear

    N, A = rewards.shape

    if target_h == 0:
        # happens sometimes, always return 0.
        zeros = np.zeros([N, A], dtype=np.float32)
        return idx, target_h, zeros

    returns = np.zeros([N, A, len(target_n_steps)], dtype=np.float32)
    m = np.zeros([N, A], dtype=np.float32)

    for i, target_n_step in enumerate(target_n_steps):

        if target_n_step > target_h:
            target_n_step = target_h

        if target_n_step == 0:
            # return is 0 by definition
            continue

        # much quicker to use the cached version of these
        s = reward_cache[target_n_step]
        discount = discount_cache[target_n_step]

        # add the bootstrap estimate
        h_remaining = target_h - target_n_step

        if h_remaining > 0:
            m[:N-target_n_step] = interpolate(value_samples[target_n_step:-1], h_remaining)
        else:
            m *= 0

        # process the final n_step estimates, this can be slow for large target_n_step
        for j in range(target_n_step):
            # small chance we have a bug here, need to test this...
            m[N-j-1] = interpolate(value_samples[-1], target_h - j - 1)

        returns[:, :, i] = s + m * discount

    returns = np.median(returns, axis=-1)

    return idx, target_h, returns


def _calculate_sampled_return_multi_threaded(
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    n_step_samples: np.ndarray=None,
    n_step_list=None,
    use_log_interpolation: bool = False,
):
    """
    New strategy, just generate these returns in parallel, and simplify the algorithm *alot*.

    @param n_step_samples: nd array of dims [K, C] with C samples for each of K horizons.
    """

    assert n_step_samples is not None or n_step_list is not None

    N, A = rewards.shape
    K = len(required_horizons)

    if n_step_list is not None:
        n_step_samples = np.asarray([list(n_step_list) for _ in range(K)])

    jobs = []
    n_step_set = set(n_step_samples.ravel())

    # this is the per horizon n_step version
    C = n_step_samples.shape[1]
    assert n_step_samples.shape == (K, C)
    for i, h in enumerate(required_horizons):
        jobs.append((n_step_samples[i], i, h))

    # build our discount / reward sum cache...
    # quicker to do the needed ones as we go
    # add up the discounted rewards,
    # that is s[n, a] = sum_{t=n}^{n+n_step} r_t
    required_horizons_set = set(required_horizons)
    s = np.zeros([N, A], dtype=np.float32)
    discount = np.ones([N, A], dtype=np.float32)
    discount_cache = {}
    reward_cache = {}

    for i in range(max(n_step_set)):
        s[:N - i] += rewards[i:] * discount[:N - i]
        discount[:N - i] *= gamma * (1 - dones[i:])
        if i + 1 in n_step_set or i + 1 in required_horizons_set:
            reward_cache[i + 1] = s.copy()
            discount_cache[i + 1] = discount.copy()

    # simple way to get this data to all the workers without having to pickle it (which would be too slow)
    global GLOBAL_CACHE
    GLOBAL_CACHE = (rewards, gamma, value_sample_horizons, value_samples, dones, discount_cache, reward_cache, use_log_interpolation)

    # turns out it's faster to just run the jobs (at least for 8 or less samples).
    result = [_n_step_estimate(job) for job in jobs]

    all_results = np.zeros([N, A, K], dtype=np.float32)
    for idx, h, returns in result:
        all_results[:, :, idx] = returns

    # just make sure there are no duplicates here...
    # as were were adding them before. Should be fine, and can delte.
    ids = ([x[0] for x in result])
    assert len(ids) == len(set(ids))


    return all_results


def test_return_estimators(seed=123):
    """
    Run a sequence of tests to make sure return estimators are correct.
    """

    st0 = np.random.get_state()
    np.random.seed(seed)

    # create random data...
    # k are horizons we need to generate
    # v are horizons we have estimates for
    N, A, K, V = [128, 16, 16, 8]
    default_args = {
        'gamma': 0.9997,
        'rewards': np.random.random_integers(-1, 2, [N, A]).astype('float32'),
        'dones': (np.random.random_integers(0, 100, [N, A]) > 98),
        'required_horizons': np.geomspace(1, 1024, num=K).astype('int32'),
        'value_sample_horizons': np.geomspace(1, 1024+1, num=V).astype('int32')-1,
        'value_samples': np.random.normal(0.1, 0.4, [N+1, A, V]).astype('float32'),
    }

    default_args['value_samples'][:, :, 0] *= 0 # h=0 must be zero

    def verify(label: str, **kwargs):

        args = default_args.copy()
        args.update(kwargs)

        start_time = clock.time()
        m1_ref = _calculate_sampled_return_multi_reference(**args)
        r1_time = clock.time() - start_time

        start_time = clock.time()
        m1 = _calculate_sampled_return_multi_threaded(**args)
        r2_time = clock.time() - start_time

        delta_m1 = np.abs(m1_ref - m1)

        e_m1 = delta_m1.max()

        ratio = r1_time / r2_time

        # note fp32 has about 7 sg fig, so rel error of around 1e-6 is expected.
        if e_m1 > 1e-5:
            print(f"Times {r1_time:.2f}s / {r2_time:.2f}s ({ratio:.1f}x), error for {label} = {e_m1:.6f}")
            return False
        return True

    n_step = 8
    eff_h = min(n_step * 3, N)
    lamb = 1 - (1 / n_step)
    weights = np.asarray([lamb ** x for x in range(eff_h)], dtype=np.float32)
    max_n = len(weights)
    probs = weights / weights.sum()
    samples = np.random.choice(range(1, max_n + 1), 80, replace=True, p=probs)

    if not verify("n_step=8", n_step_list=[8]) or \
       not verify("exponential=8", n_step_list=samples):
        raise Exception("Return estimator does not match reference.")

    np.random.set_state(st0)

