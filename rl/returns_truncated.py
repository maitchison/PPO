"""
Library to help with calculating estimates for truncated returns.
"""

import numpy as np
from typing import Optional
from collections import defaultdict


def get_return_estimate(
        distribution: str,
        mode: str,
        gamma: float,
        rewards: np.ndarray,
        dones: np.ndarray,
        required_horizons: np.ndarray,
        value_sample_horizons: np.ndarray,
        value_samples: np.ndarray,
        n_step: int = 40,
        max_samples: int = 40,
        use_log_interpolation: bool = False,
        seed=None,
):
    """
    Very slow reference version of return calculation. Calculates a weighted average of multiple n_step returns

    @param distribution: [...]
    @param mode: [standard|advanced|full]
    @param gamma: discount factor
    @param rewards: float32 ndarray of dims [N, A] containing rewards
    @param dones: bool ndarray of dims [N, A] containing true iff terminal state.
    @param required_horizons: int32 ndarray of dims [K] containing list of horizons to calculate returns for
    @param value_sample_horizons: int32 ndarray of dims [V] containing list of horizons for which a value estimate was calculated
    @param value_samples: float32 ndarray of dims [N+1, A, V] bootstrap first moment estimates
    @param n_step: horizon to use for fixed / exponential estimator
    @param max_samples: maximum number of samples to use for the weighted average estimators

    returns
        E(r),                       (if only value_samples are provided)        

    Where return estimates are a float32 numpy array of dims [N, A, K]

    """

    N, A = rewards.shape
    K = len(required_horizons)

    def calc_return(samples: np.ndarray):
        return _calculate_sampled_return_multi_fast(
            n_step_samples=samples,
            gamma=gamma,
            rewards=rewards,
            dones=dones,
            required_horizons=required_horizons,
            value_sample_horizons=value_sample_horizons,
            value_samples=value_samples,
            use_log_interpolation=use_log_interpolation,
        )

    lamb = 1 - (1 / n_step)

    def get_weights(f):
        return np.asarray([f(n) for n in range(1, N + 1)], dtype=np.float32)

    if distribution == "fixed":
        # fixed is a special case
        samples = np.zeros([K, 1], dtype=np.int32) + n_step
        return calc_return(samples)
    elif distribution == "exponential":
        weights = get_weights(lambda x: lamb ** x)
    elif distribution == "uniform":
        weights = get_weights(lambda x: 1)
    elif distribution == "hyperbolic":
        weights = get_weights(lambda x: 1 / x)
    elif distribution == "quadratic":
        weights = get_weights(lambda x: 1 / (N + (x * x)))
    else:
        raise ValueError(f"Invalid distribution {distribution}")

    weights /= np.sum(weights)

    if seed is not None:
        np.random.seed(seed)

    if mode == "standard":
        # all horizons get same sample
        samples = np.random.choice(range(1, len(weights) + 1), size=(1, max_samples), replace=True, p=weights)
        samples = np.repeat(samples, K, axis=0)
        return calc_return(samples)
    elif mode == "advanced":
        # each horizon gets a random sample.
        samples = np.random.choice(range(1, len(weights) + 1), size=(K, max_samples), replace=True, p=weights)
        return calc_return(samples)
    elif mode == "clipped":
        # each horizon gets a random sample, but zero out n_steps larger than h.
        C = max_samples
        samples = np.zeros([K, C], dtype=np.int32)
        for k in range(K):
            max_h = max(required_horizons[k], 1)
            scaled_weights = weights.copy()
            scaled_weights[max_h:] = 0
            scaled_weights = scaled_weights / scaled_weights.sum()
            samples[k, :] = np.random.choice(range(1, len(weights) + 1), size=C, replace=True, p=scaled_weights)
        return calc_return(samples)
    elif mode == "adaptive":
        # each horizon gets a random sample, but shorter horizons get smaller n_step
        C = max_samples
        samples = np.zeros([K, C], dtype=np.int32)
        for k in range(K):
            # todo adjust weights properly
            # weights = get_weights(lambda x: lamb ** x)
            max_h = max(required_horizons[k] // 2, 1)
            scaled_weights = weights.copy()
            scaled_weights[max_h:] = 0
            scaled_weights = scaled_weights / scaled_weights.sum()
            samples[k, :] = np.random.choice(range(1, len(weights) + 1), size=C, replace=True, p=scaled_weights)
        return calc_return(samples)
    elif mode == "mcx":
        # MC up to n_step*2 then exponential
        C = max_samples
        samples = np.zeros([K, C], dtype=np.int32)
        for k in range(K):
            # todo adjust weights properly
            # weights = get_weights(lambda x: lamb ** x)
            if required_horizons[k] <= 2*n_step:
                samples[k, :] = required_horizons[k]
            else:
                samples[k, :] = np.random.choice(range(1, len(weights) + 1), size=C, replace=True, p=weights)
        return calc_return(samples)
    elif mode == "full":
        # calculate each horizon and do a weighted average
        # very slow...
        returns = np.zeros([N, A, K], dtype=np.float32)
        for n_step, weight in zip(range(1, N + 1), weights):
            n_step_samples = np.zeros([K, 1], dtype=np.int32) + n_step
            returns += calc_return(n_step_samples) * weight
        return returns
    else:
        raise ValueError(f"Invalid return mode {mode}")


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
        n_step_weights: list = None,
        masked=False,
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
    cumulative_rewards = np.zeros_like(rewards, dtype=np.float32)
    discount = np.ones_like(rewards, dtype=np.float32)

    if masked:
        h_weights = []
        n_step_copy = n_step_list.copy()
        n_step_weight_copy = n_step_weights.copy()
        weight = total_weight
        for n in range(N):
            while len(n_step_copy) > 0 and n_step_copy[-1] + n > N:
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
            cumulative_rewards[:(N - current_n_step)] += rewards[current_n_step:] * discount[:(N - current_n_step)]
            discount[:(N - current_n_step)] *= gamma
            discount[:(N - current_n_step)] *= (1 - dones[current_n_step:])

            current_n_step += 1

            # set short horizons as we go...
            for h in h_lookup.keys():
                # this is a bit dumb, but one horizon might be in the list twice, so we need to update
                # both indexes to it. This could happen with random sampling I guess?
                for i in h_lookup[h]:
                    if current_n_step < h and current_n_step == n_step:
                        returns[:, :, i] += cumulative_rewards * weight  # this could be done in bulk...
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
                        interpolated_value = _interpolate(value_sample_horizons, value_samples[t + steps_made],
                                                          h - steps_made)
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
        use_log_interpolation: bool = False, # ignored
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

    assert value_samples.shape == (N + 1, A, V)

    # this allows us to map to our 'sparse' returns table
    h_multi_lookup = {}
    for index, h in enumerate(required_horizons):
        if h not in h_multi_lookup:
            h_multi_lookup[h] = [index]
        else:
            h_multi_lookup[h].append(index)

    # maps from h, to the first instance of h in our array
    h_lookup = {k: v[0] for k, v in h_multi_lookup.items()}

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
            if h - steps_made <= 0:
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


def reweigh_samples(n_step_list: list, weights=None, max_n: Optional[int] = None):
    """
    Takes a list of n_step lengths, and (optinaly) their weights, and returns a new list of samples / weights
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
        gamma: float,
        rewards: np.ndarray,
        dones: np.ndarray,
        required_horizons: np.ndarray,
        value_sample_horizons: np.ndarray,
        value_samples: np.ndarray,
        value_samples_m2: np.ndarray = None,
        value_samples_m3: np.ndarray = None,
        n_step_weights: list = None,
        n_step_samples: np.ndarray = None,
        n_step_list: list = None,
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
    @param n_step_samples: ndarray if dims [K, C] (use list or samples, but not both)

    returns
        E(r),                       (if only value_samples are provided)
        (E(r), E(r^2)),             (if value_samples, value_samples_m2 are provided)
        (E(r), E(r^2), E(r^3))      (if value_samples, value_samples_m2, value_samples_m3 are provided)

    Where return estimates are a float32 numpy array of dims [N, A, K]

    """

    assert n_step_list is None or n_step_samples is None
    assert n_step_list is not None or n_step_samples is not None

    if n_step_samples is not None:
        # this reference method does not support per horizon samples, so calculate each horizon individually
        K, C = n_step_samples.shape
        assert value_samples_m2 is None
        assert value_samples_m3 is None
        N, A = rewards.shape
        K = len(required_horizons)
        result = np.zeros([N, A, K], dtype=np.float32)
        for k in range(K):
            result_part = _calculate_sampled_return_multi_reference(
                gamma=gamma, rewards=rewards, dones=dones,
                required_horizons=np.asarray([required_horizons[k]]),
                value_sample_horizons=value_sample_horizons, value_samples=value_samples, n_step_weights=n_step_weights,
                n_step_list=n_step_samples[k, :],
            )
            result[:, :, k:k + 1] = result_part
        return result

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

                n_step = min(target_n_step, N - t, h)

                # calculate s part
                for i in range(n_step):
                    reward_sum += rewards[t + i] * discount
                    discount *= gamma
                    discount *= (1 - dones[t + i])

                s = reward_sum
                m = 0
                m2 = 0
                m3 = 0

                if h - n_step > 0:
                    m = _interpolate(value_sample_horizons, value_samples[t + n_step], h - n_step) * discount
                    if moment in [2, 3]:
                        m2 = _interpolate(value_sample_horizons, value_samples_m2[t + n_step], h - n_step) * (
                                    discount ** 2)
                    if moment == 3:
                        m3 = _interpolate(value_sample_horizons, value_samples_m3[t + n_step], h - n_step) * (
                                    discount ** 3)
                else:
                    # essentially a MC estimate
                    pass

                returns_m1[t, :, h_index] += (s + m) * weight
                if moment in [2, 3]:
                    returns_m2[t, :, h_index] += (s ** 2 + 2 * s * m + m2) * weight
                if moment == 3:
                    returns_m3[t, :, h_index] += (s ** 3 + 3 * (s ** 2) * m + 3 * s * m2 + m3) * weight

    if moment == 1:
        return returns_m1
    elif moment == 2:
        return returns_m1, returns_m2
    else:
        return returns_m1, returns_m2, returns_m3


def _n_step_estimate(job, rewards, value_sample_horizons, value_samples, discount_cache, reward_cache, use_log_interpolation):
    """
    Processes rewards [N, A] and value_estimates [N+1, A, K] into returns [N, A]
    """

    target_n_steps, idx, target_h = job

    log_value_sample_horizons = np.log10(10 + value_sample_horizons) - 1

    def interpolate_linear(value_estimates: np.ndarray, horizon: int):
        return _interpolate(value_sample_horizons, value_estimates, horizon)
        # not faster, and matches identically with my interpolate.
        # from scipy import interpolate as sp_interpolate
        # return sp_interpolate.interp1d(value_sample_horizons, value_estimates)(horizon)

    def interpolate_log(value_estimates: np.ndarray, horizon: int):
        return _interpolate(log_value_sample_horizons, value_estimates, np.log10(10 + horizon) - 1)

    interpolate = interpolate_log if use_log_interpolation else interpolate_linear

    N, A = rewards.shape

    if target_h == 0:
        # happens sometimes, always return 0.
        zeros = np.zeros([N, A], dtype=np.float32)
        return idx, target_h, zeros

    returns = np.zeros_like(rewards, dtype=np.float32)
    m = np.zeros([N, A], dtype=np.float32)

    for target_n_step in target_n_steps:

        if target_n_step > target_h:
            target_n_step = target_h

        if target_h == 0:
            # return is 0 by definition
            continue

        assert 1 <= target_n_step <= N

        # much quicker to use the cached version of these
        s = reward_cache[target_n_step]
        discount = discount_cache[target_n_step]

        # add the bootstrap estimate
        h_remaining = target_h - target_n_step

        m *= 0

        if h_remaining > 0:
            m[:N - target_n_step] = interpolate(value_samples[target_n_step:-1], h_remaining)

        # process the final n_step estimates, this can be slow for large target_n_step
        for i in range(target_n_step):
            # small chance we have a bug here, need to test this...
            m[N - i - 1] = interpolate(value_samples[-1], target_h - i - 1)

        returns += s + m * discount

    returns *= 1 / len(target_n_steps)

    return idx, target_h, returns


def _calculate_sampled_return_multi_fast(
        gamma: float,
        rewards: np.ndarray,
        dones: np.ndarray,
        required_horizons: np.ndarray,
        value_sample_horizons: np.ndarray,
        value_samples: np.ndarray,
        n_step_samples: np.ndarray = None,
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

    # just to be safe..
    for k, v in discount_cache.items():
        v.setflags(write=False)
    for k, v in reward_cache.items():
        v.setflags(write=False)

    result = [_n_step_estimate(
        job,rewards, value_sample_horizons, value_samples, discount_cache, reward_cache, use_log_interpolation
    ) for job in jobs]

    all_results = np.zeros([N, A, K], dtype=np.float32)
    for idx, h, returns in result:
        all_results[:, :, idx] = returns

    # just make sure there are no duplicates here...
    # as we were adding them before. Should be fine, and can delete.
    ids = ([x[0] for x in result])
    assert len(ids) == len(set(ids))

    return all_results
