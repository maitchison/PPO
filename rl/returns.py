import numpy as np
import time
import bisect
import math
from collections import defaultdict

def get_return_estimate(
    mode: str,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    c: float = 6,
    n_step: int = 40,
    lamb: float = 0.95,
    max_samples:int = 40,
    rho: float = 1.5,
    adaptive: bool = False,
    masked: bool = False
):

    args = {
        'gamma': gamma,
        'rewards': rewards,
        'dones': dones,
        'required_horizons': required_horizons,
        'value_sample_horizons': value_sample_horizons,
        'value_samples': value_samples,
        'masked': masked,
    }

    N, A = rewards.shape
    K = len(required_horizons)

    if adaptive:
        # if adaptive works well I
        assert mode != "geometric", "Geometric does not support adaptive mode."
        returns = np.zeros([N, A, K], dtype=np.float32)
        # this h_map allows us to work out each horizon once, but then duplicate the result into multiple
        # slots, which speeds up performance if there are duplicates in the required horizons (which happens
        # a fair bit)
        h_map = defaultdict(list)
        for i, h in enumerate(required_horizons):
            h_map[h].append(i)
        # note: adaptive mode potentially could collate horizons that have the same parameters,
        # and could also notice when horizons are duplicated
        for h, indexes in h_map.items():
            if h == 0:
                continue
            args['required_horizons'] = [h]
            result = get_return_estimate(
                mode=mode,
                **args,
                **_get_adaptive_args(mode, h, c),
                adaptive=False
            )[:, :, 0]
            for i in indexes:
                returns[:, :, i] = result
        return returns

    if mode == "fixed":
        return _calculate_sampled_return(n_step=n_step, **args)
    elif mode == "uniform":
        weights = np.ones(n_step)
        return get_weighted_sample_estimate(n_samples=max_samples, weighting=weights, **args)
    elif mode == "linear":
        weights = 1 - (np.arange(0, n_step) / n_step)
        return get_weighted_sample_estimate(n_samples=max_samples, weighting=weights, **args)
    elif mode == "exponential":
        weights = np.asarray([lamb ** x for x in range(N)], dtype=np.float32)
        return get_weighted_sample_estimate(n_samples=max_samples, weighting=weights, **args)
    elif mode == "geometric":
        # this was the old 'exponential' method, that actually seems to work quite well
        # maybe because it weights smaller n_steps more highly...
        return _calculate_sampled_return_multi(
            n_step_list=_get_geometric_n_steps(rho, N),
            **args
        )
    elif mode == "test":
        # this is just a check to make sure multi is working properly
        start_time = time.time()
        return1 = _calculate_sampled_return_multi(
            n_step_list=_get_geometric_n_steps(rho, N),
            **args
        )
        time1 = time.time() - start_time

        start_time = time.time()
        return2 = _calculate_sampled_return_slow(
            n_step_list=_get_geometric_n_steps(rho, N),
            **args
        )
        time2 = time.time() - start_time
        scale = np.std(return1)
        dif = np.abs(return1-return2) / scale

        print(f"{np.max(dif):.6f}/{np.mean(dif):.6f} {time1:.1f}s vs {time2:.1f}s")
        return return2
    else:
        raise ValueError(f"Invalid returns mode {mode}")

def get_weighted_sample_estimate(n_samples:int, weighting: np.ndarray, **kwargs):
    """
    Generates a return estimate from by taking n_samples from the given weighting curve, where
    distribution is a np.ndarray of length N and distribution(n) is the weight of the (n+1)'th n_step
    """
    max_n = len(weighting)
    probs = weighting / weighting.sum()
    samples = np.random.choice(range(1, max_n + 1), n_samples, replace=True, p=probs)
    return _calculate_sampled_return_multi(n_step_list=samples, **kwargs)


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


def _interpolate(horizons, values, target_horizon: int):
    """
    Returns linearly interpolated value from source_values

    horizons: sorted ndarray of shape[K] of horizons, must be in *strictly* ascending order
    values: ndarray of shape [*shape, K] where values[...,h] corresponds to horizon horizons[h]
    target_horizon: the horizon we would like to know the interpolated value of

    """

    if target_horizon <= 0:
        # by definition value of a 0 horizon is 0.
        return values[..., 0] * 0

    index = bisect.bisect_left(horizons, target_horizon)
    if index == 0:
        return values[..., 0]
    if index == len(horizons):
        return values[..., -1]
    value_pre = values[..., index - 1]
    value_post = values[..., index]
    dx = (horizons[index] - horizons[index - 1])
    if dx == 0:
        # this happens if there are repeated values, in this case just take leftmost result
        return value_pre
    factor = (target_horizon - horizons[index - 1]) / dx
    return value_pre * (1 - factor) + value_post * factor


def _calculate_sampled_return_slow(n_step_list: list, **kwargs):
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
        n_step:int,
        gamma:float,
        rewards: np.ndarray,
        dones: np.ndarray,
        required_horizons: np.ndarray,
        value_sample_horizons: np.ndarray,
        value_samples: np.ndarray,
        masked:bool = False,
    ):
    """
    This is a fancy n-step sampled returns calculation

    gamma: discount to use
    reward: nd array of dims [N, A]
    dones: nd array of dims [N, A]
    required_horizons: nd array of dims [K]
    value_samples: nd array of dims [N, A, K], where value_samples[n, a, k] is the value of the nth timestep ath agent
        for horizon required_horizons[k]
    n_step: n-step to use in calculation
    masked: ignored (used in calculate_sampled_return_multi)

    If n_step td_lambda is negative it is taken as
    """

    N, A = rewards.shape
    K = len(required_horizons)

    # this allows us to map to our 'sparse' returns table
    h_lookup = {}
    for index, h in enumerate(required_horizons):
        if h not in h_lookup:
            h_lookup[h] = [index]
        else:
            h_lookup[h].append(index)

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
