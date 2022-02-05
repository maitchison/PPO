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
    sqr_value_samples: np.ndarray = None,
    n_step: int = 40,
    max_samples: int = 40,
    square: bool = False,
    masked: bool = False,
    log=None,
):
    if square:
        assert sqr_value_samples is not None, "Must include bootstrapped square value samples"

    N, A = rewards.shape
    samples = None
    weights = None

    args = {
        'gamma': gamma,
        'rewards': rewards,
        'dones': dones,
        'required_horizons': required_horizons,
        'value_sample_horizons': value_sample_horizons,
        'value_samples': value_samples,
        'sqr_value_samples': sqr_value_samples,
        'masked': masked,
    }

    # fixed is a special case
    if mode == "fixed":
        samples = [n_step]
    elif mode == "exponential":
        lamb = 1-(1/n_step)
        weights = np.asarray([lamb ** x for x in range(N)], dtype=np.float32)
    else:
        raise ValueError(f"Invalid returns mode {mode}")

    if samples is None:
        max_n = len(weights)
        probs = weights / weights.sum()
        samples = np.random.choice(range(1, max_n + 1), max_samples, replace=True, p=probs)

    if square:
        # stub for the moment verify the algorithm is working
        # if log is not None:
        #     random_h = np.random.randint(1, 10)
        #     r1 = _calculate_sampled_sqr_return_multi_reference(n_step_list=[random_h], **args)
        #     r2 = _calculate_sampled_sqr_return_multi(n_step_list=[random_h], **args)
        #     delta = np.abs(r1-r2)
        #     if delta.max() > 1e-6:
        #         log.warn(f"Error for h={random_h} was {delta.mean()}/{delta.max()}")

        # the fast version is almost there, but isn't quite right yet...
        return _calculate_sampled_sqr_return_multi_reference(n_step_list=samples, **args)
    else:
        return _calculate_sampled_return_multi(n_step_list=samples, **args)

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
    sqr_value_samples: np.ndarray, # ignored
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


def _calculate_sampled_sqr_return_multi_reference(
    n_step_list: list,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    sqr_value_samples: np.ndarray,
    n_step_weights: list = None,
    masked=False, # ignored
):
    """
    Very slow reference version of square return calculation
    """

    if n_step_weights is None:
        n_step_weights = [1] * len(n_step_list)

    N, A = rewards.shape
    K = len(required_horizons)

    returns = np.zeros([N, A, K], dtype=np.float32)

    for h_index, h in enumerate(required_horizons):

        for target_n_step, weight in zip(n_step_list, n_step_weights):

            # calculate the n_step squared return estimate
            for t in range(N):

                reward_sum = np.zeros([A], dtype=np.float32)
                discount = np.ones([A], dtype=np.float32)

                n_step = min(target_n_step, N-t, h)

                # calculate s part
                for i in range(n_step):
                    reward_sum += rewards[t+i]
                    discount *= gamma
                    discount *= (1 - dones[t+i])

                s = reward_sum

                if h-n_step > 0:
                    m = _interpolate(value_sample_horizons, value_samples[t+n_step], h-n_step)
                    m2 = _interpolate(value_sample_horizons, sqr_value_samples[t+n_step], h-n_step)
                    squared_estimate = s ** 2 + 2 * discount * s * m + (discount ** 2) * m2
                else:
                    # essentially a MC estimate
                    squared_estimate = s ** 2

                returns[t, :, h_index] += squared_estimate * weight

    return returns / np.sum(n_step_weights)


def _calculate_sampled_sqr_return_multi(
    n_step_list: list,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    sqr_value_samples: np.ndarray,
    n_step_weights: list = None,
    masked=False,
):
    """
    Calculate square returns for a list of n_steps values and weightings

    #param rewards: np array of dims [N, A]

    note: we break the reward sequence into to parts (r_1+r_2+...r_k) + (r_k+1, r_k+2, ... )
    We call these S and M
    then compute S^2 + \gamma^{2n} M^{(2)} + 2 \gamma^{n} S M
    S^2 is just the square of the reward sums (MC)
    M^2 is the (discounted) bootstrapped second moment
    2SM is a problem, for this I assume indepdendance, which is clearly not true

    For n_step >= h this reduces to the square of the MC estimate, which is correct. For longer h we might
    have problems.

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

    sqr_returns = np.zeros([N, A, K], dtype=np.float32)
    # S is defined as S_{k} sum_{i=0}^{k} gamma^i r_{t+i}, and is essentially just the cumulative reward, but capped
    # to some horizon k.
    S = np.zeros([N, A, K], dtype=np.float32) # partial reward sums
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
            for h in h_lookup.keys():
                if h >= current_n_step:
                    for i in h_lookup[h]:
                        S[:(N-current_n_step), :, i] += rewards[current_n_step:] * discount[:(N-current_n_step)]

            discount[:(N - current_n_step)] *= gamma
            discount[:(N - current_n_step)] *= (1 - dones[current_n_step:])

            current_n_step += 1

            # -----------------------------------
            # this is the S^2 part...

            # set short horizons as we go...
            for h in h_lookup.keys():
                # this is a bit dumb, but one horizon might be in the list twice, so we need to update
                # both indexes to it. This could happen with random sampling I guess?
                for i in h_lookup[h]:
                    if current_n_step < h and current_n_step == n_step:
                        sqr_returns[:, :, i] += (cumulative_rewards**2) * weight # this could be done in bulk...
                    elif current_n_step == h:
                        # this is the final one, give it all the remaining weight...
                        if masked:
                            weight_so_far = total_weight - remaining_weight
                            weight_to_go = np.clip(h_weights[:, :, 0] - weight_so_far, 0, float('inf'))
                            sqr_returns[:, :, i] += (cumulative_rewards**2) * weight_to_go
                        else:
                            sqr_returns[:, :, i] += (cumulative_rewards**2) * remaining_weight

        # -----------------------------------
        # this is the M^2 part...

        # # we can do most of this with one big update, however near the end of the rollout we need to account
        # # for the fact that we are actually using a shorter n-step
        steps_made = current_n_step
        block_size = 1 + N - current_n_step
        for h_index, h in enumerate(required_horizons):
            if h - steps_made <= 0:
                continue
            interpolated_sqr_value = _interpolate(
                value_sample_horizons,
                sqr_value_samples[steps_made:block_size + steps_made],
                h - steps_made
            )
            sqr_returns[:block_size, :, h_index] += interpolated_sqr_value * (discount[:block_size]**2) * weight

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
                        interpolated_sqr_value = _interpolate(value_sample_horizons, sqr_value_samples[t + steps_made], h - steps_made)
                        sqr_returns[t, :, h_index] += interpolated_sqr_value * (discount[t]**2) * weight

        # -----------------------------------
        # this is the 2SM part...

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
            sqr_returns[:block_size, :, h_index] += \
                2 * S[:block_size, :, h_index] * \
                interpolated_value * \
                discount[:block_size] * \
                weight
        # # next do the remaining few steps
        # # this is a bit slow for large n_step, but most of the estimates are low n_step anyway
        # # (this could be improved by simply ignoring these n-step estimates
        # # the problem with this is that it places *a lot* of emphasis on the final value estimate
        if not masked:
            for t in range(1 + N - current_n_step, N):
                steps_made = min(current_n_step, N - t)
                if t + steps_made > N:
                    continue
                for h_index, h in enumerate(required_horizons):
                    if h - steps_made <= 0:
                        continue
                    interpolated_value = _interpolate(
                        value_sample_horizons,
                        value_samples[t + steps_made],
                        h - steps_made
                    )
                    sqr_returns[t, :, h_index] += \
                        2 * S[t, :, h_index] * \
                        interpolated_value * \
                        discount[t] * \
                        weight

        remaining_weight -= weight

    return sqr_returns / h_weights
