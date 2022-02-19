import numpy as np
import time as clock
import bisect
import math
from collections import defaultdict
from .logger import Logger


def test_return_estimators(log=None):
    """
    Run a sequence of tests to make sure return estimators are correct.
    """

    st0 = np.random.get_state()
    np.random.seed(123)

    # create random data...
    N, A, K, V = [128, 64, 16, 16]
    default_args = {
        'gamma': 0.997,
        'rewards': np.random.random_integers(0, 2, [N, A]).astype('float32'),
        'dones': (np.random.random_integers(0, 100, [N, A]) > 95),
        'required_horizons': np.geomspace(1, 1024, num=K).astype('int32'),
        'value_sample_horizons': np.geomspace(1, 1024+1, num=V).astype('int32')-1,
        'value_samples': np.random.normal(0.1, 0.4, [N+1, A, V]).astype('float32'),
        'value_samples_m2': np.random.normal(0.1, 1.0, [N+1, A, V]).astype('float32') ** 2,
    }

    default_args['value_samples'][:, :, 0] *= 0 # h=0 must be zero
    default_args['value_samples_m2'][:, :, 0] *= 0  # h=0 must be zero

    m3 = np.random.normal(0.1, 0.4, [N+1, A, V]) ** 3,

    def verify(label: str, **kwargs):

        args = default_args.copy()
        args.update(kwargs)

        start_time = clock.time()
        m1_ref, m2_ref = _calculate_sampled_return_multi_reference(**args)
        r1_time = clock.time() - start_time

        start_time = clock.time()
        m1, m2 = _calculate_sampled_return_multi_fast(**args)
        r2_time = clock.time() - start_time

        delta_m1 = np.abs(m1_ref - m1)
        delta_m2 = np.abs(m2_ref - m2)

        e_m1 = delta_m1.max()
        e_m2 = delta_m2.max()

        ratio = r1_time / r2_time

        # note fp32 has about 6 sg fig, so rel error of around 1e-6 is expected.

        print(f"Times {r1_time:.2f}s / {r2_time:.2f}s ({ratio:.1f}x), error for {label} = {e_m1:.6f}/{e_m2:.6f}")

    n_step = 8
    eff_h = min(n_step * 3, N)
    lamb = 1 - (1 / n_step)
    weights = np.asarray([lamb ** x for x in range(eff_h)], dtype=np.float32)
    max_n = len(weights)
    probs = weights / weights.sum()
    samples = np.random.choice(range(1, max_n + 1), 40, replace=True, p=probs)

    verify("n_step=8", n_step_list=[8])
    verify("exponential=8", n_step_list=samples)

    np.random.set_state(st0)


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
    @param value_samples_m2: float32 ndarray of dims [N+1, A, V] bootstrap second moment estimates
    @param value_samples_m3: float32 ndarray of dims [N+1, A, V] bootstrap third moment estimates
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
        'value_samples_m2': value_samples_m2,
        'value_samples_m3': value_samples_m3,
        'use_log_interpolation': use_log_interpolation,
    }

    # fixed is a special case
    if mode == "fixed":
        samples = [n_step]
    elif mode == "adaptive":
        # we do this by repeated calling exponential, which can be a bit slow...
        result = np.zeros([N, A, K], dtype=np.float32)
        target_n_step = n_step
        for h_index, h in enumerate(required_horizons):
            args_copy = args.copy()
            args_copy['required_horizons'] = [h]
            args_copy['max_samples'] = max_samples
            args_copy['estimator_mode'] = estimator_mode
            args_copy['log'] = log
            n_step = int(np.clip(h, 1, target_n_step))  # the magic heuristic...
            result[:, :, h_index] = get_return_estimate(mode="exponential", n_step=n_step, **args_copy)[:, :, 0]
        return result
    elif mode == "adaptive_cap":
        # we do this by repeated calling exponential, which can be a bit slow...
        result = np.zeros([N, A, K], dtype=np.float32)
        target_n_step = n_step
        for h_index, h in enumerate(required_horizons):
            args_copy = args.copy()
            args_copy['required_horizons'] = [h]
            args_copy['max_samples'] = max_samples
            args_copy['estimator_mode'] = estimator_mode
            args_copy['log'] = log
            n_step = int(np.clip(h, 1, target_n_step))  # the magic heuristic...
            result[:, :, h_index] = get_return_estimate(mode="exponential_cap", n_step=n_step, **args_copy)[:, :, 0]
        return result
    elif mode == "exponential":
        # this just makes things a bit faster for small n_step horizons
        # essentially we ignore the very rare, very long n_steps.
        max_h = min(n_step * 3, N)
        lamb = 1-(1/n_step)
        weights = np.asarray([lamb ** x for x in range(max_h)], dtype=np.float32)
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

    if samples is None:
        max_n = len(weights)
        probs = weights / weights.sum()
        samples = np.random.choice(range(1, max_n + 1), max_samples, replace=True, p=probs)

    if estimator_mode == 'default':
        return _calculate_sampled_return_multi_fast(n_step_list=samples, **args)
    elif estimator_mode == 'reference':
        return _calculate_sampled_return_multi_reference(n_step_list=samples, **args)
    elif estimator_mode == 'verify':
        assert log is not None
        if value_samples_m2 is None:
            m1 = _calculate_sampled_return_multi_fast(n_step_list=samples, **args)
            verified_m1 = _calculate_sampled_return_multi_reference(n_step_list=samples, **args)
            delta_m1 = np.abs(verified_m1 - m1).max()
            if delta_m1 > 1e-4:
                log.warn(f"Errors in return estimation {delta_m1:.5f}")
            return m1
        else:
            m1, m2 = _calculate_sampled_return_multi_fast(n_step_list=samples, **args)
            verified_m1,verified_m2 = _calculate_sampled_return_multi_reference(n_step_list=samples, **args)
            delta_m1 = np.abs(verified_m1 - m1).max()
            delta_m2 = np.abs(verified_m2 - m2).max()
            if delta_m1 > 1e-4 or delta_m2 > 1e-4:
                log.warn(f"Errors in return estimation {delta_m1:.5f}/{delta_m2:.5f}")
            else:
                log.info(f"Errors in return estimation {delta_m1:.5f}/{delta_m2:.5f}")
            return m1, m2
    else:
        raise ValueError(f"Invalid estimator_mode {estimator_mode}")




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


def reweigh_samples(n_step_list:list, weights=None):
    """
    Takes a list of n_step lengths, and (optinally) their weights, and returns a new list of samples / weights
    such that duplicate n_steps have had their weight added together.
    I.e. reweigh_samples = [1,1,1,2] -> ([1,2], [1:3,2:1])
    """
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


def _calculate_sampled_return_multi_fast(
    n_step_list: list,
    gamma: float,
    rewards: np.ndarray,
    dones: np.ndarray,
    required_horizons: np.ndarray,
    value_sample_horizons: np.ndarray,
    value_samples: np.ndarray,
    value_samples_m2: np.ndarray = None,
    value_samples_m3: np.ndarray = None,
    n_step_weights: list = None,
    use_log_interpolation: bool = False,
):
    """
        Fast version of return calculation. Calculates a weighted average of multiple n_step returns

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

    # the idea here is to progressively update a running total of the n-step rewards, then apply bootstrapping
    # quickly in a vectorized way. Unlike the previous algorithm, this is O(n) rather then O(n^2) where n is n_steps
    # I also take advantage of the fact that nstep(n`,h) = nstep(n,h) for all n`>=n. This allows me to eliminate
    # many of the horizon updates once we get into the high n_steps.

    # if we have n_step requests that exceed the longest horizon we can cap these to the longest horizon.
    max_h = max(required_horizons)
    max_n = len(rewards)
    n_step_list = np.clip(n_step_list, 1, max_h)
    n_step_list = np.clip(n_step_list, 1, max_n)
    n_step_list = list(n_step_list)

    # calculate the weight for each n_step
    n_step_list, n_step_weights = reweigh_samples(n_step_list, n_step_weights)

    # remove any duplicates (these will be handled by the weight calculation)
    n_step_list = list(set(n_step_list))
    n_step_list.sort()

    moment = 1
    if value_samples_m2 is not None:
        moment = 2

    assert value_samples_m3 is None, "Third moment not supported on fast mode yet."

    N, A = rewards.shape
    K = len(required_horizons)

    log_value_sample_horizons = np.log10(10+value_sample_horizons.astype('float32'))-1

    total_weight = sum(n_step_weights.values())
    remaining_weight = 1.0

    # this allows us to map to our 'sparse' returns table
    h_lookup = {}
    for index, h in enumerate(required_horizons):
        if h not in h_lookup:
            h_lookup[h] = [index]
        else:
            h_lookup[h].append(index)

    returns_m1 = np.zeros([N, A, K], dtype=np.float32)
    returns_m2 = np.zeros([N, A, K], dtype=np.float32)

    # S is defined as S_{k} sum_{i=0}^{k} gamma^i r_{t+i}, and is essentially just the cumulative reward, but capped
    # to some horizon k.
    S = np.zeros([N, A, K], dtype=np.float32) # partial reward sums

    cumulative_rewards = np.zeros_like(rewards)
    discount = np.ones_like(rewards)

    current_n_step = 0

    def interpolate_linear(value_estimates:np.ndarray, horizon: int):
        return _interpolate(value_sample_horizons, value_estimates, horizon)

    def interpolate_log(value_estimates:np.ndarray, horizon: int):
        return _interpolate(log_value_sample_horizons, value_estimates, np.log10(10+horizon)-1)

    interpolate = interpolate_log if use_log_interpolation else interpolate_linear

    # S [N, A, K], sum_{i=0}^{current_n_step} [ gamma^i r_{t+i} ]

    for n_step in n_step_list:
        weight = n_step_weights[n_step] / total_weight
        # step 1, update our cumulative reward count
        while current_n_step < n_step:
            cumulative_rewards[:(N-current_n_step)] += rewards[current_n_step:] * discount[:(N-current_n_step)]
            for h in h_lookup.keys():
                if h >= (current_n_step+1):
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
                        returns_m1[:, :, i] += (cumulative_rewards) * weight # this could be done in bulk...
                        if moment >= 2:
                            returns_m2[:, :, i] += (S[:,:,i]**2) * weight # this could be done in bulk...
                    elif current_n_step == h:
                        # this is the final one, give it all the remaining weight...
                        # S[:,:,i] should be == cumulative_rewards at this point
                        returns_m1[:, :, i] += (cumulative_rewards) * remaining_weight
                        if moment >= 2:
                            returns_m2[:, :, i] += (S[:,:,i]**2) * remaining_weight

        # -----------------------------------
        # this is the M^2 part...

        # # we can do most of this with one big update, however near the end of the rollout we need to account
        # # for the fact that we are actually using a shorter n-step
        steps_made = current_n_step
        block_size = 1 + N - current_n_step
        for h_index, h in enumerate(required_horizons):
            if h - steps_made <= 0:
                continue
            interpolated_value = interpolate(
                value_samples[steps_made:block_size + steps_made],
                h - steps_made
            )
            returns_m1[:block_size, :, h_index] += interpolated_value * (discount[:block_size]) * weight
            if moment >= 2:
                interpolated_value_m2 = interpolate(
                    value_samples_m2[steps_made:block_size + steps_made],
                    h - steps_made
                )
                returns_m2[:block_size, :, h_index] += interpolated_value_m2 * (discount[:block_size]**2) * weight

        # next do the remaining few steps
        # this is a bit slow for large n_step, but most of the estimates are low n_step anyway
        # (this could be improved by simply ignoring these n-step estimates
        # the problem with this is that it places *a lot* of emphasis on the final value estimate
        for t in range(1 + N - current_n_step, N):
            steps_made = min(current_n_step, N - t)
            if t + steps_made > N:
                continue
            for h_index, h in enumerate(required_horizons):
                if h - steps_made > 0:
                    interpolated_value = interpolate(value_samples[t + steps_made], h - steps_made)
                    returns_m1[t, :, h_index] += interpolated_value * (discount[t]) * weight
                    if moment >= 2:
                        interpolated_value_m2 = interpolate(value_samples_m2[t + steps_made], h - steps_made)
                        returns_m2[t, :, h_index] += interpolated_value_m2 * (discount[t]**2) * weight

        # -----------------------------------
        # this is the 2SM part...

        # we can do most of this with one big update, however near the end of the rollout we need to account
        # for the fact that we are actually using a shorter n-step
        if moment >= 2:
            steps_made = current_n_step
            block_size = 1 + N - current_n_step
            for h_index, h in enumerate(required_horizons):
                if h - steps_made <= 0:
                    continue
                interpolated_value = interpolate(
                    value_samples[steps_made:block_size + steps_made],
                    h - steps_made
                )
                returns_m2[:block_size, :, h_index] += \
                    2 * S[:block_size, :, h_index] * \
                    interpolated_value * \
                    discount[:block_size] * \
                    weight

        # # next do the remaining few steps
        # # this is a bit slow for large n_step, but most of the estimates are low n_step anyway
        # # (this could be improved by simply ignoring these n-step estimates
        # # the problem with this is that it places *a lot* of emphasis on the final value estimate
        for t in range(1 + N - current_n_step, N):
            steps_made = min(current_n_step, N - t)
            if t + steps_made > N:
                continue
            for h_index, h in enumerate(required_horizons):
                if h - steps_made <= 0:
                    continue
                interpolated_value = interpolate(
                    value_samples[t + steps_made],
                    h - steps_made
                )
                returns_m2[t, :, h_index] += \
                    2 * S[t, :, h_index] * \
                    interpolated_value * \
                    discount[t] * \
                    weight

        remaining_weight -= weight

    if moment == 1:
        return returns_m1
    else:
        return returns_m1, returns_m2


