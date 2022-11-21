import torch
from typing import Union
import numpy as np

from config import args
import rollout

import math
import collections

import time as clock
import utils
from . returns import calculate_bootstrapped_returns, get_return_estimate


class TVFRunnerModule(rollout.RunnerModule):

    # todo: move tvf_value etc to here...

    def trim_horizons(self, tvf_value_estimates, time, method: str = "timelimit", mode: str = "interpolate"):
        """
        Adjusts horizons by reducing horizons that extend over the timeout back to the timeout.
        This is for a few reasons.
        1. it makes use of the other heads, which means errors might average out
        2. it can be that shorter horizons get learned more quickly, and this will make use of them earlier on
        so long as the episodes are fairly long. If episodes are short compared to max_horizon this might not
        help at all.
        @param tvf_value_estimates: np array of dims [A, K, VH]
        @param time: np array of dims [A] containing time associated with the states that generated these estimates.
        @returns new trimmed estimates of [A, K, VH],
        @returns predicted_time_till_termination of [A, K]
        """

        old_value_estimates = tvf_value_estimates
        tvf_value_estimates = tvf_value_estimates.copy()  # don't modify input

        # by definition h=0 is 0.0
        assert self.runner.tvf_horizons[0] == 0, "First horizon must be zero"
        tvf_value_estimates[:, 0, :] = 0  # always make sure h=0 is fixed to zero.

        # step 1: work out the trimmed horizon
        # output is [A]
        if method == "off":
            return tvf_value_estimates, None
        elif method == "timelimit":
            time_till_termination = np.maximum((args.timeout / args.frame_skip) - time, 0)
        elif method == "est_term":
            est_ep_length = np.percentile(list(self.runner.episode_length_buffer) + list(time), args.tvf_at_percentile).astype(
                int)
            est_ep_length += args.tvf_at_minh / 4  # apply small buffer
            time_till_termination = np.maximum((args.timeout / args.frame_skip) - time, 0)
            est_time_till_termination = np.maximum(est_ep_length - time, args.tvf_at_minh)
            time_till_termination = np.minimum(time_till_termination, est_time_till_termination)
            self.runner.log.watch_mean("*ttt_ep_length",
                                np.percentile(self.runner.episode_length_buffer, args.tvf_at_percentile).astype(int))
            self.runner.log.watch_mean("*ttt_ep_std", np.std(self.runner.episode_length_buffer))
        else:
            raise ValueError(f"Invalid trimming method {method}")

        self.runner.log.watch_stats("ttt", time_till_termination, display_width=0, history_length=4)

        # step 2: calculate new estimates
        if mode == "interpolate":
            # this can work a little bit better if trimming is only minimal.
            # however, all horizons still end up sharing the same estimate.
            # note: we now interpolate on log scale.

            def log_scale(x):
                return np.log10(10 + x) - 1

            scale = log_scale

            A, K, VH = tvf_value_estimates.shape
            trimmed_ks = np.searchsorted(self.runner.tvf_horizons, time_till_termination)
            trimmed_value_estimate = interpolate(
                scale(np.asarray(self.runner.tvf_horizons)),
                old_value_estimates[..., 0],  # select final value head
                scale(time_till_termination)
            )
            for a in range(A):
                tvf_value_estimates[a, trimmed_ks[a]:, 0] = trimmed_value_estimate[a]
        elif mode == "average":
            # average up to h but no further
            # implementation is a bit slow, drop if it's not better.
            trimmed_ks = np.searchsorted(self.runner.tvf_horizons, time_till_termination)
            for a, trimmed_k in enumerate(trimmed_ks):
                if trimmed_k >= len(self.runner.tvf_horizons) - 1:
                    # this means no trimming
                    continue
                acc = 0
                counter = 0
                for k in range(trimmed_k, self.runner.K):
                    # note: this could be tvf_value_estimate, but I want to make it explicit that we're using
                    # the original values.
                    acc += old_value_estimates[a, k, :]
                    counter += 1
                    tvf_value_estimates[a, k, :] = acc / counter
        elif mode == "substitute":
            # just use the smallest h we can, very simple.
            untrimmed_ks = np.arange(self.runner.K)[None, :]
            trimmed_ks = np.searchsorted(self.runner.tvf_horizons, time_till_termination)[:, None]
            trimmed_ks = np.minimum(trimmed_ks, untrimmed_ks)
            tvf_value_estimates = np.take_along_axis(tvf_value_estimates[:, :, 0], trimmed_ks, axis=1)
            tvf_value_estimates = tvf_value_estimates[:, :, None]
        else:
            raise ValueError(f"Invalid trimming mode {mode}")

        # monitor how much we modified the return estimates...
        # if max is large then maybe there's an off-by-one bug on time.
        delta = np.abs(tvf_value_estimates - old_value_estimates)
        self.runner.log.watch_stats("ved", delta.ravel(), display_width=0, history_length=4)
        return tvf_value_estimates, time_till_termination

    def calculate_tvf_returns(
            self,
            value_head: str = "ext", # ext|int
            obs=None,
            rewards=None,
            dones=None,
            tvf_return_mode=None,
            tvf_return_distribution=None,
            tvf_n_step=None,
    ):
        """
        Calculates and returns the (tvf_gamma discounted) (transformed) return estimates for given rollout.

        prev_states: ndarray of dims [N+1, B, *state_shape] containing prev_states
        rewards: float32 ndarray of dims [N, B] containing reward at step n for agent b
        value_sample_horizons: int32 ndarray of dims [K] indicating horizons to generate value estimates at.
        value_head: which head to use for estimate, i.e. ext_value, int_value, ext_sqr_value etc

        """

        # setup
        obs = obs if obs is not None else self.runner.all_obs
        rewards = rewards if rewards is not None else self.runner.ext_rewards
        dones = dones if dones is not None else self.runner.terminals
        tvf_return_mode = tvf_return_mode or args.tvf_return_mode
        tvf_return_distribution = tvf_return_distribution or args.tvf_return_distribution
        tvf_n_step = tvf_n_step or args.tvf_return_n_step

        N, A, *state_shape = obs[:-1].shape

        assert obs.shape == (N + 1, A, *state_shape)
        assert rewards.shape == (N, A)
        assert dones.shape == (N, A)

        # step 2: calculate the returns
        start_time = clock.time()

        # setup return estimator mode, but only verify occasionally.
        re_mode = args.tvf_return_estimator_mode
        if re_mode == "verify" and self.runner.batch_counter % 31 != 1:
            re_mode = "default"

        # we must unnormalize the value estimates, then renormalize after
        values = self.runner.tvf_value[..., self.runner.value_heads.index(value_head)]

        returns = get_return_estimate(
            mode=tvf_return_mode,
            distribution=tvf_return_distribution,
            gamma=args.tvf_gamma,
            rewards=rewards,
            dones=dones,
            required_horizons=np.asarray(self.runner.tvf_horizons),
            value_sample_horizons=np.asarray(self.runner.tvf_horizons),
            value_samples=values,
            n_step=tvf_n_step,
            max_samples=args.tvf_return_samples,
            estimator_mode=re_mode,
            log=self.runner.log,
            use_log_interpolation=args.tvf_return_use_log_interpolation,
        )

        return_estimate_time = clock.time() - start_time
        self.runner.log.watch_mean(
            "time_return_estimate",
            return_estimate_time,
            display_precision=3,
            display_name="t_re",
        )
        return returns

    @torch.no_grad()
    def log_tvf_curve_quality(self):
        """
        Writes value quality stats to log
        """

        N, A, *state_shape = self.runner.prev_obs.shape
        K = len(self.runner.tvf_horizons)

        targets = self.calculate_tvf_returns(
            value_head='ext',
            obs=self.runner.all_obs,
            rewards=self.runner.ext_rewards,
            dones=self.runner.terminals,
            tvf_return_distribution="fixed",  # <-- MC is the least bias method we can do...
            tvf_n_step=args.n_steps,
        )

        first_moment_targets = targets
        first_moment_estimates = self.runner.tvf_value[:N, :, :, 0].reshape(N, A, K)
        self.runner._log_curve_quality(first_moment_estimates, first_moment_targets)

        # also log ev_ext
        targets = calculate_bootstrapped_returns(
            self.runner.ext_rewards, self.runner.terminals, self.runner.ext_value[self.runner.N], args.gamma
        )
        values = self.runner.ext_value[:self.runner.N]
        ev = utils.explained_variance(values.ravel(), targets.ravel())
        self.runner.log.watch_mean("*ev_ext", ev, history_length=1)


    def save(self):
        pass

    def load(self):
        pass

def get_rediscounted_value_estimate(
        values: Union[np.ndarray, torch.Tensor],
        old_gamma: float,
        new_gamma: float,
        horizons,
        clipping=10,
):
    """
    Returns rediscounted return at horizon h

    values: float tensor of shape [B, K]
    horizons: int tensor of shape [K] giving horizon for value [:, k]
    returns float tensor of shape [B]
    """

    B, K = values.shape

    if old_gamma == new_gamma:
        return values[:, -1]

    assert K == len(horizons), f"missmatch {K} {horizons}"
    assert horizons[0] == 0, 'first horizon must be 0'

    if type(values) is np.ndarray:
        values = torch.from_numpy(values)
        is_numpy = True
    else:
        is_numpy = False

    prev = values[:, 0] # should be zero
    prev_h = 0
    discounted_reward_sum = torch.zeros([B], dtype=torch.float32, device=values.device)
    for i_minus_one, h in enumerate(horizons[1:]):
        i = i_minus_one + 1
        # rewards occurred at some point after prev_h and before h, so just average them. Remembering that
        # v_h includes up to and including h timesteps.
        # also, we subtract 1 as the reward given by V_h=1 occurs at t=0
        mid_h = ((prev_h+1 + h) / 2) - 1
        discounted_reward = (values[:, i] - prev)
        prev = values[:, i]
        prev_h = h
        # a clipping of 10 gets us to about 2.5k horizon before we start introducing bias. (going from 1k to 10k discounting)
        ratio = min((new_gamma ** mid_h) / (old_gamma ** mid_h), clipping) # clipped ratio
        discounted_reward_sum += discounted_reward * ratio

    return discounted_reward_sum.numpy() if is_numpy else discounted_reward_sum


def expand_to_h(h, x):
    """
    takes 2d input and returns it duplicated [H] times
    in form [*, *, h]
    """
    x = x[:, :, None]
    x = np.repeat(x, h, axis=2)
    return x


def calculate_gae_tvf(
        batch_reward: np.ndarray,
        batch_value: np.ndarray,
        final_value_estimate: np.ndarray,
        batch_terminal: np.ndarray,
        discount_fn = lambda t: 0.999**t,
        lamb: float = 0.95):

    """
    A modified version of GAE that uses truncated value estimates to support any discount function.
    This works by extracting estimated rewards from the value curve via a finite difference.

    batch_reward: [N, A] rewards for each timestep
    batch_value: [N, A, H] value at each timestep, for each horizon (0..max_horizon)
    final_value_estimate: [A, H] value at timestep N+1
    batch_terminal [N, A] terminal signals for each timestep
    discount_fn A discount function in terms of t, the number of timesteps in the future.
    lamb: lambda, as per GAE lambda.
    """

    N, A, H = batch_value.shape

    advantages = np.zeros([N, A], dtype=np.float32)
    values = np.concatenate([batch_value, final_value_estimate[None, :, :]], axis=0)

    # get expected rewards. Note: I'm assuming here that the rewards have not been discounted
    assert args.tvf_gamma == 1, "General discounting function requires TVF estimates to be undiscounted (might fix later)"
    expected_rewards = values[:, :, 0] - batch_value[:, :, 1]

    def calculate_g(t, n:int):
        """ Calculate G^(n) (s_t) """
        # we use the rewards first, then use expected rewards from 'pivot' state onwards.
        # pivot state is either the final state in rollout, or t+n, which ever comes first.
        sum_of_rewards = np.zeros([N], dtype=np.float32)
        discount = np.ones([N], dtype=np.float32) # just used to handle terminals
        pivot_state = min(t+n, N)
        for i in range(H):
            if t+i < pivot_state:
                reward = batch_reward[t+i, :]
                discount *= 1-batch_terminal[t+i, :]
            else:
                reward = expected_rewards[pivot_state, :, (t+i)-pivot_state]
            sum_of_rewards += reward * discount * discount_fn(i)
        return sum_of_rewards

    def get_g_weights(max_n: int):
        """
        Returns the weights for each G estimate, given some lambda.
        """

        # weights are assigned with exponential decay, except that the weight of the final g_return uses
        # all remaining weight. This is the same as assuming that g(>max_n) = g(n)
        # because of this 1/(1-lambda) should be a fair bit larger than max_n, so if a window of length 128 is being
        # used, lambda should be < 0.99 otherwise the final estimate carries a significant proportion of the weight

        weight = (1-lamb)
        weights = []
        for _ in range(max_n-1):
            weights.append(weight)
            weight *= lamb
        weights.append(lamb**max_n)
        weights = np.asarray(weights)

        assert abs(weights.sum() - 1.0) < 1e-6
        return weights

    # advantage^(n) = -V(s_t) + r_t + r_t+1 ... + r_{t+n-1} + V(s_{t+n})

    for t in range(N):
        max_n = N - t
        weights = get_g_weights(max_n)
        for n, weight in zip(range(1, max_n+1), weights):
            if weight <= 1e-6:
                # ignore small or zero weights.
                continue
            advantages[t, :] += weight * calculate_g(t, n)
        advantages[t, :] -= batch_value[t, :, -1]

    return advantages

def _test_interpolate():
    horizons = np.asarray([0, 1, 2, 10, 100])
    values = np.asarray([0, 5, 10, -1, 2])[None, :].repeat(11, axis=0)
    results = interpolate(horizons, values, np.asarray([-100, -1, 0, 1, 2, 3, 4, 99, 100, 101, 200]))
    expected_results = [0, 0, 0, 5, 10, (7/8)*10+(1/8)*-1, (6/8)*10+(2/8)*-1, 1.96666667, 2, 2, 2]
    if np.max(np.abs(np.asarray(expected_results) - results)) > 1e-6:
        print("Expected:", expected_results)
        print("Found:", results)
        raise ValueError("Interpolation check failed")


def interpolate(horizons: np.ndarray, values: np.ndarray, target_horizons: np.ndarray):
    """
    Returns linearly interpolated value from source_values

    horizons: sorted ndarray of shape [K] of horizons, must be in *strictly* ascending order
    values: ndarray of shape [*shape, K] where values[...,h] corresponds to horizon horizons[h]
    target_horizons: np array of dims [*shape], the horizon we would like to know the interpolated value of for each
        example

    """

    # I did this custom, as I could not find a way to get numpy to interpolate the way I needed it to.
    # the issue is we interpolate nd data with non-uniform target x's.

    assert len(set(horizons)) == len(horizons), f"Horizons duplicates not supported {horizons}"
    assert np.all(np.diff(horizons) > 0), f"Horizons must be sorted and unique horizons:{horizons}, targets:{target_horizons}"

    assert horizons[0] == 0, "first horizon must be 0"

    # we do not extrapolate...
    target_horizons = np.clip(target_horizons, horizons[0], horizons[-1])

    *shape, K = values.shape
    shape = tuple(shape)
    assert horizons.shape == (K,)
    assert target_horizons.shape == shape, f"{target_horizons.shape} != {shape}"

    # put everything into 1d
    N = np.prod(shape)
    values = values.reshape(N, K)
    target_horizons = target_horizons.reshape(N)

    post_index = np.searchsorted(horizons, target_horizons, side='left')

    # select out our values
    pre_index = np.maximum(post_index-1, 0)
    value_pre = values[range(N), pre_index]
    value_post = values[range(N), post_index]

    dx = (horizons[post_index] - horizons[pre_index])
    dx[dx == 0] = 1.0 # this only happens when we are at the boundaries, in which case we have 0/dx, and we just want 0.
    factor = (target_horizons - horizons[pre_index]) / dx
    result = value_pre * (1 - factor) + value_post * factor
    result[post_index == 0] = 0 # these have h<=0 which by definition has value 0
    result = result.reshape(*shape)

    return result


def get_value_head_horizons(n_heads: int, max_horizon: int, spacing: str="geometric", include_weight=False):
    """
    Provides a set of horizons spaces (approximately) geometrically, with the H[0] = 1 and H[-1] = max_horizon.
    Some early horizons may be duplicated due to rounding.
    """

    target_n_heads = n_heads

    if spacing == "linear":
        result = np.asarray(np.round(np.linspace(0, max_horizon, n_heads)), dtype=np.int32)
        return (result, np.ones([n_heads], dtype=np.float32)) if include_weight else result
    elif spacing == "geometric":
        try_heads = target_n_heads

        def get_heads(x, remove_duplicates=True):
            result = np.asarray(np.round(np.geomspace(1, max_horizon + 1, x)) - 1, dtype=np.int32)
            if remove_duplicates:
                return np.asarray(sorted(set(result)))
            else:
                return np.asarray(sorted(result))

        while len(get_heads(try_heads)) != target_n_heads:
            if len(get_heads(try_heads)) < target_n_heads:
                try_heads += int(math.sqrt(n_heads))
            else:
                try_heads -= 1
        all_heads = get_heads(try_heads, remove_duplicates=False)
        counts = collections.defaultdict(int)
        for head in all_heads:
            counts[head] += 1
        result = np.asarray(list(counts.keys()), dtype=np.int32)
        counts = np.asarray(list(counts.values()), dtype=np.float32)
        return (result, counts) if include_weight else result
    else:
        raise ValueError(f"Invalid spacing value {spacing}")

