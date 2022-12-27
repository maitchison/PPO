import torch
from typing import Union
import numpy as np

import rl

from . config import args
from . import utils
from . returns import calculate_bootstrapped_returns
from . returns_truncated import get_return_estimate

import math
import collections

import time as clock


class TVFRunnerModule(rl.rollout.RunnerModule):

    def __init__(self, parent):
        super().__init__(parent)

        N, A = parent.ext_rewards.shape
        VH = len(parent.value_heads)
        K = len(parent.tvf_horizons)

        self.tvf_value = np.zeros([N + 1, A, K, VH], dtype=np.float32)
        self.tvf_untrimmed_value = np.zeros([N + 1, A, K, VH], dtype=np.float32)  # our ext value estimate
        self.tvf_final_value = np.zeros([N + 1, A], dtype=np.float32) # our final value estimate
        self.tvf_returns = np.zeros([N, A, K, VH], dtype=np.float32)

    def on_train_value_minibatch(self, model_out, data, **kwargs):

        assert "tvf_returns" in data, "TVF returns were not uploaded with batch."

        required_tvf_heads = kwargs['required_tvf_heads']
        single_value_head = kwargs['single_value_head']

        # targets "tvf_returns" are [B, K]
        # predictions "tvf_value" are [B, K, VH]
        # predictions need to be generated... this could take a lot of time so just sample a few..
        targets = data["tvf_returns"] # locked to "ext" head for the moment [B, K]
        predictions = model_out["tvf_value"][:, :, 0] # locked to ext for the moment [B, K, VH] -> [B, K]

        if required_tvf_heads is not None:
            targets = targets[:, required_tvf_heads]

        # this will be [B, K]
        tvf_loss = args.tvf.coef * 0.5 * torch.square(targets - predictions)

        # account for weights due to duplicate head removal
        head_filter = required_tvf_heads if required_tvf_heads is not None else slice(None, None)
        tvf_loss = tvf_loss * torch.from_numpy(self.runner.tvf_weights[head_filter])[None, :].to(self.runner.model.device)

        # h_weighting adjustment
        if args.tvf.head_weighting == "h_weighted" and single_value_head is None:
            def h_weight(h):
                # roughly the number of times an error will be copied, plus the original error
                return 1 + ((args.tvf.max_horizon - h) / args.tvf_return_n_step)
            weighting = np.asarray([h_weight(h) for h in self.runner.tvf_horizons], dtype=np.float32)[None, :]
            adjustment = 2 / (np.min(weighting) + np.max(weighting)) # try to make MSE roughly the same scale as before
            tvf_loss = tvf_loss * torch.tensor(weighting).to(device=tvf_loss.device) * adjustment

        if args.tvf.horizon_dropout > 0:
            # note: we weight the mask so that after the average the loss per example will be approximately the same
            # magnitude.
            keep_prob = (1-args.tvf.horizon_dropout)
            mask = torch.bernoulli(torch.ones_like(tvf_loss)*keep_prob) / keep_prob
            tvf_loss = tvf_loss * mask

        # mean over horizons generates a loss that is far too small, dividing by sqrt of the number of heads seems
        # like a good compromise and seems to give a gradient scale that remains invariant to the number of heads.
        tvf_loss = (tvf_loss.shape[-1]**0.5) * tvf_loss.mean(dim=-1)

        self.runner.log.watch_mean("loss_tvf", tvf_loss.mean(), history_length=64 * args.value_opt.epochs, display_name="ls_tvf", display_width=8)

        return tvf_loss

    def _reset(self):
        self.tvf_value *= 0
        self.tvf_untrimmed_value *= 0
        self.tvf_final_value *= 0
        self.tvf_returns *= 0

    def on_reset(self):
        self._reset()

    def on_before_generate_rollout(self):
        self._reset()

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
        @returns ext_value_estimate, estimate of value at infinite horizon of dims [A]
        """

        old_value_estimates = tvf_value_estimates.copy()  # don't modify input
        tvf_value_estimates = tvf_value_estimates.copy()  # don't modify input

        # by definition h=0 is 0.0
        assert self.runner.tvf_horizons[0] == 0, "First horizon must be zero"
        tvf_value_estimates[:, 0, :] = 0  # always make sure h=0 is fixed to zero.
        old_value_estimates[:, 0, :] = 0  # always make sure h=0 is fixed to zero.

        # step 1: work out the trimmed horizon
        # output is [A]
        if method == "off":
            return tvf_value_estimates, 0, None
        elif method == "timelimit":
            time_till_termination = np.maximum(args.env.timeout - time, 0)
        elif method == "est_term":
            est_ep_length = np.percentile(list(self.runner.episode_length_buffer) + list(time), args.tvf.eta_percentile).astype(
                int) + args.tvf.eta_buffer
            est_ep_length += args.tvf.eta_minh / 4  # apply small buffer
            time_till_termination = np.maximum(args.env.timeout - time, 0)
            est_time_till_termination = np.maximum(est_ep_length - time, args.tvf.eta_minh)
            time_till_termination = np.minimum(time_till_termination, est_time_till_termination)
            self.runner.log.watch_mean("*ttt_ep_length",
                                np.percentile(self.runner.episode_length_buffer, args.tvf.eta_percentile).astype(int))
            self.runner.log.watch_mean("*ttt_ep_std", np.std(self.runner.episode_length_buffer))
        else:
            raise ValueError(f"Invalid trimming method {method}")

        self.runner.log.watch_stats("ttt", time_till_termination, display_width=0, history_length=4)

        A, K, VH = tvf_value_estimates.shape
        trimmed_ks = np.searchsorted(self.runner.tvf_horizons, time_till_termination)

        # step 2: calculate new estimates
        if mode == "interpolate":
            # this can work a little bit better if trimming is only minimal.
            # however, all horizons still end up sharing the same estimate.
            # note: we now interpolate on log scale.

            def log_scale(x):
                return np.log10(10 + x) - 1

            scale = log_scale
            trimmed_value_estimate = horizon_interpolate(
                scale(np.asarray(self.runner.tvf_horizons)),
                old_value_estimates[..., 0],  # select final value head
                scale(time_till_termination)
            )
            for a in range(A):
                tvf_value_estimates[a, trimmed_ks[a]:, 0] = trimmed_value_estimate[a]
        elif mode == "average":
            # average up to h but no further
            # implementation is a bit slow, drop if it's not better.
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
            modified_trimmed_ks = trimmed_ks[:, None].copy()
            modified_trimmed_ks = np.minimum(modified_trimmed_ks, untrimmed_ks)
            tvf_value_estimates = np.take_along_axis(tvf_value_estimates[:, :, 0], modified_trimmed_ks, axis=1)
            tvf_value_estimates = tvf_value_estimates[:, :, None]
        elif mode == "random":
            # randomly pick a valid horizon
            # the idea here is that each horizon (on each agent) gets a slightly different estimate.
            # if this works well optimize...
            for a, trimmed_k in zip(range(A), trimmed_ks):
                new_ks = np.arange(K)
                for k in range(trimmed_k, K):
                    # leave horizons before trimming point as is,
                    # latter horizons use a random horizon less than or equal to their own horizon
                    new_ks[k] = np.random.randint(trimmed_k, k+1)
                tvf_value_estimates[a, range(K)] = old_value_estimates[a, new_ks]
        else:
            raise ValueError(f"Invalid trimming mode {mode}")

        # calculate ext_value used for advantages by averaging over all valid horizons using
        # untrimmed estimates
        final_value_estimates = np.zeros([A], dtype=np.float32)
        for a, trimmed_k in enumerate(trimmed_ks):
            trimmed_k = min(trimmed_k, K-1) # make sure there is always one sample.
            final_value_estimates[a] = old_value_estimates[a, trimmed_k:, 0].mean() # just the ext_value head.

        # clip differences if needed...
        if args.tvf.trim_clip >= 0:
            clipped_delta = np.clip(tvf_value_estimates - old_value_estimates, -args.tvf.trim_clip, +args.tvf.trim_clip)
            tvf_value_estimates = old_value_estimates + clipped_delta

        # monitor how much we modified the return estimates...
        # if max is large then maybe there's an off-by-one bug on time.
        delta = np.abs(tvf_value_estimates - old_value_estimates)

        self.runner.log.watch_stats("ved", delta.ravel(), display_width=0, history_length=4)
        return tvf_value_estimates, final_value_estimates, time_till_termination

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
        tvf_return_mode = tvf_return_mode or args.tvf.return_mode
        tvf_return_distribution = tvf_return_distribution or args.tvf.return_distribution
        tvf_n_step = tvf_n_step or args.tvf_return_n_step

        N, A, *state_shape = obs[:-1].shape

        assert obs.shape == (N + 1, A, *state_shape)
        assert rewards.shape == (N, A)
        assert dones.shape == (N, A)

        # step 2: calculate the returns
        start_time = clock.time()

        # we must unnormalize the value estimates, then renormalize after
        values = self.tvf_value[..., self.runner.value_heads.index(value_head)]

        returns = get_return_estimate(
            mode=tvf_return_mode,
            distribution=tvf_return_distribution,
            gamma=args.tvf.gamma,
            rewards=rewards,
            dones=dones,
            required_horizons=np.asarray(self.runner.tvf_horizons),
            value_sample_horizons=np.asarray(self.runner.tvf_horizons),
            value_samples=values,
            n_step=tvf_n_step,
            max_samples=args.tvf.return_samples,
            use_log_interpolation=args.tvf.return_use_log_interpolation,
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
        first_moment_estimates = self.tvf_value[:N, :, :, 0].reshape(N, A, K)
        self.runner._log_curve_quality(first_moment_estimates, first_moment_targets)

        # also log ev_ext
        targets = calculate_bootstrapped_returns(
            self.runner.ext_rewards, self.runner.terminals, self.runner.ext_value[self.runner.N], args.gamma
        )
        values = self.runner.ext_value[:self.runner.N]
        ev = utils.explained_variance(values.ravel(), targets.ravel())
        self.runner.log.watch_mean("*ev_ext", ev, history_length=1)

    @torch.no_grad()
    def get_tvf_ext_value_estimate(self, new_gamma: float):
        """

        Returns rediscounted value estimate for given rollout (i.e. rewards + value if using given gamma)
        Usually this is just GAE, but if gamma != tvf_gamma, then rediscounting is applied.

        We expect:
        self.tvf_value: np array of dims [N+1, A, K, VH] containing value estimates for each horizon K and each value head VH

        @returns value estimate for gamma=gamma for example [N+1, A]
        """

        assert args.tvf.enabled
        N, A, K, VH = self.tvf_value[:self.runner.N].shape

        VALUE_HEAD_INDEX = self.runner.value_heads.index('ext')

        # [N, A, K]
        trimmed_tvf_values = self.tvf_value[:, :, :, VALUE_HEAD_INDEX]
        untrimmed_tvf_values = self.tvf_untrimmed_value[:, :, :, VALUE_HEAD_INDEX]

        if abs(new_gamma - args.tvf.gamma) < 1e-8:
            if args.tvf.trimming_mode == "off":
                return untrimmed_tvf_values[:, :, -1]

            # simple case no discounting...
            if args.tvf.trim_advantages == "trimmed":
                # just return the longest trimmed horizon...
                return trimmed_tvf_values[:, :, -1]
            elif args.tvf.trim_advantages == "average":
                # during trimming we store the average of all valid horizons in this variable.
                return self.tvf_final_value
            elif args.tvf.trim_advantages == "untrimmed":
                # use final untrimmed horizon
                return untrimmed_tvf_values[:, :, -1]
            else:
                raise ValueError(f"Invalid advantage trimming mode {args.tvf.trim_advantages}.")

        assert args.tvf.trim_advantages != "average", "Average advantage trimming not supported with rediscounting."

        if args.tvf.trim_advantages == "trimmed":
            tvf_values = trimmed_tvf_values
        elif args.tvf.trim_advantages == "untrimmed":
            tvf_values = untrimmed_tvf_values
        else:
            raise ValueError()

        # stub:
        print("FYI we are rediscounting...")

        # otherwise... we need to rediscount...
        return rl.tvf.get_rediscounted_value_estimate(
            values=tvf_values.reshape([(N + 1) * A, -1]),
            old_gamma=args.tvf.gamma,
            new_gamma=new_gamma,
            horizons=self.runner.tvf_horizons,
        ).reshape([(N + 1), A])

    def rediscount_horizons(self, old_value_estimates):
        """
        Input is [B, K]
        Output is [B, K]
        """
        if args.tvf.gamma == args.gamma:
            return old_value_estimates

        # old_distil_targets = batch_data["distil_targets"].copy()  # B, K
        new_value_estimates = old_value_estimates.copy()
        B, K = old_value_estimates.shape
        for k in range(K):
            new_value_estimates[:, k] = rl.tvf.get_rediscounted_value_estimate(
                old_value_estimates[:, :k+1],
                args.tvf.gamma,
                args.gamma,
                self.runner.tvf_horizons[:k+1]
            )
        return new_value_estimates

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
    assert args.tvf.gamma == 1, "General discounting function requires TVF estimates to be undiscounted (might fix later)"
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


def horizon_interpolate(horizons: np.ndarray, values: np.ndarray, target_horizons: np.ndarray):
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
