import blosc
import time as clock
import pickle
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

from rl.returns import get_return_estimate
from rl.utils import is_sorted
from rl.returns import _interpolate

from rl.returns import _get_geometric_n_steps

class Checkpoint():

    def __init__(self, log_folder, step:int, samples=64, use_float16: bool=True, verify: bool = True, max_n=None):
        """
            Loads checkpoint at given step.
            @param use_float16 stores value and return estiamtes as float16, which is usually ok, and halves the memory
                requirements.
            @param max_n caps rollout to n steps.
        """

        very_start_time = clock.time()

        results = {}

        no_stack_vars = ["seed", "gamma", "value_sample_horizons", "required_horizons", "reward_scale"]

        # 1. load from disk
        start_time = clock.time()
        for seed in range(samples):
            with open(f"{log_folder}/rollouts_{step}_{seed}.dat", "rb") as f:
                data = pickle.load(f)
                for k, v in data.items():
                    if type(v) is bytes:
                        v = blosc.unpack_array(v)
                        if not use_float16 and v.dtype == np.float16:
                            v = v.astype('float32') # saved as float16 just for space saving, but much faster with float32.
                    if k not in no_stack_vars:
                        if k not in results:
                            results[k] = []
                        if max_n is None:
                            cap = None
                        else:
                            cap = max_n + 1 if "value_samples" in k else max_n
                        results[k].append(v[:cap])
                    else:
                        results[k] = v
        load_time = clock.time() - start_time

        # 2. merge together with NSA order
        start_time = clock.time()
        for k in results.keys():
            v = results[k]
            if k not in no_stack_vars:
                results[k] = np.stack(v, axis=1)
        merge_time = clock.time() - start_time


        self.rewards = results['rewards']                           # [N, S, A]
        self.dones = results['dones']                               # [N, S, A]
        self.required_horizons = results['required_horizons']       # [K]
        self.value_sample_horizons = results['value_sample_horizons'] # [V]
        value_samples = results['value_samples']                    # [N+1, S, A, V]
        value_samples_m2 = results['value_samples_m2']              # [N+1, S, A, V]
        self.all_times = results['all_times']                       # [N, S, A]
        self.mask = results['mask']                                 # [N, S, A]
        self.gamma:float = results['gamma']
        self.reward_scale:float = results['reward_scale']

        N, S, A = self.rewards.shape
        K = len(self.required_horizons)

        # required horizons is simply a list of horizons from 1 to max_n spaced geometrically
        # I generate n_step estimates only at these horizons, so evaluations can only be done
        # at required_horizons.
        if max_n is None:
            max_n = len(self.rewards)
        self.required_horizons = np.round(np.geomspace(1, max_n, num=1+2*int(math.log2(max_n)))).astype('int32')
        self.required_horizons = sorted(set(self.required_horizons)) # remove duplicates

        self.MAX_N = min(1024, max_n) # this is the largest N that can be used when generating estimates (but not for targets)

        # value sample horizons are the horizons at which we took value samples during the rollout
        assert is_sorted(self.value_sample_horizons)

        # quick verify
        if verify:
            re1 = self.generate_return_moments_reference(value_samples[:10])
            re2, _ = self.generate_return_moments(value_samples[:10], value_samples_m2[:10])
            err = np.abs(re1-re2).max()
            if err > 1e-6:
                print(f"Return moments delta from reference was {err}")

        # 3. calculate all return estimates
        start_time = clock.time()
        # note: return estimates are for the first state only (this is required due to the nature of the samples
        self.n_step_return_estimate, self.n_step_return_estimate_m2 = self.generate_return_moments(value_samples, value_samples_m2)
        # all seeds should agree on the first state prediction, so just pick seed 0
        self.second_moment_predictions = value_samples_m2[0, 0, :, :].astype('float32').copy() # NSAV -> AV
        self.first_moment_predictions = value_samples[0, 0, :, :].astype('float32').copy()  # NSAV -> AV
        return_time = clock.time() - start_time

        self.est_return_var = np.zeros([self.MAX_N, A], dtype=np.float32)
        for n in range(self.MAX_N):
            m1 = _interpolate(self.value_sample_horizons, self.first_moment_predictions, n)
            m2 = _interpolate(self.value_sample_horizons, self.second_moment_predictions, n)
            var_estimates = np.clip(m2 ** 2 - m1 ** 2, 0, float('inf'))
            self.est_return_var[n, :] = var_estimates

        full = np.mean(self.mask)

        total_time = clock.time() - very_start_time

        # 4. calculate ground truth returns for first state
        # note that the final n_step estimate is simply the sum of rewards
        self.true_return_mean = self.n_step_return_estimate[:, -1, :, :].astype('float32').mean(axis=1).transpose(1, 0)  # KNSA->KA->AK
        self.true_sqrt_return_m2 = (self.n_step_return_estimate[:, -1, :, :].astype('float32')**2).mean(axis=1).transpose(1, 0) ** 0.5 # KNSA->KA->AK
        self.true_return_var = self.n_step_return_estimate[:, -1, :, :].astype('float32').var(axis=1).transpose(1, 0)  # KNSA->KA->AK
        self.true_return_var_from_moment = self.true_sqrt_return_m2 ** 2 - self.true_return_mean ** 2 # AK

        # 5. set other variables
        self.CACHE = {} # used to cache return estimations, might not be needed now??

        # 6. store some useful calculations for later
        # this is an estimate of the variance of the truncated return of agent a up to step n.
        self.true_interpolated_return_var = np.zeros([self.MAX_N, A], dtype=np.float32)
        for n in range(self.MAX_N):
            # todo: check these are right..
            self.true_interpolated_return_var[n, :] = self.interpolate_variance(n + 1, slice(None, None))

        # estimate of the mean-squared-error of the n_step=n estimator on horizon required_horizons[k] for agent a.
        # slow and not helpful...
        # self.mse_estimates = np.zeros([self.MAX_N, len(self.required_horizons), A], dtype=np.float32)
        # for a in range(A):
        #     for n in range(self.MAX_N):
        #         for k, h in enumerate(self.required_horizons):
        #             self.mse_estimates[n, k, a] = self.interpolate_mse(n+1, h, a)

        if verify:
            self.quick_check(value_samples)

        print(f"Loading finished, took {load_time:.1f}s/{merge_time:.1f}s/{return_time:.1f}s [{total_time:.1f}s total] ({100*full:.1f}% full)")

    def quick_check(self, value_samples: np.ndarray):
        """
        Verifies generated return estimates against returns.py estimates.
        """
        # tests
        r1 = self.generate_return_estimates('fixed', n_step=4)
        r2 = self.generate_return_estimates_reference('fixed', n_step=4, value_samples=value_samples)
        delta = np.abs(r1 - r2)
        print("n_step 4 (fixed) error: ", delta.mean(), "/", delta.max())

        r1 = self.generate_return_estimates('fixed', n_step=16)
        r2 = self.generate_return_estimates_reference('fixed', n_step=16, value_samples=value_samples)
        delta = np.abs(r1 - r2)
        print("n_step 16 (fixed) error: ", delta.mean(), "/", delta.max())

        # r1 = checkpoint.generate_return_estimates('linear', n_step=4)
        # r2 = checkpoint.generate_return_estimates_old('linear', n_step=4)
        # delta = np.abs(r1 - r2)
        # print("Linear Error: ", delta.mean(), "/", delta.max())  # will be off a little due to sampling used in old method
        # # print(delta)
        # print(r1.shape)
        # print(r1[0, 0, :])
        # print("---")
        # print(r2[0, 0, :])
        # print((r1 - r2)[0, 0, :])
        # print("---")
        # r1 = checkpoint.generate_return_estimates('linear', n_step=1)
        # r2 = checkpoint.generate_return_estimates_old('fixed', n_step=1)
        # print((r1 - r2)[0, 0, :])

    def generate_return_moments_reference(self, value_samples: np.ndarray, verbose: bool = False):
        """
        This is the old version, I'll use this to check against for the newer faster version
        """

        N, S, A, V = value_samples.shape
        N = N - 1
        K = len(self.required_horizons)

        n_step_return_estimate = np.zeros([K, N+1, S, A], dtype=np.float32)
        discount = np.ones([S, A], dtype=np.float32)

        for t in tqdm(range(N), disable=not verbose):
            for i, h in enumerate(self.required_horizons):
                if t < h:
                    n_step_return_estimate[i, t + 1:] += (self.rewards[t] * discount)[None, :, :]
            discount *= self.gamma
            discount *= (1 - self.dones[t])
            for i, h in enumerate(self.required_horizons):
                if h - (t + 1) > 0:
                    boot_strap_values = _interpolate(self.value_sample_horizons, value_samples[t + 1].astype('float32'),
                                                     h - (t + 1))
                    n_step_return_estimate[i, t + 1] += boot_strap_values * discount

        # we calculated the n_step=0 result, which we remove here...
        n_step_return_estimate = n_step_return_estimate[:, 1:].copy()  # make contig

        return n_step_return_estimate

    def generate_return_moments(self, value_samples: np.ndarray, value_samples_m2: np.ndarray, verbose=False):
        """
        New faster version, with support for second moment.
        """

        N, S, A, V = value_samples.shape
        N = N - 1
        K = len(self.required_horizons)

        # Generate all n_step estimates
        # this is the kth horizon (n+1)th n_step estimate for sample s, and agent a.

        # first work out the cumulative sum [N, S, A]
        # cum_sum [N,S,A] = sum_{i=0}^{n+1} [ gamma^i r_i ]
        cum_sum = np.zeros([N, S, A], dtype=np.float32)
        current_sum = np.zeros([S, A], dtype=np.float32)
        discount = np.ones([S, A], dtype=np.float32)
        for n in range(N):
            current_sum += self.rewards[n] * discount
            discount *= self.gamma
            discount *= (1 - self.dones[n])
            cum_sum[n] = current_sum

        S_part = np.zeros([K, N, S, A], dtype=value_samples.dtype)
        M_part = np.zeros([K, N, S, A], dtype=value_samples.dtype)
        M2_part = np.zeros([K, N, S, A], dtype=value_samples.dtype)
        discount = np.ones([S, A], dtype=np.float32)
        for n in tqdm(range(N), disable=not verbose):

            # S part
            for i, h in enumerate(self.required_horizons):
                S_part[i, n] = cum_sum[min(n+1, h)-1]

            # M part
            discount *= self.gamma
            discount *= (1 - self.dones[n])

            # this happens when all agents have completed, no need to bootstrap in these cases...
            if discount.sum() == 0:
                continue

            for i, h in enumerate(self.required_horizons):
                if h - (n + 1) > 0:
                    boot_strap_values = _interpolate(self.value_sample_horizons, value_samples[n + 1],
                                                     h - (n + 1)) * discount
                    boot_strap_values_m2 = _interpolate(self.value_sample_horizons, value_samples_m2[n + 1],
                                                        h - (n + 1)) * (discount ** 2)
                    M_part[i, n] = boot_strap_values
                    M2_part[i, n] = boot_strap_values_m2

        first_moment = (S_part + M_part)
        second_moment = (S_part**2 + 2*S_part*M_part + M2_part)

        return first_moment, second_moment

    def interpolate_variance(self, h, a=None):
        """
        Returns (interpolated) variance of return for agent a at horizon h.
        """
        return _interpolate(self.required_horizons, self.true_return_var[a, :], h)

    def interpolate_mse(self, n_step, h, a=None):
        """
        Returns (interpolated) mse of n_step return estimate for agent a at horizon h.
        """
        mse_errors = np.asarray([self.calc_metric(metric='mse', mode='fixed', n_step=n_step, agent=a, h=h) for h in self.required_horizons])
        return _interpolate(self.required_horizons, mse_errors, h)

    def get_dist_weights(self, mode: str, n_step: int):
        """
        Get n_step weights for given return estimator 'mode' with parameter n_steps

        @returns np.ndarray of dims [<MAX_N] containing weights for each n_step estimate. May not be full length,
            in which case remaining weights are 0. These weights sum to 1.0.
        """

        if mode == "fixed":
            weights = None
        elif mode == "geometric":
            rho = float(np.clip(1 + (n_step / 100), 1.05, 100))
            weights = np.asarray([1 if (1 + n) in _get_geometric_n_steps(rho, self.MAX_N) else 0 for n in range(self.MAX_N)],
                                 dtype=np.float32)
        elif mode == "hyperbolic":
            k = 10 / n_step
            weights = 1 / (1 + k * np.arange(self.MAX_N))
        elif mode == "inv_sqr":
            k = 0.1 / n_step
            weights = 1 / (1 + k * np.arange(self.MAX_N) ** 2)
        elif mode == "linear":
            # linear has average of n_step, so but goes to n_step*2
            weights = (1 - (np.arange(0, n_step * 2) / (n_step * 2)))
        elif mode == "uniform":
            weights = np.ones([n_step], dtype=np.float32)
        elif mode == "exponential":
            lamb = 1 - (1 / n_step)
            cap_n = min(self.MAX_N, n_step*10) # much faster... and probably won't make much difference
            weights = np.asarray([lamb ** x for x in range(cap_n)], dtype=np.float32)
        elif mode == "exponential_cap":
            lamb = 1 - (1 / n_step)
            cap_n = min(self.MAX_N, n_step*10) # much faster... and probably won't make much difference
            weights = np.asarray([lamb ** x for x in range(cap_n)], dtype=np.float32)
            remaining_weight = (lamb ** cap_n)/(1-lamb)
            weights[-1] += remaining_weight
        elif mode == "exponential_cap2":
            lamb = 1 - (1 / n_step)
            cap_n = min(self.MAX_N, n_step*10) # much faster... and probably won't make much difference
            weights = np.asarray([lamb ** x for x in range(cap_n)], dtype=np.float32)
            remaining_weight = (lamb ** cap_n)/(1-lamb)
            pos = len(weights)-1
            while remaining_weight > 1e-6 and pos > 0:
                space_free = 4-weights[pos]
                transfer_amount = min(space_free, remaining_weight, 4)
                weights[pos] += transfer_amount
                remaining_weight -= transfer_amount
                pos -= 1
        else:
            raise ValueError(f"Invalid mode {mode}")

        if weights is not None:
            weights = weights.astype("float32")[:self.MAX_N]
            weights /= weights.sum()

        return weights

    def generate_return_estimates(self, mode: str, n_step: int, no_cache=False, samples=None, k=None):
        """
        Returns a np array of dims [S, A, K] containing return estimates using given mode for each of the seeds.
        (if k is set returns [S, A])
        """

        K, N, S, A = self.n_step_return_estimate.shape

        if n_step <= 0:
            raise ValueError(f"n_step must be >= 1 but was {n_step}")

        if samples is None and (not no_cache):
            key = ('gre', mode, n_step, k)
            if key in self.CACHE:
                return self.CACHE[key]
        else:
            key = None

        if k is None:
            k_selector = slice(None, None)
        else:
            k_selector = slice(k, k+1)

        if mode == "var_weighted":
            assert samples is None, "not supported yet"
            assert k is not None
            h = self.required_horizons[k]
            # this is a special case, as weights are state dependant (not just horizon dependant)
            weights = 1/((0.01/(n_step/h)) + self.true_interpolated_return_var)[:self.MAX_N] # NA

            # this worked quite well, but n_step needed tuning...
            # weights = 1/(0.01+10*n_step*self.var_estimates) # NA
            # weights = np.clip(weights, 0, 10)

            #max_h = self.required_horizons[k]*2
            #weights[max_h:] *= 0 # do not use longer n_step estimates than the horizon.
            weights = weights / weights.sum(axis=0)[None, :] # normalize so N dim sums to 1
            weights = weights[None, :, None, :] # NA->1N1A
            weighted_estimates = self.n_step_return_estimate[k_selector, :self.MAX_N].astype('float32') * weights # KNSA -> KNSA
            weighted_estimates = weighted_estimates.sum(axis=1) # weighted average over N, KNSA->KSA
            weighted_estimates = weighted_estimates.transpose((1, 2, 0))  # KSA-> SAK
            if k is not None:
                weighted_estimates = weighted_estimates[:, :, 0]  # last dim will be 1 anyway.
            if key is not None:
                self.CACHE[key] = weighted_estimates
            return weighted_estimates

        if mode == "est_var_weighted":
            # estimated variance weighted...

            assert samples is None, "not supported yet"
            assert k is not None
            h = self.required_horizons[k]
            # work out out estiamted variance

            # this is a special case, as weights are state dependant (not just horizon dependant)
            weights = 1/((0.01/(n_step/h))+self.est_return_var)[:self.MAX_N] # NA

            # this worked quite well, but n_step needed tuning...
            # weights = 1/(0.01+10*n_step*self.var_estimates) # NA
            # weights = np.clip(weights, 0, 10)

            #max_h = self.required_horizons[k]*2
            #weights[max_h:] *= 0 # do not use longer n_step estimates than the horizon.
            weights = weights / weights.sum(axis=0)[None, :] # normalize so N dim sums to 1
            weights = weights[None, :, None, :] # NA->1N1A
            weighted_estimates = self.n_step_return_estimate[k_selector, :self.MAX_N].astype('float32') * weights # KNSA -> KNSA
            weighted_estimates = weighted_estimates.sum(axis=1) # weighted average over N, KNSA->KSA
            weighted_estimates = weighted_estimates.transpose((1, 2, 0))  # KSA-> SAK
            if k is not None:
                weighted_estimates = weighted_estimates[:, :, 0]  # last dim will be 1 anyway.
            if key is not None:
                self.CACHE[key] = weighted_estimates
            return weighted_estimates

        if mode == "mse_weighted":
            assert samples is None, "not supported yet"
            assert k is not None

            # this should be var(s) + MSE(m) we can estimate MSE(m) by substracting off the MSE of the first part...
            # e.g. approximate error for returns 50...125 as error to 125 minus error to 50
            max_h = self.required_horizons[k]
            var_part = self.true_interpolated_return_var # NA
            m_error = np.clip(self.mse_estimates[max_h-1:max_h, k, :] - self.mse_estimates[:, k, :], 0, float('inf'))
            m_error *= 0
            #m_error = self.mse_estimates[:, k, :] # NKA->NA

            # this is a special case, as weights are state dependant (not just horizon dependant)
            weights = 1/(0.00001+n_step*(var_part+m_error))  # NKA -> NA

            weights[max_h:] *= 0 # do not use longer n_step estimates than the horizon.
            weights = weights / weights.sum(axis=0)[None, :] # normalize so N dim sums to 1
            weights = weights[None, :, None, :] # NA->1N1A
            weighted_estimates = self.n_step_return_estimate[k_selector, :self.MAX_N].astype('float32') * weights # KNSA -> KNSA
            weighted_estimates = weighted_estimates.sum(axis=1) # weighted average over N, KNSA->KSA
            weighted_estimates = weighted_estimates.transpose((1, 2, 0))  # KSA-> SAK
            if k is not None:
                weighted_estimates = weighted_estimates[:, :, 0]  # last dim will be 1 anyway.
            if key is not None:
                self.CACHE[key] = weighted_estimates
            return weighted_estimates

        if mode == "heuristic2":
            assert samples is None, "not supported yet"
            # the idea here is to keep extending the horizon until we meet high variance, then stop
            k_length = len(np.arange(K)[k_selector])
            estimates = np.zeros([S, A, k_length], dtype=np.float32)
            for a in range(A):
                accumulated_variance = 0
                j = 0
                for j in range(len(self.required_horizons)):
                    accumulated_variance += self.true_return_var[a, j]
                    if accumulated_variance > (0.01/n_step):  # this used to be 0.001
                        break
                heuristic2 = int(self.required_horizons[j] * 1)
                # next we generate the estimate using exponential returns...
                # note: it's quite slow generating estimates for all agents, but then using only one.
                exp_est = self.generate_return_estimates(mode='exponential', n_step=heuristic2, no_cache=no_cache, k=k)
                if len(exp_est.shape) == 2:
                    exp_est = exp_est[:, :, None] # expand to SAK
                estimates[:, a, :] = exp_est[:, a, :]
            if k is not None:
                estimates = estimates[:, :, 0]
            return estimates

        weights = self.get_dist_weights(mode, n_step)

        # n_step_return_estimate is KNSA
        if mode == "fixed":
            result = self.n_step_return_estimate[k_selector, np.clip(n_step, 1, self.MAX_N) - 1].astype('float32') # KNSA-> KSA
        else:
            if samples is None:
                result = np.average(self.n_step_return_estimate[k_selector, :len(weights)].astype('float32'), axis=1, weights=weights)
            else:
                horizon_samples = np.random.choice(len(weights), size=[samples], p=weights, replace=True)
                result = np.mean(self.n_step_return_estimate[k_selector, horizon_samples].astype('float32'), axis=1)

        result = result.transpose((1, 2, 0))  # KSA-> SAK

        if k is not None:
            result = result[:, :, 0]  # last dim will be 1 anyway.

        if key is not None:
            self.CACHE[key] = result
        return result

    def _get_seeded_return_estimate(self, mode: str, seed: int, n_step:int, value_samples: np.ndarray):
        """
        Gets return estimate for given mode, with given parameters, on given rollout seed.
        returns np array of dims [N, A, K] representing the return estimates for each state, agent, horizon
        """

        if mode == "fixed":
            max_n = n_step # much faster
        else:
            max_n = len(self.rewards)

        return get_return_estimate(
            mode,
            n_step=n_step,
            gamma=self.gamma,
            rewards=self.rewards[:max_n, seed],
            dones=self.dones[:max_n, seed],
            required_horizons=np.asarray(self.required_horizons),
            value_sample_horizons=self.value_sample_horizons,
            value_samples=value_samples[:max_n+1, seed].astype('float32'),
        )

    def generate_return_estimates_reference(self, mode: str, n_step:int, value_samples:np.ndarray):
        """
        Returns a np array of dims [S, A, K] containing return estimates using given mode for each of the seeds,
        (for the first state only).
        """

        K, N, S, A = self.n_step_return_estimate.shape

        results = np.zeros([S, A, K], dtype=np.float32)
        for seed in range(S):
            results[seed] = self._get_seeded_return_estimate(
                mode=mode, seed=seed, n_step=n_step,
                value_samples=value_samples)[0]
        return results

    def calc_metric(self, metric: str, mode: str, n_step: int, h: int, samples=None, agent=None):
        """
        Calculates a return estimate for given n_step value and horizon
        e.g. calc_metric('tse', 'fixed', 80, 300)

        @param agent: if defined selects only one of the agents to calculate metric on.

        """

        K, N, S, A = self.n_step_return_estimate.shape

        if samples is None:
            key = ('cm', metric, mode, n_step, h, agent)
            if key in self.CACHE:
                return self.CACHE[key]
        else:
            key = None

        if metric.startswith("log_"):
            result = self.calc_metric(metric[4:], mode, n_step, h, samples, agent)
            return np.log10(np.clip(result, 1e-8, float('inf')))
        if metric.startswith("1m_"):
            result = self.calc_metric(metric[3:], mode, n_step, h, samples, agent)
            return 1 - result

        if metric.startswith("mean_"):
            result = 0
            metric = metric[4:]
            for h in self.required_horizons:
                result += self.calc_metric(metric, mode, n_step, h, samples, agent)
            return result / len(self.required_horizons)

        if metric.startswith("weighted_"):
            metric = metric[len("weighted_"):]
            MAX_H = 2048
            w = 1 + ((MAX_H - h) / 100)  # approximately how many times an error will an error be copied with average n_step of 100
            result = self.calc_metric(metric, mode, n_step, h, samples, agent)
            return result * w

        assert h in self.required_horizons

        k = list(self.required_horizons).index(h)
        return_estimates = self.generate_return_estimates(mode, n_step, samples=samples, k=k) # SA

        def get_mse():
            # including sigma...
            if agent is None:
                return np.square(return_estimates - self.true_return_mean[None, :, k]).mean()  # SA -> [1]
            else:
                return np.square(return_estimates[:, agent] - self.true_return_mean[None, agent, k]).mean()  # SA -> [1]

        def get_sq_bias():
            # squared_bias is the (expected prediction - true_return)^2 (averaged over all randomly sampled states)
            if agent is None:
                return np.square(return_estimates.mean(axis=0) - self.true_return_mean[:, k]).mean(axis=0)  # SA -> [1]
            else:
                return np.square(return_estimates[:, agent].mean(axis=0) - self.true_return_mean[agent, k])  # SA -> [1]

        def get_variance():
            # variance is the var(predictions) (averaged over all randomly sampled states)
            if agent is None:
                # average over all agents
                return np.var(return_estimates, axis=0).mean(axis=0)  # SAK -> [1]
            else:
                # select out one agent
                return np.var(return_estimates[:, agent])  # SAK -> [1]

        if metric == 'ev':
            total_var_y = 0.0 + 1e-6  # bias this a little... will avoid / 0.
            unexplained_var = 0
            for s in range(S):
                y_pred = return_estimates[s]
                y_true = self.true_return_mean[:, k]
                total_var_y += np.var(y_true)
                unexplained_var += np.var(y_true - y_pred)
            if total_var_y == 0:
                result = 0  # seems most fitting... ?
            else:
                result = np.clip(1 - unexplained_var / total_var_y, -1, 1)
        elif metric == "sq_bias":
            result = get_sq_bias()
        elif metric == "var":
            result = get_variance()
        elif metric == "mse":
            result = get_mse()
        elif metric == "sigma":
            result = get_mse() - (get_sq_bias() + get_variance())
        elif metric == "tse":
            result = get_sq_bias() + get_variance()
        elif metric == "av_bias":
            assert agent is None
            # todo: check if this is right still..
            result = (return_estimates.mean(axis=0) - self.true_return_mean).mean(axis=0)
        else:
            raise ValueError(f"Invalid metric {metric}")

        if key is not None:
            self.CACHE[key] = result
        return result


# ------------------------------------------------------------------------------------------------------------------
# plotting

# show how well we perform at each horizon with a given method, over a set of hyperparameters
def plot_n_step_return(data, metric='ev', mode="fixed", hold=False, fig=True, title=None, n_steps=None):
    if fig:
        plt.figure(figsize=(8, 3))
        plt.grid(True, alpha=0.25)
        plt.title(title or metric)

    n_steps = [1, 2, 4, 6, 8, 16, 32, 64, 128, 256, 512, 1024] if n_steps is None else n_steps

    for n in n_steps:
        scores = []
        # calculate ev and log results
        for h in data.required_horizons:
            scores.append(data.calc_metric(metric, mode, n, h))

        label = f"n_step_{n} ({metric})"
        label = None
        plt.plot(scores, label=label, color=data.get_h_color(n))

    plt.xticks(range(len(data.required_horizons)), labels=data.required_horizons)
    plt.xlabel("h")
    plt.legend()
    if not hold:
        plt.show()


def plot_horizon_curve(data, metric='ev', mode="fixed", hold=False, fig=(12, 5), title=None, n_steps=None, style="-",
                       horizons=None, c=None, alpha=1.0, label=None, max_n=2048):
    if fig:
        plt.figure(figsize=fig)
        plt.grid(True, alpha=0.25)
        if title is None:
            title = f"{metric} {mode}"
        plt.title(title)

    results = {}

    if horizons is None:
        horizons = data.required_horizons
    if n_steps is None:
        n_steps = data.required_horizons

    for n in n_steps:
        # calculate ev and log results
        for h in horizons:
            results[(n, h)] = data.calc_metric(metric, mode, n, h)

    for h in horizons:
        l = f"h={h}" if label is None else label
        xs = n_steps
        ys = [results[n, h] for n in n_steps]
        plt.plot(xs, ys, label=l, color=c or data.get_h_color(h), ls=style, alpha=alpha)
        minx = xs[np.argmin(ys)]
        plt.scatter([minx], np.min(ys), color=c or data.get_h_color(h), alpha=alpha)

    plt.gca().set_xscale('log')
    ticks = [2 ** int(x) for x in range(int(1 + math.log2(max_n)))]
    plt.xticks(ticks, labels=ticks)
    plt.xlabel("n_steps")
    plt.ylabel(metric)
    plt.legend()
    if not hold:
        plt.show()


def plot_hp_curve(data, horizon: int, metric='tse', mode="fixed", hold=False, fig=(8, 4), color='red', include_ref=True,
                  max_n: int = 2048, include_refs:bool = False, **kwargs):
    """
    Plot hyperparmater curve for given return estimator, using n_step=1, n_step=max, and n_step=optimal as references
    """

    plot_horizon_curve(data, metric=metric, mode=mode, horizons=[horizon], hold=True, fig=fig, c=color, label=mode, **kwargs)
    if include_ref:
        plot_horizon_curve(data, metric=metric, mode='fixed', horizons=[horizon], **kwargs, hold=True, style="-", fig=False, c="gray", alpha=0.25, label='')

        # mark the optimal n_step(s)
    #     for n in range(1, MAX_N+1):
    #         if abs(n_step_results[n-1] - best_n_step_result) < 1e-6:
    #             plt.scatter([n], [best_n_step_result], color="yellow")

    plt.title(f"{metric} h={horizon}")

    if not hold:
        n_step_results = []
        if include_refs:
            for n in set(np.geomspace(1, 1024, num=40, dtype=np.int32)):
                n_step_results.append(data.calc_metric(metric, "fixed", n, horizon))
            best_n_step = 1 + np.argmax(n_step_results)
            best_n_step_result = min(n_step_results)

            xs = [1, max_n]

            plt.hlines(n_step_results[0], *xs, ls="--", color="gray", label="td(0)", zorder=100)
            plt.hlines(n_step_results[-1], *xs, ls="-", color="silver", label="MC", zorder=100)
            plt.hlines(best_n_step_result, *xs, ls="--", color="yellow", label="best n_step", zorder=100)

        plt.legend()
        plt.show()


def get_h_color(h):
    return plt.get_cmap('tab20')(int(math.log2(h)))

def hp_plot(data, mode: str, metric: str = "tse", max_n=2048, hold=False):

    plt.figure(figsize=(8, 4))
    plt.title(f"{mode} - {metric}")

    # no need to check h=1
    xs = np.asarray(data.required_horizons)
    ys = []

    window = set(range(1, max_n + 1))

    for h in xs:
        scores = []
        for n in data.required_horizons:
            scores.append(data.calc_metric(metric, mode, n, h))
        best_n = data.required_horizons[np.argmin(scores)]
        ys.append(best_n)

        # show values better than ref
        n_step_best = np.min([data.calc_metric(metric, "fixed", n, h) for n in data.required_horizons])
        better_ys = []
        for n, score in zip(data.required_horizons, scores):
            delta = n_step_best - score
            if delta > -1e-6:
                better_ys.append(n)
        plt.scatter([h] * len(better_ys), better_ys, color='yellow', marker='x', alpha=0.25)
        window = window.intersection(better_ys)

    ys = np.asarray(ys)
    plt.xscale('log')
    plt.yscale('log')

    if mode == "geometric":
        ys = 1 + (ys / 100)
        plt.ylabel('rho')
    else:
        plt.ylabel('n_step')
        ticks = [2 ** int(x) for x in range(int(1 + math.log2(max_n)))]
        plt.xticks(ticks, labels=ticks)

    if len(window) > 0:
        plt.hlines(min(window), 0, 2048, color='gray')
        plt.hlines(max(window), 0, 2048, color='gray')

    # plot y=x
    plt.plot(xs, xs, ls='--', alpha=0.5, color='gray')

    plt.plot(xs, ys)
    plt.xlabel('horizon')

    if not hold:
        plt.grid(True, alpha=0.25)
        plt.plot()
        if len(window) > 0:
            print(f"{mode} {metric} {min(window)}'-'{max(window)} ({(max(window) + min(window)) / 2:.1f}) ")


if __name__ == "__main__":
    # 2 samples is 6s / 2.3g
    # 8 samples is 41s / 10g
    # 8 samples is 25s / 5g (float16 version)
    LOG_FOLDER = "./Run/DVQ1/env=CrazyClimber [1adebac8]/"
    data = Checkpoint(LOG_FOLDER, step=0, samples=8, max_n=256) # just to limit the memory requirements for testing.
    hp_plot(data, "exponential")
