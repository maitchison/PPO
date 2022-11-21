import unittest

import rl.returns as returns
import rl.tvf as tvf
import numpy as np
import time as clock

class TestTVF(unittest.TestCase):

    def test_return_estimators(self):
        # create random data...
        # k are horizons we need to generate
        # v are horizons we have estimates for
        N, A, K, V = [128, 16, 16, 32]
        default_args = {
            'gamma': 0.9997,
            'rewards': np.random.random_integers(-1, 2, [N, A]).astype('float32'),
            'dones': (np.random.random_integers(0, 100, [N, A]) >= 98),
            'required_horizons': np.geomspace(1, 1024, num=K).astype('int32'),
            'value_sample_horizons': np.geomspace(1, 1024, num=V).astype('int32') - 1,
            'value_samples': np.random.normal(0.1, 0.4, [N + 1, A, V]).astype('float32'),
        }

        default_args['value_samples'][:, :, 0] *= 0  # h=0 must be zero

        def verify(label: str, **kwargs):

            args = default_args.copy()
            args.update(kwargs)

            start_time = clock.time()
            m1_ref = returns._calculate_sampled_return_multi_reference(**args)
            r1_time = clock.time() - start_time

            start_time = clock.time()
            m1 = returns._calculate_sampled_return_multi_threaded(**args)
            r2_time = clock.time() - start_time

            delta_m1 = np.abs(m1_ref - m1)

            e_m1 = delta_m1.max()

            ratio = r1_time / r2_time

            # note fp32 has about 7 sg fig, so rel error of around 1e-6 is expected.
            if e_m1 > 1e-5 * (m1_ref.max()):
                print(f"Times {r1_time:.2f}s / {r2_time:.2f}s ({ratio:.1f}x), error for {label} = {e_m1:.6f}")
                return False
            return True

        def print_sample_error():
            """
            Returns the approximate error due to sampling.
            """
            n_step = 20
            args = default_args.copy()
            ref = returns.get_return_estimate(
                distribution='exponential',
                mode='full',
                n_step=n_step,
                **args
            )

            xs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
            y_adv = []
            y_std = []

            for samples in xs:
                adv = []
                std = []
                for _ in range(10):
                    adv.append(returns.get_return_estimate(
                        distribution='exponential',
                        max_samples=samples,
                        mode='advanced',
                        n_step=n_step,
                        **args
                    ))
                    std.append(returns.get_return_estimate(
                        distribution='exponential',
                        max_samples=samples,
                        mode='standard',
                        n_step=n_step,
                        **args
                    ))
                adv = np.asarray(adv)  # S,N,A,K
                std = np.asarray(std)  # S,N,A,K

                adv_mae = np.mean([np.mean(np.abs(ref - adv[i])) for i in range(len(adv))])
                std_mae = np.mean([np.mean(np.abs(ref - std[i])) for i in range(len(std))])
                adv_bias = np.mean([np.mean(ref - adv[i]) for i in range(len(adv))])
                std_bias = np.mean([np.mean(ref - std[i]) for i in range(len(std))])

                print(f"{samples}:",
                      adv_mae, std_mae, adv_bias, std_bias
                      )

                y_adv.append(adv_mae)
                y_std.append(std_mae)
            import matplotlib.pyplot as plt
            xs = [str(x) for x in xs]
            plt.plot(xs, y_adv, label='adv')
            plt.plot(xs, y_std, label='std')
            plt.legend()
            plt.grid(alpha=0.25)
            plt.show()

        n_step = 20
        n_samples = 8
        lamb = 1 - (1 / n_step)
        weights = np.asarray([lamb ** x for x in range(128)], dtype=np.float32)
        max_n = len(weights)
        probs = weights / weights.sum()
        samples = np.random.choice(range(1, max_n + 1), [K, n_samples], replace=True, p=probs)

        self.assertTrue(verify("n_step:1", n_step_list=[1]))
        self.assertTrue(verify("n_step:8", n_step_list=[8]))
        self.assertTrue(verify("n_step:128", n_step_list=[128]))
        self.assertTrue(verify("exponential:20", n_step_samples=samples))

    def test_interpolation(self):

        horizons = np.asarray([0, 1, 2, 10, 100])
        values = np.asarray([0, 5, 10, -1, 2])[None, :].repeat(11, axis=0)
        results = tvf.horizon_interpolate(horizons, values, np.asarray([-100, -1, 0, 1, 2, 3, 4, 99, 100, 101, 200]))
        expected_results = [0, 0, 0, 5, 10, (7 / 8) * 10 + (1 / 8) * -1, (6 / 8) * 10 + (2 / 8) * -1, 1.96666667, 2,
                            2, 2]
        max_abs_error = np.max(np.abs(np.asarray(expected_results) - results))
        self.assertLess(max_abs_error, 1e-6, f"Expected {expected_results} found {results}")


