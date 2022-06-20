# test for v-trace (and gae and others I guess)

import torch
import numpy as np
from rl import ppo, rollout
from rl.vtrace import importance_sampling_v_trace, v_trace_trust_region
from rl.utils import entropy, log_entropy, log_kl, kl
from rl.rollout import  interpolate

def is_similar(a, b):
    delta = np.abs(a-b)
    return np.max(delta) < 1e-4

def assert_is_similar(x, truth, message = "Failed test."):
    x = np.asarray(x)
    truth = np.asarray(truth)
    assert x.shape == truth.shape, f"Shapes must match, {x.shape}, {truth.shape}"

    assert is_similar(x, truth), message+" expected \n{}\n found \n{}\n error \n{}\n".format(truth,x, truth-x)

def select_from(a,b):
    N,B = b.shape
    r = np.zeros_like(b)
    for i in range(N):
        for j in range(B):
            r[i,j] = a[i,j,b[i,j]]
    return r

def _ground_truth_vtrace_calculation(discounts, log_rhos, rewards, values,
                                     bootstrap_value, clip_rho_threshold,
                                     clip_pg_rho_threshold):
  """Calculates the ground truth for V-trace in Python/Numpy."""

  # this is the ground truth calculation from https://github.com/deepmind/scalable_agent/blob/master/vtrace_test.py

  vs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  clipped_rhos = rhos
  if clip_rho_threshold:
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
  clipped_pg_rhos = rhos
  if clip_pg_rho_threshold:
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

  # This is a very inefficient way to calculate the V-trace ground truth.
  # We calculate it this way because it is close to the mathematical notation of
  # V-trace.
  # v_s = V(x_s)
  #       + \sum^{T-1}_{t=s} \gamma^{t-s}
  #         * \prod_{i=s}^{t-1} c_i
  #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
  # Note that when we take the product over c_i, we write `s:t` as the notation
  # of the paper is inclusive of the `t-1`, but Python is exclusive.
  # Also note that np.prod([]) == 1.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    v_s = np.copy(values[s])  # Very important copy.
    for t in range(s, seq_len):
      v_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t],
                                                    axis=0) * clipped_rhos[t] *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
    vs.append(v_s)
  vs = np.stack(vs, axis=0)
  pg_advantages = (
      clipped_pg_rhos * (rewards + discounts * np.concatenate(
          [vs[1:], bootstrap_value[None, :]], axis=0) - values))

  return vs, pg_advantages

def test_information_theory_functions():

    tests = [
        ([1 / 3, 1 / 3, 1 / 3], 1.0986),
        ([0.25, 0.75], 0.56233),
        ([[0.25, 0.75],
         [0.50, 0.50]
         ], 0.69314718+0.56233)
    ]

    for x, y in tests:
        assert_is_similar(entropy(x), y, "Entropy test failed.")
        assert_is_similar(entropy(torch.tensor(x, dtype=torch.float32)), y, "Entropy test failed.")

    for x, y in tests:
        assert_is_similar(log_entropy(np.log(x)), y, "Entropy log_test failed.")
        assert_is_similar(log_entropy(torch.tensor(np.log(x), dtype=torch.float32)), y, "Entropy log_test failed.")

    # check kl

    tests = [
        ([1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3], 0),
        # example from wikipedia
        ([0.36, 0.48, 0.16], [1/3, 1/3, 1/3], 0.0852996),
        ([1/3, 1/3, 1/3], [0.36, 0.48, 0.16], 0.097455)
    ]

    for x1, x2, y in tests:
        assert_is_similar(kl(x1, x2), y, "KL test failed.")
        assert_is_similar(log_kl(np.log(x1), np.log(x2)), y, "Log KL test failed.")

    return True

def test_vtrace():

    value_estimates = np.asarray([[0.1, -0.1],[0.0, 0.4],[0.4, -0.2],[-0.2, 0.6] ,[0.3, 0.9]])
    rewards = np.asarray([[1, -2],[3, 4],[5, 1],[6, 12] ,[-5, 2]])

    final_value_estimate = np.asarray([3,1])

    # just for moment... will many this non-false later.
    dones = np.asarray([[False, False], [False, False], [True, False], [False, False], [False, False]])

    gamma = 0.90
    lamb = 1.0 # doesn't work with lamb != 1, not sure who's correct though, might be V-Trace?

    # first calculate the returns
    returns = rollout.calculate_returns(rewards, dones, final_value_estimate, gamma)

    gae = rollout.gae(rewards, value_estimates, final_value_estimate, dones, gamma, lamb=lamb, normalize=False)

    behaviour_log_policy = np.zeros([5,2,1], dtype=np.float)
    target_log_policy = np.zeros([5,2,1], dtype=np.float)
    actions = np.zeros([5, 2], dtype=np.int)

    vs, pg_adv, cs = ppo.importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions, rewards, dones,
                                                     value_estimates, final_value_estimate, gamma, lamb=lamb)

    assert_is_similar(returns, vs, "V-trace returns do not match.")
    assert_is_similar(pg_adv, gae, "V-trace advantages do not match.")

    # check mine matches theres...

    behaviour_log_policy = np.zeros([5,2,1], dtype=np.float)
    target_log_policy = np.zeros([5,2,1], dtype=np.float)

    # set some off polcyness
    behaviour_log_policy[:,0,0] = [-3,-2,3,-2,-5.5]
    behaviour_log_policy[:,1,0] = [-2,0,5,-4,-2]
    target_log_policy[:,0,0] = [-2,-4,-4,2,-1]
    target_log_policy[:,1,0] = [-6,-5,-4,-4,-3]

    discounts = np.ones_like(value_estimates) * gamma * (1-dones)


    log_rhos = (select_from(target_log_policy,actions) - select_from(behaviour_log_policy,actions))

    gt_vs, gt_adv = _ground_truth_vtrace_calculation(discounts, log_rhos, rewards, value_estimates,
                                                     final_value_estimate, 1, 1)

    vs, pg_adv, cs = importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions, rewards, dones,
                                                     value_estimates, final_value_estimate, gamma, lamb=lamb)

    assert_is_similar(gt_adv, pg_adv, "V-trace advantages do not match reference.")
    assert_is_similar(gt_vs, vs, "V-trace values do not match reference.")

    return True

def test_trust_region():
    """ Tests for trust region calculations. """
    pass

def get_tvf_test_params_one():
    """
    Get the parameters for the truncated value function test data.
    """
    rewards = [1, 0, 2, 4, 6]
    dones = [0, 0, 1, 0, 0]
    final_value_estimates = [0, 5, 10, 15, 20]
    value_estimates = [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [0, 3, 6, 9, 12], [0, 4, 8, 12, 16]]
    return (
        np.asarray(rewards)[:, None],
        np.asarray(dones)[:, None],
        np.asarray(value_estimates)[:, None, :],
        np.asarray(final_value_estimates)[None, :],
        0.5,
    )

def get_tvf_test_params_two():
    """
    Get the parameters for the truncated value function test data.
    This one has dones at begining and end
    """
    rewards = [1, 0, 2, 4, 6]
    dones = [1, 0, 1, 0, 1]
    final_value_estimates = [0, 5, 10, 15, 20]
    value_estimates = [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8], [0, 3, 6, 9, 12], [0, 4, 8, 12, 16]]
    return (
        np.asarray(rewards)[:, None],
        np.asarray(dones)[:, None],
        np.asarray(value_estimates)[:, None, :],
        np.asarray(final_value_estimates)[None, :],
        0.5,
    )

def get_tvf_test_params_three():
    """
    Get the parameters for the truncated value function test data.
    This one has dones at begining and end
    """
    rewards = [[1,0], [0,0], [2,0], [4,0], [6,0]]
    dones = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    final_value_estimates = [[0, 5, 10, 15, 20, 25], [0, 5, 10, 15, 20, 25]]
    value_estimates = np.zeros([5,2,6])
    for i in range(5):
        for j in range(6):
            for k in range(2):
                value_estimates[i,k,j] = i*j
    return (
        np.asarray(rewards),
        np.asarray(dones),
        np.asarray(value_estimates),
        np.asarray(final_value_estimates),
        1.0,
    )

def test_rediscounted_value_estimate():
    values = np.asarray([[0, -2, 5], [0, 4, 12]], dtype=np.float32)
    horizons = np.asarray([0, 1, 20])

    func = rollout.get_rediscounted_value_estimate

    print(func(values, old_gamma=0.99, new_gamma=0.99, horizons=horizons))
    print(func(values, old_gamma=1.0, new_gamma=0.99, horizons=horizons))

    # try list of rewards that match value estimates and make sure it works
    rewards = np.asarray([-1, 1, 4, 3, 2, 5, 6, 7, 1], dtype=np.float32)
    horizons = np.arange(len(rewards)+1)
    values_99 = []
    values_1 = []
    values_9 = []
    for h in horizons:
        values_9.append(np.sum([(0.9 ** i) * rewards[i] for i in range(h)]))
        values_99.append(np.sum([(0.99 ** i) * rewards[i] for i in range(h)]))
        values_1.append(np.sum([(1.0 ** i) * rewards[i] for i in range(h)]))
    values_99 = np.asarray(values_99)[None, :]
    values_1 = np.asarray(values_1)[None, :]
    values_9 = np.asarray(values_9)[None, :]
    for new_gamma in [1.0, 0.99, 0.9]:
        print(func(values_9, old_gamma=0.9, new_gamma=new_gamma, horizons=horizons)[0], end=', ')
        print(func(values_99, old_gamma=0.99, new_gamma=new_gamma, horizons=horizons)[0], end=', ')
        print(func(values_1, old_gamma=1.0, new_gamma=new_gamma, horizons=horizons)[0])

    return True

def test_gae():
    rewards = np.asarray([1, 0, 2, 4, 6], dtype=np.float32)[:, None]
    dones = np.asarray( [0, 0, 1, 0, 0], dtype=np.float32)[:, None]
    final_value_estimate = np.asarray(5, dtype=np.float32)
    value_estimates = np.asarray([0, 0.5, 0.5, 3, 4], dtype=np.float32)[:, None]
    result = rollout.gae(rewards, value_estimates, final_value_estimate, dones, gamma=0.5, lamb=1.0)
    assert_is_similar(result, np.asarray([1.5, 0.5, 1.5, 5.25, 4.5])[:, None])
    return True

def test_calculate_tvf_n_step():

    params = get_tvf_test_params_one()
    params_str = tuple(repr(x) for x in params)
    ref_result = rollout.calculate_tvf_td(*params)
    n_step_result = rollout.calculate_tvf_n_step(*params, n_step=1)
    assert_is_similar(n_step_result, ref_result)

    # ref_result = rollout.calculate_tvf_td0(*params)
    n_step_result = rollout.calculate_tvf_n_step(*params, n_step=2)
    ref_result = np.asarray(
        [
            [0, 1, 1, 1.5, 2],
            [0, 0, 1, 1,   1],
            [0, 2, 2, 2,   2],
            [0, 4, 7, 8.25, 9.5],
            [0, 6, 8.5, 11, 13.5],
        ]
    )[:, None, :]
    assert_is_similar(n_step_result, ref_result)
    assert params_str == tuple(repr(x) for x in params)

    return True

def test_interpolate():
    h = np.asarray([0, 300, 1500, 30000])
    values = np.zeros(shape=(2, 2, len(h)), dtype=np.float32)
    targets = np.zeros(shape=(2, 2), dtype=np.float32)

    targets[0, 0] = -100
    targets[0, 1] = 25
    targets[1, 0] = 51
    targets[1, 1] = 30000

    values[:, :, 0] = 50
    values[:, :, 1] = 100
    values[:, :, 2] = 3000
    values[:, :, 3] = 60000

    interpolate(h, values, targets)


# def test_calculate_tvf_lambda():
#
#     # just test against weighted n_step
#
#     params = get_tvf_test_params()
#
#     # calculate our n_step returns
#     g = []
#     for i in range(100):
#         g.append(rollout.calculate_tvf_n_step(*params, n_step=i+1))
#
#     # compute weighted average
#     for lamb in [0, 0.9, 0.95]:
#         ref_result = g[0] * (1-lamb)
#         for i in range(1, 100):
#             ref_result += g[i] * (lamb**i) * (1-lamb)
#
#         lambda_result = rollout.calculate_tvf_lambda(*params, lamb=lamb)
#         assert_is_similar(lambda_result, ref_result)
#
#
#     return True


def test_calculate_tvf_mc():

    for params in [get_tvf_test_params_one(), get_tvf_test_params_two(), get_tvf_test_params_three()]:
        h = params[3].shape[-1]
        ref_result = rollout.calculate_tvf_n_step(*params, n_step=h-1)
        mc_result = rollout.calculate_tvf_mc(*params)
        assert_is_similar(mc_result, ref_result)
    return True


def test_calculate_tvf_td():

    result = rollout.calculate_tvf_td(*get_tvf_test_params_one())

    assert_is_similar(result[0, 0], [0, 1, 1.5, 2, 2.5], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[1, 0], [0, 0, 1,   2,   3], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[2, 0], [0, 2, 2,   2,   2], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[3, 0], [0, 4, 6,   8,  10], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[4, 0], [0, 6, 8.5, 11, 13.5], "Truncated Value Function estimates do not match.")

    result = rollout.calculate_tvf_td(*get_tvf_test_params_two())

    assert_is_similar(result[0, 0], [0, 1, 1,   1,   1], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[1, 0], [0, 0, 1,   2,   3], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[2, 0], [0, 2, 2,   2,   2], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[3, 0], [0, 4, 6,   8,  10], "Truncated Value Function estimates do not match.")
    assert_is_similar(result[4, 0], [0, 6, 6,   6,   6], "Truncated Value Function estimates do not match.")

    return True

# print("V-trace: ", end='')
# print("Pass" if test_vtrace() else "Fail!")
#
# print("Information Theory Functions: ", end='')
# print("Pass" if test_information_theory_functions() else "Fail!")

# print("ROLLOUT_INTERPOLATE: ", end='')
# print("Pass" if test_interpolate() else "Fail!")

test_rediscounted_value_estimate()

# print("TVF_MC: ", end='')
# print("Pass" if test_calculate_tvf_mc() else "Fail!")
# print("TVF_TD: ", end='')
# print("Pass" if test_calculate_tvf_td() else "Fail!")
# print("TVF_N_STEP: ", end='')
# print("Pass" if test_calculate_tvf_n_step() else "Fail!")
# print("GAE: ", end='')
# print("Pass" if test_gae() else "Fail!")
#print("Rediscount: ", end='')
#print("Pass" if test_rediscounted_value_estimate() else "Fail!")
#print("TVF_Lambda: ", end='')
#print("Pass" if test_calculate_tvf_lambda() else "Fail!")