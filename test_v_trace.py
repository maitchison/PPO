# test for v-trace (and gae and others I guess)

import numpy as np
from rl import ppo

def is_similar(a, b):
    delta = np.abs(a-b)
    return np.max(delta) < 1e-4


def select_from(a,b):
    N,B = b.shape
    r = np.zeros_like(b)
    for i in range(N):
        for j in range(B):
            r[i,j] = a[i,j,b[i,j]]
    return r

def _ground_truth_calculation(discounts, log_rhos, rewards, values,
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

value_estimates = np.asarray([[0.1, -0.1],[0.0, 0.4],[0.4, -0.2],[-0.2, 0.6] ,[0.3, 0.9]])
rewards = np.asarray([[1, -2],[3, 4],[5, 1],[6, 12] ,[-5, 2]])

final_value_estimate = np.asarray([3,1])

# just for moment... will many this non-false later.
dones = np.asarray([[False, False], [False, False], [True, False], [False, False], [False, False]])

gamma = 0.90
lamb = 1.0 # doesn't work with lamb != 1, not sure who's correct though, might be V-Trace?

# first calculate the returns
returns = ppo.calculate_returns(rewards, dones, final_value_estimate, gamma)

gae = ppo.calculate_gae(rewards, value_estimates, final_value_estimate, dones, gamma, lamb=lamb, normalize=False)

behaviour_log_policy = np.zeros([5,2,1], dtype=np.float)
target_log_policy = np.zeros([5,2,1], dtype=np.float)
actions = np.zeros([5, 2], dtype=np.int)

vs, pg_adv, cs = ppo.importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions, rewards, dones,
                                                 value_estimates, final_value_estimate, gamma, lamb=lamb)

print("------------- Returns -----------------")
print(returns)
print("------------- VS -----------------")
print(vs)
print("------------- GAE -----------------")
print(gae)
print("------------- PG_ADV -----------------")
print(pg_adv)
print("------------- CS -----------------")
print(cs)

print(pg_adv-gae)

assert is_similar(returns, vs), "V-trace returns do not match."
assert is_similar(pg_adv, gae), "V-trace advantages do not match."

# check mine matches theres...

behaviour_log_policy = np.zeros([5,2,1], dtype=np.float)
target_log_policy = np.zeros([5,2,1], dtype=np.float)

# set some off polcyness
behaviour_log_policy[3,0,0] = -3
behaviour_log_policy[3,1,0] = -2
target_log_policy[3,1,0] = -5
behaviour_log_policy[1,0,0] = -3
behaviour_log_policy[1,1,0] = -2
target_log_policy[1,1,0] = -5

discounts = np.ones_like(value_estimates) * gamma * (1-dones)

print(discounts)

log_rhos = (select_from(target_log_policy,actions) - select_from(behaviour_log_policy,actions))

print(log_rhos.ravel())

gt_vs, gt_adv = _ground_truth_calculation(discounts, log_rhos, rewards, value_estimates,
        final_value_estimate, 1, 1)

vs, pg_adv, cs = ppo.importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions, rewards, dones,
                                                 value_estimates, final_value_estimate, gamma, lamb=lamb)

print("PASS! initial (:")

print("Check against reference")

print(gt_adv)
print(pg_adv)

print(gt_vs)
print(vs)


assert is_similar(gt_adv, pg_adv), "V-trace advantages do not match reference."
assert is_similar(gt_vs, vs), "V-trace values do not match reference."

print()
print("PASS! reference :)")
