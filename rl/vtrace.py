import numpy as np

def v_trace_trust_region(behaviour_log_policy, target_log_policy):
    """
    apply trust region
    this is from https://www.groundai.com/project/off-policy-actor-critic-with-shared-experience-replay/1
    we filter out example that would introduce too much bias,
    that is examples for which implied policy pi^hat, deviates from the true policy by too much
    because we condition on the policy distribution and not the actions we introduce no additional bias by doing this
    
    note: unlike in the paper we return a weighting, [0,1] where 1 indicates the sample should be included, and
    0 indicates it should not. This weighting can be used to randomly select samples, or as an importance weight
    in a sampling routine. (the paper just uses a cut-off)

    :param behaviour_log_policy     [N, B, A] Log of policy
    :param target_log_policy        [N, B, A] Log of policy

    :returns weights                [N, B]

    """

    # note: this would work much better on a large shared experience replay as it would allow us to use the
    # same batch size, as it is the batch size may become very small. This could cause significant issues
    # during training.

    # might be a way to do this all in log space, but for the moment get the actual policies back
    behaviour_policy = np.exp(behaviour_log_policy)
    target_policy = np.exp(target_log_policy)

    # first we calculate the implied policy pi
    implied_policy = np.minimum(1 * behaviour_policy, target_policy)

    # next normalize the policies
    implied_policy /= np.sum(implied_policy, axis=2, keepdims=True)

    # now calculate KL between the two policies
    kl = (np.log(target_policy/implied_policy) * target_policy).sum(axis=2)

    return 1/(1+kl)




def importance_sampling_v_trace(behaviour_log_policy, target_log_policy, actions, rewards, dones,
                                target_value_estimates, target_value_final_estimate, gamma, lamb=1.0):
    """
    Calculate importance weights, and value estimates from off-policy data.

    N is number of time steps, B the batch size, and A the number of actions.

    Based on https://arxiv.org/pdf/1802.01561.pdf

    :param behaviour_log_policy         np array [N, B, A], (i.e. the policy that generated the experience)
    :param target_log_policy            np array [N, B, A], (i.e. the policy that we want to update)
    :param actions                      np array [N, B], action taken by behaviour policy at each step
    :param rewards                      np array [N, B], reward received at each step
    :param dones                        np array [N, B], boolean terminals

    :param target_value_estimates       np array [N, B], the value estimates for the target policy.
    :param target_value_final_estimate  np array [B], the value estimate for target policy at time N

    :param gamma                        float, discount rate
    :param lamb                         float, lambda for generalized advantage function

    :return:
        vs                              np array [N, B], the value estimates for pi learned from off-policy data. These
                                        can be used as targets for policy gradient learning.
        weighted_advantages             np array [N, B], weighted (by rho) advantage estimates that can be used in
                                        policy gradient updates.
        cs                              the importance sampling correction adjustments

    """

    N, B, A = behaviour_log_policy.shape

    vs = np.zeros_like(target_value_estimates)
    cs = np.zeros_like(target_value_estimates)

    not_terminal = np.asarray(1-dones, dtype=np.float32)

    target_log_policy_actions = np.zeros_like(target_value_estimates)
    behaviour_log_policy_actions = np.zeros_like(target_value_estimates)

    # get just the part of the policy we need
    # stub: there will be a faster way of doing this with indexing..
    # for i in range(N):
    #     for j in range(B):
    #         target_log_policy_actions[i,j] = target_log_policy[i,j,actions[i, j]]
    #         behaviour_log_policy_actions[i, j] = behaviour_log_policy[i, j, actions[i, j]]
    for b in range(B):
      target_log_policy_actions[:, b] = target_log_policy[range(N), b, actions[:, b]]
      behaviour_log_policy_actions[:, b] = behaviour_log_policy[range(N), b, actions[:, b]]

    rhos = np.minimum(1, np.exp(target_log_policy_actions - behaviour_log_policy_actions))

    for i in reversed(range(N)):
        next_target_value_estimate = target_value_estimates[i + 1] if i + 1 != N else target_value_final_estimate
        # adding an epsilon to the denominator introduce a small amount of bias, this probably would not be necessary
        # if I use logs instead.
        tdv = rhos[i] * (rewards[i] + gamma * not_terminal[i] * next_target_value_estimate - target_value_estimates[i])
        cs[i] = rhos[i]  # in the paper these seem to be different, but I have them the same here?
        next_vs = vs[i + 1] if i + 1 != N else 0
        vs[i] = tdv + gamma * not_terminal[i] * (lamb * cs[i]) * next_vs

    # add value estimates back in
    vs = vs + target_value_estimates
    vs_plus_one = np.concatenate((vs[1:], [target_value_final_estimate]))

    # calculate advantage estimates
    weighted_advantages = rhos * (rewards + gamma * not_terminal * vs_plus_one - target_value_estimates)

    # cast to float because for some reason goes to float64
    weighted_advantages = weighted_advantages.astype(dtype=np.float32)

    return vs, weighted_advantages, cs
