"""
Library to help with calculating return estimates.
"""

import numpy as np

def gae(
        batch_rewards,
        batch_value,
        final_value_estimate,
        batch_terminal,
        gamma: float,
        lamb=0.95,
        normalize=False
    ):
    """
    Calculates GAE based on rollout data.
    """
    N, A = batch_rewards.shape

    batch_advantage = np.zeros_like(batch_rewards, dtype=np.float32)
    prev_adv = np.zeros([A], dtype=np.float32)
    for t in reversed(range(N)):
        is_next_new_episode = batch_terminal[
            t] if batch_terminal is not None else False  # batch_terminal[t] records if prev_state[t] was terminal state)
        value_next_t = batch_value[t + 1] if t != N - 1 else final_value_estimate
        delta = batch_rewards[t] + gamma * value_next_t * (1.0 - is_next_new_episode) - batch_value[t]
        batch_advantage[t] = prev_adv = delta + gamma * lamb * (
                1.0 - is_next_new_episode) * prev_adv
    if normalize:
        batch_advantage = (batch_advantage - batch_advantage.mean()) / (batch_advantage.std() + 1e-8)
    return batch_advantage


def calculate_bootstrapped_returns(rewards, dones, final_value_estimate, gamma) -> np.ndarray:
    """
    Calculates returns given a batch of rewards, dones, and a final value estimate.

    Input is vectorized so it can calculate returns for multiple agents at once.
    :param rewards: nd array of dims [N,A]
    :param dones:   nd array of dims [N,A] where 1 = done and 0 = not done.
    :param final_value_estimate: nd array [A] containing value estimate of final state after last action.
    :param gamma:   discount rate.
    :return: np array of dims [N,A]
    """

    N, A = rewards.shape

    returns = np.zeros([N, A], dtype=np.float32)
    current_return = final_value_estimate

    if type(gamma) is float:
        gamma = np.ones([N, A], dtype=np.float32) * gamma

    for i in reversed(range(N)):
        returns[i] = current_return = rewards[i] + current_return * gamma[i] * (1.0 - dones[i])

    return returns


def td_lambda(
        batch_rewards,
        batch_value,
        final_value_estimate,
        batch_terminal,
        gamma: float,
        lamb=0.95,
):
    advantages = gae(batch_rewards, batch_value, final_value_estimate, batch_terminal, gamma, lamb, normalize=False)
    return advantages + batch_value

