"""
Estimates the true return distribution for an agent over an episode.

Memory and complexity is O(h^3) where h is the maximum horizon.
Environments / policies with low stochastisity or many duplicate states will be much faster.

The basic idea is to take an agent run it through an environment, and at each time step generate
1000 particles that give a distribution over the agents true return.

"""

import torch
import numpy as np

# Maps from (s, h) to R_h(s) where R_h(s) is a return sample up to horizon h for state s.
RETURN_CACHE = {}

N_SAMPLES = 100
ENV_IS_DETERMANISTIC = False


def reset_environment_to(state):
    pass

def get_model_action_sample(state):
    """
    Returns a sample of actions taken from the models policy at given state.
    """
    pass

def sample_return_distribution(state, horizon=30000):

    if horizon == 0:
        return np.zeros([N_SAMPLES])

    # get first step rewards
    actions = get_model_action_sample(state)

    if ENV_IS_DETERMANISTIC:
        raise NotImplementedError()
        # unique_actions = list(set(actions))
        # new_states = []
        # counts = [actions.count(x) for x in unique_actions]
        # for action in unique_actions:
        #     env.reset_to(state)
        #     env.step(action)
        #     new_states.append(env.state)
    else:
        for action in actions:
            env.load(state)
            reward, new_state = env.step(action)
            new_states.append(env.state)

    remaining_horizon = horizon - 1
    if remaining_horizon == 0:
        return



def run_agent_in_multiverse():
    while True:

        with torch.no_grad():
            model_out = model.forward(
                states,
                **kwargs,
                **({'policy_temperature':temperature} if temperature is not None else {})
            )

        probs = model_out["argmax_policy"].detach().cpu().numpy()
        action = np.asarray([np.argmax(prob) for prob in probs], dtype=np.int32)

        states, rewards, dones, infos = env.step(action)

        return_distrbition = sample_return(state)



if __name__ == "__main__":
    pass


