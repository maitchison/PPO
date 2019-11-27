import gym
import numpy as np

class HybridAsyncVectorEnv(gym.vector.AsyncVectorEnv):
    """ Async vector env, that limits the number of worker threads spawned """

    def __init__(self, env_fns, max_cpus=8, verbose=False, **kwargs):
        if len(env_fns) <= max_cpus:
            # this is just a standard vec env
            super(HybridAsyncVectorEnv, self).__init__(env_fns, **kwargs)
            self.is_batched = False
        else:
            # create sequential envs for each worker
            assert len(env_fns) % max_cpus == 0, "Number of environments ({}) must be a multiple of the CPU count ({}).".format(len(env_fns), max_cpus)
            self.n_sequential = len(env_fns) // max_cpus
            self.n_parallel = max_cpus
            vec_functions = []
            for i in range(self.n_parallel):
                vec_functions.append(lambda : gym.vector.SyncVectorEnv(env_fns[i*self.n_sequential:(i+1)*self.n_sequential], **kwargs))

            if verbose:
                print("Creating {} cpu workers with {} environments each.".format(self.n_parallel, self.n_sequential))
            super(HybridAsyncVectorEnv, self).__init__(vec_functions, **kwargs)

            self.is_batched = True

    def reset(self):
        if self.is_batched:
            obs = super(HybridAsyncVectorEnv, self).reset()
            return np.reshape(obs, [-1, *obs.shape[2:]])
        else:
            return super(HybridAsyncVectorEnv, self).reset()

    def step(self, actions):
        if self.is_batched:

            # put actions into 2d python array.
            actions = np.reshape(actions, [self.n_parallel, self.n_sequential])
            actions = [list(actions[i]) for i in range(len(actions))]

            observations_list, rewards, dones, infos = super(HybridAsyncVectorEnv, self).step(actions)

            return (
                np.reshape(observations_list, [-1, *observations_list.shape[2:]]),
                np.reshape(rewards, [-1]),
                np.reshape(dones, [-1]),
                np.reshape(infos, [-1])
            )
        else:
            return super(HybridAsyncVectorEnv, self).step(actions)
