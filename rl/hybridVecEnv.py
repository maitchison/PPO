import gym
import numpy as np
import functools
import sys

import gym.vector.async_vector_env
from gym.vector.utils import write_to_shared_memory

# modified to support vector environments...

class HybridAsyncVectorEnv(gym.vector.async_vector_env.AsyncVectorEnv):
    """ Async vector env, that limits the number of worker threads spawned """

    def __init__(self, env_fns, max_cpus=8, verbose=False, **kwargs):
        if len(env_fns) <= max_cpus:
            # this is just a standard vec env
            super().__init__(env_fns, **kwargs)
            self.is_batched = False
        else:
            # create sequential envs for each worker
            assert len(env_fns) % max_cpus == 0, "Number of environments ({}) must be a multiple of the CPU count ({}).".format(len(env_fns), max_cpus)
            self.n_sequential = len(env_fns) // max_cpus
            self.n_parallel = max_cpus
            vec_functions = []
            for i in range(self.n_parallel):
                # I prefer the lambda, but it won't work with pickle, and I want to multiprocessor this...
                constructor = functools.partial(gym.vector.SyncVectorEnv, env_fns[i*self.n_sequential:(i+1)*self.n_sequential], **kwargs)
                vec_functions.append(constructor)

            if verbose:
                print("Creating {} cpu workers with {} environments each.".format(self.n_parallel, self.n_sequential))

            use_shared_memory = kwargs.get("shared_memory", True)
            worker_function = _worker_shared_memory if use_shared_memory else _worker

            super().__init__(vec_functions, worker=worker_function, **kwargs)

            self.is_batched = True
            # super will set num_envs to number of workers, so we fix it here.
            self.num_envs = len(env_fns)

    def reset(self):
        if self.is_batched:
            obs = super(HybridAsyncVectorEnv, self).reset()
            return np.reshape(obs, [-1, *obs.shape[2:]])
        else:
            return super(HybridAsyncVectorEnv, self).reset()

    def step(self, actions):
        """
        mask: (optional) boolean nd array of shape [A] indicating which environments should accept this input.
        """
        if self.is_batched:

            # put actions into 2d python array.
            if type(actions[0]) is tuple:
                n = len(actions[0])
                actions = np.reshape(actions, [self.n_parallel, self.n_sequential, n])
                actions = [list(actions[i]) for i in range(len(actions))]
            else:
                actions = np.reshape(actions, [self.n_parallel, self.n_sequential])
                actions = [list(actions[i]) for i in range(len(actions))]

            observations_list, rewards, dones, infos = super().step(actions)

            return (
                np.reshape(observations_list, [-1, *observations_list.shape[2:]]),
                np.reshape(rewards, [-1]),
                np.reshape(dones, [-1]),
                np.reshape(infos, [-1])
            )
        else:
            return super().step(actions)

def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                pipe.send((observation, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                # Vectorized environments will reset by themselves so we don't need to auto reset them here.
                if type(done) != np.ndarray and done:
                    observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                # Vectorized environments will reset by themselves so we don't need to auto reset them here.
                if type(done) != np.ndarray and done:
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, '
                    '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
