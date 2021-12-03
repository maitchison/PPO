"""
Experience Replay used for distillation
"""

import numpy as np
import hashlib
import math

class ExperienceReplayBuffer():
    """
    Class to handle experience replay. Buffer is limited to N entries, and holds a uniform sample of all (unique)
    experience passed to it. For performance rollout.obs should be set to readonly.

    The buffer will make sure that no duplicate frames are added (via hashing)

    Usage:
        buffer = ExperienceReplayBuffer(N)

        while True:
            rollout = generate_rollout()
            model.train(rollout)
            buffer.add_experience(rollout.obs)
            process_replay_experience(buffer.data)

    """

    def __init__(self, N:int, obs_shape: tuple, obs_dtype: np.dtype):
        """
        @param N Size of replay
        @param state_shape Shape of states
        """
        self.N = N
        self.experience_seen = 0
        self.data = np.zeros([N, *obs_shape], dtype=obs_dtype)
        self.time = np.zeros([N], dtype=np.float32) # this is annoying, maybe there's a better way?
        self.hashes = np.zeros([N], dtype=object)
        self.hash_set = set()

    def save_state(self):
        return {
            'N': self.N,
            'experience_seen': self.experience_seen,
            'data': self.data.copy(),
            'time': self.time.copy(),
            'hashes': self.hashes.copy(),
            'hash_set': self.hash_set.copy(),
        }

    def load_state(self, state_dict: dict):
        self.N = state_dict["N"]
        self.experience_seen = state_dict["experience_seen"]
        self.data = state_dict["data"].copy()
        self.time = state_dict["time"].copy()
        self.hashes = state_dict["hashes"].copy()
        self.hash_set = state_dict["hashes"].copy()

    def add_experience(self, new_experience: np.ndarray, new_time: np.ndarray = None):
        """
        Adds new experience to the experience replay

        @param new_experience: ndarray of dims [N, *state_shape]
        """

        assert new_experience.shape[1:] == self.data.shape[1:]
        assert new_experience.dtype == self.data.dtype

        # 1. filter our examples that have already been seen
        hash_fn = lambda x: int(hashlib.sha256(x.data.tobytes()).hexdigest(), 16)

        ids = np.asarray(range(len(new_experience)))

        hashes = np.asarray([hash_fn(new_experience[i]) for i in range(len(new_experience))])
        mask = [hash not in self.hash_set for hash in self.hashes]

        ids = ids[mask]

        # 2. work out how many new entries we want to use, and resample them
        new_entries = len(ids)
        if self.experience_seen == 0:
            entries_to_add = new_entries
        else:
            entries_to_add = math.ceil((new_entries / (self.experience_seen + new_entries)) * self.N)

        ids = np.random.choice(ids, size=[entries_to_add], replace=False)
        np.sort(ids) # faster?

        # 3. add the new entries
        new_spots = np.random.choice(range(self.N), size=[entries_to_add], replace=False)
        np.sort(new_spots) # faster?

        for i, destination in enumerate(new_spots):
            # remove its hash
            if self.hashes[destination] != 0:
                self.hash_set.remove(self.hashes[destination])
            # add new entry
            source = ids[i]
            self.data[destination] = new_experience[source]
            self.hashes[destination] = hashes[source]
            self.hash_set.add(hashes[source])
            if new_time is not None:
                self.time[destination] = new_time[source]



