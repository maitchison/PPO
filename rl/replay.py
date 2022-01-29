"""
Experience Replay used for distillation
"""

import numpy as np
import math
import time
import hashlib
import typing

import torch

from rl.compression import BufferSlot


class ExperienceReplayBuffer:
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

    def __init__(self,
                 max_size:int,
                 obs_shape: tuple,
                 obs_dtype,
                 filter_duplicates: bool = False,
                 mode="uniform",
                 name="replay"
                 ):
        """
        @param max_size Size of replay
        @param state_shape Shape of states
        @param filter_duplicates Hashes input and filters out any entries that already exist in the replay buffer.
        """
        self.max_size = max_size # N is the maximum size of the buffer.
        self.experience_seen = 0
        self.ring_buffer_location = 0
        self.data = np.zeros([0, *obs_shape], dtype=obs_dtype)
        self.time = np.zeros([0], dtype=np.float32) # this is annoying, maybe there's a better way?
        self.hashes = np.zeros([0], dtype=np.uint64)
        self.filter_duplicates = filter_duplicates
        self.mode = mode
        self.stats_total_duplicates_removed: int = 0
        # how many (submitted) entries where removed as duplicates last update.
        self.stats_last_duplicates_removed: int = 0
        # how many entries where added to replay buffer last update.
        self.stats_last_entries_added: int = 0
        self.stats_last_entries_submitted: int = 0
        self.name = name

    def save_state(self, force_copy=True):
        return {
            'max_size': self.max_size,
            'experience_seen': self.experience_seen,
            'ring_buffer_location': self.ring_buffer_location,
            'data': self.data.copy() if force_copy else self.data,
            'time': self.time.copy() if force_copy else self.time,
            'hashes': self.hashes.copy() if force_copy else self.hashes,
            'stats_total_duplicates_removed': self.stats_total_duplicates_removed,
            'name': self.name,
        }

    def load_state(self, state_dict: dict):
        self.max_size = state_dict["max_size"]
        self.experience_seen = state_dict["experience_seen"]
        self.ring_buffer_location = state_dict['ring_buffer_location']
        self.data = state_dict["data"].copy()
        self.time = state_dict["time"].copy()
        self.hashes = state_dict["hashes"].copy()
        self.stats_total_duplicates_removed = self.stats_total_duplicates_removed

    @torch.no_grad()
    def estimate_replay_diversity(self, max_samples=32):
        """
        Returns an estimate of the diversity within the replay buffer using L2 distance in pixel space.
        max_samples: maximum number of samples to use. All pairs will be tested, so number of pairs is this number
            squared.
        """

        results = {}

        sample = np.random.choice(
            self.current_size,
            size=min(max_samples, self.current_size),
            replace=False
        )
        data = self.data[sample]
        if data.dtype == np.object:
            data = np.asarray([data[i].decompress() for i in range(len(data))], dtype=np.float32) / 255
        else:
            data = np.asarray(data, dtype=np.float32) / 255

        # might have been faster if we did this on the gpu...
        data = torch.from_numpy(data)
        B, *state_shape = data.shape
        data = data.reshape([1, B, np.prod(state_shape)])

        distances = torch.cdist(data, data, p=2)
        distances = distances[0, :, :].cpu().numpy()

        # diversity is average distance between x and the elements in the set
        # density is an estimate of the minimum distance between x and nearest elements of the set

        results['mean'] = float(distances.mean())
        # matrix will include the distance to oneself, remove these when calculating the min
        for i in range(len(distances)):
            distances[i, i] = float('inf')

        for k in [1, 2, 3, 4, 5]:
            results[f'k{k}'] = float(np.sort(distances, axis=0)[:k].mean())

        return results

    def log_stats(self, log):

        # takes around 1.7s for a 16x(128*128) replay (for 32 samples...)
        start_time = time.time()
        diversity_results = self.estimate_replay_diversity(128)  # note, sample count changes density estimate.
        log.watch_mean(f"{self.name}_diversity", diversity_results["mean"], display_width=0)
        log.watch_mean(f"{self.name}_density", diversity_results["k1"], display_width=0)
        for k, v in diversity_results.items():
            log.watch_mean(f"{self.name}_diversity_{k}", v, display_width=0)
        if self.data.dtype == np.object:
            # duplicate counting is far too slow unless compression is enabled.
            log.watch_mean(f"{self.name}_duplicates", self.count_duplicates(), display_width=0)
            log.watch_mean(f"{self.name}_current_duplicates_frac", self.count_duplicates() / len(self.data), display_width=0)
        log.watch_mean(f"{self.name}_total_duplicates_removed", self.stats_total_duplicates_removed, display_width=0)
        log.watch_mean(f"{self.name}_last_duplicates_removed", self.stats_last_duplicates_removed, display_width=0)
        log.watch_mean(f"{self.name}_last_entries_added", self.stats_last_entries_added, display_width=0)
        log.watch_mean(f"{self.name}_last_duplicates_removed_frac", self.stats_last_duplicate_frac, display_width=0)
        log.watch_mean(f"{self.name}_size", len(self.data), display_width=0)
        replay_log_time = time.time() - start_time
        log.watch_mean(f"{self.name}_log_time", replay_log_time, display_width=0)

    def count_duplicates(self):
        """
        Counts the number of duplicates in the replay buffer (a bit slow, especially if compression is not used).
        """
        hashes = np.asarray(
            [ExperienceReplayBuffer.get_observation_hash(x) for x in self.data],
            dtype=np.uint64
        )
        return len(hashes) - len(set(hashes))

    @property
    def current_size(self):
        return len(self.data)

    @property
    def stats_last_duplicate_frac(self):
        if self.stats_last_entries_submitted > 0:
            return self.stats_last_duplicates_removed / self.stats_last_entries_submitted
        else:
            return 0

    @staticmethod
    def get_observation_hash(x: typing.Union[np.ndarray, BufferSlot]) -> np.uint64:
        """
        Return has for given observation.
        If compressed will hash the compressed data.
        """
        if type(x) is np.ndarray:
            return int(hashlib.sha256(x.tobytes()).hexdigest(), 16) % (2 ** 64)
        elif type(x) is BufferSlot:
            # using the hash of the compressed bytes is much quicker.
            return x.compressed_hash
        else:
            raise ValueError("Invalid type for hashing {type(x)}")

    def _remove_duplicates(self, data, time, hashes):
        """
        Returns a new data, time with duplicates removed.
        Duplicates are either duplicates within the data or duplicates that are already in the replay
        """

        old_hashes = set(self.hashes)
        mask = []
        seen_hashes = set()
        for i in range(len(data)):
            # remove duplicates that are in the buffer already, or which occur multiple times in new experience.
            this_hash = hashes[i]
            is_duplicate = False
            if this_hash in old_hashes or this_hash in seen_hashes:
                is_duplicate = True
            seen_hashes.add(this_hash)
            mask.append(not is_duplicate)

        data = data[mask]
        time = time[mask]
        hashes = hashes[mask]

        self.stats_last_duplicates_removed = len(mask) - sum(mask)
        self.stats_total_duplicates_removed += self.stats_last_duplicates_removed
        self.stats_last_entries_added = sum(mask)

        return data, time, hashes


    def add_experience(self, new_experience: np.ndarray, new_time: np.ndarray = None):
        """
        Adds new experience to the experience replay

        @param new_experience: ndarray of dims [N, *state_shape]
        @param new_time: times for states.
        """

        assert new_experience.shape[1:] == self.data.shape[1:], \
            f"Invalid shape for new experience, expecting {new_experience.shape[1:]} but found {self.data.shape[1:]}."
        assert new_experience.dtype == self.data.dtype, \
            f"Invalid dtype for new experience, expecting {self.data.dtype} but found {self.data.dtype}."

        self.stats_last_entries_submitted = len(new_experience)

        new_hashes = np.asarray(
            [ExperienceReplayBuffer.get_observation_hash(x) for x in new_experience],
            dtype=np.uint64
        )
        if self.filter_duplicates:
            new_experience, new_time, new_hashes = self._remove_duplicates(new_experience, new_time, new_hashes)

        # 2. work out how many new entries we want to use, and resample them
        new_entry_count = len(new_experience)

        if self.mode == "uniform":
            # uniform tries to maintain the buffer so that it is a uniform sample over all experiences observed
            # this is done by decreasing the number of experiences added by 1/t where t is the total number of
            # samples 'observed'. Observed here does not mean added to the buffer.
            if self.experience_seen + new_entry_count < self.max_size:
                # simple just add them all
                number_of_entries_to_add = len(new_experience)
            else:
                # work out what proportion to keep
                wanted_ratio = len(new_experience) / (self.experience_seen + len(new_experience))
                number_of_entries_to_add = math.ceil(wanted_ratio * self.max_size)
        elif self.mode in ["overwrite", "sequential"]:
            number_of_entries_to_add = len(new_experience)
        else:
            raise ValueError(f"Invalid mode {self.mode}")

        ids = np.random.choice(len(new_experience), size=[number_of_entries_to_add], replace=False)
        ids = sorted(ids)  # faster to insert if source is in sequential order.

        self.stats_last_entries_added = 0

        # if we can, just increase the size of the buffer and add as many samples as we can
        new_buffer_size = min(self.max_size, len(self.data) + len(ids))
        if len(self.data) != new_buffer_size:
            old_n, *obs_shape = self.data.shape
            self.data.resize([new_buffer_size, *obs_shape], refcheck=False)
            self.time.resize([new_buffer_size], refcheck=False)
            self.hashes.resize([new_buffer_size], refcheck=False)
            new_slots = new_buffer_size - old_n
            self.data[old_n:] = new_experience[ids[:new_slots]]
            self.time[old_n:] = new_time[ids[:new_slots]]
            if new_hashes is not None:
                self.hashes[old_n:] = new_hashes[ids[:new_slots]]
            ids = ids[new_slots:]
            self.ring_buffer_location += new_slots
            self.stats_last_entries_added += new_slots

        if len(ids) > 0:
            # the remaining samples get written according to the replay strategy method.
            if self.mode == "sequential":
                new_spots = np.asarray(range(self.ring_buffer_location, self.ring_buffer_location + len(ids)))
                new_spots = np.mod(new_spots, new_buffer_size)
            else:
                new_spots = np.random.choice(range(new_buffer_size), size=[len(ids)], replace=False)

            for source, destination in zip(ids, new_spots):
                # add new entry
                self.data[destination] = new_experience[source]
                if new_hashes is not None:
                    self.hashes[destination] = new_hashes[source]
                if new_time is not None:
                    self.time[destination] = new_time[source]

            self.ring_buffer_location += len(ids)
            self.stats_last_entries_added += len(ids)

        # keep track of how many (non-duplicate) frames we've seen.
        self.experience_seen += new_entry_count


def smart_sample(x, n):
    """
    Samples n samples from x.
    If x <= len(x) samples ones without replacement
    If x > len(x) samples len(x)//n times without replacement

    This makes sure if n >= len(x) then every x is choosen at least once, and bounds the delta in duplicates to 1.

    """
    if n <= len(x):
        return np.random.choice(x, size=[n], replace=False)
    else:
        return np.concatenate([smart_sample(x, len(x)), smart_sample(x, n-len(x))], axis=0)


