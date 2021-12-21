import lz4.frame
import time
import numpy as np
from typing import List

"""
Handles compressed data for rollout.
"""

# lz4 is by far the quickest...
COMPRESSION_LIBRARY = lz4.frame
COMPRESSION_PARAMS = {
    'compression_level': 0
}


class BufferSlot():

    def __init__(self, initial_data=None):
        self._compressed_data: bytes
        self._compression_time: float = None
        self._decompression_time: float = None
        self._dtype: np.dtype
        self._shape: tuple
        self._thread_job = None

        if initial_data is not None:
            self.compress(initial_data)

    @property
    def _compressed_size(self):
        return len(self._compressed_data) if self._compressed_data is not None else 0

    @property
    def _uncompressed_size(self):
        if self._shape == None:
            return 0
        return np.prod(self._shape) * self._dtype.itemsize

    def compress(self, x: np.ndarray):
        start_time = time.time()
        self._compressed_data = COMPRESSION_LIBRARY.compress(x.tobytes(), **COMPRESSION_PARAMS)
        self._shape = x.shape
        self._dtype = x.dtype
        self._compression_time = (time.time() - start_time)

    def decompress(self) -> np.ndarray:

        # if we have not finished compression wait for it to finish...
        if self._thread_job is not None:
            if not self._thread_job.done():
                print('join')
                _ = self._thread_job.result(10.0)
            self._thread_job = None

        start_time = time.time()
        result = np.frombuffer(COMPRESSION_LIBRARY.decompress(self._compressed_data), dtype=self._dtype).reshape(
            self._shape)
        self._decompression_time = (time.time() - start_time)

        return result


class StateBuffer():
    """
    Buffer for observations that can (optionally) use compression

    usage:

    buffer = StateBuffer()

    buffer
    """

    def __init__(self):
        self.buffer: List[BufferSlot] = []

    def append(self, x: np.ndarray):
        self.buffer.append(BufferSlot(x))

    @property
    def ratio(self):
        """ Returns the compress ratio. """
        return np.sum([slot._compressed_size for slot in self.buffer]) / np.sum(
            [slot._uncompressed_size for slot in self.buffer])

    @property
    def av_compression_time(self):
        """ Returns the average time (in seconds) to compress a slot. """
        return np.mean([slot._compression_time for slot in self.buffer if slot._compression_time is not None])

    @property
    def av_decompression_time(self):
        """ Returns the average time (in seconds) to compress a slot. """
        return np.mean([slot._decompression_time for slot in self.buffer if slot._decompression_time is not None])

    def __len__(self):
        return len(self.buffer)

    def __setitem__(self, index, value: np.ndarray):
        if type(index) is int:
            self.buffer[index].compress(value)
        elif type(index) is slice:
            for i in range(slice.start, slice.stop, slice.step):
                self.buffer[i].compress(value[i])
        else:
            raise ValueError()

    def __getitem__(self, index) -> np.ndarray:
        if type(index) is int:
            return self.buffer[index].decompress()
        if type(index) is slice:
            index = range(index.start or 0, index.stop or len(self), index.step or 1)
        try:
            result = [self[i] for i in index]
            return np.asarray(result)
        except:
            raise ValueError()

