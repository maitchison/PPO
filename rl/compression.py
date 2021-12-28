import lz4.frame
import blosc
import time
import numpy as np
from typing import List
from collections import deque

"""
Handles compressed data for rollout.

On my computer LZ4 drops IPS from 1,733 down to 1,499 (-13.5%)

for benchmarks see:
http://alimanfoo.github.io/2016/09/21/genotype-compression-benchmark.html
 
"""


def lz4_compress(x: np.ndarray):
    return lz4.frame.compress(x.tobytes(), compression_level=0)


def lz4_decompress(x, dtype, shape) -> np.ndarray:
    return np.frombuffer(lz4.frame.decompress(x), dtype=dtype).reshape(shape)


def blosc_compress(x: np.ndarray):
    return blosc.compress(
        x.tobytes(),
        typesize=8,
        clevel=1,
        shuffle=0,
        cname='lz4',
    )


def blosc_decompress(x, dtype, shape) -> np.ndarray:
    return np.frombuffer(blosc.decompress(x), dtype=dtype).reshape(shape)

COMPRESS = lz4_compress
DECOMPRESS = lz4_decompress

HISTORY_LENGTH = 100
_compression_times = deque(maxlen=HISTORY_LENGTH)
_decompression_times = deque(maxlen=HISTORY_LENGTH)
_compressed_sizes = deque(maxlen=HISTORY_LENGTH)
_uncompressed_sizes = deque(maxlen=HISTORY_LENGTH)

class BufferSlot():

    def __init__(self, initial_data=None):
        self._compressed_data: bytes
        self._dtype: np.dtype
        self._shape: tuple

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
        self._compressed_data = COMPRESS(x)
        self._shape = x.shape
        self._dtype = x.dtype
        _compression_times.append(time.time() - start_time)
        _compressed_sizes.append(self._compressed_size)
        _uncompressed_sizes.append(self._uncompressed_size)

    def decompress(self) -> np.ndarray:
        start_time = time.time()
        result = DECOMPRESS(self._compressed_data, self._dtype, self._shape)
        _decompression_times.append(time.time() - start_time)
        return result


def ratio():
    """ Returns the compression ratio. """
    if np.sum(_uncompressed_sizes) > 0:
        return np.sum(_uncompressed_sizes) / np.sum(_compressed_sizes)
    else:
        return 0


def av_compression_time():
    """ Returns the average time (in seconds) to compress a slot. """
    return np.mean(_compression_times)


def av_decompression_time():
    """ Returns the average time (in seconds) to compress a slot. """
    return np.mean(_decompression_times)
