"""
Tools for hashing state space
"""

import numpy as np
import torch
import torch as pt
from .utils import prod

class StateHasher():
    def __init__(self):
        pass

class LinearStateHasher(pt.nn.Module):

    def __init__(self, input_space:tuple, output_bits: int=16, device="cpu"):
        super().__init__()
        in_d = prod(input_space)
        out_d = output_bits
        self.input_space = tuple(input_space)
        self.device = device
        self.projection = pt.nn.Linear(in_d, out_d, bias=False) # bias or not?

        # make sure we always get the same weights (so we don't have to store them)
        g = torch.Generator()
        g.manual_seed(12345)
        s = 0.01
        self.projection.weight.data.uniform_(-s, s, generator=g)

        self.to(device)

    def forward(self, x):

        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        x = x.to(self.device)
        x = x.to(torch.float32)

        b, *state_shape = x.shape
        x = x.reshape(b, -1)
        x = self.projection(x)
        bits = torch.greater(x, 0).detach().cpu().numpy()

        acc = np.zeros([b], dtype=np.int32)
        for i, bit in enumerate(range(bits.shape[-1])):
            acc += bits[:, i] * (2 ** i)

        return acc


