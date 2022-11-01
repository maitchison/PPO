"""
Tools for hashing state space
"""

import numpy as np
import torch
import torch as pt
from .utils import prod

def convert_to_state(x):

    b = len(x)
    bits = torch.greater(x, 0).detach().cpu().numpy()

    acc = np.zeros([b], dtype=np.int32)
    for i, bit in enumerate(range(bits.shape[-1])):
        acc += bits[:, i] * (2 ** i)

    return acc


class LinearStateHasher(pt.nn.Module):

    def __init__(self, input_space:tuple, output_bits: int=16, device="cpu", bias=0.0):
        super().__init__()
        in_d = prod(input_space)
        out_d = output_bits
        self.input_space = tuple(input_space)
        self.device = device
        self.projection = pt.nn.Linear(in_d, out_d, bias=True)

        # make sure we always get the same weights (so we don't have to store them)
        g = torch.Generator()
        g.manual_seed(12345)
        s = 0.01
        self.projection.weight.data.uniform_(-s, s, generator=g)
        self.projection.bias.data.uniform_(-bias, bias, generator=g)

        self.to(device)

    def forward(self, x):

        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        x = x.to(self.device)
        x = x.to(torch.float32)

        b, *state_shape = x.shape
        x = x.reshape(b, -1)
        x = self.projection(x)

        acc = convert_to_state(x)

        return acc


class ConvStateHasher(pt.nn.Module):

    def __init__(self, input_space:tuple, output_bits: int=16, device="cpu", bias=0.0):
        """
        bias is not used.
        """

        super().__init__()
        out_d = output_bits

        self.input_space = tuple(input_space)

        C, H, W = input_space

        self.device = device

        self.conv1 = pt.nn.Conv2d(C, 16, kernel_size=(5, 5), stride=(3, 3))
        self.conv2 = pt.nn.Conv2d(16, 1, kernel_size=(1, 1), stride=(1, 1)) # squeeze
        in_d = prod([(H-2)//3, (W-2)//3, 1]) # this might be slightly wrong, works for 84x84 though.
        self.projection = pt.nn.Linear(in_d, out_d)

        # make sure we always get the same weights (so we don't have to store them)
        g = torch.Generator()
        g.manual_seed(12345)
        s = 0.01
        self.projection.weight.data.uniform_(-s, s, generator=g)
        self.projection.bias.data.uniform_(-s, s, generator=g)
        self.conv1.weight.data.uniform_(-s, s, generator=g)
        self.conv1.bias.data.uniform_(-s, s, generator=g)

        self.to(device)

    def forward(self, x):

        B = len(x)

        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
        x = x.to(self.device)
        x = x.to(torch.float32)

        x = pt.relu(self.conv1(x))
        x = self.conv2(x) # no conv here, it's just a random projection down.
        x = x.reshape([B, -1])
        x = self.projection(x)

        acc = convert_to_state(x)

        return acc


