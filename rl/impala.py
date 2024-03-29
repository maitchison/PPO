"""
Modified from https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/impala_cnn.py
"""

import math

from torch import nn
from torch.nn import functional as F
from . import tensor_utilities as tu

class Encoder(nn.Module):
    """
    Takes in seq of observations and outputs sequence of codes
    Encoders can be stateful, meaning that you pass in one observation at a
    time and update the state, which is a separate object. (This object
    doesn't store any state except parameters)
    """

    def __init__(self):
        super().__init__()

    def initial_state(self, batchsize):
        raise NotImplementedError

    def empty_state(self):
        return None

    def stateless_forward(self, obs):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
        returns:
            codes: array or dict, all with preshape (B, T)
        """
        code, _state = self(obs, None, self.empty_state())
        return code

    def forward(self, obs, first, state_in):
        """
        inputs:
            obs: array or dict, all with preshape (B, T)
            first: float array shape (B, T)
            state_in: array or dict, all with preshape (B,)
        returns:
            codes: array or dict
            state_out: array or dict
        """
        raise NotImplementedError

class CnnBasicBlock(nn.Module):
    """
    Residual basic block (without batchnorm), as in ImpalaCNN
    Preserves channel number and shape
    """

    def __init__(self, inchan, scale=1.0, batch_norm=False, **ignore_args):
        super().__init__()
        self.inchan = inchan
        self.batch_norm = batch_norm
        s = math.sqrt(scale)
        self.conv0 = tu.NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        self.conv1 = tu.NormedConv2d(self.inchan, self.inchan, 3, padding=1, scale=s)
        if self.batch_norm:
            self.bn0 = nn.BatchNorm2d(self.inchan)
            self.bn1 = nn.BatchNorm2d(self.inchan)

    def residual(self, x):
        # inplace should be False for the first relu, so that it does not change the input,
        # which will be used for skip connection.
        # getattr is for backwards compatibility with loaded models
        if getattr(self, "batch_norm", False):
            x = self.bn0(x)
        x = F.relu(x, inplace=False)
        x = self.conv0(x)
        if getattr(self, "batch_norm", False):
            x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        return x

    def forward(self, x):
        return x + self.residual(x)


class CnnDownStack(nn.Module):
    """
    Downsampling stack from Impala CNN
    """

    def __init__(self, inchan, nblock, outchan, scale=1.0, down_sample='pool', **kwargs):
        super().__init__()
        assert down_sample in ['pool', 'stride', 'none']
        self.inchan = inchan
        self.outchan = outchan
        self.down_sample = down_sample
        self.firstconv = tu.NormedConv2d(inchan, outchan, 3, padding=1, stride=2 if down_sample == 'stride' else 1)
        s = scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [CnnBasicBlock(outchan, scale=s, **kwargs) for _ in range(nblock)]
        )

    def forward(self, x):
        x = self.firstconv(x)
        if self.down_sample == 'pool':
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, inshape):
        c, h, w = inshape
        assert c == self.inchan
        if self.down_sample == 'pool':
            return (self.outchan, (h + 1) // 2, (w + 1) // 2)
        elif self.down_sample == 'none':
            return (self.outchan, h, w)
        elif self.down_sample == 'stride':
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            out_h = math.floor((h + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1)
            out_w = math.floor((w + 2 * 1 - 1 * (3 - 1) - 1) / 2 + 1)
            return (self.outchan, out_h, out_w)
        else:
            raise ValueError(f"Invalid down_sample mode {self.down_sample}")



def intprod(xs):
    """
    Product of a sequence of integers
    """
    out = 1
    for x in xs:
        out *= x
    return out


def transpose(x, before, after):
    """
    Usage: x_bca = transpose(x_abc, 'abc', 'bca')
    """
    assert sorted(before) == sorted(after), f"cannot transpose {before} to {after}"
    assert x.ndim == len(
        before
    ), f"before spec '{before}' has length {len(before)} but x has {x.ndim} dimensions: {tuple(x.shape)}"
    return x.permute(tuple(before.index(i) for i in after))

def flatten_image(x):
    """
    Flattens last three dims
    """
    *batch_shape, h, w, c = x.shape
    return x.reshape((*batch_shape, h * w * c))


def sequential(layers, x, *args, diag_name=None):
    for (i, layer) in enumerate(layers):
        x = layer(x, *args)
    return x