"""
Just a collection of utilities,
Mostly taken from https://github.com/openai/phasic-policy-gradient/blob/master/phasic_policy_gradient/torch_util.py
"""

import torch as th
import torch.nn
import torch.nn as nn
import numpy as np


def parse_dtype(x):
    if isinstance(x, th.dtype):
        return x
    elif isinstance(x, str):
        if x == "float32" or x == "float":
            return th.float32
        elif x == "float64" or x == "double":
            return th.float64
        elif x == "float16" or x == "half":
            return th.float16
        elif x == "uint8":
            return th.uint8
        elif x == "int8":
            return th.int8
        elif x == "int16" or x == "short":
            return th.int16
        elif x == "int32" or x == "int":
            return th.int32
        elif x == "int64" or x == "long":
            return th.int64
        elif x == "bool":
            return th.bool
        else:
            raise ValueError(f"cannot parse {x} as a dtype")
    else:
        raise TypeError(f"cannot parse {type(x)} as dtype")


def NormedLinear(*args, scale=1.0, dtype=th.float32, **kwargs):
    """
    nn.Linear but with normalized fan-in init
    bias: if true (default) will zero the bias initalization.
    """
    dtype = parse_dtype(dtype)
    if dtype == th.float32:
        out = nn.Linear(*args, **kwargs)
    elif dtype == th.float16:
        #out = LinearF16(*args, **kwargs)
        raise NotImplementedError("Float16 not implemented yet.")
    else:
        raise ValueError(dtype)
    out.weight.data *= scale / out.weight.norm(dim=1, p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out

def NormedConv2d(*args, scale=1.0, **kwargs):
    """
    nn.Conv2d but with normalized fan-in init
    bias: if true (default) will zero the bias initalization.
    """
    out = nn.Conv2d(*args, **kwargs)
    out.weight.data *= scale / out.weight.norm(dim=(1, 2, 3), p=2, keepdim=True)
    if kwargs.get("bias", True):
        out.bias.data *= 0
    return out

def process_weights(param:torch.nn.Module, method="default", scale:float=1.0):
    if param.bias is not None:
        param.bias.data *= 0  # zero bias weights
    if method == "default":
        param.weight.data *= scale
    elif method == "xavier":
        torch.nn.init.xavier_uniform_(param.weight.data, gain=scale)
    elif method == "orthogonal":
        torch.nn.init.orthogonal_(param.weight.data, gain=scale)


def CustomLinear(*args, scale=1.0, weight_init="default", **kwargs):
    """
    nn.Linear with (the option for) custom weight initialization
    """
    out = nn.Linear(*args, **kwargs)
    process_weights(out, method=weight_init, scale=scale)
    return out


def CustomConv2d(*args, scale=1.0, weight_init="default", **kwargs):
    """
    nn.Conv2d with (the option for) custom weight initialization
    """
    out = nn.Conv2d(*args, **kwargs)
    process_weights(out, method=weight_init, scale=scale)
    return out

