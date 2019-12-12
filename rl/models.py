import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from . import utils
from .utils import RunningMeanStd

# ----------------------------------------------------------------------------------------------------------------
# Heads (feature extractors)
# ----------------------------------------------------------------------------------------------------------------

class BaseHead(nn.Module):
    def __init__(self, input_dims, hidden_units):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units

class NatureCNNHead(BaseHead):
    """ Takes stacked frames as input, and outputs features.
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.fc = nn.Linear(self.d, hidden_units)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        N = len(x)
        D = self.d
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.reshape(x, [N,D])
        x = self.fc(x)
        return x

class RNDTargetHead(BaseHead):
    """ Used to predict output of random network.
        see https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        for details.
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.out = nn.Linear(self.d, hidden_units)

        # we scale the weights so the output varience is about right. The sqrt2 is because of the relu,
        # bias is disabled as per OpenAI implementation.
        # I found I need to bump this up a little to get the varience required (0.4)
        # maybe this is due to weight initialization differences between TF and Torch?
        scale_weights(self, weight_scale=np.sqrt(2)*1.3, bias_scale=0.0)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        N = len(x)
        D = self.d

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.reshape(x, [N,D])
        x = self.out(x)
        return x


class RNDPredictorHead(BaseHead):
    """ Used to predict output of random network.
        see https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        for details.
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.fc1 = nn.Linear(self.d, 512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, hidden_units)

        # we scale the weights so the output varience is about right. The sqrt2 is because of the relu,
        # bias is disabled as per OpenAI implementation.
        scale_weights(self, weight_scale=np.sqrt(2)*1.3, bias_scale=0.0)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        N = len(x)
        D = self.d
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.reshape(x, [N,D])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# ----------------------------------------------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_dims, actions):
        super().__init__()
        self.input_dims = input_dims
        self.actions = actions
        self.device = None
        self.dtype = None

    def set_device_and_dtype(self, device, dtype):
        self.to(device)
        if dtype == torch.half:
            self.half()
        elif dtype == torch.float:
            self.float()
        elif dtype == torch.double:
            self.double()
        else:
            raise Exception("Invalid dtype {} for model.".format(dtype))

        self.device, self.dtype = device, dtype

    def prep_for_model(self, x, scale_int=True):
        """ Converts data to format for model (i.e. uploads to GPU, converts type).
            Can accept tensor or ndarray.
            scale_int scales uint8 to [0..1]
         """

        assert self.device is not None, "Must call set_device_and_dtype."

        validate_dims(x, (None, *self.input_dims))

        # if this is numpy convert it over
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)

        # move it to the correct device
        x = x.to(self.device, non_blocking=True)

        # then covert the type (faster to upload uint8 then convert on GPU)
        if x.dtype == torch.uint8:
            x = x.to(dtype=self.dtype, non_blocking=True)
            if scale_int:
                x = x / 255
        elif x.dtype == self.dtype:
            pass
        else:
            raise Exception("Invalid dtype {}".format(x.dtype))

        return x

class ActorCriticModel(BaseModel):
    """ Actor critic model, outputs policy, and value estimate."""

    def __init__(self, head: str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)
        self.name = "AC-"+head
        self.head = constructHead(head, input_dims, **kwargs)
        self.fc_policy = nn.Linear(self.head.hidden_units, actions)
        self.fc_value = nn.Linear(self.head.hidden_units, 1)
        self.set_device_and_dtype(device, dtype)

    def forward(self, x):
        x = self.prep_for_model(x)
        x = F.relu(self.head.forward(x))
        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'ext_value': value
        }


class AttentionModel(BaseModel):
    """ Has extra value and policy heads for fovea attention."""

    def __init__(self, head: str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)
        self.name = "AC-"+head
        self.head = constructHead(head, input_dims, **kwargs)
        self.fc_policy = nn.Linear(self.head.hidden_units, actions)
        self.fc_value = nn.Linear(self.head.hidden_units, 1)
        self.fc_policy_atn = nn.Linear(self.head.hidden_units, 25)
        self.fc_value_atn = nn.Linear(self.head.hidden_units, 1)
        self.set_device_and_dtype(device, dtype)

    def forward(self, x):
        x = self.prep_for_model(x)
        x = F.relu(self.head.forward(x))
        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)
        log_policy_atn = F.log_softmax(self.fc_policy_atn(x), dim=1)
        value_atn = self.fc_value_atn(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'atn_log_policy': log_policy_atn,
            'ext_value': value,
            'atn_value': value_atn
        }

class RNDModel(BaseModel):
    """
    Random network distilation model
    """

    def __init__(self, head:str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)

        self.name = "RND-" + head

        single_channel_input_dims = (1, *input_dims[1:])

        self.head = constructHead(head, input_dims, **kwargs)
        self.prediction_net = RNDPredictorHead(single_channel_input_dims)
        self.target_net = RNDTargetHead(single_channel_input_dims)

        self.fc_policy = nn.Linear(self.head.hidden_units, actions)
        self.fc_value_ext = nn.Linear(self.head.hidden_units, 1)
        self.fc_value_int = nn.Linear(self.head.hidden_units, 1)

        self.obs_rms = RunningMeanStd(shape=(single_channel_input_dims))

        self.features_mean = 0
        self.features_std = 0

        self.set_device_and_dtype(device, dtype)

    def perform_normalization(self, x):
        """ Applies normalization transform, and updates running mean / std. """

        # update normalization constants
        if type(x) is torch.Tensor:
            x = x.cpu().numpy()

        assert x.dtype == np.uint8

        x = np.asarray(x[:, 0:1], np.float32)
        self.obs_rms.update(x)
        mu = torch.tensor(self.obs_rms.mean.astype(np.float32)).to(self.device)
        sigma = torch.tensor(self.obs_rms.var.astype(np.float32) ** 0.5).to(self.device)

        # normalize x
        x = torch.tensor(x).to(self.device)
        x = torch.clamp((x - mu) / (sigma + 1e-5), -5, 5) 

        # note: clipping reduces the std down to 0.3 so we multiply the output so that it is roughly
        # unit normal.
        x *= 3.0

        return x

    def prediction_error(self, x):
        """ Returns prediction error for given state. """

        # only need first channel for this.
        x = self.perform_normalization(x)

        # random features have too low varience due to weight initialization being different from the OpenAI implementation
        # we adjust it here by simply multiplying the output to scale the features to have a max of around 3
        random_features = self.target_net.forward(x).detach()
        predicted_features = self.prediction_net.forward(x)

        # note: I really want to normalize these... otherwise scale just makes such a difference.
        # or maybe tanh the features?

        self.features_mean = float(random_features.mean().detach().cpu())
        self.features_var = float(random_features.var(axis=0).mean().detach().cpu())
        self.features_max = float(random_features.abs().max().detach().cpu())

        errors = (random_features - predicted_features).pow(2).mean(axis=1)

        return errors

    def forward(self, x):
        x = self.prep_for_model(x)
        x = F.relu(self.head.forward(x))
        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        value_ext = self.fc_value_ext(x).squeeze(dim=1)
        value_int = self.fc_value_int(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'ext_value': value_ext,
            'int_value': value_int
        }

# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------

def constructHead(head_name, input_dims, **kwargs) -> BaseHead:
    head_name = head_name.lower()
    if head_name == "nature":
        return NatureCNNHead(input_dims, **kwargs)
    else:
        raise Exception("No model head named {}".format(head_name))


def validate_dims(x, dims, dtype=None):
    """ Makes sure x has the correct dims and dtype.
        None will ignore that dim.
    """

    if dtype is not None:
        assert x.dtype == dtype, "Invalid dtype, expected {} but found {}".format(str(dtype), str(x.dtype))

    assert len(x.shape) == len(dims), "Invalid dims, expected {} but found {}".format(dims, x.shape)

    assert all((a is None) or a == b for a,b in zip(dims, x.shape)), "Invalid dims, expected {} but found {}".format(dims, x.shape)


def get_CNN_output_size(input_size, kernel_sizes, strides, max_pool=False):
    """ Calculates CNN output size, if max_pool is true uses max_pool instead of stride."""
    size = input_size
    for kernel_size, stride in zip(kernel_sizes, strides):

        if max_pool:
            size = (size - (kernel_size - 1) - 1) // 1 + 1
            size = size // stride
        else:
            size = (size - (kernel_size - 1) - 1) // stride + 1
    return size

def scale_weights(model: nn.Module, weight_scale, bias_scale):
    for child in model.children():
        child.weight.data *= weight_scale
        child.bias.data *= bias_scale
