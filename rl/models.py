import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union

from . import utils

# ----------------------------------------------------------------------------------------------------------------
# Heads (feature extractors)
# ----------------------------------------------------------------------------------------------------------------

class Base_Net(nn.Module):
    def __init__(self, input_dims, hidden_units):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units

class NatureCNN_Net(Base_Net):
    """ Takes stacked frames as input, and outputs features.
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8,8), stride=(4,4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2,2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1,1))

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units
        if self.hidden_units > 0:
            self.fc = nn.Linear(self.d, hidden_units)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        N = len(x)
        D = self.d
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = torch.reshape(x3, [N,D])
        if self.hidden_units > 0:
            x5 = self.fc(x4)
            return x5
        else:
            return x4


class RNDTarget_Net(Base_Net):
    """ Used to predict output of random network.
        see https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        for details.
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

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


class RNDPredictor_Net(Base_Net):
    """ Used to predict output of random network.
        see https://github.com/openai/random-network-distillation/blob/master/policies/cnn_policy_param_matched.py
        for details.
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1))

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

class TVFModel(nn.Module):
    """
    Truncated Value Function model
    based on PPG, and (optionaly) using RND

    network: the network to use, [nature_cnn]
    input_dims: tuple containing input dims with channels first, e.g. (4, 84, 84)
    actions: number of actions
    device: device to use for model
    dtype: dtype to use for model

    use_rnd: enables random network distilation for exploration
    use_rnn: enables recurrent model (not implemented yet)

    tvf_horizon_transform: transform to use on horizons before input.
    tvf_hidden_units: number of hidden units to use on FC layer before output
    tvf_activation: activation to use on TVF FC layer
    tvf_h_scale: scales horizon output [constant|linear|square], where linear predicts average reward

    network_args: dict containing arguments for network

    """
    def __init__(
            self,
            network: str,
            input_dims: tuple,
            actions: int,
            device: str = "cuda",
            dtype: torch.dtype=torch.float32,

            use_rnd:bool = False,
            use_rnn:bool = False,

            tvf_horizon_transform = lambda x : x / 1000,
            tvf_hidden_units:int = 512,
            tvf_activation:str = "relu",
            tvf_h_scale:str = 'constant',
            network_args:Union[dict, None] = None,
    ):

        assert not use_rnn, "RNN not supported yet"

        super().__init__()

        self.input_dims = input_dims
        self.actions = actions
        self.device = device
        self.dtype = dtype

        self.name = "PPG-" + network
        single_channel_input_dims = (1, *input_dims[1:])
        self.tvf_h_scale = tvf_h_scale
        self.use_rnd = use_rnd

        self.policy_net = construct_network(network, input_dims, **(network_args or {}))
        self.value_net = construct_network(network, input_dims, **(network_args or {}))
        if self.use_rnd:
            self.prediction_net = RNDPredictor_Net(single_channel_input_dims)
            self.target_net = RNDTarget_Net(single_channel_input_dims)
            self.obs_rms = utils.RunningMeanStd(shape=single_channel_input_dims)
            self.features_mean = 0
            self.features_std = 0

        self.tvf_activation = tvf_activation
        self.tvf_hidden_units = int(tvf_hidden_units)
        self.horizon_transform = tvf_horizon_transform

        # policy outputs policy, but also value so that we can train it to predict value as an aux task
        # value outputs extrinsic and intrinsic value
        self.policy_net_policy = nn.Linear(self.policy_net.hidden_units, actions)
        self.policy_net_value = nn.Linear(self.policy_net.hidden_units, 2)

        # value net outputs a basic value estimate as well as the truncated value estimates (for extrinsic only)
        self.value_net_value = nn.Linear(self.policy_net.hidden_units, 2)
        self.value_net_hidden = nn.Linear(self.policy_net.hidden_units + 1, self.tvf_hidden_units)
        self.value_net_tvf = nn.Linear(self.tvf_hidden_units, 1)

        self.set_device_and_dtype(device, dtype)

    def log_policy(self, x, state=None):
        """ Returns detached log_policy for given input. """
        return self.forward(x, state, output="policy")["log_policy"].detach().cpu().numpy()

    def perform_normalization(self, x):
        """
        Applies normalization transform, and updates running mean / std.
        """

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

    def set_device_and_dtype(self, device, dtype):
        """
        Set the device and type for model
        """
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

    def forward(self, x, horizons=None, output:str = "both", policy_temperature: float = 1.0):
        """

        Forward input through model and return dictionary containing

            log_policy: log policy
            policy_int_value: policy networks intrinsic value estimate
            policy_ext_value: policy networks extrinsic value estimate

            int_value: intrinsic value estimate
            ext_value: extrinsic value estimate (often trained as most distant horizon estimate)
            tvf_value: (if horizons given) truncated horizon value estimates

        x: tensor of dims [B, *obs_shape]
        horizons: (optional) int32 tensor of dims [B, H]
        output: which network(s) to run ["both", "policy", "value"]
        """

        assert output in ["both", "policy", "value"]

        result = {}
        x = self.prep_for_model(x)

        # policy part
        if output in ["both", "policy"]:

            policy_features = F.relu(self.policy_net.forward(x))
            policy_values = self.policy_net_value(policy_features)
            unscaled_policy = self.policy_net_policy(policy_features)
            result['raw_policy'] = unscaled_policy

            rescaled_policy = unscaled_policy / policy_temperature

            result['log_policy'] = F.log_softmax(rescaled_policy, dim=1)

            result['policy_ext_value'] = policy_values[:, 0]
            result['policy_int_value'] = policy_values[:, 1]

        # value part
        if output in ["both", "value"]:
            value_features = F.relu(self.value_net.forward(x))
            value_values = self.value_net_value(value_features)
            result['ext_value'] = value_values[:, 0]
            result['int_value'] = value_values[:, 1]

            # horizon generation
            if horizons is not None:

                # upload horizons to GPU and cast to float
                if type(horizons) is np.ndarray:
                    horizons = torch.from_numpy(horizons)

                horizons = horizons.to(device=value_values.device, dtype=torch.float32)

                _, H = horizons.shape

                transformed_horizons = self.horizon_transform(horizons)

                # x is [B, 512], make it [B, H, 512]
                x_duplicated = value_features[:, None, :].repeat(1, H, 1)

                x_with_side_info = torch.cat([
                    x_duplicated,
                    transformed_horizons[:, :, None]
                ], dim=-1)

                if self.tvf_activation == "relu":
                    activation = F.relu
                elif self.tvf_activation == "tanh":
                    activation = F.tanh
                elif self.tvf_activation == "sigmoid":
                    activation = F.sigmoid
                else:
                    raise Exception("invalid activation")

                tvf_h = activation(self.value_net_hidden(x_with_side_info))
                tvf_values = self.value_net_tvf(tvf_h)[..., 0]

                if self.tvf_h_scale == "constant":
                    tvf_values = tvf_values
                elif self.tvf_h_scale == "linear":
                    tvf_values = tvf_values * transformed_horizons
                elif self.tvf_h_scale == "squared":
                    # this is a linear interpolation between constant and squared
                    tvf_values = tvf_values * (2 * transformed_horizons - transformed_horizons ** 2)
                else:
                    # todo: implement the linear then constant version
                    raise ValueError(f"invalid h_scale: {self.tvf_h_scale}")

                result['tvf_value'] = tvf_values

        return result

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



# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------

def construct_network(head_name, input_dims, **kwargs) -> Base_Net:
    head_name = head_name.lower()
    if head_name == "nature":
        return NatureCNN_Net(input_dims, **kwargs)
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
