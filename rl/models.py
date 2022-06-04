import ast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from typing import Union
from enum import Enum

from . import utils, impala

# rough speeds for these models (on a 2080 TI, and with DNA)

# Encoder      IPS       Params     Hidden Units   Notes
# impala_fast  818       1.3M       256            Not actually much faster...
# impala       767       2.2M       256
# nature       1847      3.4M       512
# nature_fat   1451      7.1M       512

JIT = False # causes problems with impala
AMP = False # not helpful

# ----------------------------------------------------------------------------------------------------------------
# Heads (feature extractors)
# ----------------------------------------------------------------------------------------------------------------

def get_memory_format():
    return torch.channels_last if AMP else torch.contiguous_format

def get_dtype():
    return torch.float16 if AMP else torch.float32

class TVFMode(Enum):
    OFF = 'off'
    DYNAMIC = 'dynamic'
    FIXED = 'fixed'


class Base_Net(nn.Module):

    def __init__(self, input_dims, hidden_units):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.trace_module = None

    def jit(self, device):
        assert self.trace_module is None, "Multiple calls to jit."
        fake_input = torch.zeros([256, *self.input_dims], device=device, dtype=get_dtype())
        fake_input = fake_input.to(memory_format=get_memory_format())
        self.trace_module = torch.jit.trace(self, example_inputs=fake_input)

class ImpalaCNN_Net(Base_Net):
    """
        Drop in replacement for Nature CNN network.
        input_dims: dims of input (C, H, W)
    """

    name = "ImpalaCNN"  # put it here to preserve pickle compat

    def __init__(self,
                 input_dims:tuple,
                 hidden_units:int = 256,
                 channels=(16, 32, 32),
                 n_block:int = 2,
                 down_sample='pool', # [pool|stride]
                 **ignore_args
                 ):

        super().__init__(input_dims, hidden_units)

        curshape = input_dims

        s = 1 / math.sqrt(len(channels))  # per stack scale
        self.stacks = nn.ModuleList()
        for out_channel in channels:
            stack = impala.CnnDownStack(curshape[0], nblock=n_block, outchan=out_channel, scale=s, down_sample=down_sample)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        self.dense = impala.NormedLinear(utils.prod(curshape), hidden_units, scale=1.4)

        self.out_shape = curshape
        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units

    def forward(self, x):
        """ forwards input through model, returns features (without relu)
            input should be in B,C,H,W format
         """
        #with torch.torch.autocast(device_type='cuda', enabled=AMP):
        x = impala.sequential(self.stacks, x, diag_name=self.name)
        x = impala.flatten_image(x)
        x = torch.relu(x)
        x = self.dense(x)
        return x

class NatureCNN_Net(Base_Net):
    """ Takes stacked frames as input, and outputs features.
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, input_dims, hidden_units=512, base_channels=32, norm='off'):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, base_channels, kernel_size=(8, 8), stride=(4, 4))
        self.conv2 = nn.Conv2d(base_channels, 2 * base_channels, kernel_size=(4, 4), stride=(2, 2))
        self.conv3 = nn.Conv2d(2 * base_channels, 2 * base_channels, kernel_size=(3, 3), stride=(1, 1))

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units
        if self.hidden_units > 0:
            self.fc = nn.Linear(self.d, hidden_units)

        self.norm = norm

        if self.norm == "off":
            pass
        elif self.norm == 'layer':
            # just on the convolutions... less parameters... less over fitting...
            x = self.conv1(fake_input)
            _, c, w, h = x.shape
            self.n1 = nn.LayerNorm([c, h, w])
            x = self.conv2(x)
            _, c, w, h = x.shape
            self.n2 = nn.LayerNorm([c, h, w])
            x = self.conv3(x)
            _, c, w, h = x.shape
            self.n3 = nn.LayerNorm([c, h, w])
        elif self.norm == 'batch':
            momentum = 0.01 # we have a noisy setting...
            self.n1 = nn.BatchNorm2d(base_channels, momentum)
            self.n2 = nn.BatchNorm2d(2 * base_channels, momentum)
            self.n3 = nn.BatchNorm2d(2 * base_channels, momentum)
        else:
            raise ValueError("Invalid normalization")


    # this causes everything to be on cuda:1... hmm... even when it's disabled...
    #@torch.autocast(device_type='cuda', enabled=AMP)
    def forward(self, x):
        """ forwards input through model, returns features (without relu) """

        D = self.d

        if self.norm == "off":
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        elif self.norm in ["layer", "batch"]:
            x = F.relu(self.conv1(x))
            x = self.n1(x)
            x = F.relu(self.conv2(x))
            x = self.n2(x)
            x = F.relu(self.conv3(x))
            x = self.n3(x)

        x = torch.reshape(x, [-1, D])
        if self.hidden_units > 0:
            x = self.fc(x)
        return x


class MLP_Net(Base_Net):
    """ Based on https://arxiv.org/pdf/1707.06347.pdf
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, input_dims, hidden_units=64):

        super().__init__(input_dims, hidden_units)

        self.fc1 = nn.Linear(input_dims[0], hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)


    # this causes everything to be on cuda:1... hmm... even when it's disabled...
    #@torch.autocast(device_type='cuda', enabled=AMP)
    def forward(self, x):
        """ forwards input through model, returns features (without relu) """

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x


class RTG_Net(Base_Net):
    """ Takes stacked frames as input, and outputs features.
        Based on model from rescue the general
    """

    def __init__(self, input_dims, hidden_units=512, **ignore_args):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.forward(fake_input, ignore_head=True).shape
        self.out_shape = (c, w, h)

        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units
        if self.hidden_units > 0:
            self.fc = nn.Linear(self.d, hidden_units)

    # this causes everything to be on cuda:1... hmm... even when it's disabled...
    #@torch.autocast(device_type='cuda', enabled=AMP)
    def forward(self, x, ignore_head=False):
        """ forwards input through model, returns features (without relu) """

        N = x.shape[0]

        x = F.relu(torch.max_pool2d(self.conv1(x), 2, 2))
        x = F.relu(torch.max_pool2d(self.conv2(x), 2, 2))
        x = F.relu(torch.max_pool2d(self.conv3(x), 2, 2))

        if ignore_head:
            return x

        x = torch.reshape(x, [N, -1])
        if self.hidden_units > 0:
            x = self.fc(x)
        return x


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

        # we scale the weights so the output variance is about right. The sqrt2 is because of the relu,
        # bias is disabled as per OpenAI implementation.
        # I found I need to bump this up a little to get the variance required (0.4)
        # maybe this is due to weight initialization differences between TF and Torch?
        scale_weights(self, weight_scale=np.sqrt(2)*1.3, bias_scale=0.0)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        N = len(x)
        D = self.d

        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.reshape(x, [N, D])
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

        # we scale the weights so the output variance is about right. The sqrt2 is because of the relu,
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

class DualHeadNet(nn.Module):
    """
    Network has both policy and (multiple) value heads.
    """

    def __init__(
            self,
            encoder: str,
            input_dims: tuple,
            n_actions: int,

            hidden_units: int = 512,
            tvf_mode:Union[TVFMode, str] = TVFMode.DYNAMIC,
            tvf_hidden_units: int = 512,
            tvf_horizon_transform=lambda x: x,
            tvf_time_transform=lambda x: x,
            tvf_max_horizon: int = 30000,

            tvf_value_scale_fn: str = "identity",
            tvf_value_scale_norm: str = "max",

            activation_fn="relu",
            tvf_activation_fn: str = "relu",

            tvf_fixed_head_horizons: Union[None, list] = None,

            # value_head_names: Union[list, tuple] = ('ext', 'int', 'ext_m2', 'int_m2', 'uni'),
            value_head_names: Union[list, tuple] = ('ext',), # keeping it simple

            device=None,
            **kwargs
    ):
        """
        @encoder: the encoder type to use [nature|impala]
        @input_dims: the expected input dimensions (for RGB input this is (C,H,W))
        @n_actions: the number of actions model should produce a policy for

        @hidden_units: number of encoder hidden features

        @tvf_mode: enable truncated value estimates [off|dynamic|fixed].
        @tvf_hidden_units: number of units in the tvf hidden layer (set to 0 to disable extra layer)
        @tvf_horizon_transform: function to transform horizon before processing
        @tvf_time_transform: function to transform time before processing
        @tvf_max_horizon: the maximum (unscaled) horizon for tvf (h_max)

        @tvf_value_scale_fn: how to scale value, setting to "linear" lets model predict the average reward.
            [identity|linear|log|sqrt]
        @tvf_value_scale_norm: how to normalize values, [none|max|half_max]

        @activation_fn: the activation function to use after feature encoder
        @tvf_activation_fn: the activation function to use after truncated value hidden layer

        @tvf_fixed_head_horizons: if given then model switches to fixed heads mode for TVF, which means that
            only estimates for the horizons given in this list are permissible.

        @value_heads: list of value heads to output, a standard and tvf output will be created.

        @device: the device to allocate model to
        """

        # network was the old parameter name.
        if "network" in kwargs:
            encoder = encoder
            del kwargs["network"]

        if type(tvf_mode) is not TVFMode:
            tvf_mode = TVFMode(tvf_mode)

        if tvf_mode == TVFMode.DYNAMIC:
            assert tvf_fixed_head_horizons is not None

        if len(kwargs) > 0:
            # this is just to checkwe didn't accidently ignore some parameters
            print(f" - additional encoder args: {kwargs}")

        super().__init__()

        self.encoder = construct_network(encoder, input_dims, hidden_units=hidden_units, **kwargs)
        # jit is in theory a little faster, but can be harder to debug
        self.encoder = self.encoder.to(device)
        if JIT:
            self.encoder.jit(device)

        assert self.encoder.hidden_units == hidden_units

        self.tvf_max_horizon = tvf_max_horizon
        self.tvf_value_scale_fn = tvf_value_scale_fn
        self.tvf_value_scale_norm = tvf_value_scale_norm
        self.feature_activation_fn = activation_fn
        self.tvf_mode = tvf_mode

        self.policy_head = nn.Linear(self.encoder.hidden_units, n_actions)

        self.tvf_activation = tvf_activation_fn
        self.horizon_transform = tvf_horizon_transform
        self.time_transform = tvf_time_transform
        self.value_head_names = list(value_head_names)
        self.tvf_fixed_head_horizons = tvf_fixed_head_horizons
        self.tvf_hidden_units = tvf_hidden_units

        # value net outputs a basic value estimate as well as the truncated value estimates
        self.value_head = nn.Linear(self.encoder.hidden_units, len(value_head_names))

        if tvf_mode != TVFMode.OFF:
            heads_multiplier = 1 if tvf_fixed_head_horizons is None else len(tvf_fixed_head_horizons)
            if tvf_hidden_units > 0:
                self.tvf_hidden = nn.Linear(self.encoder.hidden_units, tvf_hidden_units)
                self.tvf_hidden_aux = nn.Linear(2, tvf_hidden_units, bias=False)  # for time and horizon

                # because we are adding aux to hidden we want the weight initialization to be roughly the same scale
                torch.nn.init.uniform_(
                    self.tvf_hidden_aux.weight,
                    -1 / (self.encoder.hidden_units ** 0.5),
                    1 / (self.encoder.hidden_units ** 0.5)
                )
                if self.tvf_hidden_aux.bias is not None:
                    torch.nn.init.uniform_(
                        self.tvf_hidden_aux.bias,
                        -1 / (self.encoder.hidden_units ** 0.5),
                        1 / (self.encoder.hidden_units ** 0.5)
                    )
                tvf_n_features = tvf_hidden_units
            else:
                # in this case we just concat,
                tvf_n_features = 2 + self.encoder.hidden_units

            # we want intrinsic / extrinsic versions of first and second moments.
            self.tvf_head = nn.Linear(tvf_n_features, heads_multiplier * len(value_head_names), bias=True)


    @property
    def tvf_activation_function(self):
        if self.tvf_activation == "relu":
            return F.relu
        elif self.tvf_activation == "tanh":
            return F.tanh
        elif self.tvf_activation == "sigmoid":
            return F.sigmoid
        else:
            raise Exception("invalid activation")

    def apply_value_scale(self, values, horizons):
        """
        Applies value scaling.
        values: tensor of dims [B, H, 1] # final dim might be 2 in which case it's just ext_value and int_value, but horizon will be matched.
        horizons: tensor of dims [B, H]
        """

        horizons = horizons[:, :, np.newaxis] # match final dim.

        fn_map = {
            'identity': lambda x: 1,
            'linear': lambda x: x,
            'log': lambda x: torch.log10(10+x)-1,
            'sqrt': lambda x: torch.sqrt(x),
        }

        assert self.tvf_value_scale_fn in fn_map, f"invalid scale fn {self.tvf_value_scale_fn}"
        fn = fn_map[self.tvf_value_scale_fn]

        values = values * fn(horizons)

        max_horizon = fn(torch.tensor(self.tvf_max_horizon))

        if self.tvf_value_scale_norm == "none":
            pass
        elif self.tvf_value_scale_norm == "max":
            values = values / max_horizon
        elif self.tvf_value_scale_norm == "half_max":
            values = values / (max_horizon/2)
        else:
            raise ValueError(f"Invalid tvf_value_scale_norm {self.tvf_value_scale_norm}")

        return values

    def process_tvf_hidden_layer(self, encoder_features, transformed_aux_features, force_single_h: bool=False):
        """
        Takes encoder features, and auxilary features and produces the features used for the tvf network.

        @encoder_features: The features from the encoder (with relu applied) [B, F_encoder]
        @transformed_aux_features: aux features transformed (e.g. logged and scaled) [B, H, 2]
        @force_single_h: Returns [B, F] instead of [B, H, F], where first h is used.

        @returns: TVF hidden features of dims [B, H, F_tvf] or [B, F_tvf]

        """

        if self.tvf_hidden_units > 0:
            # this is the version where we use a hidden layer
            # generate one value estimate per input horizon, horizons can be anything.
            tvf_features = self.tvf_hidden(encoder_features)
            if force_single_h:
                aux_part = self.tvf_hidden_aux(transformed_aux_features[:, 0])  # [B, F]
                tvf_h = self.tvf_activation_function(tvf_features + aux_part)  # [B, H, F] = [B, -, F] + [B, H, F]
            else:
                aux_part = self.tvf_hidden_aux(transformed_aux_features)  # [B, H, F_tvf]
                tvf_h = self.tvf_activation_function(tvf_features[:, None, :] + aux_part)  # [B, H, F] = [B, -, F] + [B, H, F]
        else:
            # in this case we concatinate
            if force_single_h:
                tvf_h = torch.concat([encoder_features, transformed_aux_features[:, 0, :]], dim=1) # [B, F] + [B, 2]
            else:
                _, H, _  = transformed_aux_features.shape
                encoder_features = encoder_features[:, None, :].repeat(1, H, 1)
                tvf_h = torch.concat([encoder_features, transformed_aux_features], dim=2)  # [B, H, F] + [B, H, 2]
        return tvf_h


    def forward(
            self, x, aux_features=None, policy_temperature=1.0,
            exclude_value=False, exclude_policy=False, include_features: bool = False
        ):
        """
        x is [B, *state_shape]
        aux_features, if given are [B, H, 2] where [...,0] are horizons and [...,1] are fractions of remaining time
        """

        result = {}
        if self.encoder.trace_module is not None:
            # faster path, precompiled
            encoder_features = self.encoder.trace_module(x)
        else:
            encoder_features = self.encoder(x)

        # convert back to float32, and also switch to channels first, not that that should matter.
        encoder_features = encoder_features.float(memory_format=torch.contiguous_format)

        if self.feature_activation_fn == "relu":
            af = F.relu
        elif self.feature_activation_fn == "tanh":
            af = torch.tanh
        else:
            raise ValueError(f"Invalid activation function {self.feature_activation_fn}")

        if include_features:
            # used for debugging sometimes
            result['raw_features'] = encoder_features
            encoder_features = af(encoder_features)
            result['features'] = encoder_features
        else:
            encoder_features = af(encoder_features)

        if not exclude_policy:
            unscaled_policy = self.policy_head(encoder_features)
            result['raw_policy'] = unscaled_policy

            assert len(unscaled_policy.shape) == 2

            if policy_temperature <= 0:
                # interpret negative policy temperatures as policy blending.
                # temp=-1 is standard policy, temp=0 is argmax policy, with bending in between
                # grads probably won't work here... ?
                argmax_policy = torch.zeros_like(unscaled_policy)
                argmax_policy[range(len(x)), torch.argmax(unscaled_policy, dim=1)] = 1.0
                base_policy = torch.exp(F.log_softmax(unscaled_policy, dim=1))
                epsilon = 1+policy_temperature
                mixed_strategy = epsilon * argmax_policy + (1 - epsilon) * base_policy
                result['log_policy'] = torch.log(mixed_strategy + 1e-8)
                result['argmax_policy'] = argmax_policy
            elif policy_temperature > 0:
                # standard temperature scaling
                result['log_policy'] = F.log_softmax(unscaled_policy / policy_temperature, dim=1)

        if not exclude_value:

            value_values = self.value_head(encoder_features)
            for i, name in enumerate(self.value_head_names):
                result[f'{name}_value'] = value_values[:, i]

            using_fixed_heads = self.tvf_mode == TVFMode.FIXED

            # if auxiliary features are present generate a result per h using tvf heads
            if aux_features is not None:
                if type(aux_features) is np.ndarray:
                    aux_features = torch.from_numpy(aux_features)
                aux_features = aux_features.to(device=value_values.device, dtype=torch.float32)
                _, H, _ = aux_features.shape

                transformed_aux_features = torch.zeros_like(aux_features)
                horizon_in = aux_features[:, :, 0]
                time_in = aux_features[:, :, 1]
                transformed_aux_features[:, :, 0] = self.horizon_transform(horizon_in)  # [B, H]
                transformed_aux_features[:, :, 1] = self.time_transform(time_in)  # [B, H]

                if using_fixed_heads:
                    transformed_aux_features[:, :, 0] = 0

                tvf_h = self.process_tvf_hidden_layer(encoder_features, transformed_aux_features, force_single_h=using_fixed_heads)

                if using_fixed_heads:
                    # generate all fixed horizons then map to requested order. Any horizons which do not match will
                    # cause an error.

                    tvf_values = self.tvf_head(tvf_h) # [B, F, H_fixed]

                    # work out the mapping from output heads to the order requested
                    h_requested = torch.round(aux_features[0, :, 0]).to(torch.int)
                    assert torch.all(h_requested[None, :] == aux_features[:, :, 0]), "horizons must match in fixed head mode."

                    head_mapping = {}
                    for i, head in enumerate(self.tvf_fixed_head_horizons):
                        head_mapping[int(head)] = i
                    for h in h_requested:
                        if int(h) not in head_mapping.keys():
                            raise ValueError(f"Requested horizon {h}, but not found in {self.tvf_fixed_head_horizons}")
                    ordering = np.asarray([head_mapping[int(h)] for h in h_requested])

                    # sort though heads...
                    for i, name in enumerate(self.value_head_names):
                        result[f'tvf_{name}_value'] = tvf_values[:, ordering + (i*len(self.tvf_fixed_head_horizons))]

                else:
                    tvf_values = self.tvf_head(tvf_h)
                    tvf_values = self.apply_value_scale(tvf_values, horizon_in)
                    for i, name in enumerate(self.value_head_names):
                        result[f'tvf_{name}_value'] = tvf_values[..., i]

        return result

class TVFModel(nn.Module):



    def __init__(
            self,
            encoder: str,
            encoder_args: Union[str, dict],
            input_dims: tuple,
            actions: int,
            device: str = "cuda",
            architecture: str = "dual",
            dtype: torch.dtype=torch.float32,

            use_rnd:bool = False,

            tvf_mode:TVFMode=TVFMode.OFF,
            tvf_max_horizon:int = 65536,
            tvf_horizon_transform = lambda x : x,
            tvf_time_transform = lambda x: x,
            hidden_units:int = 512,
            tvf_hidden_units: int = 512,
            tvf_activation:str = "relu",
            feature_activation_fn: str = "relu",
            tvf_value_scale_fn: str = "identity",
            tvf_value_scale_norm: str = "max",
            observation_normalization=False,
            freeze_observation_normalization=False,
            tvf_fixed_head_horizons: Union[None, list] = None,
    ):
        """
            Truncated Value Function model
            based on PPG, and (optionally) using RND

            encoder: feature encoder to use [nature|impala]
            encoder_args: dict, or dict as string giving any encoder specific args.
            input_dims: tuple containing input dims with channels first, e.g. (4, 84, 84)
            actions: number of actions
            device: device to use for model
            dtype: dtype to use for model

            use_rnd: enables random network distilation for exploration

            tvf_horizon_transform: transform to use on horizons before input.
            hidden_units: number of hidden units to use on FC layer before output
            tvf_activation: activation to use on TVF FC layer

            network_args: dict containing arguments for network

        """

        super().__init__()

        self.input_dims = input_dims
        self.actions = actions
        self.device = device
        self.dtype = dtype
        self.observation_normalization = observation_normalization

        # todo: rename this..
        if architecture == "single":
            self.name = "PPO-" + encoder
        if architecture == "dual":
            if tvf_mode == TVFMode.OFF:
                self.name = "DNA-" + encoder
            else:
                self.name = "TVF-" + encoder

        single_channel_input_dims = (1, *input_dims[1:])
        self.use_rnd = use_rnd
        self.freeze_observation_normalization = freeze_observation_normalization
        self._mu = None
        self._std = None

        if type(encoder_args) is str:
            encoder_args = ast.literal_eval(encoder_args)

        def make_net():
            return DualHeadNet(
                encoder=encoder,
                input_dims=input_dims,
                tvf_mode=tvf_mode,
                tvf_activation_fn=tvf_activation,
                hidden_units=hidden_units,
                tvf_hidden_units=tvf_hidden_units,
                tvf_horizon_transform=tvf_horizon_transform,
                tvf_time_transform=tvf_time_transform,
                tvf_max_horizon=tvf_max_horizon,
                tvf_value_scale_fn=tvf_value_scale_fn,
                tvf_value_scale_norm=tvf_value_scale_norm,
                activation_fn=feature_activation_fn,
                tvf_fixed_head_horizons=tvf_fixed_head_horizons,
                n_actions=actions,
                device=device,
                **(encoder_args or {})
            )

        self.policy_net = make_net()
        if architecture == "dual":
            self.value_net = make_net()
        elif architecture == "single":
            self.value_net = self.policy_net
        else:
            raise Exception("Invalid architecture, use [dual|single]")

        self.architecture = architecture

        if self.observation_normalization:
            # we track each channel separately. Helps if one channel is watermarked, or if we are using color.
            self.obs_rms = utils.RunningMeanStd(shape=input_dims)

        if self.use_rnd:
            assert self.observation_normalization, "rnd requires observation normalization."
            self.prediction_net = RNDPredictor_Net(single_channel_input_dims)
            self.target_net = RNDTarget_Net(single_channel_input_dims)
            self.rnd_features_mean = 0.0
            self.rnd_features_std = 0.0
            self.rnd_features_var = 0.0
            self.rnd_features_max = 0.0

        self.set_device_and_dtype(device, dtype)

    def model_size(self, trainable_only: bool = True):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) if trainable_only else self.parameters()
        return sum([np.prod(p.size()) for p in model_parameters])

    def log_policy(self, x, state=None):
        """ Returns detached log_policy for given input. """
        return self.forward(x, state, output="policy")["log_policy"].detach().cpu().numpy()

    def refresh_normalization_constants(self):
        self._mu = torch.tensor(self.obs_rms.mean.astype(np.float32)).to(self.device, dtype=get_dtype())
        self._std = torch.tensor(self.obs_rms.var.astype(np.float32) ** 0.5).to(self.device, dtype=get_dtype())

    @torch.no_grad()
    def perform_normalization(self, x: torch.tensor, update_normalization: bool = False):
        """
        x: input as [B, C, H, W]
        Applies normalization transform, and updates running mean / std.
        x should be processed (i.e. via prep_for_model)
        if ignore_update is true then no stats will be updated.
        """

        # update normalization constants
        assert type(x) is torch.Tensor, "Input for normalization should be tensor"
        assert x.dtype == get_dtype()

        B, *state_shape = x.shape
        assert tuple(state_shape) == self.input_dims

        if update_normalization and not self.freeze_observation_normalization:
            batch_mean = torch.mean(x, dim=0).detach().cpu().numpy()
            # unbiased=False to match numpy
            batch_var = torch.var(x, dim=0, unbiased=False).detach().cpu().numpy()
            batch_count = x.shape[0]
            self.obs_rms.update_from_moments(batch_mean, batch_var, batch_count)
            self.refresh_normalization_constants()

        if self._mu is None:
            self.refresh_normalization_constants()

        # normalize x
        x = torch.clamp((x - self._mu) / (self._std + 1e-5), -5, 5)

        # note: clipping reduces the std down to 0.3, therefore we multiply the output so that it is roughly
        # unit normal.
        x = x * 3.0

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

    def rnd_prediction_error(self, x, already_normed=False):
        """ Returns prediction error for given state.
            input should be preped and normalized
        """

        B, C, H, W = x.shape

        if not already_normed:
            x = self.prep_for_model(x)
            x = self.perform_normalization(x)

        x = x[:, 0:1, :, :]  # rnd works on just one channel

        # random features have too low varience due to weight initialization being different from the OpenAI implementation
        # we adjust it here by simply multiplying the output to scale the features to have a max of around 3
        random_features = self.target_net.forward(x).detach()
        predicted_features = self.prediction_net.forward(x)

        # note: I really want to normalize these... otherwise scale just makes such a difference.
        # or maybe tanh the features?
        self.rnd_features_mean = float(random_features.mean().detach().cpu())
        self.rnd_features_var = float(random_features.var(axis=0).mean().detach().cpu())
        self.rnd_features_max = float(random_features.abs().max().detach().cpu())

        errors = torch.square(random_features - predicted_features).mean(dim=1)

        return errors

    def forward(
            self,
            x,
            aux_features=None,
            output: str = "default",
            policy_temperature: float = 1.0,
            include_rnd=False,
            include_features=False,
            update_normalization=False,
        ):
        """
        Forward input through model and return dictionary containing

            log_policy: log policy
            policy_int_value: policy networks intrinsic value estimate
            policy_ext_value: policy networks extrinsic value estimate

            int_value: intrinsic value estimate
            ext_value: extrinsic value estimate (often trained as most distant horizon estimate)
            tvf_value: (if horizons given) truncated horizon value estimates

        x: tensor of dims [B, *obs_shape]
        aux_features: (optional) int32 tensor of dims [B, H]
        output: which network(s) to run ["both", "policy", "value"]
        is_test: if true disables normalization constant updates.

        Outputs are:
            policy: policy->policy
            value: value->value
            default: policy->policy, value->value
            full: policy->policy, value->value, value->policy, policy->value (with prefix)
        """

        assert output in ["default", "full", "policy", "value"]

        args = {
            'include_features': include_features,
            'aux_features': aux_features,
            'policy_temperature': policy_temperature,
        }

        result = {}
        x = self.prep_for_model(x)

        if self.observation_normalization:
            x = self.perform_normalization(x, update_normalization=update_normalization)

        if include_rnd:
            result["rnd_error"] = self.rnd_prediction_error(x, already_normed=True)

        # special case for single model version (faster)
        if self.architecture == "single":
            network_output = self.policy_net(x, **args)
            for k, v in network_output.items():
                result["policy_" + k] = v
                result["value_" + k] = v
                result[k] = v
            return result

        if output == "full":
            # this is a special case where we return all heads from both networks
            # required for distillation.
            policy_part = self.policy_net(x, **args)
            value_part = self.value_net(x, **args)
            for k,v in policy_part.items():
                result["policy_" + k] = v
            for k, v in value_part.items():
                result["value_" + k] = v
            return result

        if output in ["default", "policy"]:
            result.update(self.policy_net(
                x,
                **args,
                exclude_value=output == 'default',
            ))
        if output in ["default", "value"]:
            result.update(self.value_net(
                x,
                **args,
                exclude_policy=output == 'default',
                ))

        return result

    @torch.no_grad()
    def prep_for_model(self, x, scale_int=True):
        """ Converts data to format for model (i.e. uploads to GPU, converts type).
            Can accept tensor or ndarray.
            scale_int scales uint8 to [-1..1]
         """

        assert self.device is not None, "Must call set_device_and_dtype."

        validate_dims(x, (None, *self.input_dims))

        # if this is numpy convert it over
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)

        # move it to the correct device
        assert x.dtype in [torch.uint8, torch.float16, torch.float32], Exception("Invalid dtype {}".format(x.dtype))
        was_uint8 = x.dtype == torch.uint8

        x = x.to(self.device, non_blocking=True)
        x = x.to(dtype=get_dtype(), memory_format=get_memory_format(), non_blocking=True)

        # then covert the type (faster to upload uint8 then convert on GPU)
        if was_uint8 and scale_int:
            x = (x / 127.5)-1.0

        return x


# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------

def construct_network(head_name, input_dims, **kwargs) -> Base_Net:
    head_name = head_name.lower()
    if head_name == "nature":
        return NatureCNN_Net(input_dims, **kwargs)
    if head_name == "impala":
        return ImpalaCNN_Net(input_dims, channels=(16, 32, 32), down_sample='pool', **kwargs)
    if head_name == "impala_fast": # not that fast, not really.
        return ImpalaCNN_Net(input_dims, channels=(16, 32, 32, 48), down_sample='stride', **kwargs)
    if head_name == "rtg":
        return RTG_Net(input_dims, **kwargs)
    if head_name == "mlp":
        return MLP_Net(input_dims, **kwargs)

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
