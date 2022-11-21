import ast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import tensor_utilities as tu

from typing import Union
from enum import Enum

from . import utils, impala

# rough speeds for these models (on a 2080 TI, and with DNA)

# Encoder      IPS       Params     Hidden Units   Notes
# impala       767       2.2M       256
# nature       1847      3.4M       256

# (these numbers might be out of date)

# ----------------------------------------------------------------------------------------------------------------
# Heads (feature extractors)
# ----------------------------------------------------------------------------------------------------------------

AMP = False

def get_memory_format():
    return torch.channels_last if AMP else torch.contiguous_format

def get_dtype():
    return torch.float16 if AMP else torch.float32

class TVFMode(Enum):
    OFF = 'off'
    FIXED = 'fixed'


class BaseNet(nn.Module):

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

class ImpalaCNN(BaseNet):
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
                 **extra_args
                 ):

        super().__init__(input_dims, hidden_units)

        curshape = input_dims

        s = 1 / math.sqrt(len(channels))  # per stack scale
        self.stacks = nn.ModuleList()
        for out_channel in channels:
            stack = impala.CnnDownStack(curshape[0], nblock=n_block, outchan=out_channel, scale=s, down_sample=down_sample, **extra_args)
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)

        # super weird, they have it setup so that bias=false enables the bias... which we definitely want here
        # (just not on the convolutions, [except the first])
        self.dense = tu.NormedLinear(utils.prod(curshape), hidden_units, scale=1.414)

        self.out_shape = curshape
        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units

    def forward(self, x):
        """ forwards input through model, returns features (without relu)
            input should be in B,C,H,W format
         """
        # with torch.torch.autocast(device_type='cuda', enabled=AMP):
        x = impala.sequential(self.stacks, x, diag_name=self.name)
        x = impala.flatten_image(x)
        x = torch.relu(x)
        x = self.dense(x)
        return x

class NatureCNN(BaseNet):
    """ Takes stacked frames as input, and outputs features.
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
        Note: it's much better to just get the weight initialization right than to perform normalization. 
        We're just using 3 layers here, should be no problem.
    """

    def __init__(self, input_dims, hidden_units=512, base_channels=32):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = tu.CustomConv2d(input_channels, base_channels, kernel_size=(8, 8), stride=(4, 4), scale=1.414, weight_init="orthogonal")
        self.conv2 = tu.CustomConv2d(base_channels, 2 * base_channels, kernel_size=(4, 4), stride=(2, 2), scale=1.414, weight_init="orthogonal")
        self.conv3 = tu.CustomConv2d(2 * base_channels, 2 * base_channels, kernel_size=(3, 3), stride=(1, 1), scale=1.414, weight_init="orthogonal")

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.hidden_units = hidden_units
        if self.hidden_units > 0:
            self.fc = tu.CustomLinear(self.d, hidden_units, scale=1.414, weight_init="orthogonal")


    # this causes everything to be on cuda:1... hmm... even when it's disabled...
    #@torch.autocast(device_type='cuda', enabled=AMP)
    def forward(self, x):
        """ forwards input through model, returns features (without relu) """

        D = self.d

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.reshape(x, [-1, D])
        if self.hidden_units > 0:
            features_out = self.fc(x)
        else:
            features_out = x

        return features_out


class StandardMLP(BaseNet):
    """ Based on https://arxiv.org/pdf/1707.06347.pdf
        Based on https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """

    def __init__(self, input_dims, hidden_units=64):

        super().__init__(input_dims, hidden_units)

        s = torch.nn.init.calculate_gain("tanh")

        # taken from https://github.com/alirezakazemipour/Continuous-PPO/blob/master/model.py
        self.fc1 = tu.CustomLinear(input_dims[0], hidden_units, weight_init="orthogonal", scale=s)
        self.fc2 = tu.CustomLinear(hidden_units, hidden_units, weight_init="orthogonal", scale=1.414)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """

        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x


class RTG_LSTM(BaseNet):
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


class RNDTarget(BaseNet):
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


class RNDPredictor(BaseNet):
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

            activation_fn="relu",

            tvf_fixed_head_horizons: Union[None, list] = None,
            tvf_feature_sparsity: float = 0.0,
            tvf_feature_window: int = -1,

            head_scale: float = 1.0,

            # value_head_names: Union[list, tuple] = ('ext', 'int', 'ext_m2', 'int_m2', 'uni'),
            value_head_names: Union[list, tuple] = ('ext',), # keeping it simple

            head_bias: bool=False,

            device=None,
            **kwargs
    ):
        """
        @encoder: the encoder type to use [nature|impala]
        @input_dims: the expected input dimensions (for RGB input this is (C,H,W))
        @n_actions: the number of actions model should produce a policy for
        @hidden_units: number of encoder hidden features
        @activation_fn: the activation function to use after feature encoder
        @tvf_fixed_head_horizons: if given then model enables tvf with fixed heads at these locations.
        @value_heads: list of value heads to output, a standard and tvf output will be created.
        @device: the device to allocate model to
        """

        if len(kwargs) > 0:
            # this is just to check didn't accidently ignore some parameters
            print(f" - additional encoder args: {kwargs}")

        super().__init__()

        self.encoder = construct_network(encoder, input_dims, hidden_units=hidden_units, **kwargs).to(device)

        self.hidden_units = hidden_units

        self.encoder_activation_fn = activation_fn

        def linear(*args, scale=1.0, **kwargs):
            # almost always don't want bias for the final heads.
            # the reason for this is any change in bias will apply to all states, which is not ideal.
            # also, in tvf_mode, the bias enables the model to output a simple function of h, which
            # we would like to avoid.
            # the game boss fight can do this. The noise is so high that the model just outputs the state
            # independant average, and is not able to easily improve on that.
            return tu.CustomLinear(*args, scale=scale, weight_init="orthogonal", bias=head_bias, **kwargs)

        self.policy_head = linear(self.hidden_units, n_actions, scale=head_scale)
        self.value_head = linear(self.hidden_units, len(value_head_names), scale=head_scale)
        self.advantage_head = linear(self.hidden_units, n_actions, scale=head_scale)

        self.log_std = nn.Parameter(torch.zeros(n_actions, device=device, dtype=torch.float32))

        self.value_head_names = list(value_head_names)
        self.tvf_fixed_head_horizons = tvf_fixed_head_horizons
        self.tvf_feature_sparsity = tvf_feature_sparsity
        self.tvf_feature_window = tvf_feature_window

        self.tvf_head = None
        self.tvf_features_mask = None

        if self.tvf_fixed_head_horizons is not None:
            self.tvf_head = linear(
                self.hidden_units,
                len(tvf_fixed_head_horizons) * len(value_head_names),
                device=device,
                scale=head_scale
            )

            mask = torch.ones([len(self.tvf_fixed_head_horizons), self.hidden_units], dtype=torch.float32, device=device)
            mask.requires_grad = False

            if self.tvf_feature_sparsity > 0:
                keep_prob = (1 - self.tvf_feature_sparsity)
                g = torch.Generator(device=device)
                g.manual_seed(99) # mask will be recreated on restore (it is not saved) so make sure it's always the same.
                # increase magnitude of weights that are not zeroed out.
                tvf_features_mask = torch.bernoulli(mask * keep_prob, generator=g) * math.sqrt(1 / keep_prob)
                self.tvf_head.weight.data *= tvf_features_mask
                self.tvf_features_mask = torch.gt(tvf_features_mask, 0).to(torch.uint8)

            if self.tvf_feature_window > 0:
                assert self.tvf_feature_sparsity <= 0, "sparsity and feature window not supported together"
                n_heads = len(self.tvf_fixed_head_horizons)
                n_features = self.hidden_units
                first_left = 0
                first_right = self.tvf_feature_window
                last_left = n_features - self.tvf_feature_window
                last_right = n_features

                for head in range(n_heads):
                    factor = (head / (n_heads-1))
                    left = int(first_left * (1-factor) + last_left * factor)
                    right = int(first_right * (1 - factor) + last_right * factor)
                    mask[head, :left] = 0
                    mask[head, right:] = 0

                old_scale = 1/math.sqrt(self.hidden_units)
                new_scale = 1/math.sqrt(self.tvf_feature_window)

                mask = mask * (new_scale/old_scale)
                # zero out and rescale weights
                self.tvf_head.weight.data *= mask
                # keep a copy of the mask with 0 for not used and 1 for used.
                self.tvf_features_mask = torch.gt(mask, 0).to(torch.uint8)



    def mask_feature_weights(self):
        if self.tvf_features_mask is not None:
            self.tvf_head.weight.data *= self.tvf_features_mask

    @property
    def use_tvf(self):
        return self.tvf_fixed_head_horizons is not None

    def forward(
            self, x, policy_temperature=1.0,
            exclude_value=False,
            exclude_policy=False,
            exclude_tvf=False,
            include_features: bool = False,
            required_tvf_heads: list = None
        ):
        """
        x is [B, *state_shape]

        @param required_tvf_heads: list of tvf heads required, none evaluates all heads
        """

        result = {}

        encoder_features = self.encoder(x)

        # convert back to float32, and also switch to channels first, not that that should matter.
        encoder_features = encoder_features.float(memory_format=torch.contiguous_format)

        if self.encoder_activation_fn == "relu":
            activation_function = F.relu
        elif self.encoder_activation_fn == "tanh":
            activation_function = torch.tanh
        else:
            raise ValueError(f"Invalid activation function {self.encoder_activation_fn}")

        if include_features:
            # used for debugging sometimes
            result['raw_features'] = encoder_features
            encoder_features = activation_function(encoder_features)
            result['features'] = encoder_features
        else:
            encoder_features = activation_function(encoder_features)

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
            result[f'value'] = value_values

            if not exclude_tvf and self.use_tvf:
                # note, it's a shame we have to do this every time.
                # even though they were zeroed out they still become non-zero after an optimizer update.
                self.mask_feature_weights()
                tvf_values = self.tvf_head(encoder_features)
                K = len(self.tvf_fixed_head_horizons)
                result[f'tvf_value'] = tvf_values.reshape([-1, K, len(self.value_head_names)])
                if required_tvf_heads is not None:
                    # select on the heads needed
                    result[f'tvf_value'] = result[f'tvf_value'][:, required_tvf_heads]

        # always include advantages for the moment
        result[f'advantage'] = self.advantage_head(encoder_features)

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
            hidden_units:int = 512,
            encoder_activation_fn: str = "relu",
            observation_normalization=False,
            freeze_observation_normalization=False,
            tvf_fixed_head_horizons: Union[None, list] = None,
            tvf_fixed_head_weights: Union[None, list] = None,
            tvf_feature_sparsity: float = 0.0,
            tvf_feature_window: int = -1,
            head_scale: float=1.0,
            value_head_names=('ext',),
            norm_eps: float = 1e-5,
            head_bias: bool = False,
            observation_scaling: str = "scaled"
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
            hidden_units: number of hidden units to use on FC layer before output
            network_args: dict containing arguments for network

        """

        super().__init__()

        self.input_dims = input_dims
        self.actions = actions
        self.device = device
        self.dtype = dtype
        self.observation_normalization = observation_normalization
        self.tvf_fixed_head_weights = tvf_fixed_head_weights
        self.encoder_name = encoder
        self.norm_eps = norm_eps
        self.observation_scaling = observation_scaling

        # todo: rename this..
        if architecture == "single":
            self.name = "PPO-" + encoder
        if architecture == "dual":
            if tvf_fixed_head_horizons is not None:
                self.name = "TVF-" + encoder
            else:
                self.name = "DNA-" + encoder

        single_channel_input_dims = (1, *input_dims[1:])
        self.use_rnd = use_rnd
        self.freeze_observation_normalization = freeze_observation_normalization
        self._mu = None
        self._std = None
        self.tvf_fixed_head_horizons = tvf_fixed_head_horizons

        if type(encoder_args) is str:
            encoder_args = ast.literal_eval(encoder_args)

        if use_rnd:
            assert 'int' in value_head_names, "RND requires int value head."

        def make_net(**extra_args):
            return DualHeadNet(
                encoder=encoder,
                input_dims=input_dims,
                hidden_units=hidden_units,
                activation_fn=encoder_activation_fn,
                tvf_fixed_head_horizons=tvf_fixed_head_horizons,
                tvf_feature_sparsity=tvf_feature_sparsity,
                tvf_feature_window=tvf_feature_window,
                n_actions=actions,
                device=device,
                head_scale=head_scale,
                value_head_names=value_head_names,
                head_bias=head_bias,
                **extra_args,
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
            self.prediction_net = RNDPredictor(single_channel_input_dims)
            self.target_net = RNDTarget(single_channel_input_dims)
            self.rnd_features_mean = 0.0
            self.rnd_features_std = 0.0
            self.rnd_features_var = 0.0
            self.rnd_features_max = 0.0

        self.set_device_and_dtype(device, dtype)

    def adjust_value_scale(self, factor: float, process_value=True, process_tvf=True, value_net_only=False):
        """
        Scales the value predictions of all models by given amount by scaling weights on the final layer.
        """

        if value_net_only:
            models = [self.value_net]
        elif self.architecture == "single":
            models = [self.policy_net]
        elif self.architecture == "dual":
            models = [self.policy_net, self.value_net]
        else:
            raise ValueError(f"Invalid architecture {self.architecture}")

        with torch.no_grad():
            for model in models:
                if process_value:
                    model.value_head.weight.data *= factor
                    model.value_head.bias.data *= factor
                if model.tvf_head is not None and process_tvf:
                    model.tvf_head.weight.data *= factor
                    model.tvf_head.bias.data *= factor

    def model_size(self, trainable_only: bool = True):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) if trainable_only else self.parameters()
        return sum([np.prod(p.size()) for p in model_parameters])

    def log_policy(self, x):
        """ Returns detached log_policy for given input. """
        return self.forward(x, output="policy")["log_policy"].detach().cpu().numpy()

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

        x = torch.clamp((x - self._mu) / (self._std + self.norm_eps), -5, 5)

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

        x = x[:, -1:, :, :]  # rnd works on just one channel so take the last (which is the most recent)

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
            output: str = "default",
            policy_temperature: float = 1.0,
            include_rnd=False,
            include_features=False,
            update_normalization=False,
            **kwargs,
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
            'policy_temperature': policy_temperature,
        }
        args.update(kwargs)

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
            scale_int scales uint8 to [0..1]
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
            if self.observation_scaling == "scaled":
                x = x / 255.0
            elif self.observation_scaling == "centered":
                x = ((x / 255.0)-0.5)*1
            elif self.observation_scaling == "unit":
                # approximates unit normal
                x = ((x / 255.0)-0.5)*6 # from -3 to +3, which should be close to unit norm
            else:
                raise ValueError(f"Invalid observation_scaling mode {self.observation_scaling}")
        return x


# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------

def construct_network(head_name, input_dims, **kwargs) -> BaseNet:
    head_name = head_name.lower()
    if head_name == "nature":
        return NatureCNN(input_dims, **kwargs)
    if head_name == "impala":
        return ImpalaCNN(input_dims, channels=(16, 32, 32), **kwargs)
    if head_name == "rtg":
        return RTG_LSTM(input_dims, **kwargs)
    if head_name == "mlp":
        return StandardMLP(input_dims, **kwargs)

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


def window(x, max_x, max_y):
    window_size = max_y // max_x
    first_left = 0
    first_right = window_size
    last_left = max_y - window_size
    last_right = max_y
    factor = (x / (max_x - 1))
    left = int(first_left * (1 - factor) + last_left * factor)
    right = int(first_right * (1 - factor) + last_right * factor)
    return left, right
