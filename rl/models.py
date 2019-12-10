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

    def __init__(self, input_dims, hidden_units=512, weight_scale=1.0, bias_scale=1.0):

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

        if weight_scale != 1.0:
            self.conv1.weights *= weight_scale
            self.conv2.weights *= weight_scale
            self.conv3.weights *= weight_scale
            self.fc.weights *= weight_scale

        if bias_scale != 1.0:
            self.conv1.bias *= bias_scale
            self.conv2.bias *= bias_scale
            self.conv3.bias *= bias_scale
            self.fc.bias *= bias_scale


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

class ResNetCNNHead(BaseHead):
    """ Takes stacked frames as input, and outputs features.
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.fc = nn.Linear(self.d, hidden_units)

        if weight_scale != 1.0:
            self.conv1.weights *= weight_scale
            self.conv2.weights *= weight_scale
            self.conv3.weights *= weight_scale
            self.fc.weights *= weight_scale

        if bias_scale != 1.0:
            self.conv1.bias *= bias_scale
            self.conv2.bias *= bias_scale
            self.conv3.bias *= bias_scale
            self.fc.bias *= bias_scale


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
            'value_ext': value
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
            'log_policy_atn': log_policy_atn,
            'value_ext': value,
            'value_atn': value_atn
        }


# class CNNModel(PolicyModel):
#     """ Nature paper inspired CNN
#     """
#
#     name = "CNN"
#
#     def __init__(self, input_dims, actions, device, dtype, hidden_units=512, layer_scale=1):
#
#         super().__init__()
#
#         self.input_dims = input_dims
#         self.actions = actions
#         c, w, h = input_dims
#         self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#
#         w = get_CNN_output_size(w, [8, 4, 3], [4, 2, 1])
#         h = get_CNN_output_size(h, [8, 4, 3], [4, 2, 1])
#
#         self.out_shape = (64, w, h)
#
#         self.d = utils.prod(self.out_shape)
#         self.fc = nn.Linear(self.d, hidden_units)
#         self.fc_policy = nn.Linear(hidden_units, actions)
#         self.fc_policy_atn = nn.Linear(hidden_units, 49)
#         self.fc_value_int = nn.Linear(hidden_units, 1)
#         self.fc_value_ext = nn.Linear(hidden_units, 1)
#         self.fc_value_atn = nn.Linear(hidden_units, 1)
#         self.freeze_layers = 0
#         self.layer_scale = layer_scale
#
#         self.set_device_and_dtype(device, dtype)
#
#     def forward(self, x):
#         """ forwards input through model, returns policy, and value estimates. """
#         x = F.relu(self.features(x))
#         policy = F.log_softmax(self.fc_policy(x), dim=1)
#         attention_policy = F.log_softmax(self.fc_policy_atn(x), dim=1)
#         value_ext = self.fc_value_ext(x).squeeze(dim=1)
#         value_int = self.fc_value_int(x).squeeze(dim=1)
#         value_attn = self.fc_value_atn(x).squeeze(dim=1)
#         return policy, attention_policy, value_ext, value_int, value_attn
#
#     def conv_layer(self, conv, x, detach=False):
#         x = F.relu(conv(x))
#         if self.layer_scale != 1:
#             x *= self.layer_scale
#         if detach:
#             x = x.detach()
#         return x
#
#     def features(self, x):
#         if len(x.shape) == 3:
#             # make a batch of 1 for a single example.
#             x = x[np.newaxis, :, :, :]
#
#         validate_dims(x, (None, *self.input_dims))
#
#         n,c,w,h = x.shape
#
#         x = self.prep_for_model(x)
#
#         x = self.conv_layer(self.conv1, x, self.freeze_layers >= 1)
#         x = self.conv_layer(self.conv2, x, self.freeze_layers >= 2)
#         x = self.conv_layer(self.conv3, x, self.freeze_layers >= 3)
#
#         assert x.shape[1:] == self.out_shape, "Invalid output dims {} expecting {}".format(x.shape[1:], self.out_shape)
#
#         features = self.fc(x.view(n, self.d)) * self.layer_scale
#
#         if self.freeze_layers >= 4:
#             features = features.detach()
#
#         return features
#

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



# class CNNPredictionModel(PolicyModel):
#     """ Nature paper inspired CNN
#     """
#
#     name = "CNN_Prediction"
#
#     def __init__(self, input_dims, actions, device, dtype, layer_scale=1):
#
#         super().__init__()
#
#         self.input_dims = input_dims
#         self.actions = actions
#         c, w, h = input_dims
#         self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#
#         w = get_CNN_output_size(w, [8, 4, 3], [4, 2, 1])
#         h = get_CNN_output_size(h, [8, 4, 3], [4, 2, 1])
#
#         self.out_shape = (64, w, h)
#
#         self.d = utils.prod(self.out_shape)
#         self.fc1 = nn.Linear(self.d, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.out = nn.Linear(256, 512)
#         self.layer_scale = layer_scale
#
#         self.set_device_and_dtype(device, dtype)
#
#     def forward(self, x):
#         return self.features(x)
#
#     def features(self, x):
#         if len(x.shape) == 3:
#             # make a batch of 1 for a single example.
#             x = x[np.newaxis, :, :, :]
#
#         validate_dims(x, (None, *self.input_dims))
#
#         n,c,w,h = x.shape
#
#         x = self.prep_for_model(x)
#         x = F.relu(self.conv1(x)) * self.layer_scale
#         x = F.relu(self.conv2(x)) * self.layer_scale
#         x = F.relu(self.conv3(x)) * self.layer_scale
#
#         assert x.shape[1:] == self.out_shape, "Invalid output dims {} expecting {}".format(x.shape[1:], self.out_shape)
#
#         x = F.relu(self.fc1(x.view(n, self.d))) * self.layer_scale
#         x = F.relu(self.fc2(x)) * self.layer_scale
#         predicted_features = self.out(x)
#
#         return predicted_features
#
#
# class RNDModel(PolicyModel):
#     """
#     Random network distilation model
#     """
#
#     name = "RND"
#
#     def __init__(self, input_dims, actions, device, dtype):
#         super().__init__()
#
#         single_channel_input_dims = (1, *input_dims[1:])
#
#         self.policy_model = CNNModel(input_dims, actions, device, dtype)
#         self.prediction_model = CNNPredictionModel(single_channel_input_dims, actions, device, dtype)
#         self.random_model = CNNModel(single_channel_input_dims, actions, device, dtype)
#
#         self.random_model.layer_scale = math.sqrt(2)
#         self.prediction_model.layer_scale = math.sqrt(2)
#         self.policy_model.int_value_head = True
#         self.actions = actions
#         self.set_device_and_dtype(device, dtype)
#
#         self.obs_rms = RunningMeanStd(shape=(single_channel_input_dims))
#
#         self.features_mean = 0
#         self.features_std = 0
#
#     def perform_normalization(self, x):
#         """ Applies normalization transform, and updates running mean / std. """
#
#         # update normalization constants
#         x = np.float32(x[:, 0:1])
#         self.obs_rms.update(x)
#         mu = self.prep_for_model(self.obs_rms.mean.astype(np.float32))
#         sigma = self.prep_for_model(self.obs_rms.var.astype(np.float32) ** 0.5)
#
#         # normalize x
#         x = self.prep_for_model(x)
#         x = torch.clamp((x - mu) / (sigma + 1e-5), -5, 5) * 1.5 # need more feature variance...
#
#         return x
#
#     def prediction_error(self, x):
#         """ Returns prediction error for given state. """
#
#         # only need first channel for this.
#         x = self.perform_normalization(x)
#
#         # random features have too low varience due to weight initialization being different from the OpenAI implementation
#         # we adjust it here by simply multiplying the output to scale the features to have a max of around 3
#         random_features = self.random_model.features(x).detach() * 7
#         predicted_features = self.prediction_model.features(x) * 14
#
#         # note: I really want to normalize these... otherwise scale just makes such a difference.
#         # or maybe tanh the features?
#
#         self.features_mean = float(random_features.mean().detach().cpu())
#         self.features_var = float(random_features.var(axis=0).mean().detach().cpu())
#         self.features_max = float(random_features.abs().max().detach().cpu())
#
#         errors = (random_features - predicted_features).pow(2).mean(axis=1)
#
#         return errors
#
#     def forward(self, x):
#         return self.policy_model.forward(x)
#
#
# class ICMModel(PolicyModel):
#     """ Intrinsic curiosity model
#
#     Uses forward prediction error on inverse dynamics features as an axualiary task.
#
#     See https://github.com/pathak22/noreward-rl/blob/master/src/model.py
#     """
#
#     name = "ICM"
#
#     def __init__(self, input_dims, actions, device, dtype, **kwargs):
#
#         super(ICMModel, self).__init__()
#
#         self.actions = actions
#         c, w, h = input_dims
#         self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
#
#         w = get_CNN_output_size(w, [8, 4, 3], [4, 2, 1])
#         h = get_CNN_output_size(h, [8, 4, 3], [4, 2, 1])
#
#         self.out_shape = (64, w, h)
#
#         self.d = utils.prod(self.out_shape)
#         self.fc = nn.Linear(self.d, 512)
#         self.fc_policy = nn.Linear(512, actions)
#         self.fc_value = nn.Linear(512, 1)
#
#         # ICM part
#
#         # note we should take the last frame only, not the entire stack... but alas, this is how it is in ICM.
#         self.encode1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1)
#         self.encode2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.encode3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#         self.encode4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
#
#         # output padding is needed to make sure the convolutions match dims.
#         self.decode1 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.decode2 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
#         self.decode3 = nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=1, output_padding=0)
#         self.decode4 = nn.ConvTranspose2d(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
#
#         self.idm_fc = nn.Linear(288 * 2, 256)
#         self.idm_out = nn.Linear(256, self.actions)
#
#         self.fdm_fc = nn.Linear(288 + self.actions, 256)
#         self.fdm_out = nn.Linear(256, 288)
#
#         self.set_device_and_dtype(device, dtype)
#
#     def extract_down_sampled_frame(self, x):
#         """ Converts 4x84x84 (uint8) state to a 4x42x42 (float) frame. """
#
#         if type(x) == np.ndarray:
#             # upload to gpu if needed.
#             x = torch.from_numpy(x).to(self.device, non_blocking=True)
#
#         validate_dims(x, (None, 4, 84, 84), torch.uint8)
#
#         x = self.prep_for_model(x) / 255.0
#         x = F.max_pool2d(x, 2)
#         return x
#
#     def encode(self, x):
#         """ runs a single 4x42x42 (float) frame through the encoder part of the IDM, returns the embedded (288) features """
#
#         validate_dims(x, (None, 4, 42, 42), torch.float)
#
#         n,c,h,w = x.shape
#
#         x = F.relu(self.encode1(x))
#         x = F.relu(self.encode2(x))
#         x = F.relu(self.encode3(x))
#         x = F.relu(self.encode4(x))
#         x = x.view(n, 288)
#         return x
#
#     def decode(self, x):
#         """ runs an embedding through decoder returning [n, 4, 42 ,42] (float32) images."""
#
#         validate_dims(x, (None, 288), torch.float)
#
#         n,d = x.shape
#
#         x = x.reshape((n, 32, 3, 3))
#         x = F.relu(self.decode1(x))
#         x = F.relu(self.decode2(x))
#         x = F.relu(self.decode3(x))
#         x = torch.sigmoid(self.decode4(x))
#
#         return x
#
#     def idm(self, state_1, state_2):
#         """ Predicts the action that occurred between state_1 and state_2"""
#
#         v1 = self.encode(self.extract_down_sampled_frame(state_1))
#         v2 = self.encode(self.extract_down_sampled_frame(state_2))
#
#         # concat the embeddings together, then feed into a linear layer and finally make a prediction about the input.
#         x = torch.cat((v1, v2), dim=1)
#         x = F.relu(self.idm_fc(x))
#         log_probs = self.idm_out(x)
#
#         return F.log_softmax(log_probs, dim=1)
#
#     def fdm(self, states, actions):
#         """ Predict next embedding given a state and following action. """
#         state_embeddings = self.encode(self.extract_down_sampled_frame(states))
#         action_embeddings = F.one_hot(actions, self.actions).float()
#         x = torch.cat((state_embeddings, action_embeddings), dim=1)
#
#         x = F.relu(self.fdm_fc(x))
#         return self.fdm_out(x)
#
#     def forward(self, x):
#         """ forwards input through model, returns policy and value estimate. """
#
#         if len(x.shape) == 3:
#             # make a batch of 1 for a single example.
#             x = x[np.newaxis, :, :, :]
#
#         assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
#         assert len(x.shape) == 4, "input should be (N,C,W,H)"
#
#         n,c,w,h = x.shape
#
#         x = self.prep_for_model(x) / 255.0
#
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#
#         assert x.shape[1:] == self.out_shape, "Invalid output dims {} expecting {}".format(x.shape[1:], self.out_shape)
#
#         x = F.relu(self.fc(x.view(n, self.d)))
#
#         policy = F.log_softmax(self.fc_policy(x), dim=1)
#         value = self.fc_value(x).squeeze(dim=1)
#
#         return policy, value
#


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

