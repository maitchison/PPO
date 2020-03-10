import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import hashlib

from . import utils
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

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.reshape(x, [N,D])
        if self.hidden_units > 0:
            x = self.fc(x)
        return x

class FDMEncoder_Net(Base_Net):
    """ Encoder for forward dynamics model
    """

    def __init__(self, input_dims, hidden_units=512):

        super().__init__(input_dims, hidden_units)

        input_channels = input_dims[0]

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=4, stride=2)

        fake_input = torch.zeros((1, *input_dims))
        _, c, w, h = self.conv3(self.conv2(self.conv1(fake_input))).shape

        self.out_shape = (c, w, h)
        self.d = utils.prod(self.out_shape)
        self.fc = nn.Linear(self.d, hidden_units)

    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.reshape(x, [len(x), self.d])
        x = F.relu(self.fc(x))
        return x

class FDMDecoder_Net(nn.Module):
    """ Encoder for forward dynamics model
    """

    def __init__(self, input_units, in_shape):

        super().__init__()

        self.in_shape = in_shape
        self.d = utils.prod(self.in_shape)
        self.fc = nn.Linear(input_units, self.d)
        self.conv1 = nn.ConvTranspose2d(in_shape[0], 64, kernel_size=3, stride=1)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2)


    def forward(self, x):
        """ forwards input through model, returns features (without relu) """
        x = F.relu(self.fc(x))
        x = torch.reshape(x, [len(x), *self.in_shape])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        x = x[:, :, 4:4+84, 4:4+84]
        return x


class RNDTarget_Net(Base_Net):
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


class RNDPredictor_Net(Base_Net):
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

    def __init__(self, head: str, input_dims, actions, device, dtype, use_rnn=False, **kwargs):
        super().__init__(input_dims, actions)
        self.name = "AC_RNN-"+head if use_rnn else "AC-"+head

        # disable last linear layer of CNN
        if use_rnn and "hidden_units" not in kwargs:
            kwargs["hidden_units"] = 0

        self.net = constructNet(head, input_dims, **kwargs)
        self.use_rnn = use_rnn
        if self.use_rnn:
            self.lstm = nn.LSTM(input_size=self.net.d, hidden_size=512, num_layers=1)

        final_hidden_units = 512 if use_rnn else self.net.hidden_units

        self.fc_policy = nn.Linear(final_hidden_units, actions)
        self.fc_value = nn.Linear(final_hidden_units, 1)
        self.set_device_and_dtype(device, dtype)

    def forward(self, x, state=None):
        # state, if given, should be a tuple (h,c)

        if self.use_rnn and state is None:
            raise Exception("RNN Model requires LSTM state input.")

        x = self.prep_for_model(x)
        x = F.relu(self.net.forward(x))

        result = {}

        if self.use_rnn:

            (h, c) = state

            # input should be sequence, batch, features, we use a sequence length of 1 here.
            x, (h, c) = self.lstm(x[np.newaxis], (h[np.newaxis],c[np.newaxis]))

            x = F.relu(x[0])

            result['state'] = (h[0], c[0])

        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)

        result['log_policy'] = log_policy
        result['ext_value'] = value

        return result

class AttentionModel(BaseModel):
    """ Has extra value and policy heads for fovea attention."""

    def __init__(self, head: str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)
        self.name = "AC-"+head
        self.net = constructNet(head, input_dims, **kwargs)
        self.fc_policy = nn.Linear(self.net.hidden_units, actions)
        self.fc_value = nn.Linear(self.net.hidden_units, 1)
        self.fc_policy_atn = nn.Linear(self.net.hidden_units, 25)
        self.fc_value_atn = nn.Linear(self.net.hidden_units, 1)
        self.set_device_and_dtype(device, dtype)

    def forward(self, x):
        x = self.prep_for_model(x)
        x = F.relu(self.net(x))
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


class EMIModel(BaseModel):
    """
    Estimated Model Improvement model
    """

    def __init__(self, head:str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)

        self.name = "EMI-" + head

        # note:
        # shouldn't we use the same network (the encoder?) to do improvement prediction...

        self.net = constructNet(head, input_dims, **kwargs)
        self.fdm_encoder = FDMEncoder_Net(input_dims, hidden_units=1024)
        self.fdm_decoder = FDMDecoder_Net(input_units=1024+actions, in_shape=self.fdm_encoder.out_shape)
        self.improvement_net = FDMEncoder_Net(input_dims)

        self.fc_policy = nn.Linear(self.net.hidden_units, actions)
        self.fc_ext_value = nn.Linear(self.net.hidden_units, 1)
        self.fc_int_value = nn.Linear(self.net.hidden_units, 1)
        self.fc_pred_improvement = nn.Linear(self.net.hidden_units, 1)

        self.set_device_and_dtype(device, dtype)
        self.counter = 0

    def predict_model_improvement(self, states):
        """ Returns the predicted model improvement from landing in given state. """
        states = self.prep_for_model(states)
        return self.fc_pred_improvement(F.relu(self.improvement_net(states))).squeeze(dim=1)

    def forward(self, x):
        x = self.prep_for_model(x)
        x = F.relu(self.net(x))
        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        ext_value = self.fc_ext_value(x).squeeze(dim=1)
        int_value = self.fc_int_value(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'ext_value': ext_value,
            'int_value': int_value
        }

    def fdm(self, prev_states, actions):
        """
        Predicts the next state given the previous
        """
        prev_states = self.prep_for_model(prev_states)
        if type(actions) is np.ndarray:
            actions = torch.from_numpy(actions)
        actions = actions.to(dtype=torch.int64, device=self.device)
        state_embeddings = self.fdm_encoder(prev_states)
        action_embeddings = F.one_hot(actions, self.actions).float()
        x = torch.cat((state_embeddings, action_embeddings), dim=1)
        x = self.fdm_decoder(x)
        return x

    def fdm_error(self, prev_states, actions, next_states):

        # stub this is modified to just be an auto-encoder...
        prev_states = self.prep_for_model(prev_states)
        next_states = self.prep_for_model(next_states)
        predicted_next_states = self.fdm(prev_states, actions)

        # only look at error on first frame as others are trivial to reconstruct from input.
        error = ((predicted_next_states[:,0] - next_states[:,0])**2).mean() * 256 # increase error as we transformed scale during preprocessing

        return error

    def generate_debug_image(self, prev_states, actions, next_states):
        predicted_next_states = self.fdm(prev_states[0:1], actions[0:1])
        prev_img = prev_states[0, 0]
        next_img = next_states[0, 0]
        pred_img = predicted_next_states[0, 0]
        delt_img = torch.abs(predicted_next_states[0, 0] - next_states[0, 0])

        tst1_img = self.fdm(prev_states * 0, actions * 0)[0, 0]
        tst2_img = self.fdm(prev_states, actions * 0)[0, 0]

        images = [x[np.newaxis] for x in [prev_img, next_img, pred_img, delt_img, tst1_img, tst2_img]]
        debug_image = torchvision.utils.make_grid(images, nrow=3)
        debug_image = F.interpolate(debug_image[np.newaxis], scale_factor=(4, 4))[0]
        return debug_image


class RNDModel(BaseModel):
    """
    Random network distillation model
    """

    def __init__(self, head:str, input_dims, actions, device, dtype, **kwargs):
        super().__init__(input_dims, actions)

        self.name = "RND-" + head

        single_channel_input_dims = (1, *input_dims[1:])

        self.net = constructNet(head, input_dims, **kwargs)
        self.prediction_net = RNDPredictor_Net(single_channel_input_dims)
        self.target_net = RNDTarget_Net(single_channel_input_dims)

        self.fc_policy = nn.Linear(self.net.hidden_units, actions)
        self.fc_ext_value = nn.Linear(self.net.hidden_units, 1)
        self.fc_int_value = nn.Linear(self.net.hidden_units, 1)

        self.obs_rms = utils.RunningMeanStd(shape=(single_channel_input_dims))

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
        x = F.relu(self.net.forward(x))
        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        ext_value = self.fc_ext_value(x).squeeze(dim=1)
        int_value = self.fc_int_value(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'ext_value': ext_value,
            'int_value': int_value
        }

class TokenNet(nn.Module):

    def __init__(self, token_count=32, token_length=16, token_filters=128):
        super().__init__()

        # this is the number of different tokens the model can recognise
        self.conv = nn.Conv2d(1, token_filters, kernel_size=(1, token_length))

    def forward(self, x):
        """ forwards input through model, returns token indicators.
            X tensor of shape [N, token_count, token_length]
            outputs tensor of shape [N, token_channels]
        """

        x = x[:, np.newaxis, :, :] # need to add an empty channel to the tokens
        x = torch.sigmoid(self.conv(x))

        # max pool means each filter will register if the token it's looking for exists anywhere.
        x = torch.max_pool2d(x, kernel_size=(32,1))
        x = x[:,:,0, 0] # remove that last 1 dims.

        return x


class RARModel(BaseModel):
    """
    Random auxiliary rewards model.
    """

    TOKEN_LENGTH = 16   # length of tokens in bits.
    TOKEN_COUNT = 32    # number of tokens in the models side channel input.
    TOKEN_FILTERS = 64  # this is roughly how many tokens the model is able to identify.

    def __init__(self, head:str, input_dims, actions, device, dtype, seed=0, super_state_size=64, **kwargs):
        super().__init__(input_dims, actions)

        self.name = "RAR-" + head

        torch.manual_seed(seed)

        self.net = constructNet(head, input_dims, **kwargs)
        self.state_mapping = constructNet("nature", input_dims, hidden_units=super_state_size)
        self.token_net = TokenNet(
            token_count=self.TOKEN_COUNT,
            token_length=self.TOKEN_LENGTH,
            token_filters=self.TOKEN_FILTERS
        )

        h = self.net.hidden_units + self.TOKEN_FILTERS

        self.fc_policy = nn.Linear(h, actions)
        self.fc_ext_value = nn.Linear(h, 1)
        self.fc_int_value = nn.Linear(h, 1)

        self.set_device_and_dtype(device, dtype)

    def get_mapped_states(self, states):
        """ Applies a mapping from the environment observational space to a smaller state space. """

        # convert this state into a 16bit state index
        states = self.prep_for_model(states)
        states = self.state_mapping.forward((states-0.5)*10) # increase the variance in the states a bit.
        states = states.detach().cpu().numpy()

        mapped_states = []
        for state_bits in states:
            state = [1 if x >= 0 else 0 for x in state_bits]
            state_hex = hashlib.md5(("".join(str(i) for i in state)).encode()).hexdigest()
            state = int(state_hex[-4:],16)
            mapped_states.append(state)

        return np.asarray(mapped_states)

    def token_encode(self, x):
        """ Encodes a state number as a bit vector. """
        return np.asarray([int(bit) for bit in '{0:016b}'.format(x)], dtype=np.uint8)

    def make_tokens(self, visited_reward_sets):
        """ Converts visited state sets to tokens.

            output is [batch_size, token_count, token_length]
        """
        N = len(visited_reward_sets)
        # convert state into tokens
        tokens = torch.zeros((N, self.TOKEN_COUNT, self.TOKEN_LENGTH), dtype=torch.float)
        for i in range(N):
            for j, state in enumerate(visited_reward_sets[i]):
                tokens[i, j % self.TOKEN_COUNT] = torch.from_numpy(self.token_encode(state))
        return tokens

    def forward(self, x, reward_tokens=None):
        """
        Optionally takes a set of reward reward tokens as input.
        """
        x = self.prep_for_model(x)

        x = F.relu(self.net.forward(x))

        if reward_tokens is None:
            token_activations = torch.zeros((len(x), self.TOKEN_FILTERS), dtype=self.dtype, device=self.device)
        else:
            if type(reward_tokens) is np.ndarray:
                reward_tokens = torch.from_numpy(reward_tokens)
            reward_tokens = reward_tokens.to(dtype=self.dtype, device=self.device)
            token_activations = self.token_net.forward(reward_tokens)

        x = torch.cat((x, token_activations), dim=1)

        log_policy = F.log_softmax(self.fc_policy(x), dim=1)
        ext_value = self.fc_ext_value(x).squeeze(dim=1)
        int_value = self.fc_int_value(x).squeeze(dim=1)
        return {
            'log_policy': log_policy,
            'ext_value': ext_value,
            'int_value': int_value
        }



# ----------------------------------------------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------------------------------------------

def constructNet(head_name, input_dims, **kwargs) -> Base_Net:
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
