import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class PolicyModel(nn.Module):

    def forward(self, x):
        raise NotImplemented()

    def policy(self, x):
        policy, value = self.forward(x)
        return policy

    def value(self, x):
        policy, value = self.forward(x)
        return value

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

    def prep_for_model(self, x):
        """ Converts data to format for model (i.e. uploads to GPU, converts type). """
        return torch.from_numpy(x).to(self.device, non_blocking=True).to(dtype=self.dtype)


class CNNModel(PolicyModel):
    """ Nature paper inspired CNN
    """

    name = "CNN"

    def __init__(self, input_dims, actions, device, dtype):

        super(CNNModel, self).__init__()

        self.actions = actions
        c, w, h = input_dims
        self.conv1 = nn.Conv2d(c, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        w = get_CNN_output_size(w, [8, 4, 3], [4, 2, 1])
        h = get_CNN_output_size(h, [8, 4, 3], [4, 2, 1])

        self.out_shape = (64, w, h)

        self.d = utils.prod(self.out_shape)
        self.fc = nn.Linear(self.d, 512)
        self.fc_policy = nn.Linear(512, actions)
        self.fc_value = nn.Linear(512, 1)

        self.set_device_and_dtype(device, dtype)

    def forward(self, x):
        """ forwards input through model, returns policy and value estimate. """

        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]

        assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
        assert len(x.shape) == 4, "input should be (N,C,W,H)"

        n,c,w,h = x.shape

        x = self.prep_for_model(x) / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        assert x.shape[1:] == self.out_shape, "Invalid output dims {} expecting {}".format(x.shape[1:], self.out_shape)

        x = F.relu(self.fc(x.view(n, self.d)))

        policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)

        return policy, value


class ImprovedCNNModel(PolicyModel):
    """ An improved CNN model that uses 3x3 filters and maxpool instead of strides.
    """

    name = "Improved_CNN"

    def __init__(self, input_dims, actions, device, dtype, include_xy=True):

        super(ImprovedCNNModel, self).__init__()

        self.actions = actions
        c, w, h = input_dims

        self.include_xy = include_xy

        if self.include_xy:
            c = c + 2

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        w = get_CNN_output_size(w, [3, 3, 3, 3], [2, 2, 2, 1], max_pool=True)
        h = get_CNN_output_size(h, [3, 3, 3, 3], [2, 2, 2, 1], max_pool=True)

        self.out_shape = (64, w, h)

        self.d = utils.prod(self.out_shape)
        self.fc = nn.Linear(self.d, 512)
        self.fc_policy = nn.Linear(512, actions)
        self.fc_value = nn.Linear(512, 1)

        self.set_device_and_dtype(device, dtype)

    def forward(self, x):
        """ forwards input through model, returns policy and value estimate. """

        if len(x.shape) == 3:
            # make a batch of 1 for a single example.
            x = x[np.newaxis, :, :, :]

        assert x.dtype == np.uint8, "invalid dtype for input, found {} expected {}.".format(x.dtype, "uint8")
        assert len(x.shape) == 4, "input should be (N,C,W,H)"

        n,c,w,h = x.shape

        x = utils.prep_for_model(x) / 255.0

        # give filters access to x,y location
        if self.include_xy:
            x = add_xy(x)

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv4(x))

        if x.shape[1:] != self.out_shape:
            raise Exception("Invalid output dims. Expected {} found {}.".format(x.shape, self.out_shape))

        x = F.relu(self.fc(x.view(n, self.d)))

        policy = F.log_softmax(self.fc_policy(x), dim=1)
        value = self.fc_value(x).squeeze(dim=1)

        return policy, value

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


def add_xy(x):
    """ Adds an xy channel to tensor"""

    n, c, w, h = x.shape
    # from https://gist.github.com/leVirve/0377a8fbac455bfd44e374e5cf8b1260
    xx_channel = torch.arange(w).repeat(1, h, 1)
    yy_channel = torch.arange(h).repeat(1, w, 1).transpose(1, 2)

    xx_channel = xx_channel.float() / (w - 1)
    yy_channel = yy_channel.float() / (h - 1)

    xx_channel = xx_channel * 2 - 1
    yy_channel = yy_channel * 2 - 1

    xx_channel = xx_channel.repeat(n, 1, 1, 1).transpose(2, 3)
    yy_channel = yy_channel.repeat(n, 1, 1, 1).transpose(2, 3)

    return torch.cat([
        x,
        xx_channel.type_as(x),
        yy_channel.type_as(x)], dim=1)

