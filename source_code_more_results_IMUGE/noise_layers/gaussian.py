import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
import matplotlib.pyplot as plt
import numpy as np
from config import GlobalConfig
import math
class Gaussian(nn.Module):
    '''Adds random noise to a tensor.'''

    def __init__(self, config=GlobalConfig()):
        super(Gaussian, self).__init__()
        self.config = config
        self.device = config.device

    def forward(self, tensor, mean=0, stddev=0.1):
        print("Gaussian Attack Added")
        noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(self.device), mean, stddev)
        return tensor + noise


# def gaussian(tensor, mean=0, stddev=0.1):
#     '''Adds random noise to a tensor.'''
#
#     noise = torch.nn.init.normal_(torch.Tensor(tensor.size()).to(device), mean, stddev)
#
#     return tensor + noise