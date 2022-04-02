import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from noise_layers.crop import random_float
from config import GlobalConfig

class Resize(nn.Module):
    """
    Resize the image. The target size is original size * resize_ratio
    """
    def __init__(self, config=GlobalConfig(), resize_ratio_range=(0.5,0.5), interpolation_method='nearest'):
        super(Resize, self).__init__()
        self.config = config
        self.device = config.device
        self.resize_ratio_min = resize_ratio_range[0]
        self.resize_ratio_max = resize_ratio_range[1]
        self.interpolation_method = interpolation_method


    def forward(self, noised_image):
        print("Resize Attack Added")
        #resize_ratio = random_float(self.resize_ratio_min, self.resize_ratio_max)

        resize_ratio = 0.5
        # noised_image = noised_and_cover[0]
        out = F.interpolate(
                                    noised_image,
                                    scale_factor=(resize_ratio, resize_ratio),
                                    mode=self.interpolation_method)

        recover = F.interpolate(
                                    out,
                                    scale_factor=(1/resize_ratio, 1/resize_ratio),
                                    mode=self.interpolation_method)
        return recover
