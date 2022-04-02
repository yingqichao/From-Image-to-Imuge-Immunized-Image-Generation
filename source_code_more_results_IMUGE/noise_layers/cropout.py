import torch
import torch.nn as nn
from noise_layers.crop import get_random_rectangle_inside
import matplotlib.pyplot as plt
import numpy as np
from config import GlobalConfig
import math

class Cropout(nn.Module):
    """
    Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
    the cover image. The resulting image has the same size as the original and the noised images.
    """
    def __init__(self, config=GlobalConfig()):
        super(Cropout, self).__init__()
        self.config = config
        self.device = config.device

    def forward(self, embedded_image,require_attack=None, min_size=0.1, max_size=None,Cover=None, blockNum=100):
        block = 0
        if require_attack is None:
            require_attack = self.config.attack_portion
        if max_size is None:
            max_size = self.config.crop_size
        # if cover_image is None:
        cover_image = torch.zeros_like(embedded_image)
        assert embedded_image.shape == cover_image.shape
        sum_attacked = 0
        cropout_mask = torch.zeros_like(embedded_image)


        while sum_attacked<require_attack and block<blockNum:
            h_start, h_end, w_start, w_end, ratio = get_random_rectangle_inside(
                image=embedded_image, height_ratio_range=(min_size, max_size), width_ratio_range=(min_size, max_size))
            sum_attacked += ratio
            # 被修改的区域内赋值1, dims: batch channel height width
            cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
            block += 1

        print("                                     Attacked Ratio: {0}, Max Crop Size: {1}".format(sum_attacked,max_size))
        tampered_image = embedded_image * (1-cropout_mask) + cover_image * cropout_mask

        CropWithCover = embedded_image * (1-cropout_mask) + Cover * cropout_mask


        return tampered_image, CropWithCover, cropout_mask, sum_attacked # cropout_label.to(self.device)

# class Cropout(nn.Module):
#     """
#     Combines the noised and cover images into a single image, as follows: Takes a crop of the noised image, and takes the rest from
#     the cover image. The resulting image has the same size as the original and the noised images.
#     """
#     def __init__(self, height_ratio_range, width_ratio_range):
#         super(Cropout, self).__init__()
#         self.height_ratio_range = height_ratio_range
#         self.width_ratio_range = width_ratio_range
#
#     def forward(self, noised_and_cover):
#         noised_image = noised_and_cover[0]
#         cover_image = noised_and_cover[1]
#         assert noised_image.shape == cover_image.shape
#
#         cropout_mask = torch.zeros_like(noised_image)
#         h_start, h_end, w_start, w_end = get_random_rectangle_inside(image=noised_image,
#                                                                      height_ratio_range=self.height_ratio_range,
#                                                                      width_ratio_range=self.width_ratio_range)
#         cropout_mask[:, :, h_start:h_end, w_start:w_end] = 1
#
#         noised_and_cover[0] = noised_image * cropout_mask + cover_image * (1-cropout_mask)
#         return  noised_and_cover