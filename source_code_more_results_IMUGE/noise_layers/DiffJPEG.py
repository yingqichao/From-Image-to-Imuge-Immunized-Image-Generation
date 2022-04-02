# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# Local
from util.modules import compress_jpeg, decompress_jpeg
from noise_layers.utils import diff_round, quality_to_factor
from util import util
from config import GlobalConfig

class DiffJPEG(nn.Module):
    def __init__(self, height, width, differentiable=True, quality=80, config=GlobalConfig()):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(DiffJPEG, self).__init__()
        self.config = config
        self.quality = quality
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round

        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                          factor=factor)

    def forward(self, x):
        '''
        需要先做denormalize，再normalize回来
        '''
        print("Jpeg Quality Factor: {0}".format(self.quality))
        denorm = util.denormalize_batch(x, self.config.std, self.config.mean)
        y, cb, cr = self.compress(denorm)
        recovered = self.decompress(y, cb, cr)
        recovered_norm = util.normalize_batch(recovered, self.config.std, self.config.mean)
        return recovered_norm
