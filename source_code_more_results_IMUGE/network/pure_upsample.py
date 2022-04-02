import torch.nn as nn
import torch.nn.functional as F


class PureUpsampling(nn.Module):
    def __init__(self, scale=2, mode='bilinear'):
        super(PureUpsampling, self).__init__()
        # assert isinstance(scale, int)
        self.scale = scale
        self.mode = mode

    def forward(self, x):
        h, w = int(x.size(2) * self.scale), int(x.size(3) * self.scale)
        if self.mode == 'nearest':
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode)
        else:
            xout = F.interpolate(input=x, size=(h, w), mode=self.mode, align_corners=True)
        return xout