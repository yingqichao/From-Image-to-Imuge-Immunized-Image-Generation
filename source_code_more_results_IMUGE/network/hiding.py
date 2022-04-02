# %matplotlib inline
import torch
import torch.nn as nn
from network.double_conv import DoubleConv
from network.single_conv import SingleConv

# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.initialH3 = nn.Sequential(
            DoubleConv(50+3, 50, mode=0),
            DoubleConv(50, 50, mode=0))
        self.initialH4 = nn.Sequential(
            DoubleConv(50+3, 50, mode=1),
            DoubleConv(50, 50, mode=1))
        self.initialH5 = nn.Sequential(
            DoubleConv(50+3, 50, mode=2),
            DoubleConv(50, 50, mode=2))
        self.finalH3 = DoubleConv(150, 50, mode=0)
        self.finalH4 = DoubleConv(150, 50, mode=1)
        self.finalH5 = DoubleConv(150, 50, mode=2)
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        mid = torch.cat((h1, h2, h3), 1)
        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)
        mid2 = torch.cat((h4, h5, h6), 1)
        out = self.finalH(mid2)
        # out_noise = gaussian(out.data, 0, 0.1)
        return out
