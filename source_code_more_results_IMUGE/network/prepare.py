# %matplotlib inline
import torch
import torch.nn as nn
from network.double_conv import DoubleConv

# Preparation Network (2 conv layers)
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            DoubleConv(3, 50, mode=0),
            DoubleConv(50, 50, mode=0))
        self.initialP4 = nn.Sequential(
            DoubleConv(3, 50, mode=1),
            DoubleConv(50, 50, mode=1))
        self.initialP5 = nn.Sequential(
            DoubleConv(3, 50, mode=2),
            DoubleConv(50, 50, mode=2))
        self.finalP3 = DoubleConv(150, 50, mode=0)
        self.finalP4 = DoubleConv(150, 50, mode=1)
        self.finalP5 = DoubleConv(150, 50, mode=2)

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
        mid = torch.cat((p1, p2, p3), 1)
        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)
        out = torch.cat((p4, p5, p6), 1)
        return out