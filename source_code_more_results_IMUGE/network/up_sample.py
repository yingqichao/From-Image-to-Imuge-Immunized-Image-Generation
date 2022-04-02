import torch.nn as nn
from network.double_conv import DoubleConv

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                DoubleConv(in_channels, out_channels, in_channels)
            )
        else:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.up_conv(x)