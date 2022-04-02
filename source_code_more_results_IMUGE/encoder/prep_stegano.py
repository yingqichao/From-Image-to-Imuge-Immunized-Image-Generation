import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class PrepStegano(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(PrepStegano, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:64 """
        self.downsample_1 = nn.Sequential(
            SingleConv(3, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
        )
        # 64
        self.downsample_2 = nn.Sequential(
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )

        # 64
        self.downsample_3 = nn.Sequential(
            # SingleConv(64, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(128, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 32
        self.downsample_4 = nn.Sequential(
            # SingleConv(192, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(192, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 16
        self.downsample_5 = nn.Sequential(
            # SingleConv(256, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(256, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 32
        self.upsample4 = nn.Sequential(
            # SingleConv(320, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(256, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 64
        self.upsample3 = nn.Sequential(
            # SingleConv(192, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(256, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 64
        self.upsample2 = nn.Sequential(
            # SingleConv(448, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(256, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # 256
        self.upsample1 = nn.Sequential(
            # SingleConv(512, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
            SingleConv(256, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
            SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        )

        # self.upsample0 = nn.Sequential(
        #     # SingleConv(512, out_channels=64, kernel_size=1, stride=1, dilation=1, padding=0),
        #     SingleConv(320, out_channels=64, kernel_size=5, stride=1, dilation=2, padding=4),
        #     SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=4, padding=8),
        #     SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=8, padding=16),
        #     SingleConv(64, out_channels=64, kernel_size=5, stride=1, dilation=1, padding=2)
        # )

        self.finalH1 = nn.Sequential(
            SingleConv(256, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
            SingleConv(64, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.Conv2d(64, 3, kernel_size=1, padding=0),
            # nn.Tanh()
        )


    def forward(self, p):
        # Features with Kernel Size 7
        down8 = self.downsample_1(p)
        down7 = self.downsample_2(down8)
        down7_cat = torch.cat((down8, down7), 1)
        down6 = self.downsample_3(down7_cat)
        down6_cat = torch.cat((down8, down7, down6), 1)
        down5 = self.downsample_4(down6_cat)
        down5_cat = torch.cat((down8, down7, down6, down5), 1)
        down4 = self.downsample_5(down5_cat)
        down4_cat = torch.cat((down7, down6, down5, down4), 1)
        up4 = self.upsample4(down4_cat)
        up4_cat = torch.cat((down6, down5, down4, up4), 1)
        up3 = self.upsample3(up4_cat)
        up3_cat = torch.cat((down5, down4, up4, up3), 1)
        up2 = self.upsample2(up3_cat)
        up2_cat = torch.cat((down4, up4, up3, up2), 1)
        up1 = self.upsample1(up2_cat)
        up1_cat = torch.cat((up4, up3, up2, up1), 1)
        up0 = self.finalH1(up1_cat)
        # out = p + up0
        return up0
