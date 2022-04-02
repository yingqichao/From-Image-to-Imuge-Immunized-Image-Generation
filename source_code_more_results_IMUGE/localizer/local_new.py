import torch
import torch.nn as nn

from config import GlobalConfig
from network.conv_bn_relu import ConvBNRelu
from network.double_conv import DoubleConv
from network.single_conv import SingleConv
from network.pure_upsample import PureUpsampling
from network.single_de_conv import SingleDeConv

class Localize(nn.Module):
    def __init__(self,config=GlobalConfig()):
        super(Localize, self).__init__()
        self.config = config
        # input channel: 3, output channel: 96
        """Features with Kernel Size 7---->channel:128 """
        self.downsample_8 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, dilation=1, padding=1),
            nn.ELU(inplace=True)
        )
        # 128
        self.downsample_7 = SingleConv(64, out_channels=128, kernel_size=3, stride=2, dilation=1, padding=1)
        # self.Down1_conv_7 = SingleConv(64, out_channels=64, kernel_size=7, stride=1, dilation=1, padding=3)
        # 64
        self.downsample_6 = SingleConv(128, out_channels=256, kernel_size=3, stride=2, dilation=1, padding=1)
        # self.Down2_conv_7 = SingleConv(128, out_channels=128, kernel_size=7, stride=1, dilation=1, padding=3)
        # 32
        self.downsample_5 = SingleConv(256, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # 16
        self.downsample_4 = SingleConv(512, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # 16以下的卷积用4层conv
        self.fullConv = nn.Sequential(
            SingleConv(512, out_channels=512, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(512, out_channels=512, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(512, out_channels=512, kernel_size=5, stride=1, dilation=1, padding=2),
            SingleConv(512, out_channels=512, kernel_size=5, stride=1, dilation=1, padding=2)
        )
        # # 8
        # self.downsample_3 = SingleConv(512, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 4
        # self.downsample_2 = SingleConv(512, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 2
        # self.downsample_1 = SingleConv(512, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 1
        # self.downsample_0 = SingleConv(512, out_channels=512, kernel_size=3, stride=2, dilation=1, padding=1)
        # # 2
        # self.Up8 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample8_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.Up7 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 4
        # self.upsample7_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 8
        # self.Up6 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample6_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # # 16
        # self.Up5 = nn.Sequential(
        #     PureUpsampling(scale=2),
        #     SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.upsample5_3 = nn.Sequential(
        #     # PureUpsampling(scale=2),
        #     SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        # )
        # self.pureUpsamle = PureUpsampling(scale=2)
        # 32
        self.Up4 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(512, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.upsample4_3 = nn.Sequential(
            # PureUpsampling(scale=2),
            SingleConv(1024, out_channels=512, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 64
        self.Up3 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(512, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.upsample3_3 = nn.Sequential(
            # PureUpsampling(scale=2),
            SingleConv(512, out_channels=256, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 128
        self.Up2 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.upsample2_3 = nn.Sequential(
            # PureUpsampling(scale=2),
            SingleConv(256, out_channels=128, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        # 256
        self.Up1 = nn.Sequential(
            PureUpsampling(scale=2),
            SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        )
        self.upsample1_3 = nn.Sequential(
            # PureUpsampling(scale=2),
            SingleConv(128, out_channels=64, kernel_size=3, stride=1, dilation=1, padding=1)
        )

        self.final256 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            # nn.Tanh()
        )
        # self.finalH2 = nn.Sequential(
        #     nn.Conv2d(6, 3, kernel_size=1, padding=0),
        #     nn.Tanh()
        # )

    def forward(self, p):
        # 256
        down8 = self.downsample_8(p)
        # 128
        down7 = self.downsample_7(down8)
        # 64
        down6 = self.downsample_6(down7)
        # 32
        down5 = self.downsample_5(down6)
        # 16
        down4 = self.downsample_4(down5)
        up5 = self.fullConv(down4)
        # # 8
        # down3 = self.downsample_3(down4)
        # # 4
        # down2 = self.downsample_2(down3)
        # # 2
        # down1 = self.downsample_1(down2)
        # # 1
        # down0 = self.downsample_0(down1)
        # # 2
        # up8_up = self.Up8(down0)
        # up8_cat = torch.cat((down1, up8_up), 1)
        # up8 = self.upsample8_3(up8_cat)
        # # 4
        # up7_up = self.Up7(up8)
        # up7_cat = torch.cat((down2, up7_up), 1)
        # up7 = self.upsample7_3(up7_cat)
        # # 8
        # up6_up = self.Up6(up7)
        # up6_cat = torch.cat((down3, up6_up), 1)
        # up6 = self.upsample6_3(up6_cat)
        # # 16
        # up5_up = self.Up5(up6)
        # up5_cat = torch.cat((down4, up5_up), 1)
        # up5 = self.upsample5_3(up5_cat)
        # 32
        up4_up = self.Up4(up5)
        up4_cat = torch.cat((down5, up4_up), 1)
        up4 = self.upsample4_3(up4_cat)
        # 64
        up3_up = self.Up3(up4)
        up3_cat = torch.cat((down6, up3_up), 1)
        up3 = self.upsample3_3(up3_cat)
        # 128
        up2_up = self.Up2(up3)
        up2_cat = torch.cat((down7, up2_up), 1)
        up2 = self.upsample2_3(up2_cat)
        # 256
        up1_up = self.Up1(up2)
        up1_cat = torch.cat((down8, up1_up), 1)
        up1 = self.upsample1_3(up1_cat)
        up0 = self.final256(up1)
        return up0
