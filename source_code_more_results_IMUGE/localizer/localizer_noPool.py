import torch
import torch.nn as nn
from  config import GlobalConfig
from network.double_conv import DoubleConv


class LocalizeNetwork_noPool(nn.Module):
    def __init__(self, config=GlobalConfig()):
        super(LocalizeNetwork_noPool, self).__init__()
        self.config = config
        # Level 5
        self.hiding_1_1 = nn.Sequential(
            DoubleConv(3, 256),
            DoubleConv(256, 256),
        )

        # Level 3
        self.invertLevel3_1 = nn.Sequential(
            DoubleConv(256, 128),
            DoubleConv(128, 128),
        )
        # Level 2
        self.invertLevel2_1 = nn.Sequential(
            DoubleConv(128, 64),
            DoubleConv(64, 64),
        )

        # Level 1
        self.invertLevel1_1 = nn.Sequential(
            DoubleConv(64, 3),
            DoubleConv(3, 3),
        )

        self.final = nn.Conv2d(3, 1, kernel_size=1, padding=0)
        # self.final = nn.Conv2d(120, 3, kernel_size=1, padding=0)
        # self.final = DoubleConv(120, 3, disable_last_activate=True)

    def forward(self, p):
        # Level 5
        hiding = self.hiding_1_1(p)
        # hiding_2 = self.hiding_1_2(p)

        # Level 3
        il3 = self.invertLevel3_1(hiding)

        # Level 2
        #il3_cat = torch.cat([il3, hiding_2], dim=1)
        il2 = self.invertLevel2_1(il3)

        # Level 1
        #il2_cat = torch.cat([il2, il3], dim=1)
        il1 = self.invertLevel1_1(il2)

        #il1_cat = torch.cat([il1, p], dim=1)
        out = self.final(il1)
        out = out.squeeze()

        return out