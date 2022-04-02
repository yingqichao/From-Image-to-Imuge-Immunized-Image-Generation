import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            # nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            # nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            # nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, inchannel=1, outchannel=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Globalpool = nn.MaxPool2d(kernel_size=4, stride=4)

        self.Conv1 = DoubleConv(inchannel, 16)
        self.Conv2 = DoubleConv(16, 32)
        self.Conv3 = DoubleConv(32, 64)
#        self.Conv4 = DoubleConv(64, 128)
#        self.Conv5 = DoubleConv(128, 256)

#        self.Up5 = up_conv(256, 128)
#        self.Conv6 = DoubleConv(256, 128)

        self.Up4 = up_conv(128, 64)
        self.Conv7 = DoubleConv(128, 64)

        self.Up3 = up_conv(64, 32)
        self.Conv8 = DoubleConv(64, 32)

        self.Up2 = up_conv(32, 16)
        self.Conv9 = DoubleConv(32, 16)

        self.Conv_1x1 = nn.Conv2d(16, outchannel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
#        x4 = self.Conv4(x4)

#        x5 = self.Maxpool(x4)
#        x5 = self.Conv5(x5)

        x6 = self.Globalpool(x4)
        x7 = x6.repeat(1,1,4,4)
        x4 = torch.cat((x4, x7), dim=1)


#        d5 = self.Up5(x5)
#        d5 = torch.cat((x4, d5), dim=1)
#        d5 = self.Conv6(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Conv7(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Conv8(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Conv9(d2)

        out = self.Conv_1x1(d2)

        return out
















