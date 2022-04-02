import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, mode=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if mode==0:
            kernel_size_1, padding_size_1 = 3, 1
            kernel_size_2, padding_size_2 = 3, 1
        elif mode==1:
            kernel_size_1, padding_size_1 = 4, 1
            kernel_size_2, padding_size_2 = 4, 2
        elif mode==2:
            kernel_size_1, padding_size_1 = 5, 2
            kernel_size_2, padding_size_2 = 5, 2
        elif mode==3:
            kernel_size_1, padding_size_1 = 7, 3
            kernel_size_2, padding_size_2 = 7, 3
        else:
            kernel_size_1, padding_size_1 = 9, 4
            kernel_size_2, padding_size_2 = 9, 4

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=kernel_size_1, padding=padding_size_1),
            nn.BatchNorm2d(mid_channels),
            # nn.InstanceNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.ELU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=kernel_size_2, padding=padding_size_2),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            # nn.ReLU(inplace=True)
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

