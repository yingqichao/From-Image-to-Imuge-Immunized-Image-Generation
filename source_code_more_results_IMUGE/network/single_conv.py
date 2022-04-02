import torch.nn as nn

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, dilation):
        super().__init__()
        # if kernel_size==3:
        #     padding_size = 1
        # elif kernel_size==5:
        #     padding_size = 2
        # elif kernel_size==7:
        #     padding_size = 3
        # else:
        #     kernel_size, padding_size = 9, 4

        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding,stride=stride,dilation=dilation),
            nn.BatchNorm2d(out_channels),
            # nn.InstanceNorm2d(out_channels),
            # nn.PReLU()
            nn.ELU(inplace=True)
            # nn.LeakyReLU(0.2, True)
        )

    def forward(self, x):
        return self.single_conv(x)

