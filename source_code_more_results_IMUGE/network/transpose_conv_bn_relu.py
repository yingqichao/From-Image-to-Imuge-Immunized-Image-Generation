import torch.nn as nn


class TransposeConvBNRelu(nn.Module):
    """
     Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1):
        super(TransposeConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(channels_out),
            # nn.InstanceNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

        # nn.ConvTranspose2d(channels_in, channels_out, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        return self.layers(x)
