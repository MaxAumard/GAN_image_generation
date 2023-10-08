import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels=3, base_features=64):
        """
        :param channels (int): Number of input channels (typically 3 for RGB).
        :param base_features (int): Number of features in the first convolution layer (subsequent layers double this value).
        """

        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            # Input: N x channels x 128 x 128
            nn.Conv2d(channels, base_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # State: N x base_features x 64 x 64
            nn.Conv2d(base_features, base_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # State: N x base_features*2 x 32 x 32
            nn.Conv2d(base_features * 2, base_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # State: N x base_features*4 x 16 x 16
            nn.Conv2d(base_features * 4, base_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # State: N x base_features*8 x 8 x 8
            nn.Conv2d(base_features * 8, 1, kernel_size=4, stride=2, padding=0, bias=False),
            nn.AdaptiveAvgPool2d(1),

            # Output: N x 1 x 1 x 1
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x).view(-1, 1)

