import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=3, base_features=64):
        """
        :param z_dim (int): Dimension of the noise vector.
        :param channels (int): Number of output channels (typically 3 for RGB).
        :param base_features (int): Number of features in the first convolution layer (subsequent layers double this value).
        """

        super(Generator, self).__init__()

        self.generator = nn.Sequential(
            # Input: N x z_dim x 1 x 1
            nn.ConvTranspose2d(z_dim, base_features * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(base_features * 16),
            nn.ReLU(True),

            # State: N x base_features*16 x 4 x 4
            nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 8),
            nn.ReLU(True),

            # State: N x base_features*8 x 8 x 8
            nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 4),
            nn.ReLU(True),

            # State: N x base_features*4 x 16 x 16
            nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features * 2),
            nn.ReLU(True),

            # State: N x base_features*2 x 32 x 32
            nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_features),
            nn.ReLU(True),

            # State: N x base_features x 64 x 64
            nn.ConvTranspose2d(base_features, channels, kernel_size=4, stride=2, padding=1, bias=False),

            # Output: N x channels x 128 x 128
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)


