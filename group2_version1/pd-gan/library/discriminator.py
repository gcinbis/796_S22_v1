from torch import nn


class Discriminator(nn.Module):
    """ Discriminator class for PD-GAN """

    def __init__(self, batch_size):
        """
        Initialize the Discriminator for PDGAN model architecture.
        """
        super().__init__()
        self.batch_size = batch_size

        "Convolution layers"
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, stride=1, padding=1)

        "Batch Normalization Layers"
        self.bn1 = nn.LayerNorm([64, 64, 64])
        self.bn2 = nn.LayerNorm([128, 32, 32])
        self.bn3 = nn.LayerNorm([256, 16, 16])

        "ReLU Activations"
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.lrelu2 = nn.LeakyReLU(0.1)
        self.lrelu3 = nn.LeakyReLU(0.1)
        self.lrelu4 = nn.LeakyReLU(0.1)

        "Sigmoid Activations"
        self.sigmoid1 = nn.Sigmoid()

    def forward(self, img):
        """ Given an image and generated real_or_fake_score"""

        out = self.conv1(img.clone().detach())
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.bn1(out)
        out = self.lrelu2(out)

        out = self.conv3(out)
        out = self.bn2(out)
        out = self.lrelu3(out)

        out = self.conv4(out)
        out = self.bn3(out)
        out = self.lrelu4(out)

        out = self.conv5(out)

        out = self.sigmoid1(out)

        return out
