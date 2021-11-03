import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """ conv --> BN --> relu x 2 """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """ Downscaling with maxpool then double conv """

    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        if dropout:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                nn.Dropout(0.3),
                DoubleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """ upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear use normal upsampling to reduce numbr of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='nearest')  # , align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        # pad x1 and x2 to same size
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        # padding starts with last dimension and then goes forward
        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffY // 2, diffY - diffY // 2, diffX // 2, diffX - diffX // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(). __init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, min_featuremaps=64, depth=5, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_channels = n_classes
        self.bilinear = bilinear

        """
        example for depth = 5
        self.inc = DoubleConv(n_channels, min_featuremaps)
        self.down1 = Down(min_featuremaps, min_featuremaps*2)
        self.down2 = Down(min_featuremaps*2, min_featuremaps*4)
        self.down3 = Down(min_featuremaps*4, min_featuremaps*8)
        self.down4 = Down(min_featuremaps*8, min_featuremaps * 16)
        self.up1 = Up(min_featuremaps*24, min_featuremaps*8, bilinear)
        self.up2 = Up(min_featuremaps*12, min_featuremaps*4, bilinear)
        self.up3 = Up(min_featuremaps*6, min_featuremaps*2, bilinear)
        self.up4 = Up(min_featuremaps*3, min_featuremaps, bilinear)
        self.outc = OutConv(min_featuremaps, n_classes)
        """
        # determine sizes for input and output of downward path
        in_sizes_down = [1, 2, 4, 8, 16, 32][:depth - 1]
        out_sizes_down = [2, 4, 8, 16, 32, 64][:depth - 1]

        # determine sizes for upward path
        in_sizes_up = [
            x1 + x2 for (x1, x2) in zip(reversed(in_sizes_down), reversed(out_sizes_down))]
        out_sizes_up = in_sizes_down[::-1]

        # Build model
        self.inc = DoubleConv(n_channels, min_featuremaps)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i in range(depth - 1):
            self.downs.append(
                Down(int(min_featuremaps * in_sizes_down[i]), int(min_featuremaps * out_sizes_down[i]), dropout=False))
            self.ups.append(
                Up(int(min_featuremaps * in_sizes_up[i]),
                    int(min_featuremaps * out_sizes_up[i])))

        self.outc = OutConv(min_featuremaps, n_classes)

    def forward(self, x):
        """
        example for depth = 5 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

        """

        x1 = self.inc(x)
        x_list = [x1]

        # Go through downsampling pathway
        for layer in self.downs:
            x_list.append(layer(x_list[-1]))

        # compute bottleneck layer
        x = self.ups[0](x_list[-1], x_list[-2])

        # computer upsampling pathway
        for i, layer in enumerate(self.ups[1:]):
            x = layer(x, x_list[-i - 3])

        logits = self.outc(x)
        return logits
