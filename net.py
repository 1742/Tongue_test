from dataset import *
import torch.nn as nn
import torch
import warnings

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                    nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=1),
                                    nn.ReLU(inplace=True))
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

            # nn.Conv2d(in_channels,in_channels//2,1,0),报错？
        self.conv = DoubleConv(in_channels, out_channels)
        # self.up.apply(self.init_weights)##???

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffy = torch.tensor(x2.size()[2] - x1.size()[2])
        diffx = torch.tensor(x2.size()[3] - x1.size()[3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x1 = nn.functional.pad(x1, (diffx // 2, diffx - diffx // 2,
                                        diffy // 2, diffy - diffy // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    # @staticmethod
    # def init_weights(m):
    # if type(m) == nn.Conv2d:
    # init.xavier_normal(m.weight)
    # init.constant(m.bias,0)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(unet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.drop3 = nn.Dropout2d(0.5)
        self.down4 = Down(512, 1024)
        self.drop4 = nn.Dropout2d(0.5)
        self.up1 = Up(1024, 512, True)
        self.up2 = Up(512, 256, True)
        self.up3 = Up(256, 128, True)
        self.up4 = Up(128, 64, True)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x4 = self.drop3(x4)
        x5 = self.down4(x4)
        x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        # x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    net = unet(3, 1)
    net.eval()
    print(net)
    image = torch.randn(1, 3, 224, 224)
    pred = net(image)
    print("input:", image.shape)
    print("output:", pred.shape)