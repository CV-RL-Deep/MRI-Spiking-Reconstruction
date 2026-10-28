import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock2D(nn.Module):
    """Standard non-spiking (Conv2d -> BatchNorm2d -> GELU) block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DenseUNet2p5D(nn.Module):
    """
    Standard dense 2.5D U-Net serving as experimental baseline.
    Modes:
      - Vanilla:  Input is (B, N, H, W)
      - Baseline: Input is (B, N+1, H, W) where channel N is the target mask
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_filters: int = 32):
        super().__init__()
        self.in_channels = in_channels

        # Encoder
        self.enc1 = ConvBlock2D(in_channels, base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = ConvBlock2D(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = ConvBlock2D(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = ConvBlock2D(base_filters * 4, base_filters * 8)

        # Decoder
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock2D(base_filters * 8 + base_filters * 4, base_filters * 4)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock2D(base_filters * 4 + base_filters * 2, base_filters * 2)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock2D(base_filters * 2 + base_filters, base_filters)

        # Output projection
        self.final_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck(p3)

        # Decoder (Enforce contiguous memory layout before convolution)
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1).contiguous())
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1).contiguous())
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1).contiguous())

        # Output bounded to [0.0, 1.0]
        return torch.sigmoid(self.final_conv(d1))

