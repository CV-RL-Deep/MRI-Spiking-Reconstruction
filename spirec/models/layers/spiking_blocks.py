import torch
import torch.nn as nn
from spirec.models.layers.plif import PLIF


class SpikingConvUnit(nn.Module):
    """
    Time-distributed 2D Spiking Convolutional Block: (Conv2d -> BatchNorm2d -> PLIF).
    Folds temporal dimension into batch to maximize cuDNN dense execution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.plif = PLIF(out_channels, surrogate_alpha=surrogate_alpha)

    def forward(self, s_seq: torch.Tensor) -> torch.Tensor:
        # Input shape: (T, B, C, H, W)
        t, b, c, h, w = s_seq.shape
        s_flat = s_seq.reshape(t * b, c, h, w)
        
        curr_flat = self.bn(self.conv(s_flat))
        _, c_out, h_out, w_out = curr_flat.shape
        curr_seq = curr_flat.reshape(t, b, c_out, h_out, w_out)
        
        spikes, _ = self.plif(curr_seq)
        return spikes


class SpikingResidualBlock(nn.Module):
    """
    Spiking Residual Block with identity skip connection.
    S_out = S_in + ConvBlock2(ConvBlock1(S_in))
    """

    def __init__(self, channels: int, surrogate_alpha: float = 2.0):
        super().__init__()
        self.unit1 = SpikingConvUnit(channels, channels, kernel_size=3, padding=1, surrogate_alpha=surrogate_alpha)
        self.unit2 = SpikingConvUnit(channels, channels, kernel_size=3, padding=1, surrogate_alpha=surrogate_alpha)

    def forward(self, s_seq: torch.Tensor) -> torch.Tensor:
        residual = s_seq
        out = self.unit1(s_seq)
        out = self.unit2(out)
        return residual + out

