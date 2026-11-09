from typing import Optional, Tuple

import torch
import torch.nn as nn

from spirec.models.layers.plif import PLIF
from spirec.models.layers.alif import ALIF


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
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
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
        self.unit1 = SpikingConvUnit(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            surrogate_alpha=surrogate_alpha,
        )
        self.unit2 = SpikingConvUnit(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            surrogate_alpha=surrogate_alpha,
        )

    def forward(self, s_seq: torch.Tensor) -> torch.Tensor:
        residual = s_seq
        out = self.unit1(s_seq)
        out = self.unit2(out)
        return residual + out


class SpikingConvUnitALIF(nn.Module):
    """Time-distributed 2D Spiking Convolutional Unit with ALIF dynamics."""

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
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.alif = ALIF(out_channels, surrogate_alpha=surrogate_alpha)

    def forward(
        self,
        s_seq: torch.Tensor,
        u_init: Optional[torch.Tensor] = None,
        w_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t, b, c, h, w = s_seq.shape
        s_flat = s_seq.reshape(t * b, c, h, w)
        curr_flat = self.bn(self.conv(s_flat))
        curr_seq = curr_flat.reshape(
            t, b, -1, h // self.conv.stride[0], w // self.conv.stride[1]
        )
        return self.alif(curr_seq, u_init=u_init, w_init=w_init)


class MembraneShortcutResBlock(nn.Module):
    """Membrane-Shortcut Residual Block (MS-ResBlock).

    Eliminates integer spike addition (Sin + Sout in {0, 1, 2}).
    Shortcut summation occurs in the continuous synaptic current space.
    """

    def __init__(self, channels: int, surrogate_alpha: float = 2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.alif1 = ALIF(channels, surrogate_alpha=surrogate_alpha)

        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.alif2 = ALIF(channels, surrogate_alpha=surrogate_alpha)

    def forward(self, s_seq: torch.Tensor) -> torch.Tensor:
        t, b, c, h, w = s_seq.shape

        # Branch 1
        s_flat = s_seq.reshape(t * b, c, h, w)
        i1_flat = self.bn1(self.conv1(s_flat))
        s1_seq, _, _ = self.alif1(i1_flat.reshape(t, b, c, h, w))

        # Branch 2 with direct linear current integration of input spikes
        s1_flat = s1_seq.reshape(t * b, c, h, w)
        i2_flat = self.bn2(self.conv2(s1_flat))

        # Inject residual shortcut current from identity spikes
        i_total = (i2_flat + s_flat).reshape(t, b, c, h, w)
        s_out, _, _ = self.alif2(i_total)
        return s_out
