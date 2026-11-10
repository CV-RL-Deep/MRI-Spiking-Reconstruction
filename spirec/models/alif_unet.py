from typing import Dict, Tuple

import torch
import torch.nn as nn

from spirec.models.layers.alif import ALIF
from spirec.models.layers.spiking_blocks import (
    MembraneShortcutResBlock,
    SpikingConvUnitALIF,
)


class SpikeALIFARUNet(nn.Module):
    """2.5D Spiking Autoregressive U-Net with Adaptive LIF Neurons (Branch A)."""

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        t_sim: int = 4,
        v_0: float = 1.0,
        initial_beta: float = 0.8,
        initial_rho: float = 0.85,
        initial_gamma: float = 0.5,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.t_sim = t_sim
        self.base_filters = base_filters

        # Analog current drive
        self.input_encoder = nn.Conv2d(
            in_channels, base_filters, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(base_filters)
        self.input_alif = ALIF(
            base_filters,
            v_0=v_0,
            initial_beta=initial_beta,
            initial_rho=initial_rho,
            initial_gamma=initial_gamma,
            surrogate_alpha=surrogate_alpha,
        )

        # Encoder stages
        self.enc1 = MembraneShortcutResBlock(
            base_filters, surrogate_alpha=surrogate_alpha
        )
        self.down1 = SpikingConvUnitALIF(
            base_filters,
            base_filters * 2,
            stride=2,
            surrogate_alpha=surrogate_alpha,
        )

        self.enc2 = MembraneShortcutResBlock(
            base_filters * 2, surrogate_alpha=surrogate_alpha
        )
        self.down2 = SpikingConvUnitALIF(
            base_filters * 2,
            base_filters * 4,
            stride=2,
            surrogate_alpha=surrogate_alpha,
        )

        # Bottleneck
        self.bottleneck = MembraneShortcutResBlock(
            base_filters * 4, surrogate_alpha=surrogate_alpha
        )

        # Decoder stages
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = SpikingConvUnitALIF(
            base_filters * 4 + base_filters * 2,
            base_filters * 2,
            surrogate_alpha=surrogate_alpha,
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = SpikingConvUnitALIF(
            base_filters * 2 + base_filters,
            base_filters,
            surrogate_alpha=surrogate_alpha,
        )

        # Non-spiking leaky analog readout
        self.readout_conv = nn.Conv2d(base_filters, 1, kernel_size=3, padding=1)
        self.readout_beta = nn.Parameter(
            torch.tensor([0.7], dtype=torch.float32)
        )

    def forward(
        self, ctx_stack: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b, _, h, w = ctx_stack.shape
        t = self.t_sim

        i_drive = self.input_bn(self.input_encoder(ctx_stack))
        i_seq = i_drive.unsqueeze(0).repeat(t, 1, 1, 1, 1)
        s_in, _, _ = self.input_alif(i_seq)

        s_e1 = self.enc1(s_in)
        s_d1, _, _ = self.down1(s_e1)

        s_e2 = self.enc2(s_d1)
        s_d2, _, _ = self.down2(s_e2)

        s_bot = self.bottleneck(s_d2)

        # Decoder Stage 2
        s_bot_flat = s_bot.reshape(
            t * b, s_bot.shape[2], s_bot.shape[3], s_bot.shape[4]
        )
        up2_seq = self.up2(s_bot_flat).reshape(
            t, b, -1, s_e2.shape[3], s_e2.shape[4]
        )
        s_dec2, _, _ = self.dec2(torch.cat([up2_seq, s_e2], dim=2))

        # Decoder Stage 1
        s_dec2_flat = s_dec2.reshape(
            t * b, s_dec2.shape[2], s_dec2.shape[3], s_dec2.shape[4]
        )
        up1_seq = self.up1(s_dec2_flat).reshape(
            t, b, -1, s_e1.shape[3], s_e1.shape[4]
        )
        s_dec1, _, _ = self.dec1(torch.cat([up1_seq, s_e1], dim=2))

        # Leaky Analog Voltage Integration
        dec1_flat = s_dec1.reshape(t * b, s_dec1.shape[2], h, w)
        currents = self.readout_conv(dec1_flat).reshape(t, b, 1, h, w)

        u_out = torch.zeros(
            (b, 1, h, w), device=ctx_stack.device, dtype=ctx_stack.dtype
        )
        beta_ro = torch.sigmoid(self.readout_beta)
        accumulated_slices = []

        for tau in range(t):
            u_out = beta_ro * u_out + currents[tau]
            accumulated_slices.append(torch.sigmoid(u_out))

        x_hat = torch.stack(accumulated_slices, dim=0).mean(dim=0)

        spike_tensors = [s_in, s_e1, s_d1, s_e2, s_d2, s_bot, s_dec2, s_dec1]
        mean_spike_rate = torch.stack(
            [torch.mean(s) for s in spike_tensors]
        ).mean()

        metrics = {
            "mean_spike_rate": mean_spike_rate,
            "synops_ratio": (mean_spike_rate * t).detach(),
        }
        return x_hat, metrics
