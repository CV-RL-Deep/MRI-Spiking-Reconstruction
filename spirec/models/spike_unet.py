from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from spirec.models.layers.plif import PLIF
from spirec.models.layers.spiking_blocks import (
    SpikingConvUnit,
    SpikingResidualBlock,
)


class SpikeARUNet(nn.Module):
    """
    2.5D Spiking Autoregressive U-Net executing under Paradigm B.
    Input:  Context stack C_z in R^(B, N, H, W).
    Decodes via Non-Spiking Analog Leaky Membrane Integration over T_sim steps.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        t_sim: int = 4,
        v_threshold: float = 1.0,
        initial_beta: float = 0.8,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.t_sim = t_sim
        self.base_filters = base_filters

        # 1. Direct Analog Current Drive
        self.input_encoder = nn.Conv2d(
            in_channels, base_filters, kernel_size=3, padding=1, bias=False
        )
        self.input_bn = nn.BatchNorm2d(base_filters)
        self.input_plif = PLIF(
            base_filters,
            v_threshold=v_threshold,
            initial_beta=initial_beta,
            surrogate_alpha=surrogate_alpha,
        )

        # 2. Encoder Pipeline
        self.enc1 = SpikingResidualBlock(
            base_filters, surrogate_alpha=surrogate_alpha
        )
        self.down1 = SpikingConvUnit(
            base_filters,
            base_filters * 2,
            stride=2,
            surrogate_alpha=surrogate_alpha,
        )

        self.enc2 = SpikingResidualBlock(
            base_filters * 2, surrogate_alpha=surrogate_alpha
        )
        self.down2 = SpikingConvUnit(
            base_filters * 2,
            base_filters * 4,
            stride=2,
            surrogate_alpha=surrogate_alpha,
        )

        # 3. Bottleneck
        self.bottleneck = SpikingResidualBlock(
            base_filters * 4, surrogate_alpha=surrogate_alpha
        )

        # 4. Decoder Pipeline
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = SpikingConvUnit(
            base_filters * 4 + base_filters * 2,
            base_filters * 2,
            surrogate_alpha=surrogate_alpha,
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = SpikingConvUnit(
            base_filters * 2 + base_filters,
            base_filters,
            surrogate_alpha=surrogate_alpha,
        )

        # 5. Non-Spiking Analog Readout
        self.readout_conv = nn.Conv2d(base_filters, 1, kernel_size=3, padding=1)
        self.readout_beta = nn.Parameter(
            torch.tensor([0.7], dtype=torch.float32)
        )

        # Layer tracking references for firing rate statistics
        self.spiking_layers = [
            self.input_plif,
            self.enc1.unit1.plif,
            self.enc1.unit2.plif,
            self.down1.plif,
            self.enc2.unit1.plif,
            self.enc2.unit2.plif,
            self.down2.plif,
            self.bottleneck.unit1.plif,
            self.bottleneck.unit2.plif,
            self.dec2.plif,
            self.dec1.plif,
        ]

    def forward(
        self,
        ctx_stack: torch.Tensor,
        u_init: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        b, _, h, w = ctx_stack.shape

        # A. Analog Drive Injection
        i_drive = self.input_bn(self.input_encoder(ctx_stack))
        # Replicate input current across micro-time tau in {1, ..., T_sim}
        i_seq = i_drive.unsqueeze(0).repeat(self.t_sim, 1, 1, 1, 1)
        s_in, _ = self.input_plif(i_seq)

        # B. Encoder Flow
        s_e1 = self.enc1(s_in)
        s_d1 = self.down1(s_e1)

        s_e2 = self.enc2(s_d1)
        s_d2 = self.down2(s_e2)

        # C. Bottleneck Flow
        s_bot = self.bottleneck(s_d2)

        # D. Decoder Flow with Nearest-Neighbor Upsampling
        t = self.t_sim

        # Stage 2 Up
        s_bot_flat = s_bot.reshape(
            t * b, s_bot.shape[2], s_bot.shape[3], s_bot.shape[4]
        )
        up2_flat = self.up2(s_bot_flat)
        up2_seq = up2_flat.reshape(
            t, b, up2_flat.shape[1], up2_flat.shape[2], up2_flat.shape[3]
        )
        s_dec2 = self.dec2(torch.cat([up2_seq, s_e2], dim=2))

        # Stage 1 Up
        s_dec2_flat = s_dec2.reshape(
            t * b, s_dec2.shape[2], s_dec2.shape[3], s_dec2.shape[4]
        )
        up1_flat = self.up1(s_dec2_flat)
        up1_seq = up1_flat.reshape(
            t, b, up1_flat.shape[1], up1_flat.shape[2], up1_flat.shape[3]
        )
        s_dec1 = self.dec1(torch.cat([up1_seq, s_e1], dim=2))

        # E. Non-Spiking Analog Membrane Voltage Integration
        dec1_flat = s_dec1.reshape(t * b, s_dec1.shape[2], h, w)
        currents = self.readout_conv(dec1_flat).reshape(t, b, 1, h, w)

        u_out = torch.zeros(
            (b, 1, h, w), device=ctx_stack.device, dtype=ctx_stack.dtype
        )
        beta_ro = torch.sigmoid(self.readout_beta)
        accumulated_slices = []

        for tau in range(self.t_sim):
            u_out = beta_ro * u_out + currents[tau]
            accumulated_slices.append(torch.sigmoid(u_out))

        x_hat = torch.stack(accumulated_slices, dim=0).mean(dim=0)

        # F. Differentiable Firing Rate Statistics Tracking
        spike_tensors = [s_in, s_e1, s_d1, s_e2, s_d2, s_bot, s_dec2, s_dec1]
        layer_means = [torch.mean(s) for s in spike_tensors]
        mean_spike_rate = torch.stack(layer_means).mean()

        metrics = {
            "mean_spike_rate": mean_spike_rate,
            "synops_ratio": (mean_spike_rate * self.t_sim).detach().item(),
        }

        return x_hat, metrics
