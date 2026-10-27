import math

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spirec.models.layers.surrogate import surrogate_atan


class PLIF(nn.Module):
    """
    Parametric Leaky Integrate-and-Fire (PLIF) Spiking Neuron.
    Operates on 5D tensors: (T_sim, Batch, Channels, Height, Width).
    Supports learnable membrane decay beta and soft reset-by-subtraction.
    """

    def __init__(
        self,
        channels: int,
        v_threshold: float = 1.0,
        initial_beta: float = 0.8,
        reset_mode: str = "subtract",
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.v_th = v_threshold
        self.reset_mode = reset_mode
        self.surrogate_alpha = surrogate_alpha

        # 4D parameterization matches single-step (B, C, H, W) state tensor
        raw_beta = math.log(initial_beta / (1.0 - initial_beta))
        self.w_beta = nn.Parameter(
            torch.full((1, channels, 1, 1), raw_beta, dtype=torch.float32)
        )

    def forward(
        self,
        input_current_seq: torch.Tensor,
        u_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_sim, b, c, h, w = input_current_seq.shape
        beta = torch.sigmoid(self.w_beta)  # Shape: (1, C, 1, 1)

        u = (
            torch.zeros(
                (b, c, h, w),
                device=input_current_seq.device,
                dtype=input_current_seq.dtype,
            )
            if u_init is None
            else u_init
        )
        spikes = []

        for tau in range(t_sim):
            current = input_current_seq[tau]  # (B, C, H, W)
            u = beta * u + current  # (B, C, H, W)
            s = surrogate_atan(u - self.v_th, alpha=self.surrogate_alpha)

            if self.reset_mode == "subtract":
                u = u - s * self.v_th
            elif self.reset_mode == "zero":
                u = u * (1.0 - s)
            else:
                raise ValueError(f"Unknown reset_mode: {self.reset_mode}")

            spikes.append(s)

        # Output shape strictly preserved: (T_sim, B, C, H, W)
        return torch.stack(spikes, dim=0), u
