import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from spirec.models.layers.surrogate import surrogate_atan


class ALIF(nn.Module):
    r"""Adaptive Leaky Integrate-and-Fire (ALIF) Spiking Neuron.

    Operates on 5D tensors: (T_sim, Batch, Channels, Height, Width).
    Implements dynamic threshold accommodation:
        U[\tau] = \beta U[\tau-1] + I[\tau]
        V_{th}[\tau] = V_0 + \gamma w[\tau]
        S[\tau] = \Theta(U[\tau] - V_{th}[\tau])
        U[\tau] \leftarrow U[\tau] - S[\tau] V_{th}[\tau]
        w[\tau] = \rho w[\tau-1] + S[\tau-1]
    """

    def __init__(
        self,
        channels: int,
        v_0: float = 1.0,
        initial_beta: float = 0.8,
        initial_rho: float = 0.85,
        initial_gamma: float = 0.5,
        surrogate_alpha: float = 2.0,
    ):
        super().__init__()
        self.v_0 = v_0
        self.surrogate_alpha = surrogate_alpha

        # Channel-wise parameters parameterized in unconstrained space
        raw_beta = math.log(initial_beta / (1.0 - initial_beta))
        raw_rho = math.log(initial_rho / (1.0 - initial_rho))
        raw_gamma = math.log(math.exp(initial_gamma) - 1.0)

        self.w_beta = nn.Parameter(
            torch.full((1, channels, 1, 1), raw_beta, dtype=torch.float32)
        )
        self.w_rho = nn.Parameter(
            torch.full((1, channels, 1, 1), raw_rho, dtype=torch.float32)
        )
        self.w_gamma = nn.Parameter(
            torch.full((1, channels, 1, 1), raw_gamma, dtype=torch.float32)
        )

    def forward(
        self,
        input_current_seq: torch.Tensor,
        u_init: Optional[torch.Tensor] = None,
        w_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t_sim, b, c, h, w = input_current_seq.shape

        beta = torch.sigmoid(self.w_beta)
        rho = torch.sigmoid(self.w_rho)
        gamma = F.softplus(self.w_gamma)

        u = (
            torch.zeros(
                (b, c, h, w),
                device=input_current_seq.device,
                dtype=input_current_seq.dtype,
            )
            if u_init is None
            else u_init
        )
        w_adapt = (
            torch.zeros(
                (b, c, h, w),
                device=input_current_seq.device,
                dtype=input_current_seq.dtype,
            )
            if w_init is None
            else w_init
        )

        spikes = []
        for tau in range(t_sim):
            current = input_current_seq[tau]
            u = beta * u + current
            v_th = self.v_0 + gamma * w_adapt
            s = surrogate_atan(u - v_th, alpha=self.surrogate_alpha)

            # Soft reset-by-subtraction preserving residual hyperpolarization
            u = u - s * v_th
            w_adapt = rho * w_adapt + s
            spikes.append(s)

        return torch.stack(spikes, dim=0), u, w_adapt
