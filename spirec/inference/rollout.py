from collections import deque
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from spirec.configs.config import Config
from spirec.core.geometry import GeometryOps
from spirec.core.utils import logger


class VolumeReconstructor:
    """
    Execution engine for long-horizon 2.5D autoregressive volume rollouts.
    Supports Dense U-Nets and Paradigm B Spiking Networks.
    """

    def __init__(self, model: nn.Module, config: Config, device: Optional[torch.device] = None):
        self.cfg = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _predict_single_slice(
        self,
        ctx_stack_np: np.ndarray,
        target_shape: tuple,
        brain_mask_slice: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Executes single slice inference with aspect-ratio-preserving letterboxing.
        
        Args:
            ctx_stack_np: (N, H_orig, W_orig) NumPy array.
            target_shape: (H_orig, W_orig).
        """
        # Convert to tensor: (1, N, H, W)
        ctx_tensor = torch.from_numpy(ctx_stack_np).unsqueeze(0).to(self.device)

        # Aspect-ratio preserving padding to uniform model input size (160, 160)
        ctx_pad, scale, pads = GeometryOps.resize_and_pad(
            ctx_tensor, self.cfg.data.padded_size, mode="bicubic"
        )

        if self.cfg.model.architecture == "baseline_unet":
            if brain_mask_slice is not None:
                bm_t = torch.from_numpy(brain_mask_slice).unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                bm_t = (ctx_tensor[:, -1:, :, :] > 0.01).float()
            bm_pad, _, _ = GeometryOps.resize_and_pad(bm_t, self.cfg.data.padded_size, mode="nearest")
            ctx_pad = torch.cat([ctx_pad, bm_pad], dim=1)

        # Forward execution
        if self.cfg.model.architecture == "spike_unet":
            pred_pad, _ = self.model(ctx_pad)
        else:
            pred_pad = self.model(ctx_pad)

        # Invert letterboxing back to native geometry
        pred_restored = GeometryOps.inverse_resize_pad(
            pred_pad, target_shape, scale, pads, mode="bicubic"
        )
        return torch.clamp(pred_restored, 0.0, 1.0).squeeze().cpu().numpy()

    def autoregressive_restore(
        self,
        volume: np.ndarray,
        start_idx: int,
        end_idx: int,
        direction: str = "forward",
        mask_volume: Optional[np.ndarray] = None,
        disable_pbar: bool = False,
    ) -> np.ndarray:
        """
        Synthesizes a contiguous 3D longitudinal sequence via sequential autoregression.
        
        Args:
            volume:      (Z, H, W) input volume in [0.0, 1.0].
            start_idx:   Starting slice index.
            end_idx:     Ending slice index.
            direction:   'forward' (0 -> Z) or 'backward' (Z -> 0).
            mask_volume: Optional (Z, H, W) pathology void mask for teacher-forced inpainting.
        """
        recon_vol = volume.copy()
        n_ctx = self.cfg.data.neighborhood
        total_slices = volume.shape[0]
        h_orig, w_orig = volume.shape[1], volume.shape[2]

        if direction == "forward":
            iter_range = list(range(start_idx, end_idx))
            initial_context = [volume[i] for i in range(start_idx - n_ctx, start_idx)]
            recon_vol[:start_idx] = volume[:start_idx]
        else:
            iter_range = list(range(end_idx - 1, start_idx - 1, -1))
            initial_context = [volume[i] for i in range(end_idx, end_idx + n_ctx)]
            initial_context.reverse()
            recon_vol[end_idx:] = volume[end_idx:]

        buffer = deque(initial_context, maxlen=n_ctx)

        pbar = tqdm(
            iter_range,
            desc=f"AR {direction.capitalize()} Rollout ({self.cfg.model.name})",
            leave=False,
            disable=disable_pbar,
        )

        for z in pbar:
            # Context stack: shape (N, H, W)
            ctx_stack = np.stack(list(buffer), axis=0)

            # Inpainting mode: zero out context within the target pathology mask
            if mask_volume is not None:
                ctx_indices = (
                    [z - n_ctx + k for k in range(n_ctx)]
                    if direction == "forward"
                    else [z + n_ctx - k for k in range(n_ctx)]
                )
                for buf_idx, slice_idx in enumerate(ctx_indices):
                    if 0 <= slice_idx < total_slices:
                        void_mask = (mask_volume[slice_idx] > 0).astype(np.float32)
                        ctx_stack[buf_idx] *= 1.0 - void_mask

            bm_slice = (volume[z] > 0.01).astype(np.float32)
            pred_slice = self._predict_single_slice(ctx_stack, (h_orig, w_orig), bm_slice)

            # Teacher Forcing: blend prediction strictly into pathological void if provided
            if mask_volume is not None:
                void_target = (mask_volume[z] > 0).astype(np.float32)
                final_slice = (pred_slice * void_target) + (volume[z] * (1.0 - void_target))
            else:
                # Pure Autoregression (Drift exposure mode)
                final_slice = pred_slice

            recon_vol[z] = final_slice
            buffer.append(final_slice)

        return recon_vol

