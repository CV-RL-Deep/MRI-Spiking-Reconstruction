from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import find_objects


class GeometryOps:
    """
    Pure PyTorch and NumPy spatial geometric operations for 2.5D slicing pipelines.
    Guarantees aspect-ratio-preserving letterboxing and exact inverse mapping.
    """

    @staticmethod
    def get_smart_crop_coords(
        volume_np: np.ndarray, target_spatial_shape: Tuple[int, int]
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Determines the bounding box centered around brain matter along projection axes.

        Args:
            volume_np: 3D NumPy array (Z, H, W).
            target_spatial_shape: (H_target, W_target).
        Returns:
            ((h_start, h_end), (w_start, w_end))
        """
        projection = np.max(volume_np, axis=0) > 0.01
        slices = find_objects(projection.astype(int))

        if not slices:
            h, w = volume_np.shape[1], volume_np.shape[2]
            ch, cw = h // 2, w // 2
        else:
            h_slice, w_slice = slices[0]
            ch = h_slice.start + (h_slice.stop - h_slice.start) // 2
            cw = w_slice.start + (w_slice.stop - w_slice.start) // 2

        th, tw = target_spatial_shape
        h_start = max(0, ch - th // 2)
        w_start = max(0, cw - tw // 2)

        max_h, max_w = volume_np.shape[1], volume_np.shape[2]
        if h_start + th > max_h:
            h_start = max(0, max_h - th)
        if w_start + tw > max_w:
            w_start = max(0, max_w - tw)

        return (int(h_start), int(h_start + th)), (
            int(w_start),
            int(w_start + tw),
        )

    @staticmethod
    def normalize_volume(tensor: torch.Tensor) -> torch.Tensor:
        """Min-Max tensor normalization bounded to [0.0, 1.0]."""
        tensor = tensor.float()
        t_min = torch.amin(tensor)
        t_max = torch.amax(tensor)
        scale = torch.clamp(t_max - t_min, min=1e-8)
        return (tensor - t_min) / scale

    @staticmethod
    def resize_and_pad(
        tensor: torch.Tensor,
        target_size: Tuple[int, int],
        mode: str = "bicubic",
    ) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Letterboxes tensor to target spatial resolution while preserving aspect ratio.

        Args:
            tensor: (C, H, W) or (B, C, H, W) tensor.
            target_size: (H_target, W_target).
            mode: Interpolation mode ('bicubic', 'bilinear', or 'nearest').
        Returns:
            padded_tensor, scale_factor, pad_offsets [pad_top, pad_left]
        """
        is_batched = tensor.dim() == 4
        if not is_batched:
            tensor = tensor.unsqueeze(0)

        _, _, h, w = tensor.shape
        target_h, target_w = target_size

        scale = min(target_h / h, target_w / w)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        align_corners = False if mode in ["bilinear", "bicubic"] else None
        resized = F.interpolate(
            tensor, size=(new_h, new_w), mode=mode, align_corners=align_corners
        )

        pad_h = target_h - new_h
        pad_w = target_w - new_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # F.pad format: (left, right, top, bottom)
        padded = F.pad(
            resized,
            (pad_left, pad_right, pad_top, pad_bottom),
            mode="constant",
            value=0.0,
        )

        offsets = torch.tensor([pad_top, pad_left], dtype=torch.long)
        if not is_batched:
            padded = padded.squeeze(0)

        return padded, scale, offsets

    @staticmethod
    def inverse_resize_pad(
        padded_tensor: torch.Tensor,
        original_shape: Tuple[int, int],
        scale: float,
        pads: torch.Tensor,
        mode: str = "bicubic",
    ) -> torch.Tensor:
        """
        Inverts letterbox transformation: crops padding and resamples to raw coordinates.

        Args:
            padded_tensor: (C, H_pad, W_pad) or (B, C, H_pad, W_pad).
            original_shape: (H_raw, W_raw).
            scale: Scaling factor used during forward padding.
            pads: Tensor containing [pad_top, pad_left].
            mode: Interpolation mode.
        """
        is_batched = padded_tensor.dim() == 4
        if not is_batched:
            padded_tensor = padded_tensor.unsqueeze(0)

        pad_top = int(pads[0].item())
        pad_left = int(pads[1].item())
        orig_h, orig_w = original_shape

        active_h = int(round(orig_h * scale))
        active_w = int(round(orig_w * scale))

        cropped = padded_tensor[
            :, :, pad_top : pad_top + active_h, pad_left : pad_left + active_w
        ]

        align_corners = False if mode in ["bilinear", "bicubic"] else None
        restored = F.interpolate(
            cropped,
            size=(orig_h, orig_w),
            mode=mode,
            align_corners=align_corners,
        )

        if not is_batched:
            restored = restored.squeeze(0)

        return restored
