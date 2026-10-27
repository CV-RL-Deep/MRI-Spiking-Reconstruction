import random

from typing import Dict, List, Tuple

import numpy as np

from scipy.ndimage import gaussian_filter, rotate, zoom

from spirec.configs.config import AugmentationConfig


class AugmentationLogic:
    """NumPy/SciPy spatial transformations and Markovian noise/void generators."""

    @staticmethod
    def apply_geometric(
        input_stack: np.ndarray,
        target_slice: np.ndarray,
        target_masks: List[np.ndarray],
        config: AugmentationConfig,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """
        Applies identical random horizontal flip and 2D rotation to context and target tensors.
        input_stack:  (N, H, W)
        target_slice: (H, W)
        target_masks: list of (H, W) masks
        """
        # Horizontal flip (along axis W, index 2 for stack, index 1 for 2D slices)
        if random.random() < config.prob_flip:
            input_stack = np.flip(input_stack, axis=2)
            target_slice = np.flip(target_slice, axis=1)
            target_masks = [np.flip(m, axis=1) for m in target_masks]

        # In-plane rotation
        if random.random() < config.prob_rotate:
            angle = random.uniform(-config.rotate_range, config.rotate_range)

            # Rotate input stack (N, H, W) on spatial axes (1, 2)
            input_stack = rotate(
                input_stack,
                angle,
                axes=(1, 2),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )

            # Rotate target slice (H, W) on axes (0, 1)
            target_slice = rotate(
                target_slice,
                angle,
                axes=(0, 1),
                reshape=False,
                order=1,
                mode="constant",
                cval=0.0,
            )

            # Rotate masks using nearest neighbor (order=0) to preserve integer labels
            target_masks = [
                rotate(
                    m,
                    angle,
                    axes=(0, 1),
                    reshape=False,
                    order=0,
                    mode="constant",
                    cval=0.0,
                )
                for m in target_masks
            ]

        return input_stack, target_slice, target_masks

    @staticmethod
    def apply_autoregressive_corruption(
        input_stack: np.ndarray, config: AugmentationConfig
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Applies cascaded Gaussian blur and additive noise backwards from t-1 to simulate drift.
        """
        stack = input_stack.copy()
        flags = {"blur_applied": False, "noise_applied": False}
        n_slices = stack.shape[0]

        if config.prob_blur <= 0.0 and config.prob_noise <= 0.0:
            return stack, flags

        for i in range(n_slices - 1, -1, -1):
            slice_data = stack[i]

            if config.prob_blur > 0.0 and random.random() < config.prob_blur:
                slice_data = gaussian_filter(
                    slice_data, sigma=config.blur_sigma
                )
                flags["blur_applied"] = True

            if config.prob_noise > 0.0 and random.random() < config.prob_noise:
                tissue_mask = (slice_data > 0.01).astype(np.float32)
                noise = np.random.normal(0, config.noise_std, slice_data.shape)
                slice_data = slice_data + (noise * tissue_mask)
                flags["noise_applied"] = True

            stack[i] = np.clip(slice_data, 0.0, 1.0)

        return stack, flags

    @staticmethod
    def apply_tumor_void_direct(
        input_stack: np.ndarray,
        seg_stack: np.ndarray,
        config: AugmentationConfig,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Injects an (N+1, H_seg, W_seg) BraTS tumor mask into the (N, H, W) input stack.
        Returns:
            modified_input_stack: (N, H, W) with tumor region zeroed out
            target_tumor_mask:    (H, W) binary mask of the target slice void
            applied_flag:         True if injection met tissue overlap criteria
        """
        n_ctx, h_in, w_in = input_stack.shape
        if seg_stack.shape[0] != n_ctx + 1:
            return input_stack, np.zeros((h_in, w_in), dtype=np.float32), False

        if not np.any(np.isin(seg_stack, config.brats_labels_for_void)):
            return input_stack, np.zeros((h_in, w_in), dtype=np.float32), False

        # Spatial interpolation to match input resolution
        scale_h = h_in / seg_stack.shape[1]
        scale_w = w_in / seg_stack.shape[2]
        seg_resized = zoom(seg_stack, (1, scale_h, scale_w), order=0)

        # Safety shape adjustment
        if seg_resized.shape[1:] != (h_in, w_in):
            padded = np.zeros((n_ctx + 1, h_in, w_in), dtype=seg_resized.dtype)
            min_h = min(seg_resized.shape[1], h_in)
            min_w = min(seg_resized.shape[2], w_in)
            padded[:, :min_h, :min_w] = seg_resized[:, :min_h, :min_w]
            seg_resized = padded

        input_seg = seg_resized[:n_ctx]
        target_seg = seg_resized[n_ctx]

        target_tumor_mask = np.isin(
            target_seg, config.brats_labels_for_void
        ).astype(np.float32)
        if not np.any(target_tumor_mask):
            return input_stack, np.zeros((h_in, w_in), dtype=np.float32), False

        # Overlap validation: tumor must intersect actual brain matter
        is_tumor_input = np.isin(input_seg, config.brats_labels_for_void)
        brain_matter = input_stack > 0.01
        overlap = is_tumor_input & brain_matter

        if np.sum(overlap) < 10:
            return input_stack, np.zeros((h_in, w_in), dtype=np.float32), False

        output_stack = input_stack.copy()
        output_stack[is_tumor_input] = 0.0
        return output_stack, target_tumor_mask, True
