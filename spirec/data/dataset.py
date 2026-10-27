import random

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from torch.utils.data import Dataset

from spirec.configs.config import Config
from spirec.core.geometry import GeometryOps
from spirec.data.augmentations import AugmentationLogic
from spirec.data.cache import DataLoaderBase, MedicalVolume


class AutoregressiveSliceDataset(Dataset):
    """
    Map-style dynamic sampling dataset for 2.5D autoregressive training.
    Produces random sequential slices on-the-fly from the in-RAM volume cache.
    """

    def __init__(
        self,
        data_manager: DataLoaderBase,
        config: Config,
        samples_per_epoch: int = 5000,
    ):
        self.manager = data_manager
        self.cfg = config
        self.length = samples_per_epoch
        self.w_ixi = config.data.ixi_sampling_weight

    def __len__(self) -> int:
        return self.length

    def _sample_slice_indices(
        self, vol: MedicalVolume, axis: int, n_ctx: int
    ) -> Optional[int]:
        candidates = vol.indices["clean"][axis]
        if len(candidates) == 0:
            return None

        axis_size = vol.shape[axis]
        valid_starts = []
        for s in range(axis_size - n_ctx):
            seq = set(range(s, s + n_ctx + 1))
            if not seq.intersection(
                set(vol.indices["tumor"][axis])
            ) and seq.intersection(set(vol.indices["any"][axis])):
                valid_starts.append(s)

        if not valid_starts:
            return None
        return random.choice(valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 1. Select collection
        col = "ixi" if random.random() < self.w_ixi else "brats"
        vol = self.manager.get_volume(col)

        # Fallback if selected pool is not yet populated
        if vol is None:
            col = "ixi" if col == "brats" else "brats"
            vol = self.manager.get_volume(col)

        if vol is None:
            # Yield blank safety item if cache is empty
            pad_h, pad_w = self.cfg.data.padded_size
            n_ctx = self.cfg.data.neighborhood
            return {
                "ctx_stack": torch.zeros(
                    n_ctx, pad_h, pad_w, dtype=torch.float32
                ),
                "target_slice": torch.zeros(
                    1, pad_h, pad_w, dtype=torch.float32
                ),
                "target_mask": torch.zeros(
                    1, pad_h, pad_w, dtype=torch.float32
                ),
                "tumor_mask": torch.zeros(1, pad_h, pad_w, dtype=torch.float32),
                "z_norm": torch.tensor([0.5], dtype=torch.float32),
            }

        axis = 0  # Standard Axial projection
        n_ctx = self.cfg.data.neighborhood
        start_idx = self._sample_slice_indices(vol, axis, n_ctx)

        if start_idx is None:
            start_idx = random.randint(0, max(0, vol.shape[axis] - n_ctx - 1))

        # Extract (N+1, H, W) raw stack
        stack = vol.t1[start_idx : start_idx + n_ctx + 1].copy()
        if stack.shape[0] != n_ctx + 1:
            stack = np.pad(
                stack,
                ((0, (n_ctx + 1) - stack.shape[0]), (0, 0), (0, 0)),
                mode="edge",
            )

        # Bidirectional slice sequence flip
        is_backward = random.random() < 0.5
        if is_backward:
            stack = np.flip(stack, axis=0)

        real_target_idx = start_idx if is_backward else (start_idx + n_ctx)
        axis_size = vol.shape[axis]
        z_norm = float(real_target_idx) / max(1.0, float(axis_size - 1))

        input_stack = stack[:-1]
        target_slice = stack[-1]

        target_tumor_mask = np.zeros_like(target_slice)
        brain_mask = (target_slice > 0.01).astype(np.float32)

        # Pathology void injection
        if random.random() < self.cfg.aug.prob_glioma:
            seg_stack = self.manager.get_matched_glioma_stack(
                axis, z_norm, n_ctx
            )
            if seg_stack is not None:
                if is_backward:
                    seg_stack = np.flip(seg_stack, axis=0)
                input_stack, target_tumor_mask, applied = (
                    AugmentationLogic.apply_tumor_void_direct(
                        input_stack, seg_stack, self.cfg.aug
                    )
                )

        # Geometric transformations
        masks = [target_tumor_mask, brain_mask]
        input_stack, target_slice, rotated_masks = (
            AugmentationLogic.apply_geometric(
                input_stack, target_slice, masks, self.cfg.aug
            )
        )
        target_tumor_mask, brain_mask = rotated_masks

        # Markovian noise corruption
        input_stack, _ = AugmentationLogic.apply_autoregressive_corruption(
            input_stack, self.cfg.aug
        )

        return {
            "ctx_stack_np": input_stack.astype(np.float32),  # (N, H, W)
            "target_slice_np": target_slice.astype(np.float32),  # (H, W)
            "brain_mask_np": brain_mask.astype(np.float32),  # (H, W)
            "tumor_mask_np": target_tumor_mask.astype(np.float32),  # (H, W)
            "z_norm": z_norm,
        }


class SequentialValidationDataset(Dataset):
    """
    Deterministic validation dataset extracting a fixed window of center slices.
    """

    def __init__(
        self,
        data_manager: DataLoaderBase,
        config: Config,
        slices_per_vol: int = 15,
    ):
        self.manager = data_manager
        self.cfg = config
        self.n_ctx = config.data.neighborhood
        self.items = []
        self._materialize_slice_indices(slices_per_vol)

    def _materialize_slice_indices(self, slices_per_vol: int) -> None:
        ixi_pool = self.manager.pool.get("ixi", [])
        brats_pool = self.manager.pool.get("brats", [])

        vols = [
            ("ixi", v)
            for v in (
                ixi_pool.values() if isinstance(ixi_pool, dict) else ixi_pool
            )
        ]
        vols += [
            ("brats", v)
            for v in (
                brats_pool.values()
                if isinstance(brats_pool, dict)
                else brats_pool
            )
        ]

        for col, vol in vols:
            axis = 0
            candidates = vol.indices["clean"][axis]
            if len(candidates) < self.n_ctx + 2:
                continue

            mid = len(candidates) // 2
            half = slices_per_vol // 2
            start_c = max(0, mid - half)
            end_c = min(len(candidates), mid + half)

            for target_idx in candidates[start_c:end_c]:
                s_idx = target_idx - self.n_ctx
                if s_idx < 0 or s_idx + self.n_ctx >= vol.shape[axis]:
                    continue

                # Ensure sequence does not cross pathology
                seq_set = set(range(s_idx, s_idx + self.n_ctx + 1))
                if seq_set.intersection(set(vol.indices["tumor"][axis])):
                    continue

                self.items.append((vol, s_idx, target_idx))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vol, s_idx, target_idx = self.items[idx]
        stack = vol.t1[s_idx : s_idx + self.n_ctx + 1]

        input_stack = stack[:-1].copy()
        target_slice = stack[-1].copy()

        brain_mask = (target_slice > 0.01).astype(np.float32)
        tumor_mask = np.zeros_like(target_slice)

        if vol.seg is not None:
            seg_target = vol.seg[target_idx]
            tumor_mask = np.isin(
                seg_target, self.cfg.aug.brats_labels_for_void
            ).astype(np.float32)

        z_norm = float(target_idx) / max(1.0, float(vol.shape[0] - 1))

        return {
            "ctx_stack_np": input_stack.astype(np.float32),
            "target_slice_np": target_slice.astype(np.float32),
            "brain_mask_np": brain_mask.astype(np.float32),
            "tumor_mask_np": tumor_mask.astype(np.float32),
            "z_norm": z_norm,
        }


def collate_autoregressive(
    batch: List[Dict[str, Any]],
    target_spatial_size: Tuple[int, int] = (160, 160),
) -> Dict[str, torch.Tensor]:
    """
    Letterboxes and pads NumPy arrays into uniform PyTorch batch tensors.
    """
    ctx_tensors = []
    tgt_tensors = []
    brain_masks = []
    tumor_masks = []
    z_positions = []

    for item in batch:
        # Convert NumPy arrays to PyTorch tensors
        # Context: (N, H, W)
        ctx_t = torch.from_numpy(item["ctx_stack_np"])
        # Targets: (1, H, W)
        tgt_t = torch.from_numpy(item["target_slice_np"]).unsqueeze(0)
        bm_t = torch.from_numpy(item["brain_mask_np"]).unsqueeze(0)
        tm_t = torch.from_numpy(item["tumor_mask_np"]).unsqueeze(0)

        # Apply aspect-ratio-preserving letterbox padding to (160, 160)
        ctx_pad, _, _ = GeometryOps.resize_and_pad(
            ctx_t, target_spatial_size, mode="bicubic"
        )
        tgt_pad, _, _ = GeometryOps.resize_and_pad(
            tgt_t, target_spatial_size, mode="bicubic"
        )
        bm_pad, _, _ = GeometryOps.resize_and_pad(
            bm_t, target_spatial_size, mode="nearest"
        )
        tm_pad, _, _ = GeometryOps.resize_and_pad(
            tm_t, target_spatial_size, mode="nearest"
        )

        ctx_tensors.append(ctx_pad)
        tgt_tensors.append(tgt_pad)
        brain_masks.append(bm_pad)
        tumor_masks.append(tm_pad)
        z_positions.append(torch.tensor([item["z_norm"]], dtype=torch.float32))

    return {
        "ctx_stack": torch.stack(ctx_tensors, dim=0),  # (B, N, H_pad, W_pad)
        "target_slice": torch.stack(tgt_tensors, dim=0),  # (B, 1, H_pad, W_pad)
        "brain_mask": torch.stack(brain_masks, dim=0),  # (B, 1, H_pad, W_pad)
        "tumor_mask": torch.stack(tumor_masks, dim=0),  # (B, 1, H_pad, W_pad)
        "z_norm": torch.stack(z_positions, dim=0),  # (B, 1)
    }
