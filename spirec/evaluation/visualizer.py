import os
import random

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from spirec.core.utils import logger


class VisualizationSuite:
    """Diagnostic visualization utilities for slice synthesis and error accumulation."""

    @staticmethod
    def plot_ortho_slices(
        volume: np.ndarray, title: str = "Tri-Planar Orthogonal View"
    ) -> None:
        """Plots Axial, Coronal, and Sagittal cross-sections from volume center."""
        c_z, c_y, c_x = np.array(volume.shape) // 2

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(title, fontsize=12)

        axes[0].imshow(volume[c_z, :, :], cmap="bone", aspect="equal")
        axes[0].set_title(f"Axial (Z={c_z})")
        axes[1].imshow(volume[:, c_y, :], cmap="bone", aspect="equal")
        axes[1].set_title(f"Coronal (Y={c_y})")
        axes[2].imshow(volume[:, :, c_x], cmap="bone", aspect="equal")
        axes[2].set_title(f"Sagittal (X={c_x})")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_autoregressive_comparison(
        gt_vol: np.ndarray,
        pred_vol: np.ndarray,
        k_indices: List[int],
        title: str = "Autoregressive Progression",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plots multi-step progression comparing Ground Truth against Model Rollout and Error Map.
        """
        n_steps = len(k_indices)
        fig, axes = plt.subplots(3, n_steps, figsize=(3.0 * n_steps, 7.5))
        fig.suptitle(title, fontsize=12, weight="bold")

        for i, idx in enumerate(k_indices):
            gt = gt_vol[idx]
            pr = pred_vol[idx]
            diff = np.abs(gt - pr)

            axes[0, i].imshow(gt, cmap="bone", vmin=0, vmax=1)
            axes[0, i].set_title(f"GT Slice {idx}")
            axes[0, i].axis("off")

            axes[1, i].imshow(pr, cmap="bone", vmin=0, vmax=1)
            axes[1, i].set_title(f"Pred Slice {idx}")
            axes[1, i].axis("off")

            axes[2, i].imshow(diff, cmap="inferno", vmin=0, vmax=0.3)
            mse = float(np.mean(diff**2))
            axes[2, i].set_title(f"MSE: {mse:.4f}")
            axes[2, i].axis("off")

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Figure saved to: {save_path}")
        else:
            plt.show()

    @staticmethod
    def sample_synthetic_glioma_mask(
        target_volume: np.ndarray, max_radius: int = 18
    ) -> np.ndarray:
        """Creates a smooth 3D Gaussian ellipsoid simulating a brain tumor void."""
        mask = np.zeros_like(target_volume, dtype=np.float32)
        coords = np.argwhere(target_volume > 0.1)
        if len(coords) == 0:
            return mask

        cz, cy, cx = coords[random.randint(0, len(coords) - 1)]
        rz = random.uniform(max_radius * 0.4, max_radius * 0.8)
        ry = random.uniform(max_radius * 0.6, max_radius * 1.0)
        rx = random.uniform(max_radius * 0.6, max_radius * 1.0)

        z, y, x = np.ogrid[
            : target_volume.shape[0],
            : target_volume.shape[1],
            : target_volume.shape[2],
        ]
        dist = (
            ((z - cz) ** 2 / rz**2)
            + ((y - cy) ** 2 / ry**2)
            + ((x - cx) ** 2 / rx**2)
        )
        blob = np.exp(-dist)

        mask[blob > 0.3] = 1.0
        brain_matter = (target_volume > 0.01).astype(np.float32)
        return mask * brain_matter
