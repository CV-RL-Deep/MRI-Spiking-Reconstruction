import glob
import os

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.models.video as video_models

from scipy import linalg
from tqdm import tqdm

from spirec.configs.config import Config
from spirec.core.utils import logger


class FrechetEvaluator:
    """
    Computes 3D-FID (structural realism) and FVD (longitudinal Z-axis continuity).
    Utilizes 3D ResNet-18 (R3D-18) and Multiscale Vision Transformers (MViT-v1).
    """

    def __init__(self, config: Config, device: torch.device = None):
        self.results_dir = config.data.results_dir
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        logger.info(f"Initializing 3D Feature Extractors on {self.device}...")

        # 1. FVD Extractor: MViT-v1-B trained on Kinetics-400
        try:
            self.mvit = video_models.mvit_v1_b(
                weights=video_models.MViT_V1_B_Weights.DEFAULT
            )
            self.mvit.head = torch.nn.Identity()
            self.mvit.eval().to(self.device)
        except Exception as e:
            logger.warning(
                f"Could not download MViT weights: {e}. FVD will use uninitialized extractor."
            )
            self.mvit = video_models.mvit_v1_b()
            self.mvit.head = torch.nn.Identity()
            self.mvit.eval().to(self.device)

        # 2. 3D-FID Extractor: 3D ResNet-18 (R3D-18)
        try:
            self.r3d = video_models.r3d_18(
                weights=video_models.R3D_18_Weights.DEFAULT
            )
            self.r3d.fc = torch.nn.Identity()
            self.r3d.eval().to(self.device)
        except Exception as e:
            logger.warning(
                f"Could not download R3D weights: {e}. 3D-FID will use uninitialized extractor."
            )
            self.r3d = video_models.r3d_18()
            self.r3d.fc = torch.nn.Identity()
            self.r3d.eval().to(self.device)

    @torch.no_grad()
    def _extract_sliding_features(
        self,
        volume: np.ndarray,
        model: torch.nn.Module,
        target_spatial_size: Tuple[int, int] = (112, 112),
        chunk_depth: int = 16,
        stride: int = 8,
    ) -> List[np.ndarray]:
        """
        Extracts 3D feature embeddings over overlapping 16-slice volumetric chunks.
        """
        z_len, h, w = volume.shape
        features_list = []

        for z_start in range(0, max(1, z_len - chunk_depth + 1), stride):
            vol_chunk = volume[z_start : z_start + chunk_depth]

            # Edge pad if smaller than chunk depth
            if vol_chunk.shape[0] < chunk_depth:
                pad_z = chunk_depth - vol_chunk.shape[0]
                vol_chunk = np.pad(
                    vol_chunk, ((0, pad_z), (0, 0), (0, 0)), mode="edge"
                )

            # Skip pure background noise
            if np.mean(vol_chunk) < 0.01:
                continue

            # Format for Video Models: (Batch=1, Channels=3, Depth/Time=16, Height, Width)
            chunk_t = (
                torch.from_numpy(vol_chunk)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, 3, 1, 1, 1)
            )

            chunk_resized = torch.nn.functional.interpolate(
                chunk_t,
                size=(
                    chunk_depth,
                    target_spatial_size[0],
                    target_spatial_size[1],
                ),
                mode="trilinear",
            )

            # Normalize to standard vision statistics
            chunk_norm = (chunk_resized - 0.45) / 0.225
            feat = model(chunk_norm.to(self.device))
            features_list.append(feat.squeeze().cpu().numpy())

        return features_list

    @staticmethod
    def _calculate_frechet_distance(
        mu1: np.ndarray,
        sigma1: np.ndarray,
        mu2: np.ndarray,
        sigma2: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """
        Computes Fréchet Distance: d^2 = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2)).
        Universal implementation compatible across all SciPy versions (< 1.16 and >= 1.18).
        """
        mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
        sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)

        diff = mu1 - mu2

        # Compute matrix square root with cross-version dispatch
        try:
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
        except TypeError:
            covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

        if isinstance(covmean, tuple):
            covmean = covmean[0]

        # Numerical instability handling for near-singular matrices
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            try:
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
            except TypeError:
                covmean = linalg.sqrtm(
                    (sigma1 + offset).dot(sigma2 + offset), disp=False
                )
            if isinstance(covmean, tuple):
                covmean = covmean[0]

        if np.iscomplexobj(covmean):
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        return float(
            diff.dot(diff)
            + np.trace(sigma1)
            + np.trace(sigma2)
            - 2 * tr_covmean
        )

    def evaluate(self, ablations: List[str]) -> pd.DataFrame:
        """Executes full 3D-FID and FVD benchmark across all model ablation volumes."""
        gt_files = sorted(glob.glob(os.path.join(self.results_dir, "gt_*.npy")))
        if not gt_files:
            logger.error(
                f"No Ground Truth files (gt_*.npy) found in {self.results_dir}."
            )
            return pd.DataFrame()

        feats_r3d = {k: [] for k in ["gt"] + ablations}
        feats_mvit = {k: [] for k in ["gt"] + ablations}

        for gt_path in tqdm(gt_files, desc="Extracting 3D Features"):
            vol_id = (
                os.path.basename(gt_path).replace("gt_", "").replace(".npy", "")
            )
            gt_vol = np.load(gt_path)

            feats_r3d["gt"].extend(
                self._extract_sliding_features(
                    gt_vol, self.r3d, target_spatial_size=(112, 112)
                )
            )
            feats_mvit["gt"].extend(
                self._extract_sliding_features(
                    gt_vol, self.mvit, target_spatial_size=(224, 224)
                )
            )

            for ab in ablations:
                pred_path = os.path.join(self.results_dir, f"{ab}_{vol_id}.npy")
                if not os.path.exists(pred_path):
                    continue
                pred_vol = np.load(pred_path)
                feats_r3d[ab].extend(
                    self._extract_sliding_features(
                        pred_vol, self.r3d, target_spatial_size=(112, 112)
                    )
                )
                feats_mvit[ab].extend(
                    self._extract_sliding_features(
                        pred_vol, self.mvit, target_spatial_size=(224, 224)
                    )
                )

        results = []
        mu_gt_r3d, sig_gt_r3d = (
            np.mean(feats_r3d["gt"], axis=0),
            np.cov(feats_r3d["gt"], rowvar=False),
        )
        mu_gt_mvit, sig_gt_mvit = (
            np.mean(feats_mvit["gt"], axis=0),
            np.cov(feats_mvit["gt"], rowvar=False),
        )

        for ab in ablations:
            if len(feats_r3d[ab]) == 0:
                continue

            mu_ab_r3d, sig_ab_r3d = (
                np.mean(feats_r3d[ab], axis=0),
                np.cov(feats_r3d[ab], rowvar=False),
            )
            mu_ab_mvit, sig_ab_mvit = (
                np.mean(feats_mvit[ab], axis=0),
                np.cov(feats_mvit[ab], rowvar=False),
            )

            fid_3d = self._calculate_frechet_distance(
                mu_gt_r3d, sig_gt_r3d, mu_ab_r3d, sig_ab_r3d
            )
            fvd = self._calculate_frechet_distance(
                mu_gt_mvit, sig_gt_mvit, mu_ab_mvit, sig_ab_mvit
            )

            results.append({"Ablation": ab, "3D-FID": fid_3d, "FVD": fvd})

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "frechet_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Fréchet metrics saved to: {csv_path}")
        return df
