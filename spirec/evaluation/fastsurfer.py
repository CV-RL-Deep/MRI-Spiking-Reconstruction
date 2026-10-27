import glob
import os
import subprocess

from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from spirec.configs.config import Config
from spirec.core.utils import logger


class FastSurferEvaluator:
    """
    Downstream subcortical anatomical validation using FastSurfer.
    FastSurfer segments T1 volumes into FreeSurfer DKT anatomical classes.
    """

    def __init__(self, config: Config):
        self.results_dir = config.data.results_dir
        self.nifti_dir = os.path.join(self.results_dir, "nifti")
        self.seg_dir = os.path.join(self.results_dir, "fastsurfer_masks")

        os.makedirs(self.nifti_dir, exist_ok=True)
        os.makedirs(self.seg_dir, exist_ok=True)

        # FreeSurfer DKT Atlas Macro-Region Label Mapping
        self.macro_regions = {
            "Ventricles": [4, 43],  # Left and Right Lateral Ventricles
            "Deep_GM": [
                10,
                49,
                11,
                50,
                12,
                51,
                13,
                52,
            ],  # Thalamus, Caudate, Putamen, Pallidum
            "White_Matter": [2, 41],  # Cerebral White Matter
            "Cortex": list(range(1000, 1036)) + list(range(2000, 2036)),
        }

    def setup(self) -> None:
        """Clones FastSurfer repository if not locally present."""
        if not os.path.exists("FastSurfer"):
            logger.info("Cloning FastSurfer repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/Deep-MI/FastSurfer.git"],
                check=True,
            )
        else:
            logger.info("FastSurfer repository verified.")

    def convert_to_nifti(self) -> None:
        """
        Converts (.npy) volume reconstructions to canonical (.nii.gz) files.
        Restores (Z, Y, X) -> (X, Y, Z) and scales intensity range to uint8 [0, 255].
        """
        npy_files = sorted(glob.glob(os.path.join(self.results_dir, "*.npy")))
        if not npy_files:
            logger.warning(
                f"No .npy files found in {self.results_dir} for NIfTI conversion."
            )
            return

        for f in tqdm(npy_files, desc="Converting to NIfTI"):
            out_name = os.path.basename(f).replace(".npy", ".nii.gz")
            out_path = os.path.join(self.nifti_dir, out_name)
            if os.path.exists(out_path):
                continue

            vol = np.load(f)  # Shape: (Z, Y, X) in [0.0, 1.0]

            # 1. Transpose to canonical RAS+ (X, Y, Z)
            vol_xyz = np.transpose(vol, (2, 1, 0))

            # 2. Rescale intensity to uint8 [0, 255] for FastSurfer conform module
            vol_scaled = np.clip(vol_xyz * 255.0, 0, 255).astype(np.uint8)

            nii = nib.Nifti1Image(vol_scaled, np.eye(4))
            nib.save(nii, out_path)

    def run_prediction(self) -> None:
        """Executes FastSurfer neural segmentation on generated NIfTI volumes."""
        nifti_files = sorted(
            glob.glob(os.path.join(self.nifti_dir, "*.nii.gz"))
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = (
            os.path.abspath("FastSurfer") + ":" + env.get("PYTHONPATH", "")
        )
        env["FASTSURFER_HOME"] = os.path.abspath("FastSurfer")

        # Dummy license bypass for seg_only mode
        license_path = os.path.abspath("dummy_fs_license.txt")
        if not os.path.exists(license_path):
            with open(license_path, "w") as stream:
                stream.write("research@academic.edu\n12345\n12345\n12345\n")

        for f in tqdm(nifti_files, desc="FastSurfer Segmentation"):
            sid = os.path.basename(f).replace(".nii.gz", "")
            out_file = os.path.join(
                self.seg_dir, sid, "mri", "aparc.DKTatlas+aseg.deep.mgz"
            )

            if os.path.exists(out_file):
                continue

            cmd = [
                "bash",
                "FastSurfer/run_fastsurfer.sh",
                "--allow_root",
                "--t1",
                os.path.abspath(f),
                "--seg_only",
                "--sd",
                os.path.abspath(self.seg_dir),
                "--sid",
                sid,
                "--device",
                "cuda" if torch.cuda.is_available() else "cpu",
                "--batch",
                "16",
                "--threads",
                "4",
            ]

            result = subprocess.run(
                cmd, env=env, capture_output=True, text=True
            )
            if result.returncode != 0:
                logger.error(f"FastSurfer failed for {sid}:\n{result.stderr}")

    @staticmethod
    def _dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
        intersection = np.sum(mask1 & mask2)
        total = np.sum(mask1) + np.sum(mask2)
        if total == 0:
            return 1.0
        return float(2.0 * intersection / total)

    def calculate_dsc(self, ablations: List[str]) -> pd.DataFrame:
        """Computes Dice Similarity Coefficients across Ground Truth and model predictions."""
        gt_sids = [
            os.path.basename(f).replace(".nii.gz", "")
            for f in glob.glob(os.path.join(self.nifti_dir, "gt_*.nii.gz"))
        ]

        results = []

        for gt_sid in tqdm(gt_sids, desc="Calculating Anatomical DSC"):
            vol_id = gt_sid.replace("gt_", "")
            gt_path = os.path.join(
                self.seg_dir, gt_sid, "mri", "aparc.DKTatlas+aseg.deep.mgz"
            )
            if not os.path.exists(gt_path):
                continue

            gt_seg = nib.load(gt_path).get_fdata()

            for ab in ablations:
                pred_sid = f"{ab}_{vol_id}"
                pred_path = os.path.join(
                    self.seg_dir,
                    pred_sid,
                    "mri",
                    "aparc.DKTatlas+aseg.deep.mgz",
                )
                if not os.path.exists(pred_path):
                    continue

                pred_seg = nib.load(pred_path).get_fdata()

                for region_name, labels in self.macro_regions.items():
                    gt_mask = np.isin(gt_seg, labels)
                    pred_mask = np.isin(pred_seg, labels)
                    dsc_val = self._dice_coefficient(gt_mask, pred_mask)

                    results.append(
                        {
                            "Volume_ID": vol_id,
                            "Ablation": ab,
                            "Region": region_name.replace("_", " "),
                            "DSC": dsc_val,
                        }
                    )

        df = pd.DataFrame(results)
        csv_path = os.path.join(self.results_dir, "anatomical_metrics.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Anatomical DSC metrics saved to: {csv_path}")
        return df
