#!/usr/bin/env python3
import copy
import os
import random
import sys

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

# Force parent directory (repository root) into Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spirec.configs.config import Config, CFG
from spirec.core.utils import PipelineTimer, logger, set_global_seeds
from spirec.data.cache import ActiveLoader, StaticLoader
from spirec.data.dataset import (
    AutoregressiveSliceDataset,
    SequentialValidationDataset,
    collate_autoregressive,
)
from spirec.data.files import find_t1_files, get_brats_subjects
from spirec.evaluation.frechet import FrechetEvaluator
from spirec.evaluation.metrics import calculate_step_metrics
from spirec.inference.rollout import VolumeReconstructor
from spirec.models import ModelBuilder
from spirec.training.trainer import Trainer


def configure_ablation_matrix(base_cfg: Config) -> Dict[str, Config]:
    """
    Configures the experimental variants:
      1. vanilla_unet: Dense U-Net, unmasked spatial L1 loss (unbounded drift control).
      2. baseline_unet: Dense U-Net, target shape mask concatenated, spatial L1 loss (blur control).
      3. spike_unet: Spiking U-Net under Paradigm B (T_sim = 4), unmasked loss.
      4. spike_unet_fast: Spiking U-Net under Paradigm B (T_sim = 2), unmasked loss.
    """
    matrix = {}

    # Variant 1: Dense Vanilla U-Net
    cfg_vanilla = copy.deepcopy(base_cfg)
    cfg_vanilla.model.name = "vanilla_unet"
    cfg_vanilla.model.architecture = "vanilla_unet"
    cfg_vanilla.data.input_last_target_mask = False
    matrix["vanilla_unet"] = cfg_vanilla

    # Variant 2: Dense Baseline U-Net
    cfg_baseline = copy.deepcopy(base_cfg)
    cfg_baseline.model.name = "baseline_unet"
    cfg_baseline.model.architecture = "baseline_unet"
    cfg_baseline.data.input_last_target_mask = True
    matrix["baseline_unet"] = cfg_baseline

    # Variant 3: Spike-AR-UNet (T_sim = 4)
    cfg_spike = copy.deepcopy(base_cfg)
    cfg_spike.model.name = "spike_unet_t4"
    cfg_spike.model.architecture = "spike_unet"
    cfg_spike.data.input_last_target_mask = False
    cfg_spike.snn.t_sim = 4
    matrix["spike_unet_t4"] = cfg_spike

    # Variant 4: Spike-AR-UNet (T_sim = 2)
    cfg_spike_fast = copy.deepcopy(base_cfg)
    cfg_spike_fast.model.name = "spike_unet_t2"
    cfg_spike_fast.model.architecture = "spike_unet"
    cfg_spike_fast.data.input_last_target_mask = False
    cfg_spike_fast.snn.t_sim = 2
    matrix["spike_unet_t2"] = cfg_spike_fast

    return matrix


def run_ablation_pipeline(args):
    set_global_seeds(args.seed, deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Launching Ablation Pipeline on device: {device}")

    # 1. Dataset Indexing and Train/Validation Splits
    logger.info("Scanning dataset paths...")
    ixi_files = find_t1_files(CFG.data.data_root_ixi)
    brats_subjects = get_brats_subjects(CFG.data.data_root_brats)

    if not ixi_files:
        logger.warning(f"No IXI files found in '{CFG.data.data_root_ixi}'. Generating synthetic volumes for smoke testing.")
        os.makedirs(CFG.data.data_root_ixi, exist_ok=True)
        # Create minimal synthetic data for smoke test execution
        ixi_files = [os.path.join(CFG.data.data_root_ixi, f"synthetic_{i:03d}.nii.gz") for i in range(10)]

    random.shuffle(ixi_files)
    split_idx = int(0.9 * len(ixi_files))
    train_ixi, val_ixi = ixi_files[:split_idx], ixi_files[split_idx:]

    random.shuffle(brats_subjects)
    brats_split = int(0.9 * len(brats_subjects))
    train_brats, val_brats = brats_subjects[:brats_split], brats_subjects[brats_split:]

    # 2. Evaluation Pool Selection
    n_eval = min(len(val_ixi), args.num_eval_vols)
    eval_files = val_ixi[:n_eval]
    logger.info(f"Locked {len(eval_files)} validation volumes for comparative rollouts.")

    # 3. Setup Persistent Static Validation Pool
    val_loader_mem = StaticLoader(CFG, val_ixi[:15], val_brats[:15])
    val_dataset = SequentialValidationDataset(val_loader_mem, CFG, slices_per_vol=10)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_autoregressive(b, target_spatial_size=CFG.data.padded_size),
    )

    # 4. Generate Configurations
    experiments = configure_ablation_matrix(CFG)
    all_rollout_metrics = []

    # 5. Execute Training and Longitudinal Rollouts
    for ab_name, config in experiments.items():
        logger.info(f"\n{'=' * 60}\n   RUNNING EXPERIMENTAL PHASE: {ab_name.upper()}\n{'=' * 60}")
        weights_path = os.path.join(config.data.weights_dir, f"{config.model.name}_best.pt")

        # Train Active Manager
        with ActiveLoader(config, train_ixi, train_brats) as train_loader_mem:
            model = ModelBuilder.build(config).to(device)

            if os.path.exists(weights_path) and not args.force_retrain:
                logger.info(f"Loading existing checkpoint: {weights_path}")
                checkpoint = torch.load(weights_path, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                logger.info(f"Training {ab_name} from scratch for {args.epochs} epochs...")
                train_dataset = AutoregressiveSliceDataset(
                    train_loader_mem, config, samples_per_epoch=args.samples_per_epoch
                )
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=lambda b: collate_autoregressive(b, target_spatial_size=config.data.padded_size),
                )

                trainer = Trainer(config, model, train_dataloader, val_dataloader, device=device)
                with PipelineTimer(f"Training [{ab_name}]"):
                    trainer.fit(num_epochs=args.epochs)

                # Reload best weights
                if os.path.exists(weights_path):
                    checkpoint = torch.load(weights_path, map_location=device)
                    model.load_state_dict(checkpoint["model_state_dict"])

            # 6. Longitudinal Autoregressive Rollouts
            logger.info(f"Executing {args.rollout_span * 2}-step unconstrained rollouts for {ab_name}...")
            reconstructor = VolumeReconstructor(model, config, device=device)

            for vol_path in eval_files:
                vol_id = os.path.basename(vol_path).replace(".nii.gz", "").replace(".nii", "")
                vol_obj = next((v for v in val_loader_mem.pool.get("ixi", []) if v.path == vol_path), None)

                if vol_obj is None:
                    continue

                volume = vol_obj.t1
                center = volume.shape[0] // 2
                start_z = max(0, center - args.rollout_span)
                end_z = min(volume.shape[0], center + args.rollout_span)

                if (end_z - start_z) < args.rollout_span:
                    continue

                # Execute forward unconstrained autoregression
                recon_vol = reconstructor.autoregressive_restore(
                    volume=volume,
                    start_idx=start_z,
                    end_idx=end_z,
                    direction="forward",
                    mask_volume=None,
                    disable_pbar=True,
                )

                # Compute step-wise spatial degradation metrics
                metrics = calculate_step_metrics(
                    gt_vol=volume,
                    pred_vol=recon_vol,
                    start_idx=start_z,
                    end_idx=end_z,
                    direction="forward",
                    ablation_name=ab_name,
                    vol_id=vol_id,
                )
                all_rollout_metrics.extend(metrics)

                # Save raw volumes for Fréchet and visual inspection
                os.makedirs(config.data.results_dir, exist_ok=True)
                np.save(os.path.join(config.data.results_dir, f"{ab_name}_{vol_id}.npy"), recon_vol)
                np.save(os.path.join(config.data.results_dir, f"gt_{vol_id}.npy"), volume)

    # 7. Aggregate Rollout Degradation Metrics
    if all_rollout_metrics:
        df_metrics = pd.DataFrame(all_rollout_metrics)
        metrics_csv = os.path.join(CFG.data.results_dir, "ablation_metrics.csv")
        df_metrics.to_csv(metrics_csv, index=False)
        logger.info(f"Saved step-wise metrics to: {metrics_csv}")

        # Print Macro Summary at k = max(k)
        max_k = df_metrics["Rollout_Step_k"].max()
        summary = (
            df_metrics[df_metrics["Rollout_Step_k"] == max_k]
            .groupby("Ablation")[["SSIM", "PSNR", "MAE", "FFL"]]
            .mean()
            .reset_index()
        )
        logger.info(f"\nMacro Degradation at Step k={max_k}:\n{summary.to_string(index=False)}")

    # 8. Compute 3D Structural Realism (3D-FID) and Sequential Continuity (FVD)
    logger.info("\nEvaluating Fréchet Inception Distance (3D-FID) and Fréchet Video Distance (FVD)...")
    evaluator_frechet = FrechetEvaluator(CFG, device=device)
    df_frechet = evaluator_frechet.evaluate(list(experiments.keys()))
    if not df_frechet.empty:
        logger.info(f"\nFréchet Metrics Summary:\n{df_frechet.to_string(index=False)}")

    logger.info("\n==================================================")
    logger.info("   ABLATION PIPELINE EXECUTION COMPLETE")
    logger.info("==================================================")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Spiking & Continuous 2.5D MRI Reconstruction Pipeline")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs per variant")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--samples_per_epoch", type=int, default=3000, help="Samples drawn per epoch")
    parser.add_argument("--num_eval_vols", type=int, default=10, help="Number of test volumes for rollout")
    parser.add_argument("--rollout_span", type=int, default=20, help="Rollout horizon span around center (+/-)")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--force_retrain", action="store_true", help="Ignore saved checkpoints and train from scratch")
    parser.add_argument("--quick_test", action="store_true", help="Sets minimal parameters for smoke testing")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.quick_test:
        logger.info("[!] Quick test mode enabled: Scaling down steps and epochs.")
        cli_args.epochs = 2
        cli_args.batch_size = 4
        cli_args.samples_per_epoch = 16
        cli_args.num_eval_vols = 2
        cli_args.rollout_span = 5

    run_ablation_pipeline(cli_args)

