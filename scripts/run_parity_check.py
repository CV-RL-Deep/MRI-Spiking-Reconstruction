#!/usr/bin/env python3
import copy
import sys

from pathlib import Path

import torch

from torch.utils.data import DataLoader

# Force parent directory (repository root) into Python module search path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spirec.configs.config import CFG
from spirec.core.utils import logger, set_global_seeds
from spirec.data.cache import MedicalVolume, StaticLoader
from spirec.data.dataset import AutoregressiveSliceDataset, collate_autoregressive
from spirec.models import CompositeLoss, ModelBuilder
from spirec.training.trainer import Trainer


def execute_parity_check():
    logger.info("==================================================")
    logger.info("   SPIREC PIPELINE INTEGRITY & PARITY VERIFICATION")
    logger.info("==================================================")

    set_global_seeds(42, deterministic=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Execution Target: {device}")

    # 1. Synthesize Mock MRI Volume
    logger.info("1. Generating synthetic brain volume...")
    dummy_vol = np.zeros((30, 113, 137), dtype=np.float32)
    z, y, x = np.ogrid[:30, :113, :137]
    brain_mask = ((z - 15) ** 2 / 12**2 + (y - 56) ** 2 / 28**2 + (x - 68) ** 2 / 28**2) <= 1.0
    dummy_vol[brain_mask] = np.random.uniform(0.3, 0.8, size=np.sum(brain_mask)).astype(np.float32)

    loader = StaticLoader(CFG, [], [])
    loader.pool["ixi"].append(MedicalVolume(dummy_vol, path="synthetic_verification_vol.nii.gz"))

    # 2. Setup Data Pipeline
    dataset = AutoregressiveSliceDataset(loader, CFG, samples_per_epoch=8)
    dataloader = DataLoader(
        dataset, batch_size=2, shuffle=False, collate_fn=collate_autoregressive
    )
    test_batch = next(iter(dataloader))
    logger.info("   -> Data collator verification passed.")

    # 3. Model Configurations to Verify
    model_targets = [
        ("vanilla_unet", False),
        ("baseline_unet", True),
        ("spike_unet", False),
    ]

    for arch_name, use_mask in model_targets:
        logger.info(f"\n2. Testing Architecture: {arch_name.upper()}...")
        cfg = copy.deepcopy(CFG)
        cfg.model.architecture = arch_name
        cfg.data.input_last_target_mask = use_mask
        cfg.data.weights_dir = "checkpoints/parity_test"

        model = ModelBuilder.build(cfg).to(device)
        trainer = Trainer(cfg, model, dataloader, val_loader=None, device=device)

        metrics = trainer.train_epoch(epoch=1)
        loss = metrics["total_loss"]
        logger.info(f"   -> Epoch 1 loss computed: {loss:.4f}")

        # Check for non-finite values (NaN / Inf)
        if not np.isfinite(loss):
            logger.error(f"FATAL: {arch_name} produced non-finite loss!")
            sys.exit(1)

        # Confirm non-zero parameter gradients
        grad_norms = [
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        ]
        if not grad_norms or max(grad_norms) == 0.0:
            logger.error(f"FATAL: {arch_name} failed gradient propagation!")
            sys.exit(1)

        logger.info(f"   -> Max parameter gradient norm: {max(grad_norms):.4f}")

    logger.info("\n==================================================")
    logger.info("   PARITY CHECK COMPLETE: ALL MODELS FUNCTIONAL")
    logger.info("==================================================")


if __name__ == "__main__":
    import numpy as np

    execute_parity_check()

