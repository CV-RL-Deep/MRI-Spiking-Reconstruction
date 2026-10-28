import os

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from spirec.configs.config import Config
from spirec.core.utils import InferenceProfiler, MemoryTelemetry, logger
from spirec.models.losses import CompositeLoss


class Trainer:
    """
    Standardized PyTorch Trainer supporting Dense Baselines and Spiking Paradigms.
    Executes mixed-precision training, gradient clipping, and validation tracking.
    """

    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = CompositeLoss(config).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scaler = GradScaler(
            enabled=config.train.use_amp and self.device.type == "cuda"
        )

        self.telemetry = MemoryTelemetry(
            os.path.join(config.data.results_dir, "train_telemetry.csv")
        )
        self.profiler = InferenceProfiler(
            config.model.name, config.data.results_dir
        )

        self.best_val_loss = float("inf")
        self.checkpoint_dir = config.data.weights_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses: Dict[str, float] = {}
        total_steps = len(self.train_loader)

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch:03d} [Train]", leave=False
        )

        for step, batch in enumerate(pbar):
            ctx_stack = batch["ctx_stack"].to(self.device, non_blocking=True)
            target_slice = batch["target_slice"].to(
                self.device, non_blocking=True
            )
            brain_mask = batch["brain_mask"].to(self.device, non_blocking=True)
            tumor_mask = batch["tumor_mask"].to(self.device, non_blocking=True)

            # Baseline baseline concatenates the target brain mask into the context
            if self.cfg.model.architecture == "baseline_unet":
                ctx_stack = torch.cat([ctx_stack, brain_mask], dim=1)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(
                enabled=self.cfg.train.use_amp and self.device.type == "cuda"
            ):
                if self.cfg.model.architecture == "spike_unet":
                    pred_slice, snn_metrics = self.model(ctx_stack)
                else:
                    pred_slice = self.model(ctx_stack)
                    snn_metrics = None

                loss, loss_dict = self.loss_fn(
                    pred_slice,
                    target_slice,
                    brain_mask,
                    tumor_mask,
                    spike_rate_metrics=snn_metrics,
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Metric tracking
            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else v
                )

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # Calculate epoch mean
        epoch_metrics = {k: v / total_steps for k, v in running_losses.items()}
        self.telemetry.log("TRAIN_EPOCH", epoch)
        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, epoch: int) -> Dict[str, float]:
        if not self.val_loader or len(self.val_loader) == 0:
            return {}

        self.model.eval()
        running_losses: Dict[str, float] = {}
        total_steps = len(self.val_loader)

        pbar = tqdm(
            self.val_loader, desc=f"Epoch {epoch:03d} [Valid]", leave=False
        )

        for batch in pbar:
            ctx_stack = batch["ctx_stack"].to(self.device, non_blocking=True)
            target_slice = batch["target_slice"].to(
                self.device, non_blocking=True
            )
            brain_mask = batch["brain_mask"].to(self.device, non_blocking=True)
            tumor_mask = batch["tumor_mask"].to(self.device, non_blocking=True)

            if self.cfg.model.architecture == "baseline_unet":
                ctx_stack = torch.cat([ctx_stack, brain_mask], dim=1)

            with autocast(
                enabled=self.cfg.train.use_amp and self.device.type == "cuda"
            ):
                if self.cfg.model.architecture == "spike_unet":
                    pred_slice, snn_metrics = self.model(ctx_stack)
                else:
                    pred_slice = self.model(ctx_stack)
                    snn_metrics = None

                _, loss_dict = self.loss_fn(
                    pred_slice,
                    target_slice,
                    brain_mask,
                    tumor_mask,
                    spike_rate_metrics=snn_metrics,
                )

            for k, v in loss_dict.items():
                running_losses[k] = running_losses.get(k, 0.0) + (
                    v.item() if isinstance(v, torch.Tensor) else v
                )

        eval_metrics = {k: v / total_steps for k, v in running_losses.items()}
        self.telemetry.log("VAL_EPOCH", epoch)
        return eval_metrics

    def fit(self, num_epochs: int) -> None:
        logger.info(
            f"Starting training run: {self.cfg.model.name} for {num_epochs} epochs."
        )

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.evaluate(epoch)

            val_loss = val_metrics.get(
                "total_loss", train_metrics.get("total_loss", 0.0)
            )
            is_best = val_loss < self.best_val_loss

            if is_best:
                self.best_val_loss = val_loss
                best_path = os.path.join(
                    self.checkpoint_dir, f"{self.cfg.model.name}_best.pt"
                )
                # To save weights cleanly when using DataParallel:
                state_dict_model = self.model.module.state_dict() if hasattr(self.model, "module") else self.model.state_dict()
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict_model,
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "config": self.cfg.to_dict(),
                        "val_loss": val_loss,
                    },
                    best_path,
                )
                logger.info(
                    f"[*] Checkpoint saved (Epoch {epoch:03d} Best Loss: {val_loss:.5f}) -> {best_path}"
                )

            metrics_summary = " | ".join(
                [f"Train Loss: {train_metrics.get('total_loss', 0.0):.4f}"]
                + [f"Val {k}: {v:.4f}" for k, v in val_metrics.items()]
            )
            logger.info(
                f"Epoch {epoch:03d}/{num_epochs:03d} -> {metrics_summary}"
            )
