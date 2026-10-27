import torch.nn as nn
from spirec.configs.config import Config
from spirec.models.dense_unet import DenseUNet2p5D
from spirec.models.losses import (
    CompositeLoss,
    FocalFrequencyLoss,
    GradientSharpnessLoss,
    SpectralLoss,
)
from spirec.models.spike_unet import SpikeARUNet


class ModelBuilder:
    """Factory to instantiate dense or spiking 2.5D autoregressive architectures."""

    @staticmethod
    def build(config: Config) -> nn.Module:
        arch = config.model.architecture.lower()

        if arch == "vanilla_unet":
            return DenseUNet2p5D(
                in_channels=config.data.neighborhood,
                out_channels=1,
                base_filters=config.model.base_filters,
            )
        elif arch == "baseline_unet":
            # Context slices N + 1 target mask channel
            return DenseUNet2p5D(
                in_channels=config.data.neighborhood + 1,
                out_channels=1,
                base_filters=config.model.base_filters,
            )
        elif arch == "spike_unet":
            return SpikeARUNet(
                in_channels=config.data.neighborhood,
                base_filters=config.model.base_filters,
                t_sim=config.snn.t_sim,
                v_threshold=config.snn.v_threshold,
                initial_beta=config.snn.initial_beta,
                surrogate_alpha=config.snn.surrogate_alpha,
            )
        else:
            raise ValueError(f"Unsupported architecture: {arch}")


__all__ = [
    "ModelBuilder",
    "DenseUNet2p5D",
    "SpikeARUNet",
    "CompositeLoss",
    "GradientSharpnessLoss",
    "SpectralLoss",
    "FocalFrequencyLoss",
]

