import json
import os

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

from spirec.core.env import ENV, KAGGLE
from spirec.core.utils import logger


@dataclass
class SpikingConfig:
    """Hyperparameters governing SNN neuronal simulation under Paradigm B."""

    t_sim: int = 4  # Micro-temporal simulation steps per slice
    v_threshold: float = 1.0  # Membrane firing threshold
    initial_beta: float = 0.8  # Initial membrane potential decay constant
    surrogate_alpha: float = (
        2.0  # Steepness parameter for ArcTan surrogate gradient
    )
    learnable_beta: bool = True  # Enable parametric decay estimation (PLIF)
    stateful_across_slices: bool = (
        False  # Carry membrane potential forward along z-axis
    )
    target_spike_rate: float = 0.10  # Target metabolic synaptic firing rate
    lambda_rate: float = 0.05  # Weight for firing-rate regularizer


@dataclass
class ModelConfig:
    """Architectural parameters for dense and spiking generators."""

    name: str = "spike_unet"
    architecture: str = (
        "spike_unet"  # 'vanilla_unet', 'baseline_unet', 'spike_unet'
    )
    in_channels: int = 3  # N historical context slices
    base_filters: int = 32
    use_positional_encoding: bool = False
    dropout_rate: float = 0.0


@dataclass
class DataConfig:
    """Dataset management, directory targets, and spatial sampling dimensions."""

    data_root_ixi: str = ENV.path_ixi
    data_root_brats: str = ENV.path_brats
    ixi_sampling_weight: float = 0.8  # Ratio of IXI to BraTS instances
    native_patch_size: Tuple[int, int] = (113, 137)
    padded_size: Tuple[int, int] = (160, 160)
    neighborhood: int = 3  # Context window size N
    input_last_target_mask: bool = (
        False  # Concatenate target spatial mask if True
    )

    # Memory pool caching limits
    ixi_cache_size: int = 120 if KAGGLE else 240
    brats_cache_size: int = 120 if KAGGLE else 240

    # Output artifact locations
    weights_dir: str = "checkpoints"
    results_dir: str = "ablation_results"
    figures_dir: str = "paper_figures"


@dataclass
class AugmentationConfig:
    """Probabilistic corruption schedules and pathology voiding."""

    prob_glioma: float = 0.65
    brats_labels_for_void: Tuple[int, ...] = (1, 2, 4)
    prob_flip: float = 0.5
    prob_rotate: float = 0.35
    rotate_range: float = 3.5  # Degrees (+/-)
    prob_noise: float = 0.0
    noise_std: float = 0.025
    prob_blur: float = 0.0
    blur_sigma: float = 0.85


@dataclass
class TrainingConfig:
    """Hyperparameters for optimization loops."""

    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 3.14159e-4
    weight_decay: float = 1e-4
    use_amp: bool = True  # Automatic Mixed Precision

    # Loss coefficients
    lambda_spatial: float = 1.0
    lambda_grad: float = 0.416
    lambda_spec: float = 0.078

    # Spatially weighted baseline weights
    weight_background: float = 1.0
    weight_brain: float = 14.0
    weight_tumor: float = 10.0


@dataclass
class Config:
    """Root configuration object containing all operational parameters."""

    model: ModelConfig = field(default_factory=ModelConfig)
    snn: SpikingConfig = field(default_factory=SpikingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = 42

    def __post_init__(self):
        # Update channel configurations dynamically
        self.model.in_channels = self.data.neighborhood + int(
            self.data.input_last_target_mask
        )

    def to_dict(self):
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as stream:
            json.dump(asdict(self), stream, indent=4)
        logger.info(f"Configuration written to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "Config":
        with open(filepath, "r") as stream:
            payload = json.load(stream)

        config = cls()
        for section, parameters in payload.items():
            if hasattr(config, section):
                sub_config = getattr(config, section)
                for key, val in parameters.items():
                    if hasattr(sub_config, key):
                        setattr(sub_config, key, val)
        return config


CFG = Config()
