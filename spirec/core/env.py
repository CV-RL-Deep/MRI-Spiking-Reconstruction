import os
import torch
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)

KAGGLE: bool = bool(
    os.environ.get("KAGGLE_URL_BASE")
    or os.environ.get("KAGGLE_KERNEL_RUN_TYPE")
)


class EnvironmentConfig:
    """Manages system paths, execution contexts, and compute devices."""

    def __init__(self):
        self.is_kaggle = KAGGLE
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if self.is_kaggle:
            self.path_ixi = (
                "/kaggle/input/preprocessed-oasis-and-epilepsy-and-ixi"
            )
            self.path_brats = "/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
            self.work_dir = "/kaggle/working"
        else:
            self.path_ixi = "data/ixi"
            self.path_brats = "data/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
            self.work_dir = "."

        self._configure_runtime()

    def _configure_runtime(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


ENV = EnvironmentConfig()
DEVICE = ENV.device
