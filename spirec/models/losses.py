import torch
import torch.nn as nn
import torch.nn.functional as F

from spirec.configs.config import Config


class GradientSharpnessLoss(nn.Module):
    """Finite-difference spatial gradient penalty enforcing edge preservation."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Spatial finite difference along Height
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_true = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])

        # Spatial finite difference along Width
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_true = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

        loss_y = torch.mean(torch.abs(dy_pred - dy_true))
        loss_x = torch.mean(torch.abs(dx_pred - dx_true))
        return loss_y + loss_x


class SpectralLoss(nn.Module):
    """L1 distance between 2D Real Fast Fourier Transform (rFFT) log-magnitudes."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Real FFT over spatial dimensions (2, 3)
        fft_pred = torch.fft.rfft2(pred.float(), norm="ortho")
        fft_true = torch.fft.rfft2(target.float(), norm="ortho")

        mag_pred = torch.log(torch.abs(fft_pred) + 1.0)
        mag_true = torch.log(torch.abs(fft_true) + 1.0)

        return torch.mean(torch.abs(mag_pred - mag_true))


class FocalFrequencyLoss(nn.Module):
    """
    Focal Frequency Loss (FFL) for high-frequency spectral fidelity (Jiang et al., ICCV 2021).
    Penalizes spectral divergence using dynamic focal weighting matrix.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Full 2D FFT with zero-frequency centered
        fft_pred = torch.fft.fftshift(torch.fft.fft2(pred.float(), norm="ortho"), dim=(-2, -1))
        fft_true = torch.fft.fftshift(torch.fft.fft2(target.float(), norm="ortho"), dim=(-2, -1))

        # Complex squared Euclidean distance matrix
        diff_matrix = torch.abs(fft_true - fft_pred) ** 2

        # Focal weighting per image in batch
        max_diff = torch.amax(diff_matrix, dim=(-2, -1), keepdim=True)
        diff_norm = diff_matrix / torch.clamp(max_diff, min=1e-8)
        weight_matrix = torch.pow(diff_norm, self.alpha)

        return torch.mean(weight_matrix * diff_matrix)


class CompositeLoss(nn.Module):
    """
    Master composite loss engine handling Vanilla, Baseline, and Spiking regularizers.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.grad_loss = GradientSharpnessLoss()
        self.spectral_loss = SpectralLoss()
        self.ffl_loss = FocalFrequencyLoss(alpha=1.0)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        brain_mask: torch.Tensor,
        tumor_mask: torch.Tensor,
        spike_rate_metrics: dict = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        pred:       (B, 1, H, W)
        target:     (B, 1, H, W)
        brain_mask: (B, 1, H, W)
        tumor_mask: (B, 1, H, W)
        """
        loss_dict = {}
        arch = self.cfg.model.architecture

        if arch == "vanilla_unet":
            # Pure unmasked spatial L1 loss (reproduces unbounded drift)
            l1_loss = F.l1_loss(pred, target)
            total_loss = l1_loss
            loss_dict["l1_vanilla"] = l1_loss.item()

        elif arch == "baseline_unet":
            # Spatially weighted L1 loss (reproduces texture blur & dragging)
            spatial_weights = (
                self.cfg.train.weight_background
                + self.cfg.train.weight_brain * brain_mask
                + self.cfg.train.weight_tumor * tumor_mask
            )
            l1_spatial = torch.mean(spatial_weights * torch.abs(pred - target))
            l_grad = self.grad_loss(pred, target)
            l_spec = self.spectral_loss(pred, target)

            total_loss = (
                self.cfg.train.lambda_spatial * l1_spatial
                + self.cfg.train.lambda_grad * l_grad
                + self.cfg.train.lambda_spec * l_spec
            )
            loss_dict["l1_spatial"] = l1_spatial.item()
            loss_dict["loss_grad"] = l_grad.item()
            loss_dict["loss_spec"] = l_spec.item()

        elif arch == "spike_unet":
            # Spiking loss: spatial reconstruction + gradient + spectral + metabolic firing rate
            l1_spatial = F.l1_loss(pred, target)
            l_grad = self.grad_loss(pred, target)
            l_ffl = self.ffl_loss(pred, target)

            total_loss = l1_spatial + self.cfg.train.lambda_grad * l_grad + self.cfg.train.lambda_spec * l_ffl

            # Neuromorphic metabolic regularizer
            if spike_rate_metrics and "mean_spike_rate" in spike_rate_metrics:
                rate = spike_rate_metrics["mean_spike_rate"]
                target_rate = self.cfg.snn.target_spike_rate
                l_rate = (rate - target_rate) ** 2
                total_loss = total_loss + self.cfg.snn.lambda_rate * l_rate
                loss_dict["spike_rate"] = rate
                loss_dict["loss_rate"] = l_rate

            loss_dict["l1_recon"] = l1_spatial.item()
            loss_dict["loss_grad"] = l_grad.item()
            loss_dict["loss_ffl"] = l_ffl.item()

        else:
            raise ValueError(f"Unknown architecture for loss calculation: {arch}")

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict

