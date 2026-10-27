from typing import Any, Dict, List
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calc_focal_frequency_error(
    gt_slice: np.ndarray, pred_slice: np.ndarray, alpha: float = 1.0
) -> float:
    """
    Computes Focal Frequency Loss (FFL) as a quantitative metric in NumPy.
    Penalizes frequency divergence with dynamic focal matrix weighting.
    """
    # 2D Fast Fourier Transform into complex domain
    fft_gt = np.fft.fft2(gt_slice)
    fft_pr = np.fft.fft2(pred_slice)

    # Shift zero-frequency DC component to center
    fft_gt = np.fft.fftshift(fft_gt)
    fft_pr = np.fft.fftshift(fft_pr)

    # Complex difference matrix d(u, v) = |F_gt - F_pred|^2
    diff_matrix = np.abs(fft_gt - fft_pr) ** 2

    max_diff = np.max(diff_matrix)
    if max_diff < 1e-8:
        return 0.0

    diff_norm = diff_matrix / max_diff
    weight_matrix = diff_norm**alpha
    ffl = np.mean(weight_matrix * diff_matrix)
    return float(ffl)


def calc_gradient_sharpness_error(gt_slice: np.ndarray, pred_slice: np.ndarray) -> float:
    """Computes mean absolute error between spatial image gradients (finite differences)."""
    dy_gt, dx_gt = np.gradient(gt_slice)
    dy_pr, dx_pr = np.gradient(pred_slice)
    grad_diff = np.abs(dy_gt - dy_pr) + np.abs(dx_gt - dx_pr)
    return float(np.mean(grad_diff))


def calculate_step_metrics(
    gt_vol: np.ndarray,
    pred_vol: np.ndarray,
    start_idx: int,
    end_idx: int,
    direction: str,
    ablation_name: str,
    vol_id: str,
) -> List[Dict[str, Any]]:
    """
    Evaluates step-by-step drift metrics along an autoregressive rollout horizon (k = 1, ..., K).
    """
    metrics_list = []
    iter_range = (
        range(start_idx, end_idx)
        if direction == "forward"
        else range(end_idx - 1, start_idx - 1, -1)
    )

    for i, z_idx in enumerate(iter_range):
        k = i + 1  # Rollout step index
        gt_slice = gt_vol[z_idx]
        pr_slice = pred_vol[z_idx]

        mae_val = float(np.mean(np.abs(gt_slice - pr_slice)))
        mse_val = float(np.mean(np.square(gt_slice - pr_slice)))
        psnr_val = float(psnr(gt_slice, pr_slice, data_range=1.0)) if mse_val > 0 else 100.0
        ssim_val = float(ssim(gt_slice, pr_slice, data_range=1.0))
        grad_err = calc_gradient_sharpness_error(gt_slice, pr_slice)
        ffl_err = calc_focal_frequency_error(gt_slice, pr_slice, alpha=1.0)

        metrics_list.append(
            {
                "Ablation": ablation_name,
                "Volume_ID": vol_id,
                "Direction": direction,
                "Rollout_Step_k": k,
                "Z_Index": z_idx,
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "MAE": mae_val,
                "MSE": mse_val,
                "Grad_Error": grad_err,
                "FFL": ffl_err,
            }
        )
    return metrics_list


def calculate_region_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> Dict[str, float]:
    """Computes MAE, MSE, PSNR, and SSIM strictly inside a region defined by mask."""
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return {"mae": 0.0, "mse": 0.0, "psnr": 0.0, "ssim": 0.0}

    true_region = y_true[mask_bool]
    pred_region = y_pred[mask_bool]

    mae = float(np.mean(np.abs(true_region - pred_region)))
    mse = float(np.mean(np.square(true_region - pred_region)))
    psnr_val = float(psnr(true_region, pred_region, data_range=1.0)) if mse > 0 else 100.0

    # Calculate SSIM on region bounding box
    rows, cols = np.where(mask_bool)
    r_min, r_max = np.min(rows), np.max(rows)
    c_min, c_max = np.min(cols), np.max(cols)

    if (r_max - r_min < 7) or (c_max - c_min < 7):
        ssim_val = 0.0
    else:
        bbox_true = y_true[r_min:r_max, c_min:c_max]
        bbox_pred = y_pred[r_min:r_max, c_min:c_max]
        ssim_val = float(ssim(bbox_true, bbox_pred, data_range=1.0))

    return {"mae": mae, "mse": mse, "psnr": psnr_val, "ssim": ssim_val}

