"""
Evaluation metrics for traffic speed prediction.

All metrics are computed on denormalized (real-unit) values in mph.

MAE and RMSE use every ground-truth entry. MAPE additionally excludes entries
whose ground-truth speed falls below `mape_min_speed` (5 mph by default):
percentage error divides by the observed speed, so near-stationary readings
produce arbitrarily large ratios that dominate the average without reflecting
forecast quality. The same threshold is applied to every configuration.
"""

import torch
import numpy as np

MAPE_MIN_SPEED = 5.0  # mph — low-speed cut-off for MAPE only


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.FloatTensor(x)


def _denorm(predictions, targets, speed_mean, speed_std):
    if speed_mean is not None and speed_std is not None:
        predictions = predictions * speed_std + speed_mean
        targets     = targets     * speed_std + speed_mean
    return predictions, targets


def compute_metrics(predictions, targets, speed_mean=None, speed_std=None,
                    mape_min_speed=MAPE_MIN_SPEED):
    """
    Compute MAE, RMSE and MAPE with optional denormalization.

    Args:
        predictions:    Tensor or ndarray (B, N, Q, 1) or (B, N, Q)
        targets:        Tensor or ndarray, same shape
        speed_mean:     Normalization mean (None = already in mph)
        speed_std:      Normalization std  (None = already in mph)
        mape_min_speed: Ground-truth speeds below this (mph) are excluded from
                        MAPE only. MAE and RMSE always use every entry.

    Returns:
        dict with keys 'mae', 'rmse', 'mape'
    """
    predictions, targets = _to_tensor(predictions), _to_tensor(targets)
    predictions, targets = _denorm(predictions, targets, speed_mean, speed_std)

    valid = ~torch.isnan(targets)
    p, t = predictions[valid], targets[valid]

    mae  = torch.mean(torch.abs(p - t)).item()
    rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()

    m = t >= mape_min_speed
    mape = (torch.mean(torch.abs(p[m] - t[m]) / t[m]) * 100).item()

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def compute_metrics_per_horizon(predictions, targets, speed_mean=None,
                                speed_std=None, mape_min_speed=MAPE_MIN_SPEED):
    """
    Compute metrics separately for each prediction horizon step.

    Returns:
        list of dicts, one per horizon step
    """
    predictions, targets = _to_tensor(predictions), _to_tensor(targets)
    predictions, targets = _denorm(predictions, targets, speed_mean, speed_std)

    horizon = predictions.shape[2] if predictions.dim() == 4 else predictions.shape[-1]
    results = []

    for q in range(horizon):
        pred_q, tgt_q = predictions[:, :, q], targets[:, :, q]
        valid = ~torch.isnan(tgt_q)
        p, t = pred_q[valid], tgt_q[valid]

        mae  = torch.mean(torch.abs(p - t)).item()
        rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()

        m = t >= mape_min_speed
        mape = (torch.mean(torch.abs(p[m] - t[m]) / t[m]) * 100).item()

        results.append({'step': q + 1, 'mae': mae, 'rmse': rmse, 'mape': mape})

    return results
