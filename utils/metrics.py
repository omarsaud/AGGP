"""
Evaluation metrics for traffic speed prediction.
All metrics are computed on denormalized (real-unit) values with zero-masking.
"""

import torch
import numpy as np


def compute_metrics(predictions, targets, speed_mean=None, speed_std=None,
                    null_val=0.0):
    """
    Compute MAE, RMSE, MAPE with optional denormalization and zero-masking.

    Args:
        predictions: Tensor or ndarray (B, N, Q, 1) or (B, N, Q)
        targets:     Tensor or ndarray, same shape
        speed_mean:  Normalization mean  (None = already in real units)
        speed_std:   Normalization std   (None = already in real units)
        null_val:    Mask out targets equal to this value (default 0.0)

    Returns:
        dict with keys 'mae', 'rmse', 'mape'
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.FloatTensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.FloatTensor(targets)

    if speed_mean is not None and speed_std is not None:
        predictions = predictions * speed_std + speed_mean
        targets     = targets     * speed_std + speed_mean

    mask = (targets != null_val) & (~torch.isnan(targets))

    pred_m = predictions[mask]
    tgt_m  = targets[mask]

    mae  = torch.mean(torch.abs(pred_m - tgt_m)).item()
    rmse = torch.sqrt(torch.mean((pred_m - tgt_m) ** 2)).item()
    mape = (torch.mean(torch.abs(pred_m - tgt_m) /
                       (torch.abs(tgt_m) + 1e-5)) * 100).item()

    return {'mae': mae, 'rmse': rmse, 'mape': mape}


def compute_metrics_per_horizon(predictions, targets, speed_mean=None,
                                 speed_std=None, null_val=0.0):
    """
    Compute metrics separately for each prediction horizon step.

    Returns:
        list of dicts, one per horizon step
    """
    if not isinstance(predictions, torch.Tensor):
        predictions = torch.FloatTensor(predictions)
    if not isinstance(targets, torch.Tensor):
        targets = torch.FloatTensor(targets)

    if speed_mean is not None and speed_std is not None:
        predictions = predictions * speed_std + speed_mean
        targets     = targets     * speed_std + speed_mean

    horizon = predictions.shape[2] if predictions.dim() == 4 else predictions.shape[-1]
    results = []

    for q in range(horizon):
        pred_q = predictions[:, :, q]
        tgt_q  = targets[:, :, q]
        mask   = (tgt_q != null_val) & (~torch.isnan(tgt_q))
        p = pred_q[mask]
        t = tgt_q[mask]
        mae  = torch.mean(torch.abs(p - t)).item()
        rmse = torch.sqrt(torch.mean((p - t) ** 2)).item()
        mape = (torch.mean(torch.abs(p - t) / (torch.abs(t) + 1e-5)) * 100).item()
        results.append({'step': q + 1, 'mae': mae, 'rmse': rmse, 'mape': mape})

    return results
