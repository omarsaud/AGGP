"""
Data loading utilities for AGGP experiments.
Supports METR-LA and PEMS-BAY datasets with optional acceleration channel.
"""

import torch
import numpy as np
import pandas as pd
import pickle
import platform
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


def load_data_simple(dataset_name, data_dir=None, use_acceleration=True):
    """
    Load traffic data for AGGP training.

    Args:
        dataset_name:     'metr-la' or 'pems-bay'
        data_dir:         Path to data directory (default: data/{dataset_name})
        use_acceleration: Whether to load acceleration channel

    Returns:
        speed_data:  (T, N) numpy array
        accel_data:  (T, N) numpy array or None
        adj_matrix:  (N, N) numpy array
        norm_params: dict with speed_mean, speed_std (and accel if available)
    """
    if data_dir is not None:
        base_dir = Path(data_dir)
    else:
        root = Path(__file__).resolve().parent.parent
        base_dir = root / "data" / dataset_name

    print(f"Loading data from: {base_dir}")

    speed_npy = base_dir / "scaled_speed.npy"
    speed_h5  = base_dir / f"{dataset_name}.h5"

    if speed_npy.exists():
        speed_data = np.load(speed_npy)
        print(f"  Speed: {speed_data.shape}  (.npy)")
    elif speed_h5.exists():
        speed_data = pd.read_hdf(speed_h5).values
        print(f"  Speed: {speed_data.shape}  (.h5)")
    else:
        raise FileNotFoundError(f"No speed data found in {base_dir}")

    accel_data = None
    if use_acceleration:
        accel_npy = base_dir / "scaled_acceleration.npy"
        accel_h5  = base_dir / f"{dataset_name}_acceleration.h5"

        if accel_npy.exists():
            accel_data = np.load(accel_npy)
            print(f"  Accel: {accel_data.shape}  (.npy)")
        elif accel_h5.exists():
            accel_data = pd.read_hdf(accel_h5).values
            print(f"  Accel: {accel_data.shape}  (.h5)")
        else:
            print("  Accel: not found — using speed only")

    adj_path = base_dir / "adj_mx.pkl"
    if adj_path.exists():
        with open(adj_path, 'rb') as f:
            _, _, adj_matrix = pickle.load(f, encoding='latin1')
        print(f"  Adj:   {adj_matrix.shape}")
    else:
        raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")

    norm_params = {}
    norm_path = base_dir / "normalization_params.json"
    if norm_path.exists():
        import json
        with open(norm_path, 'r') as f:
            norm_params = json.load(f)
        print(f"  Norm:  speed mean={norm_params.get('speed_mean', 'N/A'):.2f}, "
              f"std={norm_params.get('speed_std', 'N/A'):.2f}")

    return speed_data, accel_data, adj_matrix, norm_params


class TrafficDataset(Dataset):
    """
    Sliding-window dataset for traffic prediction.

    Returns (x, y) pairs:
        x: (N, T, F) — historical features  [speed] or [speed, accel]
        y: (N, Q, 1) — future speed targets
    """

    def __init__(self, speed_data, accel_data, seq_len, horizon, start_idx, end_idx):
        self.speed_data = speed_data
        self.accel_data = accel_data
        self.seq_len    = seq_len
        self.horizon    = horizon
        self.start_idx  = start_idx
        self.end_idx    = end_idx
        self.num_samples = end_idx - start_idx - seq_len - horizon + 1
        self.input_dim   = 2 if accel_data is not None else 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        i = self.start_idx + idx
        speed_hist   = self.speed_data[i: i + self.seq_len]
        speed_future = self.speed_data[i + self.seq_len: i + self.seq_len + self.horizon]

        if self.accel_data is not None:
            accel_hist = self.accel_data[i: i + self.seq_len]
            x = np.stack([speed_hist, accel_hist], axis=-1)  # (T, N, 2)
        else:
            x = speed_hist[:, :, np.newaxis]                 # (T, N, 1)

        y = speed_future[:, :, np.newaxis]  # (T, N, 1)

        x = np.transpose(x, (1, 0, 2))  # (N, T, F)
        y = np.transpose(y, (1, 0, 2))  # (N, Q, 1)

        return torch.FloatTensor(x), torch.FloatTensor(y)


def create_data_loaders(speed_data, accel_data, seq_len, horizon,
                        batch_size=64, train_ratio=0.7, val_ratio=0.1,
                        num_workers=4):
    """
    Create train / val / test DataLoaders.

    Args:
        speed_data:   (T, N)
        accel_data:   (T, N) or None
        seq_len:      Historical window
        horizon:      Prediction horizon
        batch_size:   Batch size
        train_ratio:  Training split (default 0.7)
        val_ratio:    Validation split (default 0.1)
        num_workers:  DataLoader worker count (auto-set to 0 on Windows)

    Returns:
        train_loader, val_loader, test_loader
    """
    use_pin_memory = torch.cuda.is_available()

    if platform.system() == 'Windows':
        num_workers = 0
    elif platform.system() == 'Linux' and num_workers < 8:
        num_workers = 8

    num_timestamps = speed_data.shape[0]
    num_samples    = num_timestamps - seq_len - horizon + 1
    train_size     = int(train_ratio * num_samples)
    val_size       = int(val_ratio * num_samples)
    train_end      = train_size
    val_end        = train_size + val_size

    print(f"  Samples — train: {train_size}  val: {val_size}  "
          f"test: {num_samples - val_end}  |  workers: {num_workers}")

    train_ds = TrafficDataset(speed_data, accel_data, seq_len, horizon, 0,          train_end)
    val_ds   = TrafficDataset(speed_data, accel_data, seq_len, horizon, train_end,  val_end)
    test_ds  = TrafficDataset(speed_data, accel_data, seq_len, horizon, val_end,    num_samples)

    dl_kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=use_pin_memory)
    if num_workers > 0:
        dl_kwargs.update(prefetch_factor=4, persistent_workers=True, drop_last=True)

    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    return train_loader, val_loader, test_loader
