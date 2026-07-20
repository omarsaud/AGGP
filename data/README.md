# Data Preparation

The AGGP experiments use two standard traffic forecasting benchmarks:

- **METR-LA** — 207 loop detectors on Los Angeles highways
- **PEMS-BAY** — 325 sensors in the San Francisco Bay Area

---

## Expected Directory Structure

```
data/
├── metr-la/
│   ├── scaled_speed.npy          # (34272, 207)  float32
│   ├── scaled_acceleration.npy   # (34272, 207)  float32
│   ├── adj_mx.pkl                # (207, 207)    adjacency matrix
│   └── normalization_params.json # speed_mean, speed_std
└── pems-bay/
    ├── scaled_speed.npy          # (52116, 325)  float32
    ├── scaled_acceleration.npy   # (52116, 325)  float32
    ├── adj_mx.pkl                # (325, 325)
    └── normalization_params.json
```

---

## Step 1 — Obtain Raw Data

### METR-LA

Download from the official DCRNN repository:

```
https://github.com/liyaguang/DCRNN
```

Files needed: `metr-la.h5`, `adj_mx.pkl`

### PEMS-BAY

Download from the same DCRNN repository:

```
https://github.com/liyaguang/DCRNN
```

Files needed: `pems-bay.h5`, `adj_mx_bay.pkl`

---

## Step 2 — Compute Acceleration

Acceleration is the first-order finite difference of speed, smoothed to reduce noise:

```python
import numpy as np
import pandas as pd

speed = pd.read_hdf('metr-la.h5').values   # (T, N)

# Finite difference: a(t) = v(t) - v(t-1)
accel = np.diff(speed, axis=0, prepend=speed[[0]])  # (T, N)

# Clip outliers (±3 std)
std = accel.std()
accel = np.clip(accel, -3 * std, 3 * std)

# Normalize speed and acceleration together
speed_mean, speed_std = speed.mean(), speed.std()
accel_mean, accel_std = accel.mean(), accel.std()

scaled_speed = (speed - speed_mean) / speed_std
scaled_accel = (accel - accel_mean) / accel_std

# Save
np.save('data/metr-la/scaled_speed.npy', scaled_speed.astype('float32'))
np.save('data/metr-la/scaled_acceleration.npy', scaled_accel.astype('float32'))

import json, shutil
with open('data/metr-la/normalization_params.json', 'w') as f:
    json.dump({'speed_mean': speed_mean, 'speed_std': speed_std,
               'accel_mean': accel_mean, 'accel_std': accel_std}, f)

shutil.copy('adj_mx.pkl', 'data/metr-la/adj_mx.pkl')
```

Repeat the same steps for PEMS-BAY, placing outputs in `data/pems-bay/`.

---


## Data Split

All experiments use a fixed 70/10/20 split (train/val/test) on consecutive timesteps,
consistent with the DCRNN and Graph WaveNet baselines.
