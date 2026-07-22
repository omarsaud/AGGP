"""
Evaluate saved predictions under the paper's reporting protocol.

Reproduces the averaged and per-horizon numbers reported in the paper from a run
directory produced by train.py, without re-running the model.

Usage
-----
    python evaluate.py --run_dir models/gwnet_aggp_metr-la_Acc_Q12 \
                       --dataset metr-la --raw data/metr-la/metr-la.h5
    python evaluate.py --run_dir <dir> --dataset pems-bay \
                       --raw data/pems-bay/pems-bay.h5 --per_horizon

Protocol
--------
* Test window    : the first N samples of the test split (METR-LA N=6784,
                   PEMS-BAY N=10368), so every configuration is scored on an
                   identical window.
* Denormalisation: inputs are standardised *per sensor*, so predictions are
                   mapped back to mph with each sensor's own mean and standard
                   deviation. Ground truth is read directly from the benchmark
                   file, avoiding a normalise/denormalise round trip.
* Missing data   : a ground-truth speed of exactly 0 marks a missing detector
                   reading and is excluded from all three metrics.
* Averaging      : every metric is computed per horizon step and then averaged
                   over the 12 steps, so all three share one convention.
* MAPE           : additionally excludes ground truth below 5 mph.
"""

import argparse
import json
import os

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover
    raise SystemExit("evaluate.py needs h5py — pip install h5py")

from utils.metrics import MAPE_MIN_SPEED

# Canonical test-window length, and the HDF5 group holding the speed table.
PROTOCOL = {
    "metr-la":  dict(n=6784,  h5_key="df"),
    "pems-bay": dict(n=10368, h5_key="speed"),
}


def read_raw(raw_path, dataset):
    """Load the benchmark speed table as (T, N) in mph."""
    key = PROTOCOL[dataset]["h5_key"]
    with h5py.File(raw_path, "r") as f:
        if key not in f:                      # tolerate either pandas layout
            key = list(f.keys())[0]
        return f[key]["block0_values"][:].astype(np.float64)


def align(raw, pred_norm, targ_norm):
    """Find where this run's test window starts in the raw series."""
    mu, sd = raw.mean(0), raw.std(0)
    scaled = (raw - mu) / sd
    first = targ_norm[0, :, 0]
    for t in range(len(scaled)):
        if np.abs(scaled[t] - first).max() < 1e-4:
            for j in (0, 1, 37):
                for q in (0, 5, 11):
                    if np.abs(scaled[t + j + q] - targ_norm[j, :, q]).max() > 1e-4:
                        break
                else:
                    continue
                break
            else:
                return t, mu, sd
    raise RuntimeError("cannot align the run's test window to the raw series")


def load_run(run_dir, dataset, raw_path):
    """Return (pred_mph, target_mph, valid_mask) with per-sensor denormalisation."""
    n = PROTOCOL[dataset]["n"]
    pn = np.load(os.path.join(run_dir, "predictions_normalized.npy"))[:n, :, :, 0].astype(np.float64)
    tn = np.load(os.path.join(run_dir, "targets_normalized.npy"))[:n, :, :, 0].astype(np.float64)
    if pn.shape[0] < n:
        raise ValueError(f"run has {pn.shape[0]} test samples, protocol needs {n}")

    raw = read_raw(raw_path, dataset)
    off, mu, sd = align(raw, pn, tn)

    J, _, Q = pn.shape
    idx = off + np.arange(J)[:, None] + np.arange(Q)[None, :]
    targ = np.transpose(raw[idx], (0, 2, 1))
    pred = pn * sd[None, :, None] + mu[None, :, None]
    return pred, targ, targ > 0


def evaluate_per_horizon(pred, targ, valid, mape_min_speed=MAPE_MIN_SPEED):
    out = []
    for q in range(pred.shape[2]):
        m = valid[:, :, q]
        e = pred[:, :, q][m] - targ[:, :, q][m]
        t = targ[:, :, q][m]
        k = t >= mape_min_speed
        out.append(dict(
            step=q + 1, minutes=(q + 1) * 5,
            mae=float(np.abs(e).mean()),
            rmse=float(np.sqrt((e ** 2).mean())),
            mape=float((np.abs(e[k]) / t[k]).mean() * 100),
        ))
    return out


def evaluate(pred, targ, valid, mape_min_speed=MAPE_MIN_SPEED):
    ph = evaluate_per_horizon(pred, targ, valid, mape_min_speed)
    return {k: float(np.mean([h[k] for h in ph])) for k in ("mae", "rmse", "mape")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="Directory produced by train.py")
    ap.add_argument("--dataset", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--raw", required=True,
                    help="Benchmark HDF5 file (metr-la.h5 or pems-bay.h5)")
    ap.add_argument("--per_horizon", action="store_true",
                    help="Also print a breakdown by horizon step")
    ap.add_argument("--json_out", default=None, help="Write the results as JSON")
    args = ap.parse_args()

    pred, targ, valid = load_run(args.run_dir, args.dataset, args.raw)
    avg = evaluate(pred, targ, valid)

    print(f"\n{os.path.basename(os.path.normpath(args.run_dir))}  [{args.dataset}]")
    print(f"  test window : {pred.shape[0]} samples x {pred.shape[1]} sensors "
          f"x {pred.shape[2]} steps")
    print(f"  valid entries: {valid.mean() * 100:.2f}%  "
          f"(zeros excluded as missing readings)")
    print(f"  MAPE mask   : ground truth >= {MAPE_MIN_SPEED:g} mph")
    print(f"\n  {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
    print(f"  {avg['mae']:8.4f} {avg['rmse']:8.4f} {avg['mape']:8.4f}")

    result = dict(run_dir=args.run_dir, dataset=args.dataset,
                  mape_min_speed=MAPE_MIN_SPEED, average=avg)

    if args.per_horizon:
        ph = evaluate_per_horizon(pred, targ, valid)
        result["per_horizon"] = ph
        print(f"\n  {'step':>4s} {'min':>4s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
        for h in ph:
            print(f"  {h['step']:4d} {h['minutes']:4d} "
                  f"{h['mae']:8.4f} {h['rmse']:8.4f} {h['mape']:8.4f}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  wrote {args.json_out}")
    print()


if __name__ == "__main__":
    main()
