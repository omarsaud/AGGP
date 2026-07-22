"""
Evaluate saved predictions under the paper's reporting protocol.

Reproduces the averaged and per-horizon numbers reported in the paper from a
run directory produced by train.py, without re-running the model.

Usage
-----
    python evaluate.py --run_dir models/gwnet_aggp_metr-la_Acc_Q12 --dataset metr-la
    python evaluate.py --run_dir <dir> --dataset pems-bay --per_horizon

Protocol
--------
* Test window   : the first N samples of the test split
                  (METR-LA N=6784, PEMS-BAY N=10368), so every configuration is
                  scored on an identical window.
* Denormalization: a single dataset-level mean/std, applied to predictions and
                  targets alike, giving speeds in mph.
* MAE           : computed per horizon step, then averaged over the 12 steps.
* RMSE          : computed over all entries at once.
* MAPE          : excludes ground-truth speeds below 5 mph (see utils/metrics).
"""

import argparse
import json
import os

import numpy as np

from utils.metrics import MAPE_MIN_SPEED

# Dataset-level denormalization statistics and canonical test-window length.
PROTOCOL = {
    "metr-la":  dict(mean=53.7190211024135, std=20.261430789619176, n=6784),
    "pems-bay": dict(mean=62.6196,          std=9.594371081990628,  n=10368),
}


def load_run(run_dir, dataset):
    """Load a run's predictions/targets and return them denormalized to mph."""
    cfg = PROTOCOL[dataset]
    pred_f = os.path.join(run_dir, "predictions_normalized.npy")
    targ_f = os.path.join(run_dir, "targets_normalized.npy")
    for f in (pred_f, targ_f):
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing {f}")

    pred = np.load(pred_f)[: cfg["n"]]
    targ = np.load(targ_f)[: cfg["n"]]
    if pred.shape[0] < cfg["n"]:
        raise ValueError(
            f"run has {pred.shape[0]} test samples, protocol needs {cfg['n']}"
        )
    return pred * cfg["std"] + cfg["mean"], targ * cfg["std"] + cfg["mean"]


def evaluate(pred, targ, mape_min_speed=MAPE_MIN_SPEED):
    """Averaged metrics under the paper's conventions."""
    q_steps = pred.shape[2]
    mae = float(np.mean([np.abs(pred[:, :, q] - targ[:, :, q]).mean()
                         for q in range(q_steps)]))
    rmse = float(np.sqrt(((pred - targ) ** 2).mean()))
    m = targ >= mape_min_speed
    mape = float((np.abs(targ[m] - pred[m]) / targ[m]).mean() * 100)
    return dict(mae=mae, rmse=rmse, mape=mape)


def evaluate_per_horizon(pred, targ, mape_min_speed=MAPE_MIN_SPEED):
    """Metrics at each of the Q horizon steps."""
    out = []
    for q in range(pred.shape[2]):
        p, t = pred[:, :, q], targ[:, :, q]
        m = t >= mape_min_speed
        out.append(dict(
            step=q + 1,
            minutes=(q + 1) * 5,
            mae=float(np.abs(p - t).mean()),
            rmse=float(np.sqrt(((p - t) ** 2).mean())),
            mape=float((np.abs(t[m] - p[m]) / t[m]).mean() * 100),
        ))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True,
                    help="Directory produced by train.py")
    ap.add_argument("--dataset", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--per_horizon", action="store_true",
                    help="Also print a breakdown by horizon step")
    ap.add_argument("--json_out", default=None,
                    help="Optional path to write the results as JSON")
    args = ap.parse_args()

    pred, targ = load_run(args.run_dir, args.dataset)
    avg = evaluate(pred, targ)

    print(f"\n{os.path.basename(os.path.normpath(args.run_dir))}  [{args.dataset}]")
    print(f"  test window : {pred.shape[0]} samples x {pred.shape[1]} sensors "
          f"x {pred.shape[2]} steps")
    print(f"  MAPE mask   : ground truth >= {MAPE_MIN_SPEED:g} mph")
    print(f"\n  {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
    print(f"  {avg['mae']:8.4f} {avg['rmse']:8.4f} {avg['mape']:8.4f}")

    result = dict(run_dir=args.run_dir, dataset=args.dataset,
                  mape_min_speed=MAPE_MIN_SPEED, average=avg)

    if args.per_horizon:
        ph = evaluate_per_horizon(pred, targ)
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
