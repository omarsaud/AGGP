"""
Diebold-Mariano significance test between two saved runs.

Reproduces the significance table reported in the paper from saved predictions,
without re-running any model. Prediction loading, per-sensor denormalisation, the
canonical test window and the missing-data mask are all taken from evaluate.py,
so the losses compared here are exactly those behind the accuracy tables.

Usage
-----
    # one comparison, all three reported horizons
    python diebold_mariano.py \
        --run_a models/gwnet_aggp_metr-la_Acc_Q12 \
        --run_b models/gwnet_metr-la_Acc_Q12 \
        --dataset metr-la --raw data/metr-la/metr-la.h5

    # a specific horizon and loss
    python diebold_mariano.py --run_a A --run_b B --dataset pems-bay \
        --raw data/pems-bay/pems-bay.h5 --steps 12 --loss mse

Method
------
Loss differential d_t = L(e_a) - L(e_b), averaged over the sensors valid at each
sample, with L absolute error (`mae`) or squared error (`mse`). The variance of
d-bar uses a Newey-West HAC estimator with a Bartlett kernel and bandwidth h-1,
where h is the forecast step. The statistic then carries the
Harvey-Leybourne-Newbold small-sample correction

    DM* = DM * sqrt( (T + 1 - 2h + h(h-1)/T) / T ),

and is referred to a Student t distribution with T-1 degrees of freedom. A
negative statistic favours run A.

References
----------
Diebold & Mariano (1995), Journal of Business & Economic Statistics 13(3).
Harvey, Leybourne & Newbold (1997), International Journal of Forecasting 13(2).
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

from evaluate import PROTOCOL, load_run


def loss_differential(pred_a, pred_b, targ, valid, step, loss="mae"):
    """Per-sample loss differential at one horizon step, over valid sensors only."""
    q = step - 1
    m = valid[:, :, q]
    power = 1 if loss == "mae" else 2
    la = np.abs(pred_a[:, :, q] - targ[:, :, q]) ** power
    lb = np.abs(pred_b[:, :, q] - targ[:, :, q]) ** power
    n_valid = m.sum(axis=1)
    keep = n_valid > 0                      # drop samples with no working sensor
    return ((la - lb) * m).sum(axis=1)[keep] / n_valid[keep]


def dm_test(d, step):
    """HLN-corrected Diebold-Mariano statistic and two-sided p-value."""
    T = len(d)
    d_bar = d.mean()
    gamma0 = np.var(d, ddof=1)

    bandwidth = max(1, step - 1)
    autocov = 0.0
    for k in range(1, bandwidth + 1):
        weight = 1.0 - k / (bandwidth + 1)          # Bartlett kernel
        autocov += 2 * weight * np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))

    var_d = (gamma0 + autocov) / T
    if var_d <= 0:
        return 0.0, 1.0, T

    stat = d_bar / np.sqrt(var_d)
    stat *= np.sqrt((T + 1 - 2 * step + step * (step - 1) / T) / T)   # HLN
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return float(stat), float(p), T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_a", required=True, help="First run (favoured if DM < 0)")
    ap.add_argument("--run_b", required=True, help="Second run")
    ap.add_argument("--dataset", required=True, choices=sorted(PROTOCOL))
    ap.add_argument("--raw", required=True,
                    help="Benchmark HDF5 file (metr-la.h5 or pems-bay.h5)")
    ap.add_argument("--steps", type=int, nargs="+", default=[3, 6, 12],
                    help="Horizon steps to test (default: 3 6 12)")
    ap.add_argument("--loss", choices=["mae", "mse", "both"], default="both")
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    pred_a, targ, valid = load_run(args.run_a, args.dataset, args.raw)
    pred_b, _, _ = load_run(args.run_b, args.dataset, args.raw)

    losses = ["mae", "mse"] if args.loss == "both" else [args.loss]
    name_a = os.path.basename(os.path.normpath(args.run_a))
    name_b = os.path.basename(os.path.normpath(args.run_b))

    print(f"\n{name_a}  vs  {name_b}   [{args.dataset}]")
    print("  negative statistic favours the first run\n")
    print(f"  {'step':>4s} {'min':>4s} {'loss':>5s} {'DM':>9s} {'p':>11s}  sig(5%)")

    rows = []
    for step in args.steps:
        for loss in losses:
            d = loss_differential(pred_a, pred_b, targ, valid, step, loss)
            stat, p, T = dm_test(d, step)
            sig = "yes" if p < 0.05 else "no"
            print(f"  {step:4d} {step * 5:4d} {loss:>5s} {stat:9.4f} {p:11.4g}  {sig}")
            rows.append(dict(step=step, minutes=step * 5, loss=loss,
                             dm_stat=stat, p_value=p, n_points=T,
                             better=name_a if stat < 0 else name_b,
                             significant_5pct=p < 0.05))

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(run_a=args.run_a, run_b=args.run_b,
                           dataset=args.dataset, results=rows), f, indent=2)
        print(f"\n  wrote {args.json_out}")
    print()


if __name__ == "__main__":
    main()
