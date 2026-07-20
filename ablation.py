"""
AGGP Ablation Study — Full Run
================================
Runs all 4 AGGP ablation configurations sequentially on all datasets and horizons.

Configurations:
  Config 1: gwnet_aggp            | 2ch [speed, accel] | gate=acceleration  (Full AGGP)
  Config 2: gwnet_aggp_gate_speed | 1ch [speed]        | gate=speed magnitude
  Config 3: gwnet_aggp_gate_accel | 2ch data, 1ch model| gate=acceleration only
  Config 4: gwnet (baseline)      | 2ch [speed, accel] | no gate

Usage:
  python ablation.py                   # all 4 configs, both datasets, Q=3/6/12
  python ablation.py --dataset metr-la # single dataset
"""

import subprocess
import sys
import time
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here before running
# ─────────────────────────────────────────────────────────────────────────────

DATASETS = ['metr-la', 'pems-bay']
HORIZONS = [3, 6, 12]

RUN_CONFIG_1 = True   # gwnet_aggp            — Full AGGP
RUN_CONFIG_2 = True   # gwnet_aggp_gate_speed — Gate=speed magnitude
RUN_CONFIG_3 = True   # gwnet_aggp_gate_accel — Gate=accel, model sees speed only
RUN_CONFIG_4 = True   # gwnet                 — Baseline (no gate)

HIDDEN_DIM    = 64
BATCH_SIZE    = 64
LEARNING_RATE = 0.001
EPOCHS        = 100
PATIENCE      = 15
DROPOUT       = 0.3
SEED          = 42
NUM_WORKERS   = 4     # Set to 0 for Windows; Linux auto-uses 8

# ─────────────────────────────────────────────────────────────────────────────


def get_data_dir(dataset, use_acceleration):
    """Resolve data directory path."""
    return f"data/{dataset}"


def build_experiments():
    experiments = []

    for dataset in DATASETS:
        for Q in HORIZONS:
            if RUN_CONFIG_1:
                experiments.append({
                    'name': f"C1_AGGP_{dataset}_Q{Q}",
                    'model': 'gwnet_aggp',
                    'dataset': dataset, 'Q': Q,
                    'use_acceleration': True,
                    'data_dir': get_data_dir(dataset, True),
                    'desc': 'Config 1 — Full AGGP (2ch, gate=accel)'
                })

            if RUN_CONFIG_2:
                experiments.append({
                    'name': f"C2_GateSpeed_{dataset}_Q{Q}",
                    'model': 'gwnet_aggp_gate_speed',
                    'dataset': dataset, 'Q': Q,
                    'use_acceleration': False,
                    'data_dir': get_data_dir(dataset, False),
                    'desc': 'Config 2 — Gate=speed magnitude (1ch)'
                })

            if RUN_CONFIG_3:
                experiments.append({
                    'name': f"C3_GateAccel_{dataset}_Q{Q}",
                    'model': 'gwnet_aggp_gate_accel',
                    'dataset': dataset, 'Q': Q,
                    'use_acceleration': True,
                    'data_dir': get_data_dir(dataset, True),
                    'desc': 'Config 3 — Gate=accel, backbone sees speed only'
                })

            if RUN_CONFIG_4:
                experiments.append({
                    'name': f"C4_Baseline_{dataset}_Q{Q}",
                    'model': 'gwnet',
                    'dataset': dataset, 'Q': Q,
                    'use_acceleration': True,
                    'data_dir': get_data_dir(dataset, True),
                    'desc': 'Config 4 — GWNet baseline (no gate)'
                })

    return experiments


def run_experiment(exp):
    cmd = [
        sys.executable, 'train.py',
        '--model', exp['model'],
        '--dataset', exp['dataset'],
        '--Q', str(exp['Q']),
        '--H', '12',
        '--hidden_dim', str(HIDDEN_DIM),
        '--batch_size', str(BATCH_SIZE),
        '--lr', str(LEARNING_RATE),
        '--epochs', str(EPOCHS),
        '--patience', str(PATIENCE),
        '--dropout', str(DROPOUT),
        '--seed', str(SEED),
        '--data_dir', exp['data_dir'],
        '--use_acceleration', 'true' if exp['use_acceleration'] else 'false',
        '--num_workers', str(NUM_WORKERS),
    ]

    print(f"\n{'='*70}")
    print(f"  {exp['desc']}")
    print(f"  Model: {exp['model']}  |  Dataset: {exp['dataset']}  |  Q={exp['Q']}")
    print(f"{'='*70}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=os.getcwd())
    elapsed = time.time() - t0

    ok = result.returncode == 0
    status = "✅ SUCCESS" if ok else "❌ FAILED"
    print(f"\n  {status} — {exp['name']} ({elapsed/60:.1f} min)")
    return ok


def main():
    print("=" * 70)
    print("  AGGP ABLATION STUDY")
    print("  4 Configurations × Datasets × Horizons")
    print("=" * 70)

    experiments = build_experiments()

    print(f"\n  Total experiments : {len(experiments)}")
    print(f"  Datasets          : {DATASETS}")
    print(f"  Horizons          : {HORIZONS}")
    print(f"\n  Experiment plan:")
    print(f"  {'-'*66}")
    for i, exp in enumerate(experiments, 1):
        ch = '2ch' if exp['use_acceleration'] else '1ch'
        print(f"  {i:2d}. [{ch}] {exp['name']}")
        print(f"       → {exp['desc']}")
    print(f"  {'-'*66}\n")

    results = []
    t_total = time.time()

    for i, exp in enumerate(experiments, 1):
        print(f"\n\n{'#'*70}")
        print(f"  EXPERIMENT {i} / {len(experiments)}")
        print(f"{'#'*70}")
        ok = run_experiment(exp)
        results.append((exp['name'], ok))

    elapsed_total = time.time() - t_total
    print(f"\n\n{'='*70}")
    print(f"  ABLATION COMPLETE — {elapsed_total/3600:.1f} hours")
    print(f"{'='*70}")
    for name, ok in results:
        print(f"  {'✅' if ok else '❌'} {name}")

    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed} / {len(results)} experiments succeeded")
    print("=" * 70)


if __name__ == '__main__':
    main()
