"""
AGGP Training Script
=====================
Train one AGGP ablation configuration on a single dataset and horizon.

Usage examples:
  # Config 1 — Full AGGP (2ch, gate=accel):
  python train.py --model gwnet_aggp --dataset metr-la --Q 3

  # Config 2 — Gate=speed magnitude (1ch):
  python train.py --model gwnet_aggp_gate_speed --dataset metr-la --Q 3

  # Config 3 — Gate=accel, backbone sees speed only:
  python train.py --model gwnet_aggp_gate_accel --dataset metr-la --Q 3

  # Config 4 — GWNet baseline (no gate):
  python train.py --model gwnet --dataset metr-la --Q 3

Model-to-data mapping:
  gwnet_aggp            → --use_acceleration true   (2ch: speed + accel)
  gwnet_aggp_gate_speed → --use_acceleration false  (1ch: speed only)
  gwnet_aggp_gate_accel → --use_acceleration true   (2ch: accel needed for gate)
  gwnet                 → --use_acceleration true   (2ch: speed + accel)
"""

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from datetime import datetime

from model.aggp import GWNetV14
from data.loader import load_data_simple, create_data_loaders
from utils.metrics import compute_metrics


# ─────────────────────────────────────────────────────────────────────────────
# GPU Optimizations
# ─────────────────────────────────────────────────────────────────────────────
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='AGGP Training')

    parser.add_argument('--model', type=str, required=True,
                        choices=['gwnet_aggp', 'gwnet_aggp_gate_speed',
                                 'gwnet_aggp_gate_accel', 'gwnet'],
                        help='Model / ablation config to train')
    parser.add_argument('--dataset', type=str, default='metr-la',
                        choices=['metr-la', 'pems-bay'])
    parser.add_argument('--Q', type=int, default=3,
                        help='Prediction horizon (3, 6, or 12)')
    parser.add_argument('--H', type=int, default=12,
                        help='Historical window length')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (default: data/{dataset})')
    parser.add_argument('--use_acceleration', type=str, default=None,
                        choices=['true', 'false'],
                        help='Override acceleration flag (auto-set if not provided)')

    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Override output directory for saved model')

    return parser.parse_args()


def build_model(model_name, num_nodes, input_dim, horizon, hidden_dim, dropout):
    """Instantiate the correct AGGP variant."""
    # Determine gate_source and use_accel_gate from model name
    if model_name == 'gwnet_aggp':
        gate_source    = 'auto'
        use_accel_gate = True
    elif model_name == 'gwnet_aggp_gate_speed':
        gate_source    = 'speed'
        use_accel_gate = True
    elif model_name == 'gwnet_aggp_gate_accel':
        gate_source    = 'accel_only'
        use_accel_gate = True
    else:  # 'gwnet' baseline
        gate_source    = 'auto'
        use_accel_gate = False

    model = GWNetV14(
        num_nodes=num_nodes,
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=hidden_dim,
        num_layers=4,
        kernel_size=2,
        dropout=dropout,
        seq_len=12,
        horizon=horizon,
        support_len=2,
        embed_dim=10,
        use_accel_gate=use_accel_gate,
        gate_boost=0.5,
        gate_source=gate_source
    )
    return model


def train_epoch(model, loader, optimizer, criterion, device, adj, scaler):
    model.train()
    total_loss = 0
    num_batches = 0
    adj_t = torch.FloatTensor(adj).to(device) if not isinstance(adj, torch.Tensor) else adj.to(device)

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(x, adj_t)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x, adj_t)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % 100 == 0:
            print(f"   [{batch_idx + 1}/{len(loader)}] loss={total_loss / num_batches:.4f}")

    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, loader, device, adj, norm_params):
    model.eval()
    preds, tgts = [], []
    adj_t = torch.FloatTensor(adj).to(device) if not isinstance(adj, torch.Tensor) else adj.to(device)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        pred = model(x, adj_t)
        preds.append(pred.cpu())
        tgts.append(y)

    preds = torch.cat(preds, dim=0)
    tgts  = torch.cat(tgts,  dim=0)

    speed_mean = norm_params.get('speed_mean') if norm_params else None
    speed_std  = norm_params.get('speed_std')  if norm_params else None

    return compute_metrics(preds, tgts, speed_mean, speed_std, null_val=0.0)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Data config: auto-determine use_acceleration from model name ──
    if args.use_acceleration is not None:
        use_acceleration = args.use_acceleration.lower() == 'true'
    else:
        use_acceleration = (args.model != 'gwnet_aggp_gate_speed')
    input_dim = 2 if use_acceleration else 1

    # ── Data directory ──
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = f"data/{args.dataset}"

    print(f"\n{'='*60}")
    print(f"  Model:   {args.model.upper()}")
    print(f"  Dataset: {args.dataset}  |  Q={args.Q}  |  H={args.H}")
    print(f"  Input:   {input_dim}ch  |  Data: {data_dir}")
    print(f"{'='*60}\n")

    # ── Load data ──
    speed_data, accel_data, adj, norm_params = load_data_simple(
        args.dataset, data_dir=data_dir, use_acceleration=use_acceleration
    )
    if not use_acceleration:
        accel_data = None

    train_loader, val_loader, test_loader = create_data_loaders(
        speed_data, accel_data,
        seq_len=args.H, horizon=args.Q,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # ── Build model ──
    num_nodes = speed_data.shape[1]
    model = build_model(args.model, num_nodes, input_dim, args.Q,
                        args.hidden_dim, args.dropout).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")

    # ── Optimizer & scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    criterion = nn.L1Loss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Output directory ──
    if args.output_dir:
        model_dir = Path(args.output_dir)
    else:
        suffix = 'Acc' if use_acceleration else 'NoAcc'
        model_dir = Path(f"models/{args.model}_{args.dataset}_{suffix}_Q{args.Q}")
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    print(f"\nTraining for {args.epochs} epochs  (patience={args.patience})\n")
    best_val_mae = float('inf')
    patience_counter = 0
    history = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_ep = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 device, adj, scaler)
        val_metrics = evaluate(model, val_loader, device, adj, norm_params)
        val_mae = val_metrics['mae']

        scheduler.step(val_mae)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_mae': val_mae,
            'val_rmse': val_metrics['rmse'],
            'val_mape': val_metrics['mape'],
            'epoch_time': time.time() - t_ep
        })

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"loss={train_loss:.4f} | "
              f"val_mae={val_mae:.4f} | "
              f"{time.time()-t_ep:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), model_dir / 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': best_val_mae,
            }, model_dir / 'best_checkpoint.pt')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Test ──
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_dir / 'best_model.pt'))
    test_metrics = evaluate(model, test_loader, device, adj, norm_params)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"{'='*60}\n")

    # ── Save results ──
    results = {
        'model': args.model,
        'dataset': args.dataset,
        'Q': args.Q,
        'H': args.H,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'test_mae': float(test_metrics['mae']),
        'test_rmse': float(test_metrics['rmse']),
        'test_mape': float(test_metrics['mape']),
        'best_val_mae': float(best_val_mae),
        'num_params': num_params,
        'epochs_trained': epoch - patience_counter,
        'total_epochs': epoch,
        'train_time_minutes': float(total_time / 60),
        'has_denormalization': bool(norm_params and 'speed_mean' in norm_params),
        'speed_mean': norm_params.get('speed_mean') if norm_params else None,
        'speed_std': norm_params.get('speed_std')  if norm_params else None,
        'timestamp': datetime.now().isoformat()
    }

    with open(model_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    import pandas as pd
    pd.DataFrame(history).to_csv(model_dir / 'train_history.csv', index=False)

    model_cfg = {
        'model_name': args.model,
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'num_nodes': num_nodes,
        'horizon': args.Q,
        'num_parameters': num_params
    }
    with open(model_dir / 'model_config.json', 'w') as f:
        json.dump(model_cfg, f, indent=2)

    print(f"Results saved to: {model_dir}/")


if __name__ == '__main__':
    main()
