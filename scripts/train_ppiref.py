"""
Training script for DT-TopoGT on the PPIRef protein-protein interface dataset.

PPIRef provides real PPI interfaces (all positives). This script generates
in-silico negatives by random chain-B permutation within each split.

Split source: ppiref_6A_filtered_clustered_04.json (whole → 80/10/10 random)
  OR supply --split-json with a file that already contains 'train'/'val'/'test'
  folds (e.g. dips_equidock.json).

Usage
─────
python scripts/train_ppiref.py \\
    --ppi-dir   path/to/data/PPIRef/ppiref/data/ppiref/ppi_6A \\
    --split-json path/to/data/PPIRef/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json \\
    --output-dir checkpoints/ppiref \\
    --epochs 50 \\
    --batch-size 32 \\
    --lr 1e-4 \\
    --d-model 256 \\
    --n-layers 4 \\
    --n-heads 8 \\
    --alpha 0.5 \\
    --beta 0.5 \\
    --neg-ratio 1 \\
    --esm-model facebook/esm2_t6_8M_UR50D \\
    --gpus 0,1,2,3 \\
    --seed 42
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep.PPIRef_dataloader import PPIRefDataset
from src.data_prep.dataset import collate_protein_pairs
from src.features.esm_encoder import ESMEncoder
from src.models.dt_topogt import DTTopoGT, DTTopoGTConfig
from src.utils.losses import ppi_loss
from src.utils.metrics import compute_metrics


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DT-TopoGT on PPIRef dataset")

    # Data
    p.add_argument("--ppi-dir", required=True,
                   help="Root of ppi_6A directory (contains two-char prefix subdirs)")
    p.add_argument("--split-json", required=True,
                   help="Path to PPIRef split JSON "
                        "(ppiref_6A_filtered_clustered_04.json or dips_equidock.json)")
    p.add_argument("--train-split", type=float, default=0.8,
                   help="Fraction of 'whole' fold used for training "
                        "(only when the JSON has a single 'whole' fold)")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of 'whole' fold used for validation")
    p.add_argument("--neg-ratio", type=int, default=1,
                   help="Number of random-negative pairs per positive")
    p.add_argument("--output-dir", default="checkpoints/ppiref",
                   help="Where to save checkpoints and logs")
    p.add_argument("--cache-dir", default=None,
                   help="Override per-chain feature cache directory")
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training keys to this many (for quick smoke-tests)")

    # Model
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--ffn-dim", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--esm-model", default="facebook/esm2_t6_8M_UR50D",
                   help="HuggingFace ESM-2 model identifier")

    # Features
    p.add_argument("--topo-max-dim", type=int, default=1)
    p.add_argument("--topo-n-landscapes", type=int, default=3)
    p.add_argument("--topo-n-points", type=int, default=50)
    p.add_argument("--no-cache", action="store_true",
                   help="Disable on-disk feature caching")

    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Contrastive loss weight")
    p.add_argument("--beta", type=float, default=0.5,
                   help="BCE loss weight")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="NT-Xent temperature")
    p.add_argument("--patience", type=int, default=10,
                   help="Early stopping patience (epochs without val improvement)")

    # Hardware
    p.add_argument("--gpus", default="0,1,2,3",
                   help="Comma-separated GPU IDs to use (e.g. 0,1,2,3). "
                        "Pass a single ID for single-GPU. Use 'cpu' to force CPU.")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Split loading ─────────────────────────────────────────────────────────────

def load_splits(args) -> tuple:
    """
    Returns (train_keys, val_keys, test_keys) as lists of PPIRef keys.

    Handles two JSON formats:
      1. 'whole' fold only → random 80/10/10 partition
      2. 'train' / 'val' / 'test' folds already present
    """
    with open(args.split_json) as f:
        data = json.load(f)
    folds = data["folds"]

    if "train" in folds and "val" in folds and "test" in folds:
        print("[split] Using pre-defined train/val/test folds from JSON.")
        return folds["train"], folds["val"], folds["test"]

    # Single 'whole' fold → random split
    keys = folds.get("whole", [])
    if not keys:
        raise ValueError("Split JSON must have 'whole', 'train', or 'val'/'test' folds.")

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(keys)).tolist()

    n_train = int(len(keys) * args.train_split)
    n_val   = int(len(keys) * args.val_split)

    train_keys = [keys[i] for i in indices[:n_train]]
    val_keys   = [keys[i] for i in indices[n_train:n_train + n_val]]
    test_keys  = [keys[i] for i in indices[n_train + n_val:]]

    print(f"[split] whole={len(keys)} → "
          f"train={len(train_keys)}, val={len(val_keys)}, test={len(test_keys)}")
    return train_keys, val_keys, test_keys


# ── Training / validation epoch ───────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, alpha, beta, temperature,
              is_train=True):
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_preds = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for list_a, list_b, labels in loader:
            if len(labels) == 0:   # all items in batch were skipped (None)
                continue
            labels = labels.to(device)

            fwd = model.module if isinstance(model, nn.DataParallel) else model
            y_preds, z_as, z_bs = fwd.forward_batch(list_a, list_b, device)
            loss, _ = ppi_loss(y_preds, z_as, z_bs, labels, alpha, beta, temperature)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(y_preds.detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds  = np.concatenate(all_preds)
    avg_loss   = total_loss / max(len(all_labels), 1)

    metrics = {}
    if len(np.unique(all_labels)) > 1:
        metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # GPU setup
    if args.gpus.lower() == "cpu":
        device = torch.device("cpu")
        gpu_ids = []
    else:
        gpu_ids = [int(g) for g in args.gpus.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Primary device : {device}")
    if gpu_ids:
        print(f"GPU IDs        : {gpu_ids}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Splits ────────────────────────────────────────────────────────────────
    train_keys, val_keys, _ = load_splits(args)

    if args.max_train is not None:
        rng = np.random.RandomState(args.seed)
        train_keys = rng.choice(train_keys, size=min(args.max_train, len(train_keys)),
                                replace=False).tolist()
        print(f"[--max-train] capped training to {len(train_keys)} keys")

    # ── ESM encoder ───────────────────────────────────────────────────────────
    print(f"Loading ESM-2: {args.esm_model} (on CPU for data loading)")
    esm_encoder = ESMEncoder(model_name=args.esm_model, device="cpu")
    d_esm = esm_encoder.hidden_size

    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        ppi_dir=args.ppi_dir,
        esm_encoder=esm_encoder,
        max_edge_length=20.0,
        max_dim=args.topo_max_dim,
        n_landscapes=args.topo_n_landscapes,
        n_points=args.topo_n_points,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )

    train_dataset = PPIRefDataset(pdb_keys=train_keys, **dataset_kwargs)
    val_dataset   = PPIRefDataset(pdb_keys=val_keys,   **dataset_kwargs)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_protein_pairs, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_protein_pairs, num_workers=0,
    )
    print(f"Train pairs: {len(train_dataset)}  |  Val pairs: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    config = DTTopoGTConfig(
        d_esm=d_esm,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        topo_max_dim=args.topo_max_dim,
        topo_n_landscapes=args.topo_n_landscapes,
        topo_n_points=args.topo_n_points,
        cross_n_heads=args.n_heads,
    )
    model = DTTopoGT(config).to(device)
    if torch.cuda.is_available() and len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))
        print(f"Using DataParallel across {len(gpu_ids)} GPUs")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_auroc   = -1.0
    patience_counter = 0
    history          = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, optimizer, device,
            args.alpha, args.beta, args.temperature, is_train=True,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, optimizer, device,
            args.alpha, args.beta, args.temperature, is_train=False,
        )
        scheduler.step()

        val_auroc = val_metrics.get("auroc", 0.0)
        row = {
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "val_loss":    round(val_loss,   5),
            **{f"train_{k}": round(v, 4) for k, v in train_metrics.items()},
            **{f"val_{k}":   round(v, 4) for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auroc={val_auroc:.4f}"
        )

        save_state = model.module if isinstance(model, nn.DataParallel) else model
        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": save_state.state_dict(),
                    "config":      config,
                    "val_auroc":   val_auroc,
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

        torch.save(
            {
                "epoch":           epoch,
                "model_state":     save_state.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":          config,
            },
            output_dir / "last_checkpoint.pt",
        )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val AUROC: {best_val_auroc:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
