"""
Training script for DT-TopoGT on the STAG TCR-pMHC dataset.

Usage
─────
python scripts/train_stag.py \\
    --train-dir path/to/data/STAG_datasplit/complete/train \\
    --output-dir checkpoints/stag \\
    --epochs 100 \\
    --batch-size 16 \\
    --lr 1e-4 \\
    --d-model 256 \\
    --n-layers 4 \\
    --n-heads 8 \\
    --alpha 0.5 \\
    --beta 0.5 \\
    --esm-model facebook/esm2_t6_8M_UR50D \\
    --device cuda \\
    --seed 42

Optional: pass --val-dir to use an explicit validation split.
If omitted, 10 % of training data is held out automatically.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep.STAG_dataloader import STAGDataset
from src.data_prep.dataset import collate_protein_pairs
from src.features.esm_encoder import ESMEncoder
from src.models.dt_topogt import DTTopoGT, DTTopoGTConfig
from src.utils.losses import ppi_loss
from src.utils.metrics import compute_metrics


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train DT-TopoGT on STAG dataset")

    # Data
    p.add_argument("--train-dir", required=True,
                   help="Path to STAG train split (contains structures/ and *_label.csv)")
    p.add_argument("--val-dir", default=None,
                   help="Path to val split directory (optional; auto-split if omitted)")
    p.add_argument("--val-split", type=float, default=0.1,
                   help="Fraction of train data used as val when --val-dir is not given")
    p.add_argument("--output-dir", default="checkpoints/stag",
                   help="Where to save checkpoints and logs")

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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Contrastive loss weight")
    p.add_argument("--beta", type=float, default=0.5,
                   help="BCE loss weight")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="NT-Xent temperature")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience (epochs without val improvement)")

    # Misc
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect if not given)")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Training / validation epoch ───────────────────────────────────────────────

def run_epoch(model, loader, optimizer, device, alpha, beta, temperature,
              is_train=True):
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_preds = [], []

    for list_a, list_b, labels in loader:
        labels = labels.to(device)

        y_preds, z_as, z_bs = model.forward_batch(list_a, list_b, device)

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

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ESM encoder ───────────────────────────────────────────────────────────
    # ESM runs on CPU to avoid cuBLAS context conflicts when called inside
    # DataLoader.__getitem__. Embeddings are cached to disk after first pass.
    print(f"Loading ESM-2: {args.esm_model} (on CPU for data loading)")
    esm_encoder = ESMEncoder(model_name=args.esm_model, device="cpu")
    d_esm = esm_encoder.hidden_size

    # ── Datasets ──────────────────────────────────────────────────────────────
    dataset_kwargs = dict(
        esm_encoder=esm_encoder,
        max_edge_length=20.0,
        max_dim=args.topo_max_dim,
        n_landscapes=args.topo_n_landscapes,
        n_points=args.topo_n_points,
        use_cache=not args.no_cache,
    )

    full_train = STAGDataset(args.train_dir, **dataset_kwargs)

    if args.val_dir:
        val_dataset   = STAGDataset(args.val_dir, **dataset_kwargs)
        train_dataset = full_train
    else:
        n_val   = max(1, int(len(full_train) * args.val_split))
        n_train = len(full_train) - n_val
        train_dataset, val_dataset = random_split(
            full_train, [n_train, n_val],
            generator=torch.Generator().manual_seed(args.seed),
        )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_protein_pairs, num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_protein_pairs, num_workers=0,
    )
    print(f"Train: {len(train_dataset)} pairs  |  Val: {len(val_dataset)} pairs")

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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Save CLI config
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
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss":   round(val_loss,   5),
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

        if val_auroc > best_val_auroc:
            best_val_auroc   = val_auroc
            patience_counter = 0
            torch.save(
                {
                    "epoch":       epoch,
                    "model_state": model.state_dict(),
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
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config":         config,
            },
            output_dir / "last_checkpoint.pt",
        )

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val AUROC: {best_val_auroc:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
