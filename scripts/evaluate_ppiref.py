"""
Test-set evaluation script for DT-TopoGT on the PPIRef dataset.

Loads a trained checkpoint, runs inference on the test split, saves:
    results/test_predictions.csv  — per-pair scores and labels
    results/metrics.txt           — AUROC, ACC, F1, AUPRC

Usage
─────
python scripts/evaluate_ppiref.py \\
    --ppi-dir    path/to/data/PPIRef/ppiref/data/ppiref/ppi_6A \\
    --split-json path/to/data/PPIRef/ppiref/data/splits/ppiref_6A_filtered_clustered_04.json \\
    --checkpoint checkpoints/ppiref/best_model.pt \\
    --output-dir results/ppiref \\
    --batch-size 32 \\
    --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep.PPIRef_dataloader import PPIRefDataset
from src.data_prep.dataset import collate_protein_pairs
from src.features.esm_encoder import ESMEncoder
from src.models.dt_topogt import DTTopoGT, DTTopoGTConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

_ESM_MAP = {
    320:  "facebook/esm2_t6_8M_UR50D",
    480:  "facebook/esm2_t12_35M_UR50D",
    640:  "facebook/esm2_t30_150M_UR50D",
    1280: "facebook/esm2_t33_650M_UR50D",
}


def load_test_keys(split_json: str, seed: int,
                   train_split: float, val_split: float) -> list:
    """
    Return the test-set keys from the split JSON.
    Mirrors the same logic used in train_ppiref.py.
    """
    with open(split_json) as f:
        data = json.load(f)
    folds = data["folds"]

    if "test" in folds:
        return folds["test"]

    keys = folds["whole"]
    rng  = np.random.RandomState(seed)
    indices = rng.permutation(len(keys)).tolist()

    n_train = int(len(keys) * train_split)
    n_val   = int(len(keys) * val_split)
    test_keys = [keys[i] for i in indices[n_train + n_val:]]
    print(f"[split] whole={len(keys)} → test={len(test_keys)}")
    return test_keys


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DT-TopoGT on PPIRef test set"
    )
    p.add_argument("--ppi-dir", required=True,
                   help="Root of ppi_6A directory (two-char prefix subdirs)")
    p.add_argument("--split-json", required=True,
                   help="Same split JSON used for training")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained .pt checkpoint (best_model.pt)")
    p.add_argument("--output-dir", default="results/ppiref",
                   help="Directory for predictions CSV and metrics TXT")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--neg-ratio", type=int, default=1,
                   help="Negatives per positive (must match training)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-split", type=float, default=0.8)
    p.add_argument("--val-split",   type=float, default=0.1)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--gpus", default=None,
                   help="Comma-separated GPU IDs (e.g. 0,1,2,3). "
                        "Single GPU recommended for eval. Overrides --device.")
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect if not given)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: DTTopoGTConfig = ckpt["config"]

    # ── ESM encoder ───────────────────────────────────────────────────────────
    esm_model_name = _ESM_MAP.get(config.d_esm, "facebook/esm2_t6_8M_UR50D")
    print(f"Loading ESM-2: {esm_model_name} (on CPU)")
    esm_encoder = ESMEncoder(model_name=esm_model_name, device="cpu")

    # ── Test split keys ───────────────────────────────────────────────────────
    test_keys = load_test_keys(
        args.split_json, args.seed, args.train_split, args.val_split
    )

    # ── Dataset & loader ──────────────────────────────────────────────────────
    test_dataset = PPIRefDataset(
        pdb_keys=test_keys,
        ppi_dir=args.ppi_dir,
        esm_encoder=esm_encoder,
        max_edge_length=20.0,
        max_dim=config.topo_max_dim,
        n_landscapes=config.topo_n_landscapes,
        n_points=config.topo_n_points,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_protein_pairs,
        num_workers=0,
    )
    print(f"Test pairs: {len(test_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DTTopoGT(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_labels = [], []

    with torch.no_grad():
        for list_a, list_b, labels in test_loader:
            if len(labels) == 0:
                continue
            y_preds, _, _ = model.forward_batch(list_a, list_b, device)
            all_preds.append(y_preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pred_binary = (all_preds >= 0.5).astype(int)

    auc   = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    acc   = accuracy_score(all_labels, pred_binary)
    f1    = f1_score(all_labels, pred_binary, zero_division=0)

    print("\n── Test Metrics ───────────────────────────────────────")
    print(f"  AUROC : {auc:.4f}")
    print(f"  AUPRC : {auprc:.4f}")
    print(f"  ACC   : {acc:.4f}")
    print(f"  F1    : {f1:.4f}")
    print("─────────────────────────────────────────────────────────")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    pair_info = test_dataset.get_pair_keys()
    csv_path  = output_dir / "test_predictions.csv"
    pd.DataFrame({
        "key_a":      [p[0] for p in pair_info],
        "chain_a":    [p[1] for p in pair_info],
        "key_b":      [p[2] for p in pair_info],
        "chain_b":    [p[3] for p in pair_info],
        "label":      all_labels.astype(int),
        "pred_score": all_preds,
        "pred_label": pred_binary,
    }).to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")

    # ── Save metrics TXT ──────────────────────────────────────────────────────
    txt_path = output_dir / "metrics.txt"
    with open(txt_path, "w") as f:
        f.write("DT-TopoGT — PPIRef Test Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint  : {args.checkpoint}\n")
        f.write(f"Split JSON  : {args.split_json}\n")
        f.write(f"N pairs     : {len(test_dataset)}\n")
        f.write(f"neg_ratio   : {args.neg_ratio}\n")
        f.write("-" * 40 + "\n")
        f.write(f"AUROC       : {auc:.4f}\n")
        f.write(f"AUPRC       : {auprc:.4f}\n")
        f.write(f"ACC         : {acc:.4f}\n")
        f.write(f"F1          : {f1:.4f}\n")
    print(f"Metrics saved to:     {txt_path}")


if __name__ == "__main__":
    main()
