"""
Test-set evaluation script for DT-TopoGT on the STAG TCR-pMHC dataset.

Loads a trained checkpoint, runs inference on the test split, then saves:
    results/test_predictions.csv  — per-pair binding scores
    results/metrics.txt           — AUROC, ACC, F1

Usage
─────
python scripts/evaluate_stag.py \\
    --test-dir   path/to/data/STAG_datasplit/complete/test \\
    --checkpoint checkpoints/stag/best_model.pt \\
    --output-dir results \\
    --batch-size 16 \\
    --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_prep.STAG_dataloader import STAGDataset
from src.data_prep.dataset import collate_protein_pairs
from src.features.esm_encoder import ESMEncoder
from src.models.dt_topogt import DTTopoGT, DTTopoGTConfig


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate DT-TopoGT on STAG test set"
    )
    p.add_argument("--test-dir", required=True,
                   help="Path to STAG test split (contains structures/ and *_label.csv)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to trained .pt checkpoint (best_model.pt)")
    p.add_argument("--output-dir", default="results",
                   help="Directory to save predictions CSV and metrics TXT")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=None,
                   help="cuda / cpu (auto-detect if not given)")
    p.add_argument("--no-cache", action="store_true",
                   help="Disable on-disk feature caching")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

# Map d_esm → ESM-2 model name (used when checkpoint doesn't store model name)
_ESM_MAP = {
    320:  "facebook/esm2_t6_8M_UR50D",
    480:  "facebook/esm2_t12_35M_UR50D",
    640:  "facebook/esm2_t30_150M_UR50D",
    1280: "facebook/esm2_t33_650M_UR50D",
}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config: DTTopoGTConfig = ckpt["config"]

    # ── ESM encoder ───────────────────────────────────────────────────────────
    esm_model_name = _ESM_MAP.get(config.d_esm, "facebook/esm2_t6_8M_UR50D")
    print(f"Loading ESM-2: {esm_model_name} (on CPU for data loading)")
    esm_encoder = ESMEncoder(model_name=esm_model_name, device="cpu")

    # ── Dataset ───────────────────────────────────────────────────────────────
    test_dataset = STAGDataset(
        root=args.test_dir,
        esm_encoder=esm_encoder,
        max_edge_length=20.0,
        max_dim=config.topo_max_dim,
        n_landscapes=config.topo_n_landscapes,
        n_points=config.topo_n_points,
        use_cache=not args.no_cache,
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
            y_preds, _, _ = model.forward_batch(list_a, list_b, device)
            all_preds.append(y_preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    pred_binary = (all_preds >= 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, pred_binary)
    f1  = f1_score(all_labels, pred_binary, zero_division=0)

    print("\n── Test Metrics ───────────────────────────────")
    print(f"  AUC : {auc:.4f}")
    print(f"  ACC : {acc:.4f}")
    print(f"  F1  : {f1:.4f}")
    print("────────────────────────────────────────────────")

    # ── Save predictions CSV ──────────────────────────────────────────────────
    csv_path = output_dir / "test_predictions.csv"
    pred_df = pd.DataFrame({
        "file_key":   test_dataset.get_file_keys(),
        "label":      all_labels.astype(int),
        "pred_score": all_preds,
        "pred_label": pred_binary,
    })
    pred_df.to_csv(csv_path, index=False)
    print(f"\nPredictions saved to: {csv_path}")

    # ── Save metrics TXT ──────────────────────────────────────────────────────
    txt_path = output_dir / "metrics.txt"
    with open(txt_path, "w") as f:
        f.write("DT-TopoGT — STAG Test Metrics\n")
        f.write("=" * 35 + "\n")
        f.write(f"Checkpoint : {args.checkpoint}\n")
        f.write(f"Test dir   : {args.test_dir}\n")
        f.write(f"N pairs    : {len(test_dataset)}\n")
        f.write("-" * 35 + "\n")
        f.write(f"AUC        : {auc:.4f}\n")
        f.write(f"ACC        : {acc:.4f}\n")
        f.write(f"F1         : {f1:.4f}\n")
    print(f"Metrics saved to:     {txt_path}")


if __name__ == "__main__":
    main()
