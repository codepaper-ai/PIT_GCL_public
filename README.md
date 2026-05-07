# PIT-GCL: Protein-Interaction Topology Graph Contrastive Learning

This repository is the official implementation of **PIT-GCL**, a topology-aware
dual-tower graph transformer for protein-protein interaction (PPI) prediction.
Each protein is encoded with three complementary signals:

- **Sequence semantics** — ESM-2 residue embeddings
- **Local geometry** — pairwise Cα distances enter as a learned RBF attention bias
- **Global topology** — persistence landscapes (Vietoris–Rips, dim 0–1)

The two towers are fused via latent-space cross-attention and trained with a
combined contrastive + BCE objective.

## Requirements

```bash
conda create -n pitgcl python=3.10 -y
conda activate pitgcl
pip install -r requirements.txt
```

The code has been tested on Linux with CUDA 11.8 and 4× NVIDIA GPUs.
GUDHI is required for persistence-landscape computation; install via pip
(`gudhi>=3.8.0`) or conda.

## Data

The model expects PDB/CIF structures organised by dataset. Download the
public datasets used in our experiments and place them under a directory of
your choice; pass the path via `--ppi-dir` / `--train-dir` / `--test-dir`.

| Dataset  | Source |
|----------|--------|
| PPIRef   | https://github.com/anton-bushuiev/PPIRef |
| STAG     | https://github.com/KavrakiLab/STAG_public |

Replace the placeholder `path/to/data/` in the example commands below with
your local data root.

## Training

PPIRef (binary PPI on real interfaces with random-permutation negatives):

```bash
python scripts/train_ppiref.py \
    --ppi-dir   path/to/data/PPIRef/ppi_6A \
    --split-json path/to/data/PPIRef/splits/ppiref_6A_filtered_clustered_04.json \
    --output-dir checkpoints/ppiref \
    --epochs 50 --batch-size 32 --lr 1e-4 \
    --d-model 256 --n-layers 4 --n-heads 8 \
    --alpha 0.5 --beta 0.5 --neg-ratio 1 \
    --esm-model facebook/esm2_t6_8M_UR50D \
    --gpus 0,1,2,3 --seed 42
```

STAG (TCR-pMHC binding):

```bash
python scripts/train_stag.py \
    --train-dir path/to/data/STAG_datasplit/complete/train \
    --output-dir checkpoints/stag \
    --epochs 100 --batch-size 16 --gpus 0,1,2,3
```

## Evaluation

```bash
python scripts/evaluate_ppiref.py \
    --ppi-dir   path/to/data/PPIRef/ppi_6A \
    --split-json path/to/data/PPIRef/splits/ppiref_6A_filtered_clustered_04.json \
    --checkpoint checkpoints/ppiref/best_model.pt \
    --output-dir results/ppiref \
    --batch-size 32 --device cuda
```

```bash
python scripts/evaluate_stag.py \
    --test-dir   path/to/data/STAG_datasplit/complete/test \
    --checkpoint checkpoints/stag/best_model.pt \
    --output-dir results/STAG_split \
    --device cuda
```

Each evaluation script writes:
- `test_predictions.csv` — per-pair `(label, pred_score, pred_label)` table
- `metrics.txt` — AUROC, AUPRC, ACC, F1 (with default threshold 0.5)

## Pre-trained Models

Two checkpoints used for the results below are included under `checkpoints/`:

| Dataset | Path                              | Config                                |
|---------|-----------------------------------|---------------------------------------|
| PPIRef  | `checkpoints/ppiref/best_model.pt`| `checkpoints/ppiref/config.json`      |
| STAG    | `checkpoints/stag/best_model.pt`  | `checkpoints/stag/config.json`        |

Both were trained with ESM-2-8M, `d_model=256`, 4 transformer layers, 8 heads,
persistence landscapes with `max_dim=1, n_landscapes=3, n_points=50`.

## Results

Test-set performance (default threshold = 0.5):

| Dataset | AUROC  | AUPRC  | ACC    | F1     | RMSE   |
|---------|:------:|:------:|:------:|:------:|:------:|
| PPIRef  | 0.9325 | 0.9377 | 0.8480 | 0.8469 | 0.3302 |
| STAG    | 0.8725 | 0.7160 | 0.8633 | 0.6999 | 0.3428 |

Threshold sensitivity (ACC / F1 across decision thresholds):

| Threshold | PPIRef ACC | PPIRef F1 | STAG ACC | STAG F1 |
|-----------|:----------:|:---------:|:--------:|:-------:|
| 0.1       | 0.7891     | 0.8198    | 0.8351   | 0.6844  |
| 0.3       | 0.8372     | 0.8473    | 0.8537   | 0.6963  |
| 0.5       | 0.8480     | 0.8469    | 0.8633   | 0.6999  |
| 0.7       | 0.8407     | 0.8285    | 0.8636   | 0.6846  |
| 0.9       | 0.8124     | 0.7777    | 0.8641   | 0.6555  |

The raw per-pair predictions used to compute these tables are provided under
`results/{ppiref,STAG_split}/test_predictions.csv` for independent verification.

## Repository Layout

```
src/
  data_prep/       # Dataset adapters: PPIRef, STAG
  features/        # ESM-2 encoder, persistence landscapes
  models/          # Dual-tower graph transformer + cross-attention
  utils/           # Loss functions and metrics
scripts/           # train_{ppiref,stag}.py / evaluate_{ppiref,stag}.py
results/           # Per-dataset test_predictions.csv and metrics.txt
checkpoints/       # Pretrained best_model.pt + config.json per dataset
```

## License

This project is released under the MIT License (see `LICENSE`).

<!-- ## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{pitgcl2026,
  title  = {PIT-GCL: Protein-Interaction Topology Graph Contrastive Learning},
  author = {Anonymous},
  year   = {2026}
}
``` -->
