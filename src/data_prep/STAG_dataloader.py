"""
STAGDataset — Dataset adapter for the STAG TCR-pMHC dataset.

Expected directory layout
─────────────────────────
dataset/
├── train/
│   ├── structures/
│   │   ├── <file_key>_TCR.pdb
│   │   ├── <file_key>_pMHC.pdb
│   │   └── ...
│   └── train_label.csv          # columns: file_key, index, ..., label
└── test/
    ├── structures/
    │   └── ...
    └── test_label.csv

The label CSV must contain at minimum:
    file_key  (str)  — matches {file_key}_TCR.pdb / {file_key}_pMHC.pdb
    label     (int)  — 1 = binding, 0 = non-binding

__getitem__ returns the same (prot_a, prot_b, label) tuple as
ProteinPairDataset, so STAGDataset is a drop-in replacement and works
with collate_protein_pairs unchanged.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_prep.pdb_parser import parse_structure
from src.features.esm_encoder import ESMEncoder
from src.features.topology import compute_persistence_landscape


def _find_label_csv(root: Path) -> Path:
    """Auto-detect the *_label.csv file in a split directory."""
    candidates = sorted(root.glob("*_label.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No *_label.csv file found in {root}. "
            "Expected train_label.csv or test_label.csv."
        )
    return candidates[0]


class STAGDataset(Dataset):
    """
    Dataset for the STAG TCR-pMHC benchmark.

    Each row in the label CSV corresponds to one TCR-pMHC pair:
        protein A  →  {structures_dir}/{file_key}_TCR.pdb
        protein B  →  {structures_dir}/{file_key}_pMHC.pdb

    Args:
        root:            Path to the split directory (e.g. .../train/)
        esm_encoder:     Pre-initialised ESMEncoder instance
        max_edge_length: Vietoris-Rips max edge length (Å) for TDA
        max_dim:         Maximum homology dimension (0 → H0, 1 → H0+H1)
        n_landscapes:    Number of landscape functions per dimension
        n_points:        Landscape discretisation resolution
        chain_ids:       Chain IDs to include from each structure (None = all)
        use_cache:       Whether to read/write per-protein feature caches
    """

    def __init__(
        self,
        root: str,
        esm_encoder: ESMEncoder,
        max_edge_length: float = 20.0,
        max_dim: int = 1,
        n_landscapes: int = 3,
        n_points: int = 50,
        chain_ids: Optional[list] = None,
        use_cache: bool = True,
    ):
        self.root = Path(root)
        self.structures_dir = self.root / "structures"
        if not self.structures_dir.exists():
            raise FileNotFoundError(
                f"structures/ directory not found in {self.root}"
            )

        self.cache_dir = (self.root / "cache") if use_cache else None
        self.esm_encoder = esm_encoder
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.n_landscapes = n_landscapes
        self.n_points = n_points
        self.chain_ids = chain_ids
        self.use_cache = use_cache

        csv_path = _find_label_csv(self.root)
        self.df = pd.read_csv(csv_path)

        for col in ("file_key", "label"):
            if col not in self.df.columns:
                raise ValueError(
                    f"Label CSV must contain a '{col}' column. "
                    f"Found: {list(self.df.columns)}"
                )

        # Reset index for clean iloc access
        self.df = self.df.reset_index(drop=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, protein_key: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{protein_key}.npz"

    def _load_protein(self, protein_key: str, pdb_path: Path) -> dict:
        """
        Return per-protein feature dict (loads from disk cache if available).

        Args:
            protein_key: Unique cache key, e.g. 'pan_peptide_7461_TCR'
            pdb_path:    Full path to the .pdb file
        """
        cache_file = self._cache_path(protein_key)
        if cache_file is not None and cache_file.exists():
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}

        # Parse Cα coords + sequence
        coords, sequence, _ = parse_structure(str(pdb_path), self.chain_ids)

        # ESM-2 embeddings
        esm_emb = self.esm_encoder.encode(sequence)

        # Align length (ESM truncates at max_length=1022)
        n = min(len(coords), esm_emb.shape[0])
        coords = coords[:n]
        esm_emb = esm_emb[:n]

        # Topology
        topo_cache = (self.cache_dir / "topo") if self.cache_dir else None
        topo = compute_persistence_landscape(
            coords,
            max_edge_length=self.max_edge_length,
            max_dim=self.max_dim,
            n_landscapes=self.n_landscapes,
            n_points=self.n_points,
            cache_dir=topo_cache,
        )

        features = {
            "coords":  coords,    # (N, 3)
            "esm_emb": esm_emb,   # (N, d_esm)
            "topo":    topo,      # (topo_dim,)
        }

        if cache_file is not None:
            np.savez(cache_file, **features)

        return features

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        file_key = str(row["file_key"])
        label = int(row["label"])

        tcr_path  = self.structures_dir / f"{file_key}_TCR.pdb"
        pmhc_path = self.structures_dir / f"{file_key}_pMHC.pdb"

        if not tcr_path.exists():
            raise FileNotFoundError(f"TCR structure not found: {tcr_path}")
        if not pmhc_path.exists():
            raise FileNotFoundError(f"pMHC structure not found: {pmhc_path}")

        feat_tcr  = self._load_protein(f"{file_key}_TCR",  tcr_path)
        feat_pmhc = self._load_protein(f"{file_key}_pMHC", pmhc_path)

        def to_tensors(feat: dict) -> dict:
            return {
                "coords":  torch.from_numpy(feat["coords"]),
                "esm_emb": torch.from_numpy(feat["esm_emb"]),
                "topo":    torch.from_numpy(feat["topo"]),
            }

        return to_tensors(feat_tcr), to_tensors(feat_pmhc), torch.tensor(label, dtype=torch.long)

    def get_file_keys(self) -> list:
        """Return the list of file_keys in dataset order (useful for saving predictions)."""
        return self.df["file_key"].tolist()

    def get_labels(self) -> list:
        """Return the list of labels in dataset order."""
        return self.df["label"].tolist()
