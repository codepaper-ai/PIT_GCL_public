"""
ProteinPairDataset — PyTorch Dataset for binary PPI prediction.

Expected directory layout
─────────────────────────
dataset/
├── train/
│   ├── structures/          # one PDB or CIF file per unique protein
│   │   ├── <id_A>.pdb
│   │   ├── <id_B>.cif
│   │   └── ...
│   ├── pairs.csv            # columns: id_A, id_B, label  (label ∈ {0, 1})
│   └── cache/               # auto-created; pre-computed per-protein features
└── test/
    ├── structures/
    ├── pairs.csv
    └── cache/

Each item returned by __getitem__ is a tuple:
    (prot_a, prot_b, label)

where prot_a / prot_b are dicts with keys:
    coords    – FloatTensor (N, 3)
    esm_emb   – FloatTensor (N, d_esm)
    topo      – FloatTensor ((max_dim+1) * n_land * n_pts,)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data_prep.pdb_parser import parse_structure
from src.features.esm_encoder import ESMEncoder
from src.features.topology import compute_persistence_landscape, topo_feature_dim


def _find_structure(structures_dir: Path, protein_id: str) -> Path:
    """Locate a PDB/CIF file for protein_id (tries .pdb, .cif, .mmcif)."""
    for ext in (".pdb", ".cif", ".mmcif", ".ent"):
        p = structures_dir / f"{protein_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No structure file for '{protein_id}' in {structures_dir}. "
        f"Tried extensions: .pdb, .cif, .mmcif, .ent"
    )


class ProteinPairDataset(Dataset):
    """
    Args:
        root: Path to the split directory (e.g. dataset/train/)
        esm_encoder: Pre-initialised ESMEncoder instance
        max_edge_length: Vietoris-Rips max edge (Å) for TDA
        max_dim: Maximum homology dimension (0 → H0, 1 → H0+H1)
        n_landscapes: Number of landscape functions per dimension
        n_points: Landscape discretisation resolution
        chain_ids: Chain IDs to include from each structure (None = all)
        use_cache: Whether to read/write per-protein feature caches
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
        self.cache_dir = self.root / "cache" if use_cache else None
        self.esm_encoder = esm_encoder
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.n_landscapes = n_landscapes
        self.n_points = n_points
        self.chain_ids = chain_ids
        self.use_cache = use_cache

        pairs_csv = self.root / "pairs.csv"
        if not pairs_csv.exists():
            raise FileNotFoundError(f"pairs.csv not found in {self.root}")
        self.pairs = pd.read_csv(pairs_csv)

        required_cols = {"id_A", "id_B", "label"}
        if not required_cols.issubset(self.pairs.columns):
            raise ValueError(
                f"pairs.csv must have columns {required_cols}; "
                f"got {set(self.pairs.columns)}"
            )

    # ── Feature computation ───────────────────────────────────────────────────

    def _protein_cache_path(self, protein_id: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{protein_id}.npz"

    def _load_protein(self, protein_id: str) -> dict:
        """
        Return per-protein feature dict (loads from cache if available).
        """
        cache_path = self._protein_cache_path(protein_id)
        if cache_path is not None and cache_path.exists():
            data = np.load(cache_path)
            return {k: data[k] for k in data.files}

        # Parse structure
        struct_path = _find_structure(self.structures_dir, protein_id)
        coords, sequence, _ = parse_structure(str(struct_path), self.chain_ids)

        # ESM-2 embeddings
        esm_emb = self.esm_encoder.encode(sequence)

        # Align sequence length: ESM may truncate long sequences
        n = min(len(coords), esm_emb.shape[0])
        coords = coords[:n]
        esm_emb = esm_emb[:n]

        # Topology
        # Use the protein-level topo cache inside the feature cache dir
        topo_cache = self.cache_dir / "topo" if self.cache_dir else None
        topo = compute_persistence_landscape(
            coords,
            max_edge_length=self.max_edge_length,
            max_dim=self.max_dim,
            n_landscapes=self.n_landscapes,
            n_points=self.n_points,
            cache_dir=topo_cache,
        )

        features = {
            "coords": coords,                              # (N, 3)
            "esm_emb": esm_emb,                           # (N, d_esm)
            "topo": topo,                                  # (topo_dim,)
        }

        if cache_path is not None:
            np.savez(cache_path, **features)

        return features

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        row = self.pairs.iloc[idx]
        id_a, id_b, label = str(row["id_A"]), str(row["id_B"]), int(row["label"])

        feat_a = self._load_protein(id_a)
        feat_b = self._load_protein(id_b)

        def to_tensors(feat: dict) -> dict:
            return {
                "coords":     torch.from_numpy(feat["coords"]),
                "esm_emb":    torch.from_numpy(feat["esm_emb"]),
                "topo":       torch.from_numpy(feat["topo"]),
            }

        return to_tensors(feat_a), to_tensors(feat_b), torch.tensor(label, dtype=torch.long)


def collate_protein_pairs(batch: list) -> tuple:
    """
    Custom collate function for ProteinPairDataset.

    Returns:
        list_a: list of prot_a dicts  (length B)
        list_b: list of prot_b dicts  (length B)
        labels: LongTensor (B,)
    """
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return [], [], torch.zeros(0, dtype=torch.long)
    list_a, list_b, labels = zip(*batch)
    return list(list_a), list(list_b), torch.stack(labels)
