"""
PPIRefDataset — Dataset adapter for the PPIRef protein-protein interface dataset.

Data layout
───────────
PPIRef PDB files are stored as:
    {ppi_6A_dir}/{key[1:3]}/{key}.pdb

where key = '{pdb_id}_{chain_a}_{chain_b}' (e.g. '1me5_A_B').
Each PDB contains the interface residues of BOTH chains within the extraction
radius (6 Å by default).

Since every PPIRef entry is a *real* interface (all positives), this loader
generates hard-negative pairs by permuting chain-B assignments:
    positive:  chain_A_i  +  chain_B_i  →  label = 1
    negative:  chain_A_i  +  chain_B_j  →  label = 0,  j ≠ i

neg_ratio controls how many negatives are created per positive (default 1).

__getitem__ returns (prot_a, prot_b, label) compatible with
collate_protein_pairs and the rest of the DT-TopoGT pipeline.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_prep.pdb_parser import parse_structure
from src.features.esm_encoder import ESMEncoder
from src.features.topology import compute_persistence_landscape


def _pdb_path(base_dir: Path, key: str) -> Path:
    """Return the full PDB path for a given PPIRef key."""
    return base_dir / key[1:3] / f"{key}.pdb"


def _resolve_chain_id(pdb_file: Path, chain_id: str) -> str:
    """
    Return the actual chain ID present in the PDB file, resolving
    case mismatches between the filename and ATOM records.

    PPIRef filenames sometimes use uppercase (e.g. 'Z') while the
    PDB ATOM records use lowercase (e.g. 'z').

    Raises ValueError if no matching chain is found.
    """
    actual_ids = set()
    with open(pdb_file) as fh:
        for line in fh:
            if line.startswith("ATOM"):
                actual_ids.add(line[21])
    if chain_id in actual_ids:
        return chain_id
    # Case-insensitive fallback
    lower = chain_id.lower()
    upper = chain_id.upper()
    if lower in actual_ids:
        return lower
    if upper in actual_ids:
        return upper
    raise ValueError(
        f"Chain '{chain_id}' not found in {pdb_file}. "
        f"Available chains: {sorted(actual_ids)}"
    )


def _parse_key(key: str):
    """
    Split 'pdbid_chainA_chainB' → (pdb_id, chain_a_id, chain_b_id).
    Uses rsplit so pdb_ids with underscores are handled correctly.
    """
    parts = key.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Cannot parse PPIRef key: {key!r}")
    return parts[0], parts[1], parts[2]


class PPIRefDataset(Dataset):
    """
    Dataset for PPIRef protein-protein interface prediction.

    Args:
        pdb_keys:        List of PPIRef keys (e.g. ['1me5_A_B', ...])
        ppi_dir:         Root directory that contains the two-char prefix subdirs
                         (e.g. /…/ppiref/data/ppiref/ppi_6A)
        esm_encoder:     Pre-initialised ESMEncoder
        max_edge_length: Vietoris-Rips radius (Å) for TDA
        max_dim:         Max homology dimension (0 → H0, 1 → H0+H1)
        n_landscapes:    Number of landscape functions per dimension
        n_points:        Landscape discretisation resolution
        use_cache:       Cache per-chain feature arrays to disk
        cache_dir:       Directory for .npz cache files (None → <ppi_dir>/cache)
        neg_ratio:       Number of negatives generated per positive (default 1)
        seed:            RNG seed for negative permutation
    """

    def __init__(
        self,
        pdb_keys: List[str],
        ppi_dir: str,
        esm_encoder: ESMEncoder,
        max_edge_length: float = 20.0,
        max_dim: int = 1,
        n_landscapes: int = 3,
        n_points: int = 50,
        use_cache: bool = True,
        cache_dir: Optional[str] = None,
        neg_ratio: int = 1,
        seed: int = 42,
    ):
        self.ppi_dir = Path(ppi_dir)
        self.esm_encoder = esm_encoder
        self.max_edge_length = max_edge_length
        self.max_dim = max_dim
        self.n_landscapes = n_landscapes
        self.n_points = n_points
        self.use_cache = use_cache
        self.neg_ratio = neg_ratio

        # Validate that PDB files exist; warn and drop missing ones
        valid_keys = []
        for key in pdb_keys:
            p = _pdb_path(self.ppi_dir, key)
            if p.exists():
                valid_keys.append(key)
            else:
                import warnings
                warnings.warn(f"[PPIRefDataset] Missing PDB, skipping: {p}")
        n_dropped = len(pdb_keys) - len(valid_keys)
        if n_dropped:
            print(f"[PPIRefDataset] Dropped {n_dropped} missing keys.")
        self.pdb_keys = valid_keys

        # Cache directory
        if use_cache:
            self._cache_dir = Path(cache_dir) if cache_dir else (self.ppi_dir / "cache")
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = None

        # ── Build pair list ───────────────────────────────────────────────────
        # Each item: (key_a, chain_a_id, key_b, chain_b_id, label)
        self._pairs: List[tuple] = []

        # Positive pairs (one per key)
        for key in self.pdb_keys:
            _, chain_a, chain_b = _parse_key(key)
            self._pairs.append((key, chain_a, key, chain_b, 1))

        # Negative pairs (neg_ratio per positive)
        rng = np.random.RandomState(seed)
        n = len(self.pdb_keys)
        chain_b_list = [_parse_key(k)[2] for k in self.pdb_keys]

        for _ in range(neg_ratio):
            perm = rng.permutation(n).tolist()
            # Repair self-matches: swap with next element
            for i in range(n):
                if perm[i] == i:
                    j = (i + 1) % n
                    perm[i], perm[j] = perm[j], perm[i]
            for i, key in enumerate(self.pdb_keys):
                _, chain_a, _ = _parse_key(key)
                j = perm[i]
                neg_key = self.pdb_keys[j]
                neg_chain_b = chain_b_list[j]
                self._pairs.append((key, chain_a, neg_key, neg_chain_b, 0))

        n_pos = sum(1 for p in self._pairs if p[4] == 1)
        n_neg = sum(1 for p in self._pairs if p[4] == 0)
        print(f"[PPIRefDataset] {n_pos} positive + {n_neg} negative pairs "
              f"({len(valid_keys)} complexes)")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _cache_path(self, key: str, chain_id: str) -> Optional[Path]:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{key}_{chain_id}.npz"

    def _load_chain(self, pdb_key: str, chain_id: str) -> dict:
        """Load (or compute + cache) features for one chain of a complex."""
        cache_file = self._cache_path(pdb_key, chain_id)
        if cache_file is not None and cache_file.exists():
            data = np.load(cache_file)
            return {k: data[k] for k in data.files}

        pdb_file = _pdb_path(self.ppi_dir, pdb_key)
        # Resolve actual chain ID (handles uppercase/lowercase mismatches)
        resolved_id = _resolve_chain_id(pdb_file, chain_id)
        coords, sequence, _ = parse_structure(str(pdb_file), chain_ids=[resolved_id])

        esm_emb = self.esm_encoder.encode(sequence)
        n = min(len(coords), esm_emb.shape[0])
        coords = coords[:n]
        esm_emb = esm_emb[:n]

        topo_cache = (self._cache_dir / "topo") if self._cache_dir else None
        topo = compute_persistence_landscape(
            coords,
            max_edge_length=self.max_edge_length,
            max_dim=self.max_dim,
            n_landscapes=self.n_landscapes,
            n_points=self.n_points,
            cache_dir=topo_cache,
        )

        features = {
            "coords":  coords,
            "esm_emb": esm_emb,
            "topo":    topo,
        }

        if cache_file is not None:
            np.savez(cache_file, **features)

        return features

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> tuple:
        key_a, chain_a, key_b, chain_b, label = self._pairs[idx]

        try:
            feat_a = self._load_chain(key_a, chain_a)
            feat_b = self._load_chain(key_b, chain_b)
        except (ValueError, FileNotFoundError) as e:
            import warnings
            warnings.warn(f"[PPIRefDataset] Skipping idx={idx} ({key_a}/{chain_a}, "
                          f"{key_b}/{chain_b}): {e}")
            return None

        def to_tensors(feat: dict) -> dict:
            return {
                "coords":  torch.from_numpy(feat["coords"]),
                "esm_emb": torch.from_numpy(feat["esm_emb"]),
                "topo":    torch.from_numpy(feat["topo"]),
            }

        return (
            to_tensors(feat_a),
            to_tensors(feat_b),
            torch.tensor(label, dtype=torch.long),
        )

    def get_pair_keys(self) -> list:
        """Return list of (key_a, chain_a, key_b, chain_b, label) for all pairs."""
        return list(self._pairs)
