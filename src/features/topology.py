"""
Persistent homology → persistence landscape features.

Uses GUDHI for Vietoris-Rips filtration and the Landscape sklearn-style
transformer, consistent with tda_sample/sample_simplex.py.

Topology feature vector layout (flat):
    [H0_landscape (n_land * n_pts), H1_landscape (n_land * n_pts), ...]
for each homology dimension 0 … max_dim.

Total length = (max_dim + 1) * n_landscapes * n_points
"""

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import gudhi
from gudhi.representations import Landscape


# Default TDA hyper-parameters (match tda_sample for H0/H1; smaller n_points
# for efficiency during training).
DEFAULT_MAX_EDGE = 20.0   # Å
DEFAULT_MAX_DIM  = 1      # H0 + H1 (H2 is expensive and rarely used in PPI)
DEFAULT_N_LAND   = 3      # λ_1 … λ_3
DEFAULT_N_PTS    = 50     # landscape discretisation resolution


def _hash_coords(coords: np.ndarray) -> str:
    return hashlib.md5(coords.tobytes()).hexdigest()[:16]


def compute_persistence_landscape(
    coords: np.ndarray,
    max_edge_length: float = DEFAULT_MAX_EDGE,
    max_dim: int = DEFAULT_MAX_DIM,
    n_landscapes: int = DEFAULT_N_LAND,
    n_points: int = DEFAULT_N_PTS,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    Compute persistence landscape feature vector for a point cloud.

    Args:
        coords: Cα coordinates, shape (N, 3)
        max_edge_length: Vietoris-Rips max edge length in Å
        max_dim: Maximum homology dimension (0 = H0, 1 = H0+H1, 2 = H0+H1+H2)
        n_landscapes: Number of landscape functions per dimension
        n_points: Discretisation resolution per landscape function
        cache_dir: If provided, cache/load results here (keyed by coord hash)

    Returns:
        Feature vector of shape ((max_dim+1) * n_landscapes * n_points,), float32
    """
    key = _hash_coords(coords)
    feat_dim = (max_dim + 1) * n_landscapes * n_points

    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"topo_{key}_md{max_dim}_nl{n_landscapes}_np{n_points}.npy"
        if cache_file.exists():
            return np.load(cache_file)

    # ── Vietoris-Rips → persistence ──────────────────────────────────────────
    rips = gudhi.RipsComplex(points=coords.astype(np.float64),
                             max_edge_length=max_edge_length)
    st = rips.create_simplex_tree(max_dimension=max_dim)
    st.compute_persistence()
    diag = st.persistence()

    # Collect finite bars per dimension
    pairs = {d: [] for d in range(max_dim + 1)}
    for dim, (b, d_val) in diag:
        if d_val != float("inf") and dim <= max_dim:
            pairs[dim].append((b, d_val))

    # ── Persistence landscape ─────────────────────────────────────────────────
    landscapes = []
    ls_transformer = Landscape(num_landscapes=n_landscapes, resolution=n_points)

    for dim in range(max_dim + 1):
        if pairs[dim]:
            dgm = np.array(pairs[dim], dtype=np.float64)   # (n, 2)
            land = ls_transformer.fit_transform([dgm])       # (1, n_land * n_pts)
            landscapes.append(land[0].astype(np.float32))
        else:
            landscapes.append(np.zeros(n_landscapes * n_points, dtype=np.float32))

    feature_vec = np.concatenate(landscapes)   # ((max_dim+1) * n_land * n_pts,)

    if cache_dir is not None:
        np.save(cache_file, feature_vec)

    return feature_vec


def topo_feature_dim(
    max_dim: int = DEFAULT_MAX_DIM,
    n_landscapes: int = DEFAULT_N_LAND,
    n_points: int = DEFAULT_N_PTS,
) -> int:
    """Return the expected flat feature vector length."""
    return (max_dim + 1) * n_landscapes * n_points
