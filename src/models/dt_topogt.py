"""
DT-TopoGT: Dual-Tower Topology-Aware Graph Transformer — full pipeline.

Forward pass for a single protein pair:

    1.  Project ESM-2 embedding:   esm_emb  (N, d_esm)  → node_feat  (N, d_model)
    2.  Project topology vector:   topo     (topo_dim,)  → topo_emb  (d_model,)
    3.  Inject topology:           H^(0) = node_feat + topo_emb.unsqueeze(0)
    4.  Graph Transformer:         H = GraphTransformer(H^(0), coords)   (N, d_model)
    5.  Cross-attention (docking): H_A, H_B = CrossAttention(H_A, H_B)
    6.  Mean pooling:              Z_A = mean(H_A),  Z_B = mean(H_B)    (d_model,)
    7.  Concatenate & predict:     Z = [Z_A ‖ Z_B]                     (2*d_model,)
                                   y_pred = Sigmoid(MLP(Z))              scalar
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from src.models.graph_tf import GraphTransformer
from src.models.cross_attn import CrossAttention
from src.features.topology import topo_feature_dim


@dataclass
class DTTopoGTConfig:
    d_esm: int = 320           # ESM-2 hidden size (t6_8M → 320)
    d_model: int = 256         # Internal model dimensionality
    n_layers: int = 4          # Graph Transformer depth
    n_heads: int = 8           # Attention heads
    ffn_dim: int = 512         # FFN hidden size (default: 2 * d_model)
    dropout: float = 0.1

    # Topology hyper-parameters (must match ProteinPairDataset settings)
    topo_max_dim: int = 1
    topo_n_landscapes: int = 3
    topo_n_points: int = 50

    # Cross-attention
    cross_n_heads: int = 8


class DTTopoGT(nn.Module):
    """
    Dual-Tower Topology-Aware Graph Transformer.

    Processes a protein pair and returns the binding probability together with
    the pooled embeddings needed for the contrastive loss.

    Args:
        config: DTTopoGTConfig instance
    """

    def __init__(self, config: DTTopoGTConfig = None):
        super().__init__()
        cfg = config or DTTopoGTConfig()
        self.config = cfg

        # ── Input projections ────────────────────────────────────────────────
        self.esm_proj = nn.Linear(cfg.d_esm, cfg.d_model)

        topo_dim = topo_feature_dim(cfg.topo_max_dim, cfg.topo_n_landscapes,
                                    cfg.topo_n_points)
        self.topo_proj = nn.Sequential(
            nn.Linear(topo_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )

        # ── Per-protein encoder ──────────────────────────────────────────────
        ffn_dim = cfg.ffn_dim or 2 * cfg.d_model
        self.graph_transformer = GraphTransformer(
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ffn_dim=ffn_dim,
            dropout=cfg.dropout,
        )

        # ── Soft-docking cross-attention ─────────────────────────────────────
        self.cross_attn = CrossAttention(
            d_model=cfg.d_model,
            n_heads=cfg.cross_n_heads,
            dropout=cfg.dropout,
        )

        # ── Prediction head ──────────────────────────────────────────────────
        self.pred_head = nn.Sequential(
            nn.LayerNorm(2 * cfg.d_model),
            nn.Linear(2 * cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
            nn.Sigmoid(),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_protein(
        self, prot: dict
    ) -> tuple:
        """
        Encode a single protein dict to contextualised node features.

        Args:
            prot: dict with keys coords, esm_emb, topo (all on same device)

        Returns:
            H:      (N, d_model) contextualised node features
            coords: (N, 3) unchanged, kept for potential downstream use
        """
        coords = prot["coords"]       # (N, 3)
        esm_emb = prot["esm_emb"]     # (N, d_esm)
        topo = prot["topo"]           # (topo_dim,)

        node_feat = self.esm_proj(esm_emb)              # (N, d_model)
        topo_emb = self.topo_proj(topo).unsqueeze(0)    # (1, d_model)
        H0 = node_feat + topo_emb                       # (N, d_model)

        H = self.graph_transformer(H0, coords)          # (N, d_model)
        return H, coords

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        prot_a: dict,
        prot_b: dict,
    ) -> tuple:
        """
        Process a single protein pair.

        Each prot dict must contain:
            coords    – FloatTensor (N, 3)
            esm_emb   – FloatTensor (N, d_esm)
            topo      – FloatTensor (topo_dim,)

        Returns:
            y_pred:  FloatTensor scalar — binding probability ∈ (0, 1)
            z_a:     FloatTensor (d_model,) — pooled embedding for protein A
            z_b:     FloatTensor (d_model,) — pooled embedding for protein B
        """
        H_a, _ = self._encode_protein(prot_a)   # (N_A, d_model)
        H_b, _ = self._encode_protein(prot_b)   # (N_B, d_model)

        # Soft-docking
        H_a, H_b = self.cross_attn(H_a, H_b)   # (N_A, d), (N_B, d)

        # Global mean pooling
        z_a = H_a.mean(dim=0)   # (d_model,)
        z_b = H_b.mean(dim=0)   # (d_model,)

        # Prediction
        z = torch.cat([z_a, z_b], dim=-1)        # (2*d_model,)
        y_pred = self.pred_head(z).squeeze(-1)   # scalar

        return y_pred, z_a, z_b

    def forward_batch(
        self,
        list_a: list,
        list_b: list,
        device: torch.device,
    ) -> tuple:
        """
        Process a batch of protein pairs (loop over pairs).

        Args:
            list_a: List of B prot_a dicts (CPU tensors)
            list_b: List of B prot_b dicts (CPU tensors)
            device: Target device

        Returns:
            y_preds: FloatTensor (B,)
            z_as:    FloatTensor (B, d_model)
            z_bs:    FloatTensor (B, d_model)
        """
        y_list, za_list, zb_list = [], [], []
        for pa, pb in zip(list_a, list_b):
            pa = {k: v.to(device) for k, v in pa.items()}
            pb = {k: v.to(device) for k, v in pb.items()}
            y, za, zb = self.forward(pa, pb)
            y_list.append(y)
            za_list.append(za)
            zb_list.append(zb)

        return (
            torch.stack(y_list),     # (B,)
            torch.stack(za_list),    # (B, d_model)
            torch.stack(zb_list),    # (B, d_model)
        )
