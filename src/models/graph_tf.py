"""
Graph Transformer for single-protein encoding.

Implements a stack of structure-aware transformer layers that encode a protein's
node features (ESM-2 embeddings + topology injection) into contextualised
per-residue representations H ∈ R^(N × d_model).

Each layer performs full self-attention over all N residues with an additive
structural bias derived from pairwise Cα distances.
"""

import torch
import torch.nn as nn

from src.models.layers import TransformerLayer


class GraphTransformer(nn.Module):
    """
    Stack of TransformerLayers for single-protein encoding.

    Forward pass:
        H = GraphTransformer(H^(0), coords)
    where H^(0) = ESM-2 embedding + topology vector (both projected to d_model).

    Args:
        d_model:  Model dimensionality
        n_layers: Number of transformer layers
        n_heads:  Attention heads per layer
        ffn_dim:  FFN hidden dimensionality (default: 4 * d_model)
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 8,
        ffn_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, n_heads, ffn_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      Node features, (N, d_model)
            coords: Cα coordinates, (N, 3) — used for structural bias
        Returns:
            H:  Contextualised node features, (N, d_model)
        """
        for layer in self.layers:
            x = layer(x, coords)
        return self.norm(x)
