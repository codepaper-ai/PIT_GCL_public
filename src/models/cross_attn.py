"""
Latent-space cross-attention for soft-docking.

Implements:
    H_A←B = softmax( Q_A K_B^T / sqrt(d) ) V_B
    H_B←A = softmax( Q_B K_A^T / sqrt(d) ) V_A

After cross-attention, each protein's representation is updated with
information from the other protein, modelling their interaction without
explicit 3-D docking.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionLayer(nn.Module):
    """
    Bidirectional multi-head cross-attention between two protein representations.

    Args:
        d_model: Model dimensionality (must be divisible by n_heads)
        n_heads: Number of attention heads
        dropout: Dropout on attention weights
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        # Separate Q projections for each protein, shared K/V projections
        self.q_a = nn.Linear(d_model, d_model, bias=False)
        self.q_b = nn.Linear(d_model, d_model, bias=False)
        self.k_a = nn.Linear(d_model, d_model, bias=False)
        self.k_b = nn.Linear(d_model, d_model, bias=False)
        self.v_a = nn.Linear(d_model, d_model, bias=False)
        self.v_b = nn.Linear(d_model, d_model, bias=False)

        self.out_a = nn.Linear(d_model, d_model)
        self.out_b = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """(N, d) → (H, N, d_head)"""
        N = x.size(0)
        return x.view(N, self.n_heads, self.d_head).transpose(0, 1)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
    ) -> tuple:
        """
        Args:
            h_a: Protein A node features, (N_A, d_model)
            h_b: Protein B node features, (N_B, d_model)

        Returns:
            h_a_upd: (N_A, d_model) — A updated with B's context
            h_b_upd: (N_B, d_model) — B updated with A's context
        """
        H, Dh = self.n_heads, self.d_head

        Q_a = self._reshape(self.q_a(h_a))   # (H, N_A, Dh)
        K_b = self._reshape(self.k_b(h_b))   # (H, N_B, Dh)
        V_b = self._reshape(self.v_b(h_b))   # (H, N_B, Dh)

        Q_b = self._reshape(self.q_b(h_b))   # (H, N_B, Dh)
        K_a = self._reshape(self.k_a(h_a))   # (H, N_A, Dh)
        V_a = self._reshape(self.v_a(h_a))   # (H, N_A, Dh)

        # A ← B
        attn_ab = F.softmax(torch.bmm(Q_a, K_b.transpose(1, 2)) / self.scale, dim=-1)
        attn_ab = self.attn_drop(attn_ab)
        ctx_ab = torch.bmm(attn_ab, V_b)     # (H, N_A, Dh)
        ctx_ab = ctx_ab.transpose(0, 1).contiguous().view(-1, self.d_model)  # (N_A, d)

        # B ← A
        attn_ba = F.softmax(torch.bmm(Q_b, K_a.transpose(1, 2)) / self.scale, dim=-1)
        attn_ba = self.attn_drop(attn_ba)
        ctx_ba = torch.bmm(attn_ba, V_a)     # (H, N_B, Dh)
        ctx_ba = ctx_ba.transpose(0, 1).contiguous().view(-1, self.d_model)  # (N_B, d)

        return self.out_a(ctx_ab), self.out_b(ctx_ba)


class CrossAttention(nn.Module):
    """
    Cross-attention module with residual connection and layer norm.

    Args:
        d_model: Model dimensionality
        n_heads: Attention heads
        dropout: Dropout rate
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = CrossAttentionLayer(d_model, n_heads, dropout)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
    ) -> tuple:
        """
        Args:
            h_a: (N_A, d_model)
            h_b: (N_B, d_model)
        Returns:
            Tuple of (h_a_updated, h_b_updated), each same shape as input
        """
        ctx_ab, ctx_ba = self.cross_attn(h_a, h_b)
        return self.norm_a(h_a + ctx_ab), self.norm_b(h_b + ctx_ba)
