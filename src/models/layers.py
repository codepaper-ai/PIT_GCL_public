"""
Structure-aware attention layer.

Implements the core attention mechanism from DT-TopoGT:

    Attention(Q, K, V) = softmax( (QK^T / sqrt(d)) + B_struct ) V

where B_struct is a per-head structural bias derived from pairwise Cα distances
encoded via learned Radial Basis Functions (RBF).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RBFStructBias(nn.Module):
    """
    Computes the structural attention bias B_struct from Cα coordinates.

    B_struct[h, i, j] = linear_h( RBF(||x_i - x_j||) )

    Args:
        n_heads:    Number of attention heads
        n_rbf:      Number of RBF kernel centres
        cutoff:     Maximum distance (Å) for RBF coverage
    """

    def __init__(self, n_heads: int, n_rbf: int = 16, cutoff: float = 20.0):
        super().__init__()
        self.n_heads = n_heads
        # Fixed RBF centres, learned widths and linear projection
        self.register_buffer(
            "centers", torch.linspace(0.0, cutoff, n_rbf)
        )
        self.log_gamma = nn.Parameter(torch.zeros(n_rbf))  # log for positivity
        self.linear = nn.Linear(n_rbf, n_heads, bias=True)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 3)
        Returns:
            B: (n_heads, N, N)
        """
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)      # (N, N, 3)
        dist = diff.norm(dim=-1, keepdim=True)                 # (N, N, 1)

        gamma = self.log_gamma.exp().unsqueeze(0).unsqueeze(0) # (1, 1, n_rbf)
        rbf = torch.exp(-gamma * (dist - self.centers) ** 2)  # (N, N, n_rbf)

        B = self.linear(rbf)          # (N, N, n_heads)
        return B.permute(2, 0, 1)    # (n_heads, N, N)


class StructureAwareAttention(nn.Module):
    """
    Multi-head self-attention with additive structural bias.

    All nodes attend to all other nodes within the same protein (full
    self-attention), with pairwise structural distance information added
    directly to the attention logits.

    Args:
        d_model:  Model dimensionality (must be divisible by n_heads)
        n_heads:  Number of attention heads
        dropout:  Dropout probability applied to attention weights
        n_rbf:    RBF kernels for structural bias
        cutoff:   RBF cutoff distance in Å
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        n_rbf: int = 16,
        cutoff: float = 20.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = math.sqrt(self.d_head)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.struct_bias = RBFStructBias(n_heads, n_rbf=n_rbf, cutoff=cutoff)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      Node features, (N, d_model)
            coords: Cα coordinates, (N, 3)
        Returns:
            out: (N, d_model)
        """
        N = x.size(0)
        H, Dh = self.n_heads, self.d_head

        QKV = self.qkv(x)                          # (N, 3*d)
        Q, K, V = QKV.split(self.d_model, dim=-1)

        # Reshape → (H, N, Dh)
        Q = Q.view(N, H, Dh).transpose(0, 1)
        K = K.view(N, H, Dh).transpose(0, 1)
        V = V.view(N, H, Dh).transpose(0, 1)

        attn = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (H, N, N)
        attn = attn + self.struct_bias(coords)                 # (H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.bmm(attn, V)                              # (H, N, Dh)
        out = out.transpose(0, 1).contiguous().view(N, self.d_model)
        return self.out_proj(out)


class TransformerLayer(nn.Module):
    """
    Pre-norm Transformer layer: attention → FFN with residual connections.

    Pre-norm (LayerNorm before sub-layer) is more stable for deep networks.

    Args:
        d_model:  Model dimensionality
        n_heads:  Attention heads
        ffn_dim:  Inner dimensionality of the feed-forward network
        dropout:  Dropout rate
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = StructureAwareAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:      (N, d_model)
            coords: (N, 3)
        Returns:
            (N, d_model)
        """
        x = x + self.attn(self.norm1(x), coords)
        x = x + self.ffn(self.norm2(x))
        return x
