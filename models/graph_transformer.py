import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GraphTransformerBlock(nn.Module):
    """A lightweight Graph Transformer block with adjacency-masked self-attention.

    Expects:
      - x: [batch, num_nodes, dim]
      - adj: [batch, num_nodes, num_nodes] with 1 for edge (directed or undirected), 0 otherwise

    Behavior:
      - Nodes attend to themselves and their graph neighbors only.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        if adj is None:
            # Fall back to full self-attention (still works as a standard Transformer block)
            attn_out, _ = self.attn(x, x, x, need_weights=False)
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ff(x))
            return x

        if adj.ndim != 3:
            raise ValueError(f"adj must be 3D [B,N,N], got {adj.shape}")

        bsz, n, _ = x.shape

        # Allow self-loops + undirected neighborhood for stability
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).unsqueeze(0)
        undirected = (adj > 0) | (adj.transpose(1, 2) > 0) | (eye > 0)

        # Build attn_mask: 0 for allowed, -inf for blocked.
        # MultiheadAttention accepts 3D mask shape [B*num_heads, N, N]
        attn_mask = torch.zeros((bsz, n, n), device=x.device, dtype=x.dtype)
        attn_mask = attn_mask.masked_fill(~undirected, float("-inf"))
        attn_mask = attn_mask.repeat_interleave(self.num_heads, dim=0)

        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class GraphTransformerEncoder(nn.Module):
    """Stacked GraphTransformer blocks."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerBlock(dim=dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, adj)
        return x
