"""
AURORA-TKG core neural layers.

  TemporalNeighborAttention  — R-GAT: aggregate K neighbors per snapshot
  SnapshotTransformer        — Transformer over H temporal snapshots
  HierarchicalEncoder        — short-term + long-term with cross-attention
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Temporal Neighbor Attention  (R-GAT over sampled neighbors)
# ─────────────────────────────────────────────────────────────────────────────

class TemporalNeighborAttention(nn.Module):
    """
    For each of H snapshots, aggregate K sampled neighbors using
    relation-biased multi-head attention.

    Input:
        h_s       (B, d)         — subject base embedding
        h_neigh   (B, H, K, d)   — neighbor entity embeddings
        r_neigh   (B, H, K)      — neighbor relation IDs
        neigh_mask(B, H, K)      — True = valid neighbor
        rel_emb   Embedding(R,d) — shared relation embedding table

    Output: h_snap (B, H, d)  — subject repr at each snapshot
    """

    def __init__(self, d: int, num_heads: int, num_relations: int,
                 dropout: float = 0.1):
        super().__init__()
        assert d % num_heads == 0
        self.d  = d
        self.nh = num_heads
        self.dh = d // num_heads

        self.W_q   = nn.Linear(d, d, bias=False)
        self.W_k   = nn.Linear(d, d, bias=False)  # applied to h_neigh + r_bias
        self.W_v   = nn.Linear(d, d, bias=False)
        self.W_o   = nn.Linear(d, d, bias=False)
        self.W_rel = nn.Linear(d, d, bias=False)  # relation → key bias

        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.ff    = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(
        self,
        h_s:        torch.Tensor,   # (B, d)
        h_neigh:    torch.Tensor,   # (B, H, K, d)
        neigh_rels: torch.Tensor,   # (B, H, K)
        neigh_mask: torch.Tensor,   # (B, H, K) bool
        rel_emb:    nn.Embedding,
    ) -> torch.Tensor:              # (B, H, d)

        B, H, K, d = h_neigh.shape
        nh, dh = self.nh, self.dh
        scale  = math.sqrt(dh)

        # Process one snapshot at a time to avoid B*H*K*d peak allocation.
        # Each step: (B, K, d) instead of (B*H, K, d) → H× less memory.
        snap_outputs = []
        for h_idx in range(H):
            h_nb   = h_neigh[:, h_idx, :, :]       # (B, K, d)
            r_nb   = neigh_rels[:, h_idx, :]        # (B, K)
            m_nb   = neigh_mask[:, h_idx, :]        # (B, K)

            # relation-biased keys
            r_bias = self.W_rel(rel_emb(r_nb.reshape(-1)))   # (B*K, d)
            r_bias = r_bias.view(B, K, d)
            keys   = h_nb + r_bias                            # (B, K, d)

            # multi-head attention
            q = self.W_q(h_s).view(B, 1, nh, dh).transpose(1, 2)   # (B, nh, 1, dh)
            k = self.W_k(keys).view(B, K, nh, dh).transpose(1, 2)   # (B, nh, K, dh)
            v = self.W_v(h_nb).view(B, K, nh, dh).transpose(1, 2)   # (B, nh, K, dh)

            scores = torch.matmul(q, k.transpose(-2, -1)) / scale    # (B, nh, 1, K)
            scores = scores.masked_fill(
                ~m_nb.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            attn = torch.softmax(scores, dim=-1)
            attn = torch.nan_to_num(attn, nan=0.0)
            attn = self.drop(attn)

            ctx     = torch.matmul(attn, v)                           # (B, nh, 1, dh)
            ctx     = ctx.squeeze(2).transpose(1, 2).reshape(B, d)    # (B, d)
            ctx     = self.W_o(ctx)

            # fall back to h_s when all neighbors are padding
            all_pad = (~m_nb).all(dim=-1, keepdim=True).float()       # (B, 1)
            h_out   = (1 - all_pad) * ctx + all_pad * h_s

            h_out = self.norm1(h_s + h_out)
            h_out = self.norm2(h_out + self.ff(h_out))
            snap_outputs.append(h_out)                                 # (B, d)

        return torch.stack(snap_outputs, dim=1)                        # (B, H, d)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Snapshot Transformer
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotTransformer(nn.Module):
    """
    Transformer encoder over a sequence of H temporal snapshots.
    Adds learnable positional encodings.
    Output: (B, d) — pooled representation (attention-weighted mean).
    """

    def __init__(self, d: int, max_len: int, num_heads: int,
                 dropout: float = 0.1, num_layers: int = 2):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=num_heads, dim_feedforward=d * 2,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool_w = nn.Linear(d, 1)

    def forward(self, h_snap: torch.Tensor) -> torch.Tensor:
        """h_snap (B, H, d)  →  (B, d)"""
        B, H, d = h_snap.shape
        pos = torch.arange(H, device=h_snap.device)
        h   = h_snap + self.pos_emb(pos).unsqueeze(0)   # (B, H, d)
        h   = self.transformer(h)                        # (B, H, d)

        # attention-weighted pooling
        w   = torch.softmax(self.pool_w(h), dim=1)      # (B, H, 1)
        out = (w * h).sum(dim=1)                         # (B, d)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hierarchical Encoder
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalEncoder(nn.Module):
    """
    AURORA's core temporal encoder:

    1. R-GAT aggregates K neighbors within each snapshot   → (B, H, d)
    2. Short-term Transformer on last S snapshots          → (B, d)
    3. Long-term aggregation over all H snapshots          → (B, d)
    4. Cross-attention: short queries long                 → (B, d)
    5. FFN fusion                                          → (B, d)
    """

    def __init__(self, d: int, num_heads: int, num_relations: int,
                 short_len: int, long_len: int, dropout: float = 0.1):
        super().__init__()
        self.short_len = short_len
        self.long_len  = long_len

        self.r_gat = TemporalNeighborAttention(d, num_heads, num_relations, dropout)

        self.short_tf = SnapshotTransformer(d, short_len, num_heads, dropout,
                                            num_layers=2)
        self.long_tf  = SnapshotTransformer(d, long_len,  num_heads, dropout,
                                            num_layers=1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d)

        self.ffn = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

    def forward(
        self,
        h_s:        torch.Tensor,   # (B, d)
        h_neigh:    torch.Tensor,   # (B, H, K, d)
        neigh_rels: torch.Tensor,   # (B, H, K)
        neigh_mask: torch.Tensor,   # (B, H, K) bool
        rel_emb:    nn.Embedding,
    ) -> torch.Tensor:              # (B, d)

        # ── R-GAT: (B, H, d) ─────────────────────────────────────────────────
        h_snap = self.r_gat(h_s, h_neigh, neigh_rels, neigh_mask, rel_emb)

        # ── short-term: last short_len snapshots ──────────────────────────────
        S = min(self.short_len, h_snap.size(1))
        h_short_seq = h_snap[:, :S, :]                  # (B, S, d)
        h_short     = self.short_tf(h_short_seq)        # (B, d)

        # ── long-term: all H snapshots ────────────────────────────────────────
        h_long_seq = h_snap                              # (B, H, d)
        h_long     = self.long_tf(h_long_seq)           # (B, d)

        # ── cross-attention: short queries long-seq ───────────────────────────
        q_cross = h_short.unsqueeze(1)                  # (B, 1, d)
        cross, _ = self.cross_attn(q_cross, h_long_seq, h_long_seq)
        cross = self.cross_norm(h_short + cross.squeeze(1))  # (B, d)

        # ── fuse ──────────────────────────────────────────────────────────────
        fused = self.ffn(torch.cat([h_short, cross], dim=-1))  # (B, d)
        return fused
