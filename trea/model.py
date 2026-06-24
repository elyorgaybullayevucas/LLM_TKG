"""
AURORA-TKG — Adaptive Unified Reasoning Over Relational Associations.

Architecture overview:
  ┌──────────────────────────────────────────────────────────┐
  │  (s, r, t) query                                         │
  │       │                                                  │
  │  ┌────┴──────────────────────┐  ┌──────────────────────┐ │
  │  │  HierarchicalEncoder      │  │  Dual Copy Path      │ │
  │  │  ─────────────────────    │  │  ─────────────────── │ │
  │  │  R-GAT per snapshot       │  │  rel-copy: (s,r,o)   │ │
  │  │  Short Transformer (S)    │  │  ent-copy: (s,*,o)   │ │
  │  │  Long  Transformer (L)    │  │  MLP decay weights   │ │
  │  │  Cross-attention          │  └──────────┬───────────┘ │
  │  └────────────┬──────────────┘             │             │
  │               │  h_ctx                     │             │
  │  ┌────────────┴────────────────────────────┤             │
  │  │  QueryProj: MLP([h_s; h_r; h_ctx] → q)  │             │
  │  │  embed_logits = q @ E^T                  │             │
  │  └────────────────────────────────────────┘             │
  │                                                          │
  │  AdaptiveGate: g · embed + (1-g) · copy                  │
  └──────────────────────────────────────────────────────────┘

Key novelties vs TREA-TKG:
  1. R-GAT over K sampled neighbors per temporal snapshot
  2. Hierarchical short+long temporal encoder with cross-attention
  3. Dual copy (relation-specific + entity co-occurrence)
  4. InfoNCE contrastive training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trea.config import AURORAConfig
from trea.layers import HierarchicalEncoder


class AURORAModel(nn.Module):

    def __init__(self, num_entities: int, num_relations: int,
                 cfg: AURORAConfig):
        super().__init__()
        d = cfg.embed_dim
        # double relations for inverse triples
        R = num_relations * (2 if cfg.use_inverse else 1)
        self.num_entities = num_entities
        self.R   = R
        self.cfg = cfg

        # ── embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)

        # Xavier init for stable training
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

        # ── hierarchical temporal encoder ──────────────────────────────────────
        self.encoder = HierarchicalEncoder(
            d=d,
            num_heads=cfg.num_heads,
            num_relations=R,
            short_len=cfg.short_len,
            long_len=cfg.long_len,
            dropout=cfg.dropout,
        )

        # ── query projection ──────────────────────────────────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # ── copy mechanism ────────────────────────────────────────────────────
        # per-relation copy temperature (learned)
        self.rel_copy_temp = nn.Embedding(R, 1)
        nn.init.ones_(self.rel_copy_temp.weight)

        # dual copy gate: how much to weight rel_copy vs ent_copy
        if cfg.use_entity_copy:
            self.copy_mix = nn.Sequential(
                nn.Linear(d, 2),
                nn.Softmax(dim=-1),
            )

        # ── adaptive gate ─────────────────────────────────────────────────────
        self.gate_mlp = nn.Sequential(
            nn.Linear(d, cfg.gate_hidden),
            nn.GELU(),
            nn.Linear(cfg.gate_hidden, 1),
        )

        self.drop = nn.Dropout(cfg.dropout)

    # ── encode query ──────────────────────────────────────────────────────────

    def encode_query(
        self,
        subs:       torch.Tensor,   # (B,)
        rels:       torch.Tensor,   # (B,)
        neigh_ents: torch.Tensor,   # (B, H, K)
        neigh_rels: torch.Tensor,   # (B, H, K)
        neigh_mask: torch.Tensor,   # (B, H, K) bool
    ) -> torch.Tensor:              # (B, d)

        h_s = self.ent_emb(subs)    # (B, d)
        h_r = self.rel_emb(rels)    # (B, d)

        # look up neighbor entity embeddings
        h_neigh = self.ent_emb(neigh_ents)   # (B, H, K, d)

        # hierarchical encoder
        h_ctx = self.encoder(h_s, h_neigh, neigh_rels, neigh_mask,
                             self.rel_emb)    # (B, d)

        # project to query vector
        q = self.query_proj(torch.cat([h_s, h_r, h_ctx], dim=-1))  # (B, d)
        return self.drop(q)

    # ── scoring ───────────────────────────────────────────────────────────────

    def embed_score(self, query_repr: torch.Tensor) -> torch.Tensor:
        """(B, d) × (N, d)^T → (B, N)"""
        return query_repr @ self.ent_emb.weight.t()

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        subs:        torch.Tensor,   # (B,)
        rels:        torch.Tensor,   # (B,)
        neigh_ents:  torch.Tensor,   # (B, H, K)
        neigh_rels:  torch.Tensor,   # (B, H, K)
        neigh_mask:  torch.Tensor,   # (B, H, K) bool
        rel_copy:    torch.Tensor,   # (B, N)  pre-computed
        ent_copy:    torch.Tensor,   # (B, N)  pre-computed (zeros if disabled)
    ) -> torch.Tensor:               # (B, N)

        query_repr = self.encode_query(subs, rels, neigh_ents,
                                       neigh_rels, neigh_mask)   # (B, d)

        embed_logits = self.embed_score(query_repr)              # (B, N)

        # copy: scale with per-relation learned temperature
        temp = F.softplus(self.rel_copy_temp(rels))              # (B, 1)
        rel_copy_scaled = rel_copy * temp

        if self.cfg.use_entity_copy:
            mix = self.copy_mix(query_repr)                       # (B, 2)
            copy_logits = (mix[:, 0:1] * rel_copy_scaled
                           + mix[:, 1:2] * ent_copy)
        else:
            copy_logits = rel_copy_scaled

        # adaptive gate: query decides how much to trust copy vs embed
        g = torch.sigmoid(self.gate_mlp(query_repr))             # (B, 1)
        return g * embed_logits + (1 - g) * copy_logits
