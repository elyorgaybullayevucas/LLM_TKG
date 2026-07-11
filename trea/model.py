"""
CREAT-TKG — Copy-Residual Encoding with Adaptive Temporal modeling.

Core insight that fixes the previous AURORA failure:
  final_score = neural_score + log1p(copy_score) * copy_scale

Copy is an ADDITIVE LOG-SPACE OFFSET, not a competing gate.
This means:
  - If copy(o) = 0  → log1p(0) = 0  → no effect on neural score
  - If copy(o) > 0  → always boosts entity o (never hurts)
  - copy_scale (learned scalar) controls how much weight to give history
  - Neural learns to predict novel entities; copy handles repetition

Architecture:
  1. Entity/relation embeddings (Xavier init)
  2. SnapshotAggregator: R-biased mean over K neighbors per snapshot → (B,H,d)
  3. 2-layer GRU over H snapshots → temporal context h_t (B,d)
  4. Query: MLP([h_s; h_t; h_r]) → q (B,d)
  5. Neural score: q @ E^T  (B,N)
  6. Copy offset: log1p(α*rel_copy + β*ent_copy) * copy_scale
  7. Final = neural + copy_offset
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from trea.config import AURORAConfig
from trea.layers import SnapshotAggregator


class CREATModel(nn.Module):

    def __init__(self, num_entities: int, num_relations: int,
                 cfg: AURORAConfig):
        super().__init__()
        d   = cfg.embed_dim
        R   = num_relations * (2 if cfg.use_inverse else 1)
        self.cfg = cfg
        self.num_entities = num_entities

        # ── embeddings ────────────────────────────────────────────────────────
        self.ent_emb = nn.Embedding(num_entities, d)
        self.rel_emb = nn.Embedding(R, d)
        nn.init.xavier_normal_(self.ent_emb.weight)
        nn.init.xavier_normal_(self.rel_emb.weight)

        # ── temporal encoder ─────────────────────────────────────────────────
        self.snap_agg = SnapshotAggregator(d, cfg.dropout)
        gru_dropout = cfg.dropout if cfg.dropout > 0 else 0.0
        self.gru = nn.GRU(d, d, num_layers=2, batch_first=True,
                          dropout=gru_dropout)

        # ── query projection ──────────────────────────────────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(d * 3, d * 2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d * 2, d),
            nn.LayerNorm(d),
        )

        # ── score projection ──────────────────────────────────────────────────
        self.score_proj = nn.Linear(d, d, bias=False)

        # ── copy mechanism ────────────────────────────────────────────────────
        # Learnable weights for rel_copy vs ent_copy blend
        self.copy_w = nn.Parameter(torch.tensor([1.0, 0.5]))
        # Global scale for the copy offset (exp so it stays positive)
        # Starts at exp(0)=1.0 — will grow larger for WIKI/YAGO
        self.log_copy_scale = nn.Parameter(torch.zeros(1))

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

        h_s     = self.ent_emb(subs)              # (B, d)
        h_r     = self.rel_emb(rels)              # (B, d)
        h_neigh = self.ent_emb(neigh_ents)        # (B, H, K, d)

        # aggregate neighbors per snapshot
        h_snap = self.snap_agg(h_s, h_neigh, neigh_rels,
                               neigh_mask, self.rel_emb)   # (B, H, d)

        # GRU over H snapshots
        _, h_gru = self.gru(h_snap)               # h_gru: (2, B, d)
        h_t = self.drop(h_gru[-1])                # (B, d)  last layer

        # query vector
        q = self.query_proj(torch.cat([h_s, h_t, h_r], dim=-1))   # (B, d)
        return q

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        subs:       torch.Tensor,   # (B,)
        rels:       torch.Tensor,   # (B,)
        neigh_ents: torch.Tensor,   # (B, H, K)
        neigh_rels: torch.Tensor,   # (B, H, K)
        neigh_mask: torch.Tensor,   # (B, H, K) bool
        rel_copy:   torch.Tensor,   # (B, N)
        ent_copy:   torch.Tensor,   # (B, N)
    ) -> torch.Tensor:              # (B, N)

        q = self.encode_query(subs, rels, neigh_ents, neigh_rels, neigh_mask)

        # neural score in logit space
        neural = self.score_proj(q) @ self.ent_emb.weight.t()     # (B, N)

        # copy: weighted blend → log-space offset
        # softmax ensures positive, normalized weights
        w = torch.softmax(self.copy_w, dim=0)
        copy_blend = w[0] * rel_copy + w[1] * ent_copy             # (B, N)
        copy_scale = torch.exp(self.log_copy_scale)
        copy_offset = torch.log1p(copy_blend) * copy_scale         # (B, N)

        # additive: copy only boosts, never hurts; neural handles novel entities
        return neural + copy_offset
