"""
CREAT-TKG core layer.

SnapshotAggregator: relation-biased masked-mean over K neighbors per snapshot.
Simpler than Transformer R-GAT — faster convergence, less overfitting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SnapshotAggregator(nn.Module):
    """
    For each of H temporal snapshots, aggregate K sampled neighbors
    using a relation-biased masked mean, then fuse with self embedding.

    Input:
        h_s        (B, d)          subject base embedding
        h_neigh    (B, H, K, d)    neighbor entity embeddings
        neigh_rels (B, H, K)       neighbor relation IDs
        neigh_mask (B, H, K) bool  True = valid neighbor
        rel_emb    Embedding(R,d)  shared relation table

    Output: h_snap (B, H, d)  — subject repr at each snapshot
    """

    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.W_msg  = nn.Linear(d * 2, d)
        self.W_self = nn.Linear(d, d)
        self.norm   = nn.LayerNorm(d)
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        h_s:        torch.Tensor,
        h_neigh:    torch.Tensor,
        neigh_rels: torch.Tensor,
        neigh_mask: torch.Tensor,
        rel_emb:    nn.Embedding,
    ) -> torch.Tensor:

        h_r  = rel_emb(neigh_rels)                                        # (B,H,K,d)
        msg  = F.relu(self.W_msg(torch.cat([h_neigh, h_r], dim=-1)))      # (B,H,K,d)

        mask  = neigh_mask.float().unsqueeze(-1)                           # (B,H,K,1)
        h_agg = (msg * mask).sum(2) / (mask.sum(2).clamp(min=1e-9))       # (B,H,d)

        B, H, d = h_agg.shape
        h_s_exp = h_s.unsqueeze(1).expand(B, H, d)
        h_snap  = self.norm(F.relu(self.W_self(h_s_exp) + h_agg))
        return self.drop(h_snap)                                           # (B,H,d)
