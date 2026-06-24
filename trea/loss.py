"""
Loss functions for TREA-TKG.

  LabelSmoothingCE     — 1-vs-N softmax with label smoothing
  HardNegativeTriplet  — in-batch hard-negative contrastive loss
  TREALoss             — combined: L = L_CE + α * L_triplet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """logits (B, N),  targets (B,) → scalar loss"""
        log_p  = F.log_softmax(logits, dim=-1)                    # (B, N)
        nll    = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)# (B,)
        smooth = -log_p.mean(dim=-1)                              # (B,)
        return (self.confidence * nll + self.smoothing * smooth).mean()


class HardNegativeTriplet(nn.Module):
    """
    In-batch hard negative mining.
    For each query, the hardest negative is the highest-scoring
    wrong entity in the batch's logits.
    """
    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self, query_repr: torch.Tensor,
                pos_emb: torch.Tensor,
                neg_emb: torch.Tensor) -> torch.Tensor:
        d_pos = ((query_repr - pos_emb) ** 2).sum(-1)   # (B,)
        d_neg = ((query_repr - neg_emb) ** 2).sum(-1)
        return F.relu(d_pos - d_neg + self.margin).mean()


class TREALoss(nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 alpha: float = 0.3, margin: float = 2.0):
        super().__init__()
        self.ce      = LabelSmoothingCE(smoothing)
        self.triplet = HardNegativeTriplet(margin)
        self.alpha   = alpha

    def forward(
        self,
        logits: torch.Tensor,         # (B, N)
        targets: torch.Tensor,        # (B,)
        query_repr: torch.Tensor,     # (B, d)
        ent_emb: nn.Embedding,
    ) -> dict:
        ce_loss = self.ce(logits, targets)

        # pick hardest negative per query from logit scores
        logits_neg = logits.clone()
        logits_neg.scatter_(1, targets.unsqueeze(1), float("-inf"))
        neg_ids    = logits_neg.argmax(dim=-1)              # (B,)

        pos_emb = ent_emb(targets)
        neg_emb = ent_emb(neg_ids)
        tri_loss = self.triplet(query_repr, pos_emb, neg_emb)

        total = ce_loss + self.alpha * tri_loss
        return {"total": total,
                "ce": ce_loss.item(),
                "triplet": tri_loss.item()}
