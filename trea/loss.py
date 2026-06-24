"""
AURORA-TKG loss functions.

  LabelSmoothingCE  — 1-vs-N softmax with label smoothing
  InfoNCELoss       — in-batch contrastive (stronger than triplet)
  AURORALoss        — CE + α * InfoNCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """
    1-vs-N cross-entropy with label smoothing.
    """
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """logits (B, N), targets (B,) → scalar"""
        log_p  = F.log_softmax(logits, dim=-1)
        nll    = -log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        smooth = -log_p.mean(dim=-1)
        return (self.confidence * nll + self.smoothing * smooth).mean()


class InfoNCELoss(nn.Module):
    """
    InfoNCE with in-batch negatives.
    With batch_size=512 this gives 511 negatives — much denser
    training signal than margin-based triplet loss.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.tau = temperature

    def forward(
        self,
        query_repr: torch.Tensor,   # (B, d)
        pos_emb:    torch.Tensor,   # (B, d)
    ) -> torch.Tensor:
        q = F.normalize(query_repr, dim=-1)
        k = F.normalize(pos_emb,    dim=-1)
        logits = torch.mm(q, k.t()) / self.tau          # (B, B)
        labels = torch.arange(logits.size(0), device=q.device)
        return F.cross_entropy(logits, labels)


class AURORALoss(nn.Module):
    """Total = CE_label_smooth + α * InfoNCE"""

    def __init__(self, smoothing: float = 0.1,
                 alpha: float = 0.5, temperature: float = 0.07):
        super().__init__()
        self.alpha   = alpha
        self.ce      = LabelSmoothingCE(smoothing)
        self.infonce = InfoNCELoss(temperature)

    def forward(
        self,
        logits:     torch.Tensor,   # (B, N)
        targets:    torch.Tensor,   # (B,)
        query_repr: torch.Tensor,   # (B, d)
        ent_emb:    nn.Embedding,
    ) -> dict:
        ce_loss      = self.ce(logits, targets)
        pos_emb      = ent_emb(targets)
        infonce_loss = self.infonce(query_repr, pos_emb)

        total = ce_loss + self.alpha * infonce_loss
        return {
            "_total":  total,
            "total":   total.item(),
            "ce":      ce_loss.item(),
            "infonce": infonce_loss.item(),
        }
