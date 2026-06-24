"""
TREA-TKG — Training entry point.

Usage:
    # ICEWS18
    python train_trea.py --dataset ICEWS18 --epochs 50 --embed_dim 256

    # YAGO (10 relations, fast)
    python train_trea.py --dataset YAGO --epochs 30 --embed_dim 128 --lr 2e-3

    # WIKI
    python train_trea.py --dataset WIKI --epochs 40 --embed_dim 256

    # CPU mode
    python train_trea.py --dataset YAGO --device cpu --batch_size 512

    # All three datasets sequentially
    python train_trea.py --dataset ICEWS18 && \\
    python train_trea.py --dataset YAGO    && \\
    python train_trea.py --dataset WIKI
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from trea.config import parse_args
from trea.trainer import TREATrainer


BANNER = """
╔══════════════════════════════════════════════════════╗
║         TREA-TKG  (2025)                            ║
║  Temporal Recurrence-Enhanced Attention for TKG     ║
╠══════════════════════════════════════════════════════╣
║  Novel contributions:                               ║
║   1. Per-relation learnable temporal decay          ║
║   2. Frequency-aware copy mechanism                 ║
║   3. Adaptive structural ↔ copy gate               ║
║   4. Hard-negative contrastive loss                 ║
╚══════════════════════════════════════════════════════╝
"""


def main():
    cfg = parse_args()
    print(BANNER)
    print(f"  Dataset   : {cfg.dataset}")
    print(f"  embed_dim : {cfg.embed_dim}")
    print(f"  history   : {cfg.history_len} timesteps")
    print(f"  epochs    : {cfg.epochs}")
    print(f"  lr        : {cfg.lr}")
    print(f"  α_contrast: {cfg.alpha_contrastive}")
    print(f"  device    : {cfg.device}\n")

    trainer = TREATrainer(cfg)
    results = trainer.train()
    return results


if __name__ == "__main__":
    main()
