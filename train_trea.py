"""
AURORA-TKG — Training entry point.

Usage:
    # ICEWS18 (GPU 1, default)
    python train_trea.py --dataset ICEWS18

    # YAGO
    python train_trea.py --dataset YAGO

    # WIKI
    python train_trea.py --dataset WIKI

    # Different GPU
    python train_trea.py --dataset ICEWS18 --gpu 2

    # CPU mode
    python train_trea.py --dataset YAGO --device cpu
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from trea.config import parse_args
from trea.trainer import AURORATrainer


BANNER = """
╔══════════════════════════════════════════════════════════╗
║          AURORA-TKG  (2025)                             ║
║  Adaptive Unified Reasoning Over Relational Assoc.      ║
╠══════════════════════════════════════════════════════════╣
║  Novel contributions:                                   ║
║   1. R-GAT: graph convolution over temporal snapshots   ║
║   2. Hierarchical encoder: short(S) + long(L) windows   ║
║   3. Cross-attention fusion of short- and long-term     ║
║   4. Dual copy: relation-specific + entity co-occur.    ║
║   5. InfoNCE contrastive loss (in-batch negatives)      ║
╚══════════════════════════════════════════════════════════╝
"""


def main():
    cfg = parse_args()
    print(BANNER)
    print(f"  Dataset    : {cfg.dataset}")
    print(f"  embed_dim  : {cfg.embed_dim}")
    print(f"  k_neighbors: {cfg.k_neighbors}")
    print(f"  short_len  : {cfg.short_len}")
    print(f"  long_len   : {cfg.long_len}")
    print(f"  rgat_layers: {cfg.rgat_layers}")
    print(f"  epochs     : {cfg.epochs}")
    print(f"  lr         : {cfg.lr}")
    print(f"  α_infonce  : {cfg.alpha_infonce}")
    print(f"  device     : {cfg.device} (GPU {cfg.gpu})\n")

    trainer = AURORATrainer(cfg)
    results = trainer.train()
    return results


if __name__ == "__main__":
    main()
