"""
TREA-TKG configuration — all hyperparameters in one place.
"""
import argparse
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class TREAConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "ICEWS18"
    data_dir: str = "C:/Users/elyor/OneDrive/Рабочий стол/data_extracted/data"

    # ── Model ────────────────────────────────────────────────────────────────
    embed_dim: int = 256
    history_len: int = 10
    num_heads: int = 4
    dropout: float = 0.2
    use_inverse: bool = True

    # ── Gate ─────────────────────────────────────────────────────────────────
    gate_hidden: int = 128

    # ── Copy Head ────────────────────────────────────────────────────────────
    copy_lambda: float = 0.5

    # ── Training ─────────────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0

    # ── Loss ─────────────────────────────────────────────────────────────────
    alpha_contrastive: float = 0.3
    margin: float = 2.0
    label_smoothing: float = 0.1

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_every: int = 1
    hits_at: Tuple[int, ...] = (1, 3, 10)

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    device: str = "cuda"
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def parse_args() -> TREAConfig:
    p = argparse.ArgumentParser(description="TREA-TKG Training")

    p.add_argument("--dataset",           type=str,   default="ICEWS18",
                   choices=["ICEWS18", "YAGO", "WIKI", "GDELT"])
    p.add_argument("--data_dir",          type=str,
                   default="C:/Users/elyor/OneDrive/Рабочий стол/data_extracted/data")
    p.add_argument("--embed_dim",         type=int,   default=256)
    p.add_argument("--history_len",       type=int,   default=10)
    p.add_argument("--num_heads",         type=int,   default=4)
    p.add_argument("--dropout",           type=float, default=0.2)
    p.add_argument("--gate_hidden",       type=int,   default=128)
    p.add_argument("--copy_lambda",       type=float, default=0.5)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=1024)
    p.add_argument("--lr",                type=float, default=1e-3)
    p.add_argument("--weight_decay",      type=float, default=1e-5)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--alpha_contrastive", type=float, default=0.3)
    p.add_argument("--margin",            type=float, default=2.0)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--eval_every",        type=int,   default=1)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--log_dir",           type=str,   default="logs")
    p.add_argument("--no_inverse",        dest="use_inverse",
                   action="store_false",  default=True)

    args = p.parse_args()
    # hits_at is fixed — not exposed as CLI arg
    cfg = TREAConfig(**{k: v for k, v in vars(args).items()
                        if k in TREAConfig.__dataclass_fields__})
    return cfg
