"""
TREA-TKG configuration — all hyperparameters in one place.

Per-dataset defaults are defined in DATASET_CONFIGS below.
Running `python train_trea.py --dataset YAGO` automatically
picks the right embed_dim, history_len, lr, etc.
"""
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

# ── Server data directory ─────────────────────────────────────────────────────
DATA_DIR = "./data"

# ── Per-dataset best hyperparameters ─────────────────────────────────────────
DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ICEWS18": {
        "embed_dim":         256,
        "history_len":       10,
        "num_heads":         4,
        "dropout":           0.2,
        "epochs":            50,
        "batch_size":        1024,
        "lr":                1e-3,
        "alpha_contrastive": 0.3,
        "copy_lambda":       0.5,
        "margin":            2.0,
    },
    "YAGO": {
        "embed_dim":         128,
        "history_len":       15,
        "num_heads":         4,
        "dropout":           0.1,
        "epochs":            30,
        "batch_size":        1024,
        "lr":                2e-3,
        "alpha_contrastive": 0.2,
        "copy_lambda":       0.3,
        "margin":            1.5,
    },
    "WIKI": {
        "embed_dim":         256,
        "history_len":       12,
        "num_heads":         4,
        "dropout":           0.2,
        "epochs":            40,
        "batch_size":        1024,
        "lr":                1e-3,
        "alpha_contrastive": 0.3,
        "copy_lambda":       0.4,
        "margin":            2.0,
    },
    "GDELT": {
        "embed_dim":         256,
        "history_len":       10,
        "num_heads":         4,
        "dropout":           0.2,
        "epochs":            40,
        "batch_size":        1024,
        "lr":                1e-3,
        "alpha_contrastive": 0.3,
        "copy_lambda":       0.5,
        "margin":            2.0,
    },
}


@dataclass
class TREAConfig:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset: str = "ICEWS18"
    data_dir: str = DATA_DIR

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
    gpu: int = 1          # GPU index (nvidia-smi: 1 and 2 have most free VRAM)
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


def parse_args() -> TREAConfig:
    p = argparse.ArgumentParser(description="TREA-TKG Training")

    p.add_argument("--dataset",  type=str, default="ICEWS18",
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--data_dir", type=str, default=DATA_DIR)

    # These can still be overridden from CLI; if not given, dataset default is used
    p.add_argument("--embed_dim",         type=int,   default=None)
    p.add_argument("--history_len",       type=int,   default=None)
    p.add_argument("--num_heads",         type=int,   default=None)
    p.add_argument("--dropout",           type=float, default=None)
    p.add_argument("--gate_hidden",       type=int,   default=128)
    p.add_argument("--copy_lambda",       type=float, default=None)
    p.add_argument("--epochs",            type=int,   default=None)
    p.add_argument("--batch_size",        type=int,   default=None)
    p.add_argument("--lr",                type=float, default=None)
    p.add_argument("--weight_decay",      type=float, default=1e-5)
    p.add_argument("--grad_clip",         type=float, default=1.0)
    p.add_argument("--alpha_contrastive", type=float, default=None)
    p.add_argument("--margin",            type=float, default=None)
    p.add_argument("--label_smoothing",   type=float, default=0.1)
    p.add_argument("--eval_every",        type=int,   default=1)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--device",            type=str,   default="cuda")
    p.add_argument("--gpu",               type=int,   default=1)
    p.add_argument("--save_dir",          type=str,   default="checkpoints")
    p.add_argument("--log_dir",           type=str,   default="logs")
    p.add_argument("--no_inverse",        dest="use_inverse",
                   action="store_false",  default=True)

    args = p.parse_args()

    # Start from dataset defaults, then apply any CLI overrides
    ds_defaults = DATASET_CONFIGS[args.dataset].copy()
    for key, val in vars(args).items():
        if val is not None and key in ds_defaults:
            ds_defaults[key] = val

    # Merge into final config
    cfg_kwargs = {k: v for k, v in vars(args).items()
                  if k in TREAConfig.__dataclass_fields__ and v is not None}
    cfg_kwargs.update(ds_defaults)
    cfg_kwargs["dataset"]  = args.dataset
    cfg_kwargs["data_dir"] = args.data_dir

    return TREAConfig(**{k: v for k, v in cfg_kwargs.items()
                         if k in TREAConfig.__dataclass_fields__})
