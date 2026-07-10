"""
AURORA-TKG configuration.
Per-dataset optimal hyperparameters are in DATASET_CONFIGS.
"""
import argparse
from dataclasses import dataclass
from typing import Tuple, Dict, Any

DATA_DIR = "./data"

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "ICEWS18": {
        "embed_dim":         512,
        "k_neighbors":       32,
        "short_len":         5,
        "long_len":          20,
        "rgat_layers":       2,
        "num_heads":         8,
        "dropout":           0.2,
        "copy_lambda":       0.5,
        "use_entity_copy":   True,
        "recency_steps":     2,
        "recency_boost":     4.0,
        "epochs":            50,
        "batch_size":        1024,
        "lr":                5e-4,
        "alpha_infonce":     0.5,
        "infonce_temp":      0.07,
        "label_smoothing":   0.1,
    },
    "YAGO": {
        "embed_dim":         512,
        "k_neighbors":       64,    # YAGO: 10K entity, can afford more
        "short_len":         5,
        "long_len":          30,
        "rgat_layers":       2,
        "num_heads":         4,
        "dropout":           0.1,
        "copy_lambda":       0.1,   # slow decay: YAGO facts persist long
        "use_entity_copy":   True,
        "recency_steps":     3,     # YAGO: very high recurrence → strong burst
        "recency_boost":     8.0,
        "epochs":            50,
        "batch_size":        512,
        "lr":                5e-4,
        "alpha_infonce":     0.3,
        "infonce_temp":      0.07,
        "label_smoothing":   0.05,
    },
    "WIKI": {
        "embed_dim":         512,
        "k_neighbors":       64,    # WIKI: 12K entity, can afford more
        "short_len":         5,
        "long_len":          30,
        "rgat_layers":       2,
        "num_heads":         4,
        "dropout":           0.15,
        "copy_lambda":       0.1,   # slow decay: WIKI facts persist long
        "use_entity_copy":   True,
        "recency_steps":     3,     # WIKI: high recurrence → strong burst
        "recency_boost":     8.0,
        "epochs":            50,
        "batch_size":        512,
        "lr":                5e-4,
        "alpha_infonce":     0.3,
        "infonce_temp":      0.07,
        "label_smoothing":   0.05,
    },
    "GDELT": {
        "embed_dim":         512,
        "k_neighbors":       32,
        "short_len":         5,
        "long_len":          20,
        "rgat_layers":       2,
        "num_heads":         4,
        "dropout":           0.2,
        "copy_lambda":       0.5,
        "use_entity_copy":   True,
        "recency_steps":     2,
        "recency_boost":     4.0,
        "epochs":            40,
        "batch_size":        256,
        "lr":                5e-4,
        "alpha_infonce":     0.5,
        "infonce_temp":      0.07,
        "label_smoothing":   0.1,
    },
}


@dataclass
class AURORAConfig:
    # dataset
    dataset:          str   = "ICEWS18"
    data_dir:         str   = DATA_DIR
    # model
    embed_dim:        int   = 512
    k_neighbors:      int   = 32
    short_len:        int   = 5
    long_len:         int   = 30
    rgat_layers:      int   = 2
    num_heads:        int   = 4
    dropout:          float = 0.2
    use_inverse:      bool  = True
    gate_hidden:      int   = 128
    # copy
    copy_lambda:      float = 0.5
    use_entity_copy:  bool  = True
    recency_steps:    int   = 2     # last N snapshots → recency burst
    recency_boost:    float = 4.0   # multiply score if within recency window
    # training
    epochs:           int   = 50
    batch_size:       int   = 1024
    lr:               float = 5e-4
    weight_decay:     float = 1e-5
    grad_clip:        float = 1.0
    # loss
    alpha_infonce:    float = 0.5
    infonce_temp:     float = 0.07
    label_smoothing:  float = 0.1
    # eval
    eval_every:       int   = 1
    hits_at:          Tuple = (1, 3, 10)
    # misc
    seed:             int   = 42
    device:           str   = "cuda"
    gpu:              int   = 1
    save_dir:         str   = "checkpoints"
    log_dir:          str   = "logs"


def parse_args() -> AURORAConfig:
    p = argparse.ArgumentParser(description="AURORA-TKG")
    p.add_argument("--dataset",         type=str,   default="ICEWS18",
                   choices=list(DATASET_CONFIGS.keys()))
    p.add_argument("--data_dir",        type=str,   default=DATA_DIR)
    p.add_argument("--embed_dim",       type=int,   default=None)
    p.add_argument("--k_neighbors",     type=int,   default=None)
    p.add_argument("--short_len",       type=int,   default=None)
    p.add_argument("--long_len",        type=int,   default=None)
    p.add_argument("--rgat_layers",     type=int,   default=None)
    p.add_argument("--num_heads",       type=int,   default=None)
    p.add_argument("--dropout",         type=float, default=None)
    p.add_argument("--copy_lambda",     type=float, default=None)
    p.add_argument("--epochs",          type=int,   default=None)
    p.add_argument("--batch_size",      type=int,   default=None)
    p.add_argument("--lr",              type=float, default=None)
    p.add_argument("--weight_decay",    type=float, default=1e-5)
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--alpha_infonce",   type=float, default=None)
    p.add_argument("--infonce_temp",    type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--eval_every",      type=int,   default=1)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          type=str,   default="cuda")
    p.add_argument("--gpu",             type=int,   default=1)
    p.add_argument("--save_dir",        type=str,   default="checkpoints")
    p.add_argument("--log_dir",         type=str,   default="logs")
    p.add_argument("--recency_steps",   type=int,   default=None)
    p.add_argument("--recency_boost",   type=float, default=None)
    p.add_argument("--no_inverse",      dest="use_inverse",
                   action="store_false", default=True)
    p.add_argument("--no_entity_copy",  dest="use_entity_copy",
                   action="store_false", default=True)

    args = p.parse_args()
    ds   = DATASET_CONFIGS[args.dataset].copy()
    # CLI overrides dataset defaults
    for k, v in vars(args).items():
        if v is not None and k in ds:
            ds[k] = v

    kwargs = {k: v for k, v in vars(args).items()
              if k in AURORAConfig.__dataclass_fields__ and v is not None}
    kwargs.update(ds)
    kwargs["dataset"]  = args.dataset
    kwargs["data_dir"] = args.data_dir
    return AURORAConfig(**{k: v for k, v in kwargs.items()
                           if k in AURORAConfig.__dataclass_fields__})
