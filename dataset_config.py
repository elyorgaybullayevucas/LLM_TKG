"""Dataset-level configuration — single source of truth.

Add a new dataset here and it works everywhere else automatically.
"""

# fmt: off
DATASET_CFG = {
    "ICEWS18": {
        "num_entities":  23033,
        "num_relations": 256,
        "period":        1,           # time display divisor
        "contrastive_temp": 0.07,     # InfoNCE temperature (dense graph)
        "epochs":        10,
        "gamma": [0.5, 0.7, 0.020, 0.15],
    },
    "YAGO": {
        "num_entities":  10623,
        "num_relations": 10,
        "period":        1,
        "contrastive_temp": 0.10,     # sparse graph → softer temperature
        "epochs":        15,
        "gamma": [0.8, 0.4, 0.001, 0.20],
    },
    "WIKI": {
        "num_entities":  12554,
        "num_relations": 24,
        "period":        1,
        "contrastive_temp": 0.10,
        "epochs":        15,
        "gamma": [0.8, 0.4, 0.001, 0.20],
    },
    "GDELT": {
        "num_entities":  7691,
        "num_relations": 240,
        "period":        1,
        "contrastive_temp": 0.07,
        "epochs":        10,
        "gamma": [0.4, 0.5, 0.005, 0.05],
    },
    "YAGOs": {
        "num_entities":  None,        # inferred at preprocess time
        "num_relations": None,
        "period":        1,
        "contrastive_temp": 0.10,
        "epochs":        15,
        "gamma": [0.8, 0.4, 0.001, 0.20],
    },
}
# fmt: on

SUPPORTED = list(DATASET_CFG.keys())


def get_cfg(dataset: str) -> dict:
    ds = dataset.upper() if dataset.upper() in DATASET_CFG else dataset
    if ds not in DATASET_CFG:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {SUPPORTED}")
    return dict(DATASET_CFG[ds])
