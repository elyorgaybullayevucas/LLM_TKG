import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="RECIPE-TKG v2 config")

    # --- Training ---
    parser.add_argument("--MICRO_BATCH_SIZE", type=int, default=2)
    parser.add_argument("--BATCH_SIZE", type=int, default=512)
    parser.add_argument("--EPOCHS", type=int, default=10)
    parser.add_argument("--WARMUP_STEPS", type=int, default=100)
    parser.add_argument("--LEARNING_RATE", type=float, default=3e-4)

    # --- Model ---
    parser.add_argument("--MODEL_NAME", type=str, default="meta-llama/Llama-2-7b-hf",
                        choices=["meta-llama/Meta-Llama-3-8B", "meta-llama/Llama-2-7b-hf"])
    parser.add_argument("--BIT_8", default=False, action="store_true")
    parser.add_argument("--BIT_4", default=False, action="store_true")

    # --- LoRA ---
    parser.add_argument("--LORA_R", type=int, default=8)
    parser.add_argument("--LORA_ALPHA", type=int, default=16)
    parser.add_argument("--LORA_DROPOUT", type=float, default=0.05)

    # --- Sequence lengths ---
    parser.add_argument("--CONTEXT_LEN", type=int, default=4096)
    parser.add_argument("--TARGET_LEN", type=int, default=128)
    parser.add_argument("--TEXT_LEN", type=int, default=256)

    # --- Contrastive loss ---
    parser.add_argument("--CONTRASTIVE", type=int, default=1,
                        help="1 = InfoNCE contrastive loss, 0 = CE only")
    parser.add_argument("--CONTRASTIVE_WEIGHT", type=float, default=0.4,
                        help="Alpha: weight of contrastive loss vs CE loss")
    parser.add_argument("--CONTRASTIVE_TEMP", type=float, default=0.07,
                        help="Initial InfoNCE temperature (learnable). "
                             "ICEWS: 0.07  |  YAGO/WIKI: 0.1")

    # --- Data ---
    parser.add_argument("--OUTPUT_DIR", type=str, default="./model/run")
    parser.add_argument("--DATA_PATH", type=str, default="./data/processed/train/icews14/icews14_rbmh_1024_align.json")
    parser.add_argument("--FULL_DATA_PATH", type=str, default="",
                        help="Optional large dataset to append (randomly sampled)")
    parser.add_argument("--APPEND_DATA_SIZE", type=int, default=5000)
    parser.add_argument("--DATA_TYPE", type=str, choices=["json", "txt"], default="json")
    parser.add_argument("--DATASET", type=str, default="",
                        help="Dataset name tag (icews14 / icews18 / yago / wiki / gdelt)")

    # --- Eval during training ---
    parser.add_argument("--EVAL_STRATEGY", type=str, default="no")
    parser.add_argument("--EVAL_BY_HF", type=int, default=1)
    parser.add_argument("--EVAL_PATH", type=str, default=None)
    parser.add_argument("--EVAL_TYPE", type=str, choices=["json", "txt"], default="txt")
    parser.add_argument("--EVAL_STEPS", type=int, default=10)

    # --- Checkpointing ---
    parser.add_argument("--SAVE_STEPS", type=int, default=4)
    parser.add_argument("--SAVE_TOTAL_LIMIT", type=int, default=None)
    parser.add_argument("--LOAD_BEST_MODEL_AT_END", type=int, default=0)

    # --- Logging ---
    parser.add_argument("--LOGGING_STEPS", type=int, default=1)
    parser.add_argument("--REPORT_TO", type=str, default=None)
    parser.add_argument("--PROJ_NAME", type=str, default=None)
    parser.add_argument("--RUN_NAME", type=str, default=None)

    # --- Resume ---
    parser.add_argument("--W_RESUME", type=int, default=0)
    parser.add_argument("--W_ID", type=str, default=0)
    parser.add_argument("--RESUME_CKPT", type=str, default="")

    return parser.parse_args()
