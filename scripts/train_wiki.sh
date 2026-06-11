#!/bin/bash
# WIKI (Wikidata12k) — yangi dataset, YAGO ga o'xshash sparse tuzilma

CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
  --DATASET        wiki \
  --MODEL_NAME     "meta-llama/Llama-2-7b-hf" \
  --CONTRASTIVE    1 \
  --CONTRASTIVE_WEIGHT 0.4 \
  --CONTRASTIVE_TEMP   0.1 \
  --MICRO_BATCH_SIZE   2 \
  --BATCH_SIZE         512 \
  --EPOCHS             15 \
  --APPEND_DATA_SIZE   5000 \
  --OUTPUT_DIR   "./model/wiki_llama2_infonce_rbmh" \
  --DATA_PATH    "./data/processed/train/wiki/wiki_rbmh_1024_align.json" \
  --FULL_DATA_PATH "./data/processed/train/wiki/raw/train_samples/wiki_rbmh.json" \
  --REPORT_TO wandb \
  --PROJ_NAME RECIPE_TKG_v2
