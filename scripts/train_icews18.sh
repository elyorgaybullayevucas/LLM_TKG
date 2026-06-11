#!/bin/bash
# ICEWS18 — InfoNCE temp=0.07 (dense, daily dataset)

CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
  --DATASET        icews18 \
  --MODEL_NAME     "meta-llama/Llama-2-7b-hf" \
  --CONTRASTIVE    1 \
  --CONTRASTIVE_WEIGHT 0.4 \
  --CONTRASTIVE_TEMP   0.07 \
  --MICRO_BATCH_SIZE   2 \
  --BATCH_SIZE         512 \
  --EPOCHS             10 \
  --APPEND_DATA_SIZE   5000 \
  --OUTPUT_DIR   "./model/icews18_llama2_infonce_rbmh" \
  --DATA_PATH    "./data/processed/train/icews18/icews18_rbmh_1024_align.json" \
  --FULL_DATA_PATH "./data/processed/train/icews18/raw/train_samples/icews18_rbmh.json" \
  --REPORT_TO wandb \
  --PROJ_NAME RECIPE_TKG_v2
