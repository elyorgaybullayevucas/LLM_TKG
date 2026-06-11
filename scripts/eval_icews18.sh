#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
  --MODEL_NAME     "meta-llama/Llama-2-7b-hf" \
  --ft             1 \
  --LORA_CHECKPOINT_DIR "./model/icews18_llama2_infonce_rbmh/model_final" \
  --input_file     "./data/processed/eval/icews18/history_facts_icews18_rbmh.txt" \
  --test_ans_file  "./data/processed/eval/icews18/test_ans_icews18.txt" \
  --fulltest       "./data/original/icews18/test.txt" \
  --time2id        "./data/original/icews18/ts2id.json" \
  --output_file    "./output/icews18_llama2_infonce_rbmh/final.txt"
