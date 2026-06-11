#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
  --MODEL_NAME     "meta-llama/Llama-2-7b-hf" \
  --ft             1 \
  --LORA_CHECKPOINT_DIR "./model/yago_llama2_infonce_rbmh/model_final" \
  --input_file     "./data/processed/eval/yago/history_facts_yago_rbmh.txt" \
  --test_ans_file  "./data/processed/eval/yago/test_ans_yago.txt" \
  --output_file    "./output/yago_llama2_infonce_rbmh/final.txt"
