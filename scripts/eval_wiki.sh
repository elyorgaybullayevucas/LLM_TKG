#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python3 inference.py \
  --MODEL_NAME     "meta-llama/Llama-2-7b-hf" \
  --ft             1 \
  --LORA_CHECKPOINT_DIR "./model/wiki_llama2_infonce_rbmh/model_final" \
  --input_file     "./data/processed/eval/wiki/history_facts_wiki_rbmh.txt" \
  --test_ans_file  "./data/processed/eval/wiki/test_ans_wiki.txt" \
  --output_file    "./output/wiki_llama2_infonce_rbmh/final.txt"
