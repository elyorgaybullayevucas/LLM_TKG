#!/bin/bash
# Train TREA-TKG on all three datasets sequentially.

set -e

DATA_DIR="C:/Users/elyor/OneDrive/Рабочий стол/data_extracted/data"
DEVICE="cuda"   # change to "cpu" if no GPU

echo "=========================================="
echo "  TREA-TKG — Training all 3 datasets"
echo "=========================================="

# ICEWS18
python train_trea.py \
  --dataset ICEWS18 \
  --data_dir "$DATA_DIR" \
  --embed_dim 256 \
  --history_len 10 \
  --num_heads 4 \
  --epochs 50 \
  --batch_size 1024 \
  --lr 1e-3 \
  --alpha_contrastive 0.3 \
  --device $DEVICE \
  --save_dir checkpoints \
  --log_dir logs

# YAGO
python train_trea.py \
  --dataset YAGO \
  --data_dir "$DATA_DIR" \
  --embed_dim 128 \
  --history_len 15 \
  --num_heads 4 \
  --epochs 30 \
  --batch_size 1024 \
  --lr 2e-3 \
  --alpha_contrastive 0.2 \
  --device $DEVICE \
  --save_dir checkpoints \
  --log_dir logs

# WIKI
python train_trea.py \
  --dataset WIKI \
  --data_dir "$DATA_DIR" \
  --embed_dim 256 \
  --history_len 12 \
  --num_heads 4 \
  --epochs 40 \
  --batch_size 1024 \
  --lr 1e-3 \
  --alpha_contrastive 0.3 \
  --device $DEVICE \
  --save_dir checkpoints \
  --log_dir logs

echo ""
echo "All done! Results saved in checkpoints/"
