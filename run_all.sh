#!/bin/bash
# Train TREA-TKG on all three datasets sequentially.
# GPU 1 and 2 have the most free VRAM (~37GB each).
# Change GPU=1 to GPU=2 to use the second free GPU.

set -e

GPU=1        # GPU index to use (1 or 2 recommended)
DATA_DIR="./data"

echo "=========================================="
echo "  TREA-TKG — Training all 3 datasets"
echo "  GPU: $GPU  |  data: $DATA_DIR"
echo "=========================================="

# ICEWS18  (256-dim, 10 history steps, 50 epochs)
python train_trea.py \
  --dataset ICEWS18 \
  --data_dir "$DATA_DIR" \
  --gpu $GPU

# YAGO  (128-dim, 15 history steps, 30 epochs)
python train_trea.py \
  --dataset YAGO \
  --data_dir "$DATA_DIR" \
  --gpu $GPU

# WIKI  (256-dim, 12 history steps, 40 epochs)
python train_trea.py \
  --dataset WIKI \
  --data_dir "$DATA_DIR" \
  --gpu $GPU

echo ""
echo "All done! Results saved in checkpoints/"
