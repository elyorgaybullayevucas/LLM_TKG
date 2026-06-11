#!/bin/bash
# Data preprocessing — history retrieval for all 3 datasets
# Run from project root: bash scripts/retrieve.sh

DS=${1:-icews18}   # default: icews18  |  usage: bash scripts/retrieve.sh yago

echo "=== Retrieving for dataset: $DS ==="
python3 data_utils/retrieve.py -d "$DS" -t rbmh

echo "Done. Files saved in data/processed/"
