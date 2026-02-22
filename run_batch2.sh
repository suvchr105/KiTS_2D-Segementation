#!/bin/bash
# KiTS23 Pipeline — Process remaining cases (batch 2)
# Run this after the first pipeline batch finishes to process
# cases that were downloaded after the first run started.

set -e
cd /mnt/raid/obed/Suvadip/KITS23_2D
source kits_env/bin/activate

echo "=== KiTS23 Pipeline Batch 2 ==="
echo "Processing all cases (will skip already-processed ones via existing output files)"

CUDA_VISIBLE_DEVICES=3 python run_pipeline.py \
    --output_dir /mnt/raid/obed/Suvadip/KITS23_2D/output \
    --gpu 3 \
    --save_format npz \
    --target_spacing 1.0 1.0 3.0 \
    --context_slices 2

echo "=== Batch 2 Complete ==="
