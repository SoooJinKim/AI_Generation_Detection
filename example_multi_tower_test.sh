#!/bin/bash

# Single-Tower Training Example
# Texture transformation을 사용한 단일 타워 학습

python test_multi_model.py \
    --input "./samples/train/fake/0.png"  \
    --model_texture "./checkpoints/single_tower_texture2025_11_12_17_04_55/model_epoch_best.pth"  \
    --model_edge "./checkpoints/single_tower_edge2025_11_12_17_10_04/model_epoch_best.pth" \
    --model_sharpen "./checkpoints/single_tower_sharpen2025_11_12_17_13_28/model_epoch_best.pth"  \
    --gpu_ids 0

echo "✅ Multi-Tower testing script completed!"

