#!/bin/bash

# Single-Tower Training Example
# Texture transformation을 사용한 단일 타워 학습

python -u train.py \
    --tower_mode single \
    --mode realfake \
    --dataroot "/Users/kimsoojin/Desktop/SKKU/AIchampion/1112/AI_Generation_Detection/samples" \
    --train_split train \
    --val_split val \
    --transform_mode texture \
    --batch_size 2 \
    --lr 0.0001 \
    --optim adam \
    --niter 5 \
    --delr_freq 20 \
    --earlystop_epoch 1 \
    --name single_tower_texture \
    --gpu_ids 0

echo "✅ Single-Tower training script completed!"

