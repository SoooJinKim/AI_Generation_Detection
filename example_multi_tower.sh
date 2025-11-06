#!/bin/bash

# Multi-Tower Training Example
# Texture, Edge, Sharpen transformation을 동시에 사용하는 멀티 타워 학습

python train.py \
    --mode realfake \
    --dataroot ./dataset \
    --train_split train \
    --val_split val \
    --transform_modes texture,edge,sharpen \
    --fusion_method concat \
    --batch_size 16 \
    --lr 0.0001 \
    --optim adam \
    --niter 100 \
    --delr_freq 20 \
    --earlystop_epoch 15 \
    --name multi_tower_all \
    --gpu_ids 0

echo "✅ Multi-Tower training script completed!"

