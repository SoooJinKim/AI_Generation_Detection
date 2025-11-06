#!/bin/bash

# Single-Tower Training Example (CPU Mode)
# 동영상 프레임 샘플링 포함

python train.py \
    --tower_mode single \
    --mode realfake \
    --dataroot ./dataset \
    --train_split train \
    --val_split val \
    --transform_mode texture \
    --batch_size 2 \
    --lr 0.0001 \
    --optim adam \
    --niter 2 \
    --delr_freq 1 \
    --earlystop_epoch 15 \
    --name single_tower_texture_cpu \
    --gpu_ids -1 \
    --num_threads 2 \
    --video_num_frames 5 \
    --video_sampling uniform \
    --video_voting max

echo "✅ Single-Tower CPU training completed!"
