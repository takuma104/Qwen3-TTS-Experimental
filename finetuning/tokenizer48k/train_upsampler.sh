#!/bin/sh

uv run accelerate launch train_upsampler.py \
    --train_shards ".../train/*.tar" \
    --val_shards ".../val/*.tar" \
    --output_dir output/upsampler_run5 \
    --batch_size 8 \
    --lr 1e-4 \
    --max_train_steps 100000 \
    --l1_weight 0.0 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --rms_weight 10.0 \
    --log_with wandb
