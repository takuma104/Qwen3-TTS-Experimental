#!/bin/bash
# DecoderBlock Addition Method - 48kHz GAN Training Script
#
# Requires pre-trained generator from reconstruction-only training (train.sh).
#
# Usage:
#   bash finetuning/decoder_block_48k/train_gan.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Training settings
TRAIN_SHARDS="/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/train/*.tar"
VAL_SHARDS="/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/val/*.tar"
OUTPUT_DIR="${SCRIPT_DIR}/output"
RUN_NUMBER=1

# Pre-trained generator checkpoint (from reconstruction-only training)
GENERATOR_CHECKPOINT="${OUTPUT_DIR}/run2/checkpoint-best"

uv run accelerate launch "${SCRIPT_DIR}/train_gan.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards "${VAL_SHARDS}" \
    --output_dir "${OUTPUT_DIR}/run_gan${RUN_NUMBER}" \
    --num_frozen 0 \
    --batch_size 6 \
    --lr_g 1e-4 \
    --lr_d 2e-4 \
    --max_train_steps 500000 \
    --gradient_accumulation_steps 10 \
    --max_audio_length 5.0 \
    --lambda_adv 1.0 \
    --lambda_fm 2.0 \
    --lambda_mel 45.0 \
    --save_every 5000 \
    --eval_every 1000 \
    --log_every 10 \
    --wandb_project qwen3-tts-decoder-block-48k \
    --wandb_run_name "run_gan${RUN_NUMBER}" \
    --mixed_precision bf16
