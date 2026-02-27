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
RUN_NUMBER=10

# Pre-trained generator checkpoint (from reconstruction-only training)
# GENERATOR_CHECKPOINT="${OUTPUT_DIR}/run2/checkpoint-best"

uv run accelerate launch "${SCRIPT_DIR}/train_gan.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards "${VAL_SHARDS}" \
    --output_dir "${OUTPUT_DIR}/run_gan${RUN_NUMBER}" \
    --batch_size 8 \
    --lr_g 1e-4 \
    --lr_d 1e-4 \
    --max_train_steps 500000 \
    --gradient_accumulation_steps 4 \
    --max_audio_length 5.0 \
    --lambda_adv 1.0 \
    --lambda_fm 1.0 \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms 1.0 \
    --lambda_d_msd 0.3 \
    --save_every 1250 \
    --eval_every 250 \
    --log_every 3 \
    --wandb_project qwen3-tts-decoder-block-48k-gan \
    --wandb_run_name "run_gan${RUN_NUMBER}" \
    --mixed_precision bf16
