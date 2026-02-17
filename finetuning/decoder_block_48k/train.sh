#!/bin/bash
# DecoderBlock Addition Method - 48kHz Training Script
#
# Usage:
#   bash finetuning/decoder_block_48k/train.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Training settings
TRAIN_SHARDS="/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/train/*.tar"
VAL_SHARDS="/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/val/*.tar"
OUTPUT_DIR="${SCRIPT_DIR}/output"
RUN_NUMBER=4

uv run accelerate launch "${SCRIPT_DIR}/train.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards "${VAL_SHARDS}" \
    --output_dir "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --resume_from "${OUTPUT_DIR}/run3/checkpoint-best" \
    --num_frozen 0 \
    --batch_size 6 \
    --lr 1e-4 \
    --max_train_steps 1000000 \
    --gradient_accumulation_steps 16 \
    --max_audio_length 5.0 \
    --l1_weight 0.0 \
    --stft_weight 0.3 \
    --mel_weight 1.0 \
    --rms_weight 3.0 \
    --save_every 5000 \
    --eval_every 1000 \
    --log_every 10 \
    --wandb_project qwen3-tts-decoder-block-48k \
    --wandb_run_name "run${RUN_NUMBER}" \
    --mixed_precision bf16
