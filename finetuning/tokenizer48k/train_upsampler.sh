#!/bin/sh

RUN_NUMBER=2

uv run accelerate launch train_upsampler.py \
    --train_shards "/mnt/4tb/takuma/dataset/speech/qwen3tts_tokenizer/train/*.tar" \
    --val_shards "/mnt/4tb/takuma/dataset/speech/qwen3tts_tokenizer/val/*.tar" \
    --output_dir output/upsampler_run${RUN_NUMBER} \
    --upsampler_hidden_dim=64 \
    --batch_size 4 \
    --lr 1e-4 \
    --max_train_steps 1000000 \
    --l1_weight 0.0 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --rms_weight 10.0 \
    --log_with wandb \
    --wandb_project qwen3tts_tokenizer48k \
    --wandb_run_name run${RUN_NUMBER}
