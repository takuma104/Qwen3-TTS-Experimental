#!/bin/sh

RUN_NUMBER=5

uv run accelerate launch train_upsampler.py \
    --train_shards "/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/train/*.tar" \
    --val_shards "/mnt/4tb/takuma/dataset/speech/gg_dataset/webdataset_upsampler/val/*.tar" \
    --output_dir output/upsampler_run${RUN_NUMBER} \
    --batch_size 16 \
    --lr 1e-4 \
    --upsampler_hidden_dim=64 \
    --max_train_steps 1000000 \
    --max_audio_length 5.0 \
    --l1_weight 0.0 \
    --stft_weight 1.0 \
    --mel_weight 1.0 \
    --rms_weight 10.0 \
    --log_with wandb \
    --wandb_project qwen3tts_tokenizer48k \
    --wandb_run_name run${RUN_NUMBER}


