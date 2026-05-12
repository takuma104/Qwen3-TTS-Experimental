#!/bin/sh

RUN_NUMBER=1

uv run accelerate launch finetuning/lora_tts/train_lora.py \
    --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
    --data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/train/data.lst \
    --eval_data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/val/data.lst \
    --output_dir output_tts/run${RUN_NUMBER} \
    --batch_size 64 \
    --max_batch_tokens 4000 \
    --gradient_accumulation_steps 2 \
    --lr 1e-4 \
    --num_epochs 100 \
    --wandb_project qwen3-tts-lora2 \
    --wandb_run_name Run${RUN_NUMBER} \
    --num_workers 4 \
    --val_every_n_steps 1000 \
    --save_every_n_steps 10000 \
    --save_total_limit 10 \
    --lora_r 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05

