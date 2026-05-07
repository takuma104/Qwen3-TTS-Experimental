#!/bin/bash

uv run finetuning/sft_asr_12hz.py \
  --init_tts_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --qwen3_model_path Qwen/Qwen3-1.7B \
  --data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/train/data.lst \
  --eval_data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/val/data.lst \
  --report_to wandb \
  --output_dir output/run3 \
  --use_acoustic_codebooks \
  --lr 1e-4 \
  --batch_size 64 \
  --max_batch_tokens 3000 \
  --eval_max_batch_tokens 3000 \
  --gradient_accumulation_steps 8 \
  --max_duration 20.0 \
  --min_duration 1.0 \
  --wandb_project qwen3-tts-asr \
  --wandb_run_name Run3 \
  --eval_every_steps 1000 \
  --save_every_steps 10000 \
  --max_eval_batches 1000 \
  --num_epochs 10000 \
  --use_8bit_optimizer
