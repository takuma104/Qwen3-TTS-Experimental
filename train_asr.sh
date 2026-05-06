#!/bin/bash

uv run finetuning/sft_asr_12hz.py \
  --init_tts_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --qwen3_model_path Qwen/Qwen3-0.6B \
  --data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/train/data.lst \
  --eval_data_lst /mnt/artifacts/qwen3-tts-stt/gg_webdataset/val/data.lst \
  --report_to wandb \
  --output_dir output/run1 \
  --lr 1e-4 \
  --batch_size 64 \
  --max_batch_tokens 6000 \
  --eval_max_batch_tokens 6000 \
  --max_duration 20.0 \
  --min_duration 1.0 \
  --wandb_project qwen3-tts-asr \
  --wandb_run_name Run1 \
  --eval_every_steps 1000 \
  --save_every_steps 10000 \
  --max_eval_batches 1000 \
  --num_epochs 10000
