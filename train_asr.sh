#!/bin/bash

uv run finetuning/sft_asr_12hz.py \
  --init_tts_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --qwen3_model_path Qwen/Qwen3-0.6B \
  --data_lst /mnt/artifacts/qwen3-tts-stt/overfit/data.lst \
  --output_dir output/run0 \
  --batch_size 2 \
  --num_epochs 10000
