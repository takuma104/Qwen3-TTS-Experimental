# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
学習済みアップサンプラーを24kHzモデルにマージして48kHzモデルを作成

Usage:
    python finetuning/tokenizer48k/merge_upsampler.py \
        --base_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
        --upsampler_checkpoint output/upsampler/checkpoint-best \
        --output_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_tts.core.tokenizer_48k.configuration import Qwen3TTSTokenizer48kConfig
from qwen_tts.core.tokenizer_48k.modeling import Qwen3TTSTokenizer48kModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge upsampler weights into 48kHz model")
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="ベースとなる24kHzモデルのパス",
    )
    parser.add_argument(
        "--upsampler_checkpoint",
        type=str,
        required=True,
        help="学習済みアップサンプラーのチェックポイントパス",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="出力する48kHzモデルのパス",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Base model: {args.base_model_path}")
    print(f"Upsampler checkpoint: {args.upsampler_checkpoint}")
    print(f"Output path: {args.output_path}")

    # 出力ディレクトリを作成
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # アップサンプラーの設定を読み込み
    upsampler_config_path = Path(args.upsampler_checkpoint) / "config.json"
    with open(upsampler_config_path) as f:
        upsampler_config = json.load(f)

    print(f"Upsampler config: {upsampler_config}")

    # ベースモデルの config.json を読み込み
    if os.path.exists(args.base_model_path):
        base_config_path = Path(args.base_model_path) / "config.json"
    else:
        # Hugging Face Hub からダウンロード
        from huggingface_hub import hf_hub_download
        base_config_path = hf_hub_download(
            repo_id=args.base_model_path,
            filename="config.json",
        )

    with open(base_config_path) as f:
        config_dict = json.load(f)

    # decoder_config を更新して48kHz設定を追加
    decoder_config = config_dict.get("decoder_config", {})
    decoder_config["enable_48khz_upsampler"] = True
    decoder_config["upsampler_hidden_dim"] = upsampler_config.get("upsampler_hidden_dim", 32)
    decoder_config["upsampler_kernel_size"] = upsampler_config.get("upsampler_kernel_size", 4)
    decoder_config["upsampler_factor"] = upsampler_config.get("upsampler_factor", 2)
    config_dict["decoder_config"] = decoder_config

    # model_type を 48k に更新
    config_dict["model_type"] = "qwen3_tts_tokenizer_48k"

    # output_sample_rate と decode_upsample_rate を更新
    upsampler_factor = decoder_config["upsampler_factor"]
    config_dict["output_sample_rate"] = config_dict.get("output_sample_rate", 24000) * upsampler_factor
    config_dict["decode_upsample_rate"] = config_dict.get("decode_upsample_rate", 1920) * upsampler_factor

    # 新しい config.json を保存
    output_config_path = output_path / "config.json"
    with open(output_config_path, "w") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {output_config_path}")

    # ベースモデルの重みをロード
    print("Loading base model weights...")
    if os.path.exists(args.base_model_path):
        base_model_files = list(Path(args.base_model_path).glob("*.safetensors"))
        if not base_model_files:
            base_model_files = list(Path(args.base_model_path).glob("*.bin"))
    else:
        from huggingface_hub import hf_hub_download
        # model.safetensors をダウンロード
        model_file = hf_hub_download(
            repo_id=args.base_model_path,
            filename="model.safetensors",
        )
        base_model_files = [model_file]

    # 重みを読み込み
    base_state_dict = {}
    for model_file in base_model_files:
        if str(model_file).endswith(".safetensors"):
            base_state_dict.update(load_file(str(model_file)))
        else:
            base_state_dict.update(torch.load(str(model_file), map_location="cpu"))

    print(f"Loaded {len(base_state_dict)} keys from base model")

    # アップサンプラーの重みをロード
    print("Loading upsampler weights...")
    upsampler_weights_path = Path(args.upsampler_checkpoint) / "upsampler.safetensors"
    upsampler_state_dict = load_file(str(upsampler_weights_path))
    print(f"Loaded {len(upsampler_state_dict)} keys from upsampler")

    # 重みをマージ
    merged_state_dict = {**base_state_dict, **upsampler_state_dict}
    print(f"Merged state dict has {len(merged_state_dict)} keys")

    # マージした重みを保存
    output_model_path = output_path / "model.safetensors"
    save_file(merged_state_dict, str(output_model_path))
    print(f"Saved merged model to {output_model_path}")

    # その他の必要なファイルをコピー
    files_to_copy = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in files_to_copy:
        if os.path.exists(args.base_model_path):
            src_path = Path(args.base_model_path) / filename
        else:
            try:
                from huggingface_hub import hf_hub_download
                src_path = hf_hub_download(
                    repo_id=args.base_model_path,
                    filename=filename,
                )
            except Exception:
                continue

        if os.path.exists(src_path):
            dst_path = output_path / filename
            shutil.copy(src_path, dst_path)
            print(f"Copied {filename}")

    # モデルをテストロード
    print("\nTesting model load...")
    try:
        model = Qwen3TTSTokenizer48kModel.from_pretrained(
            str(output_path),
            trust_remote_code=True,
        )
        print(f"Model loaded successfully!")
        print(f"  output_sample_rate: {model.config.output_sample_rate}")
        print(f"  decode_upsample_rate: {model.config.decode_upsample_rate}")
        print(f"  enable_48khz_upsampler: {model.config.decoder_config.enable_48khz_upsampler}")
        print(f"  upsampler: {model.decoder.upsampler is not None}")

        # パラメータ数
        total_params = sum(p.numel() for p in model.parameters())
        upsampler_params = sum(p.numel() for p in model.decoder.upsampler.parameters()) if model.decoder.upsampler else 0
        print(f"  Total parameters: {total_params:,}")
        print(f"  Upsampler parameters: {upsampler_params:,}")

    except Exception as e:
        print(f"Warning: Model test load failed: {e}")

    print(f"\n48kHz model saved to: {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
