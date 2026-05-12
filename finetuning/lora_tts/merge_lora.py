# coding=utf-8
# Copyright 2026 The Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Merge LoRA adapter weights into the base Qwen3-TTS model.

Usage:
    python finetuning/lora/merge_lora.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --lora_adapter_path output/lora/checkpoint-best \
        --output_path output/Qwen3-TTS-12Hz-0.6B-Finetuned
"""

import argparse
import json
import os
import shutil
import sys

import torch
from peft import PeftModel
from safetensors.torch import save_file

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")

    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Path to base Qwen3-TTS model",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        required=True,
        help="Path to LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for merged model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for merging",
    )

    return parser.parse_args()


def merge_lora(args):
    """Merge LoRA adapter into base model and save."""
    print(f"Loading base model from {args.base_model_path}")

    # Load base model
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )

    print(f"Loading LoRA adapter from {args.lora_adapter_path}")

    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        qwen3tts.model,
        args.lora_adapter_path,
        torch_dtype=torch.bfloat16,
    )

    print("Merging LoRA weights into base model...")

    # Merge LoRA weights
    model = model.merge_and_unload()

    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)

    # Copy config and other files from base model
    print(f"Copying config files from {args.base_model_path}")

    # Files to copy from base model
    files_to_copy = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "vocab.txt",
        "merges.txt",
    ]

    for filename in files_to_copy:
        src_path = os.path.join(args.base_model_path, filename)
        if os.path.exists(src_path):
            dst_path = os.path.join(args.output_path, filename)
            shutil.copy2(src_path, dst_path)
            print(f"  Copied {filename}")

    # Update config to mark as fine-tuned
    config_path = os.path.join(args.output_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

        # Add fine-tuning metadata
        config["finetuned"] = True
        config["lora_merged"] = True
        config["base_model"] = args.base_model_path
        config["lora_adapter"] = args.lora_adapter_path

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # Save merged model weights
    print("Saving merged model weights...")

    state_dict = model.state_dict()

    # Convert to CPU and proper dtype for saving
    state_dict = {
        k: v.to("cpu").to(torch.bfloat16) if v.dtype.is_floating_point else v.to("cpu")
        for k, v in state_dict.items()
    }

    # Remove speaker_encoder weights (they are frozen and unchanged)
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]

    # Save as safetensors
    save_path = os.path.join(args.output_path, "model.safetensors")
    save_file(state_dict, save_path)
    print(f"Saved merged model to {save_path}")

    # Copy speech tokenizer files if they exist
    tokenizer_subdir = os.path.join(args.base_model_path, "speech_tokenizer")
    if os.path.exists(tokenizer_subdir):
        dst_tokenizer_dir = os.path.join(args.output_path, "speech_tokenizer")
        if os.path.exists(dst_tokenizer_dir):
            shutil.rmtree(dst_tokenizer_dir)
        shutil.copytree(tokenizer_subdir, dst_tokenizer_dir)
        print("Copied speech_tokenizer directory")

    # Copy speaker_encoder files if they exist
    speaker_encoder_subdir = os.path.join(args.base_model_path, "speaker_encoder")
    if os.path.exists(speaker_encoder_subdir):
        dst_speaker_dir = os.path.join(args.output_path, "speaker_encoder")
        if os.path.exists(dst_speaker_dir):
            shutil.rmtree(dst_speaker_dir)
        shutil.copytree(speaker_encoder_subdir, dst_speaker_dir)
        print("Copied speaker_encoder directory")

    print(f"\nMerged model saved to {args.output_path}")
    print("You can now use this model directly without loading LoRA adapter.")


if __name__ == "__main__":
    args = parse_args()
    merge_lora(args)
