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
Inference script for Qwen3-TTS with LoRA adapter.

Usage:
    # With merged model
    python finetuning/lora/inference_lora.py \
        --model_path output/Qwen3-TTS-12Hz-0.6B-Finetuned \
        --text "こんにちは、音声合成のテストです。" \
        --output_audio output.wav

    # With base model + LoRA adapter
    python finetuning/lora/inference_lora.py \
        --model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --lora_adapter_path output/lora/checkpoint-best \
        --text "こんにちは、音声合成のテストです。" \
        --output_audio output.wav

    # With reference audio for voice cloning
    python finetuning/lora/inference_lora.py \
        --model_path output/Qwen3-TTS-12Hz-0.6B-Finetuned \
        --text "こんにちは、音声合成のテストです。" \
        --ref_audio reference.wav \
        --ref_text "これは参照音声のテキストです。" \
        --output_audio output.wav
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from peft import PeftModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with Qwen3-TTS LoRA model")

    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to merged model or base model",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Path to LoRA adapter (if using base model)",
    )

    # Input arguments
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Auto",
        help="Language for synthesis",
    )

    # Reference audio arguments (for voice cloning)
    parser.add_argument(
        "--ref_audio",
        type=str,
        default=None,
        help="Reference audio for voice cloning",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="Reference text for voice cloning (required if ref_audio is provided)",
    )
    parser.add_argument(
        "--x_vector_only",
        action="store_true",
        help="Use only x-vector (speaker embedding) for voice cloning",
    )

    # Output arguments
    parser.add_argument(
        "--output_audio",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )

    # Generation arguments
    parser.add_argument(
        "--do_sample",
        type=bool,
        default=True,
        help="Whether to use sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Top-p sampling",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.05,
        help="Repetition penalty",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate",
    )

    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type",
    )

    return parser.parse_args()


def get_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch dtype."""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]


def load_model(
    model_path: str,
    lora_adapter_path: Optional[str] = None,
    device: str = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Qwen3TTSModel:
    """
    Load Qwen3-TTS model with optional LoRA adapter.

    Args:
        model_path: Path to model (merged or base)
        lora_adapter_path: Optional path to LoRA adapter
        device: Device to use
        dtype: Data type

    Returns:
        Loaded Qwen3TTSModel
    """
    # Determine device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {model_path}")
    print(f"Device: {device}, dtype: {dtype}")

    # Load base model
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="flash_attention_2" if device == "cuda" else "eager",
    )

    # Load LoRA adapter if provided
    if lora_adapter_path is not None:
        print(f"Loading LoRA adapter from {lora_adapter_path}")
        qwen3tts.model = PeftModel.from_pretrained(
            qwen3tts.model,
            lora_adapter_path,
            torch_dtype=dtype,
        )
        # Optionally merge for faster inference
        # qwen3tts.model = qwen3tts.model.merge_and_unload()

    return qwen3tts


def generate_speech(
    model: Qwen3TTSModel,
    text: str,
    language: str = "Auto",
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    x_vector_only: bool = False,
    **generation_kwargs,
) -> tuple:
    """
    Generate speech from text.

    Args:
        model: Qwen3TTSModel
        text: Text to synthesize
        language: Language
        ref_audio: Optional reference audio for voice cloning
        ref_text: Optional reference text for voice cloning
        x_vector_only: Whether to use only x-vector for voice cloning
        **generation_kwargs: Generation parameters

    Returns:
        Tuple of (waveform, sample_rate)
    """
    if ref_audio is not None:
        # Voice cloning mode
        if not x_vector_only and ref_text is None:
            raise ValueError("ref_text is required when ref_audio is provided and x_vector_only is False")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
            **generation_kwargs,
        )
    else:
        # Check model type and use appropriate generation method
        if model.model.tts_model_type == "base":
            raise ValueError(
                "Base model requires reference audio for voice cloning. "
                "Please provide --ref_audio and optionally --ref_text."
            )
        elif model.model.tts_model_type == "custom_voice":
            # For custom voice model, need to specify a speaker
            # Get available speakers
            speakers = model.get_supported_speakers()
            if speakers:
                speaker = speakers[0]
                print(f"Using speaker: {speaker}")
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    speaker=speaker,
                    language=language,
                    **generation_kwargs,
                )
            else:
                raise ValueError("No speakers available in custom voice model")
        elif model.model.tts_model_type == "voice_design":
            wavs, sr = model.generate_voice_design(
                text=text,
                instruct="",
                language=language,
                **generation_kwargs,
            )
        else:
            raise ValueError(f"Unsupported model type: {model.model.tts_model_type}")

    return wavs[0], sr


def main():
    args = parse_args()

    # Load model
    dtype = get_dtype(args.dtype)
    model = load_model(
        args.model_path,
        lora_adapter_path=args.lora_adapter_path,
        device=args.device,
        dtype=dtype,
    )

    # Generate speech
    print(f"Generating speech for: {args.text}")

    generation_kwargs = {
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens,
    }

    wav, sr = generate_speech(
        model,
        text=args.text,
        language=args.language,
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only=args.x_vector_only,
        **generation_kwargs,
    )

    # Save output
    output_dir = os.path.dirname(args.output_audio)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sf.write(args.output_audio, wav, sr)
    print(f"Saved audio to {args.output_audio}")
    print(f"Sample rate: {sr} Hz")
    print(f"Duration: {len(wav) / sr:.2f} seconds")


if __name__ == "__main__":
    main()
