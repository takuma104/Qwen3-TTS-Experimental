# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler Inference Script

Loads trained upsampler.safetensors and config.json,
restores Qwen3TTSTokenizer as a complete 48kHz-compatible model, and performs inference.

Usage:
    # Encode audio file → decode to 48kHz
    python finetuning/tokenizer48k/inference_upsampler.py \
        --upsampler_checkpoint output/upsampler/checkpoint-best \
        --input_audio input.wav \
        --output_audio output_48k.wav

    # Decode from audio_codes file (.npy) to 48kHz
    python finetuning/tokenizer48k/inference_upsampler.py \
        --upsampler_checkpoint output/upsampler/checkpoint-best \
        --input_codes input_codes.npy \
        --output_audio output_48k.wav

    # Use merged 48kHz model directly
    python finetuning/tokenizer48k/inference_upsampler.py \
        --model_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz \
        --input_audio input.wav \
        --output_audio output_48k.wav
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_48k.configuration import Qwen3TTSTokenizer48kDecoderConfig
from qwen_tts.core.tokenizer_48k.modeling import Qwen3TTSTokenizer48kDecoder


def parse_args():
    parser = argparse.ArgumentParser(description="48kHz Upsampler Inference")

    # Model settings (choose one of two methods)
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to merged 48kHz model (if specified, upsampler_checkpoint is not needed)",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz model path",
    )
    parser.add_argument(
        "--upsampler_checkpoint",
        type=str,
        default=None,
        help="Trained upsampler checkpoint path",
    )

    # Input (specify one)
    parser.add_argument(
        "--input_audio",
        type=str,
        default=None,
        help="Input audio file path (encode → decode to 48kHz)",
    )
    parser.add_argument(
        "--input_codes",
        type=str,
        default=None,
        help="Input audio_codes file path (.npy format, shape: [seq_len, 16])",
    )

    # Output
    parser.add_argument(
        "--output_audio",
        type=str,
        default="output_48k.wav",
        help="Output audio file path",
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model data type",
    )

    return parser.parse_args()


class Qwen3TTSTokenizer48kHz:
    """
    Qwen3TTSTokenizer wrapper class with 48kHz support

    Adds trained upsampler to 24kHz model to enable
    48kHz output.
    """

    def __init__(
        self,
        base_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        upsampler_checkpoint: Optional[str] = None,
        merged_model_path: Optional[str] = None,
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        """
        Args:
            base_model_path: Base 24kHz model path
            upsampler_checkpoint: Trained upsampler checkpoint path
            merged_model_path: Merged 48kHz model path (ignores other parameters if specified)
            device: Device to use
            dtype: Model data type
        """
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        if merged_model_path:
            # Load merged model directly
            self._load_merged_model(merged_model_path)
        else:
            # Combination of base model + upsampler
            if upsampler_checkpoint is None:
                raise ValueError(
                    "Either 'merged_model_path' or 'upsampler_checkpoint' must be specified"
                )
            self._load_base_with_upsampler(base_model_path, upsampler_checkpoint)

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map[dtype]

    def _load_merged_model(self, model_path: str):
        """Load merged 48kHz model"""
        print(f"Loading merged 48kHz model from {model_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )
        self.output_sample_rate = self.tokenizer.get_output_sample_rate()
        print(f"Model loaded. Output sample rate: {self.output_sample_rate} Hz")

    def _load_base_with_upsampler(
        self, base_model_path: str, upsampler_checkpoint: str
    ):
        """Load base model and replace decoder with Qwen3TTSTokenizer48kDecoder"""
        print(f"Loading base model from {base_model_path}...")

        # Load base 24kHz model
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )

        # Load upsampler configuration
        checkpoint_path = Path(upsampler_checkpoint)
        config_path = checkpoint_path / "config.json"
        weights_path = checkpoint_path / "upsampler.safetensors"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        print(f"Loading upsampler config from {config_path}...")
        with open(config_path) as f:
            upsampler_config = json.load(f)

        print(f"Upsampler config: {upsampler_config}")

        # Create a 48k decoder config from the base decoder config + upsampler params
        base_decoder_config = self.tokenizer.model.decoder.config
        decoder_48k_config = Qwen3TTSTokenizer48kDecoderConfig(
            enable_48khz_upsampler=True,
            upsampler_hidden_dim=upsampler_config.get("upsampler_hidden_dim", 32),
            upsampler_kernel_size=upsampler_config.get("upsampler_kernel_size", 4),
            upsampler_factor=upsampler_config.get("upsampler_factor", 2),
            **{
                k: v
                for k, v in base_decoder_config.to_dict().items()
                if k not in ("model_type", "transformers_version")
            },
        )

        # Create 48k decoder and load base decoder weights into it
        print("Creating Qwen3TTSTokenizer48kDecoder...")
        decoder_48k = Qwen3TTSTokenizer48kDecoder(decoder_48k_config)
        base_state_dict = self.tokenizer.model.decoder.state_dict()
        decoder_48k.load_state_dict(base_state_dict, strict=False)

        # Load upsampler weights
        print(f"Loading upsampler weights from {weights_path}...")
        upsampler_state_dict = load_file(str(weights_path))

        # Remove prefix from state_dict keys if present
        cleaned_state_dict = {}
        for k, v in upsampler_state_dict.items():
            if k.startswith("upsampler."):
                new_key = k.replace("upsampler.", "", 1)
                cleaned_state_dict[new_key] = v
            else:
                cleaned_state_dict[k] = v

        decoder_48k.upsampler.load_state_dict(cleaned_state_dict)
        decoder_48k = decoder_48k.to(self.device).to(self.dtype)
        decoder_48k.eval()

        # Replace the decoder
        self.tokenizer.model.decoder = decoder_48k

        # Update sample rates
        upsampler_factor = upsampler_config.get("upsampler_factor", 2)
        original_rate = self.tokenizer.config.output_sample_rate
        new_output_rate = original_rate * upsampler_factor
        new_decode_upsample_rate = (
            self.tokenizer.config.decode_upsample_rate * upsampler_factor
        )

        self.tokenizer.config.output_sample_rate = new_output_rate
        self.tokenizer.config.decode_upsample_rate = new_decode_upsample_rate
        self.tokenizer.model.output_sample_rate = new_output_rate
        self.tokenizer.model.decode_upsample_rate = new_decode_upsample_rate

        self.output_sample_rate = new_output_rate
        print(
            f"48kHz decoder replaced. Output sample rate: {self.output_sample_rate} Hz"
        )

    def encode(self, audio_path: str):
        """
        Encode audio file to get audio_codes

        Args:
            audio_path: Input audio file path

        Returns:
            Encoding result (containing audio_codes)
        """
        return self.tokenizer.encode(audio_path, return_dict=True)

    def decode(self, encoded) -> Tuple[List[np.ndarray], int]:
        """
        Decode 48kHz audio from encoding result

        Args:
            encoded: Return value of encode(), or dict containing audio_codes

        Returns:
            (wavs, sample_rate): List of waveforms and sample rate
        """
        return self.tokenizer.decode(encoded)

    def decode_from_codes(
        self,
        audio_codes: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[List[np.ndarray], int]:
        """
        Decode 48kHz audio directly from audio_codes

        Args:
            audio_codes: Codes with shape [seq_len, 16] or [batch, seq_len, 16]

        Returns:
            (wavs, sample_rate): List of waveforms and sample rate
        """
        if isinstance(audio_codes, np.ndarray):
            audio_codes = torch.from_numpy(audio_codes).long()

        # Check and adjust shape
        if audio_codes.dim() == 2:
            # [seq_len, 16] -> [1, seq_len, 16]
            audio_codes = audio_codes.unsqueeze(0)

        return self.tokenizer.decode({"audio_codes": audio_codes})

    def encode_decode(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Encode audio file and decode to 48kHz (round-trip)

        Args:
            audio_path: Input audio file path

        Returns:
            (wav, sample_rate): Waveform and sample rate
        """
        encoded = self.encode(audio_path)
        wavs, sr = self.decode(encoded)
        return wavs[0], sr

    def get_output_sample_rate(self) -> int:
        """Get output sample rate"""
        return self.output_sample_rate


def main():
    args = parse_args()

    # Validate inputs
    if args.input_audio is None and args.input_codes is None:
        print("Error: Either --input_audio or --input_codes must be specified")
        sys.exit(1)

    if args.model_path is None and args.upsampler_checkpoint is None:
        print("Error: Either --model_path or --upsampler_checkpoint must be specified")
        sys.exit(1)

    # Load model
    print("=" * 50)
    print("Initializing 48kHz Tokenizer")
    print("=" * 50)

    tokenizer = Qwen3TTSTokenizer48kHz(
        base_model_path=args.base_model_path,
        upsampler_checkpoint=args.upsampler_checkpoint,
        merged_model_path=args.model_path,
        device=args.device,
        dtype=args.dtype,
    )

    # Inference
    print("\n" + "=" * 50)
    print("Running Inference")
    print("=" * 50)

    if args.input_audio:
        # Encode audio file → decode to 48kHz
        print(f"Input audio: {args.input_audio}")
        wav, sr = tokenizer.encode_decode(args.input_audio)
    else:
        # Decode from audio_codes to 48kHz
        print(f"Input codes: {args.input_codes}")
        audio_codes = np.load(args.input_codes)
        print(f"Audio codes shape: {audio_codes.shape}")
        wavs, sr = tokenizer.decode_from_codes(audio_codes)
        wav = wavs[0]

    # Save output
    print(f"\nOutput sample rate: {sr} Hz")
    print(f"Output duration: {len(wav) / sr:.2f} seconds")
    print(f"Saving to: {args.output_audio}")

    sf.write(args.output_audio, wav, sr)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
