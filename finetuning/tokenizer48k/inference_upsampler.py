# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler 推論スクリプト

学習済みのupsampler.safetensorsとconfig.jsonを読み込んで、
Qwen3TTSTokenizerを完全な48kHz対応モデルとして復元し、推論を行います。

Usage:
    # 音声ファイルをエンコード→48kHzデコード
    python finetuning/tokenizer48k/inference_upsampler.py \
        --upsampler_checkpoint output/upsampler/checkpoint-best \
        --input_audio input.wav \
        --output_audio output_48k.wav

    # audio_codesファイル（.npy）から48kHzデコード
    python finetuning/tokenizer48k/inference_upsampler.py \
        --upsampler_checkpoint output/upsampler/checkpoint-best \
        --input_codes input_codes.npy \
        --output_audio output_48k.wav

    # マージ済みの48kHzモデルを直接使用
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

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_48k.modeling import UpSamplerBlock


def parse_args():
    parser = argparse.ArgumentParser(description="48kHz Upsampler Inference")

    # モデル設定（2つの方法から選択）
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="マージ済みの48kHzモデルのパス（これを指定した場合、upsampler_checkpointは不要）",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="ベースとなる24kHzモデルのパス",
    )
    parser.add_argument(
        "--upsampler_checkpoint",
        type=str,
        default=None,
        help="学習済みアップサンプラーのチェックポイントパス",
    )

    # 入力（どちらか一方を指定）
    parser.add_argument(
        "--input_audio",
        type=str,
        default=None,
        help="入力音声ファイルのパス（エンコード→48kHzデコード）",
    )
    parser.add_argument(
        "--input_codes",
        type=str,
        default=None,
        help="入力audio_codesファイルのパス（.npy形式、shape: [seq_len, 16]）",
    )

    # 出力
    parser.add_argument(
        "--output_audio",
        type=str,
        default="output_48k.wav",
        help="出力音声ファイルのパス",
    )

    # デバイス設定
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="使用するデバイス（auto, cpu, cuda, cuda:0, etc.）",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="モデルのデータ型",
    )

    return parser.parse_args()


class Qwen3TTSTokenizer48kHz:
    """
    48kHz対応のQwen3TTSTokenizerラッパークラス

    学習済みのアップサンプラーを24kHzモデルに追加して、
    48kHz出力を可能にします。
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
            base_model_path: ベースの24kHzモデルのパス
            upsampler_checkpoint: 学習済みアップサンプラーのチェックポイントパス
            merged_model_path: マージ済みの48kHzモデルのパス（指定時は他のパラメータを無視）
            device: 使用するデバイス
            dtype: モデルのデータ型
        """
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        if merged_model_path:
            # マージ済みモデルを直接ロード
            self._load_merged_model(merged_model_path)
        else:
            # ベースモデル + アップサンプラーの組み合わせ
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
        """マージ済みの48kHzモデルをロード"""
        print(f"Loading merged 48kHz model from {model_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )
        self.output_sample_rate = self.tokenizer.get_output_sample_rate()
        print(f"Model loaded. Output sample rate: {self.output_sample_rate} Hz")

    def _load_base_with_upsampler(self, base_model_path: str, upsampler_checkpoint: str):
        """ベースモデルにアップサンプラーを追加してロード"""
        print(f"Loading base model from {base_model_path}...")

        # ベースの24kHzモデルをロード
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )

        # アップサンプラーの設定を読み込み
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

        # アップサンプラーを作成
        upsampler = UpSamplerBlock(
            in_channels=1,
            hidden_dim=upsampler_config.get("upsampler_hidden_dim", 32),
            kernel_size=upsampler_config.get("upsampler_kernel_size", 4),
            upsample_factor=upsampler_config.get("upsampler_factor", 2),
        )

        # 重みをロード
        print(f"Loading upsampler weights from {weights_path}...")
        upsampler_state_dict = load_file(str(weights_path))

        # state_dict のキーから "decoder.upsampler." プレフィックスを除去
        cleaned_state_dict = {}
        for k, v in upsampler_state_dict.items():
            if k.startswith("upsampler."):
                new_key = k.replace("upsampler.", "")
                cleaned_state_dict[new_key] = v
            else:
                cleaned_state_dict[k] = v

        upsampler.load_state_dict(cleaned_state_dict)
        upsampler = upsampler.to(self.device).to(self.dtype)
        upsampler.eval()

        # デコーダーにアップサンプラーを追加
        decoder = self.tokenizer.model.decoder
        decoder.upsampler = upsampler
        decoder.total_upsample *= upsampler_config.get("upsampler_factor", 2)

        # config とモデルのインスタンス変数を両方更新
        # （モデルは __init__ 時に config からコピーしているため両方必要）
        upsampler_factor = upsampler_config.get("upsampler_factor", 2)
        original_rate = self.tokenizer.config.output_sample_rate
        new_output_rate = original_rate * upsampler_factor
        new_decode_upsample_rate = self.tokenizer.config.decode_upsample_rate * upsampler_factor

        # config を更新
        self.tokenizer.config.output_sample_rate = new_output_rate
        self.tokenizer.config.decode_upsample_rate = new_decode_upsample_rate

        # モデルのインスタンス変数も直接更新（get_output_sample_rate() はこちらを参照）
        self.tokenizer.model.output_sample_rate = new_output_rate
        self.tokenizer.model.decode_upsample_rate = new_decode_upsample_rate

        self.output_sample_rate = new_output_rate
        print(f"48kHz upsampler attached. Output sample rate: {self.output_sample_rate} Hz")

    def encode(self, audio_path: str):
        """
        音声ファイルをエンコードしてaudio_codesを取得

        Args:
            audio_path: 入力音声ファイルのパス

        Returns:
            エンコード結果（audio_codesを含む）
        """
        return self.tokenizer.encode(audio_path, return_dict=True)

    def decode(self, encoded) -> Tuple[List[np.ndarray], int]:
        """
        エンコード結果から48kHz音声をデコード

        Args:
            encoded: encode()の戻り値、または audio_codes を含む dict

        Returns:
            (wavs, sample_rate): 波形のリストとサンプルレート
        """
        return self.tokenizer.decode(encoded)

    def decode_from_codes(
        self,
        audio_codes: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[List[np.ndarray], int]:
        """
        audio_codes から直接48kHz音声をデコード

        Args:
            audio_codes: shape [seq_len, 16] または [batch, seq_len, 16] のコード

        Returns:
            (wavs, sample_rate): 波形のリストとサンプルレート
        """
        if isinstance(audio_codes, np.ndarray):
            audio_codes = torch.from_numpy(audio_codes).long()

        # shape を確認・調整
        if audio_codes.dim() == 2:
            # [seq_len, 16] -> [1, seq_len, 16]
            audio_codes = audio_codes.unsqueeze(0)

        return self.tokenizer.decode({"audio_codes": audio_codes})

    def encode_decode(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        音声ファイルをエンコードして48kHzでデコード（ラウンドトリップ）

        Args:
            audio_path: 入力音声ファイルのパス

        Returns:
            (wav, sample_rate): 波形とサンプルレート
        """
        encoded = self.encode(audio_path)
        wavs, sr = self.decode(encoded)
        return wavs[0], sr

    def get_output_sample_rate(self) -> int:
        """出力サンプルレートを取得"""
        return self.output_sample_rate


def main():
    args = parse_args()

    # 入力の検証
    if args.input_audio is None and args.input_codes is None:
        print("Error: Either --input_audio or --input_codes must be specified")
        sys.exit(1)

    if args.model_path is None and args.upsampler_checkpoint is None:
        print("Error: Either --model_path or --upsampler_checkpoint must be specified")
        sys.exit(1)

    # モデルをロード
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

    # 推論
    print("\n" + "=" * 50)
    print("Running Inference")
    print("=" * 50)

    if args.input_audio:
        # 音声ファイルをエンコード→48kHzデコード
        print(f"Input audio: {args.input_audio}")
        wav, sr = tokenizer.encode_decode(args.input_audio)
    else:
        # audio_codesから48kHzデコード
        print(f"Input codes: {args.input_codes}")
        audio_codes = np.load(args.input_codes)
        print(f"Audio codes shape: {audio_codes.shape}")
        wavs, sr = tokenizer.decode_from_codes(audio_codes)
        wav = wavs[0]

    # 出力を保存
    print(f"\nOutput sample rate: {sr} Hz")
    print(f"Output duration: {len(wav) / sr:.2f} seconds")
    print(f"Saving to: {args.output_audio}")

    sf.write(args.output_audio, wav, sr)

    print("\n" + "=" * 50)
    print("Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
