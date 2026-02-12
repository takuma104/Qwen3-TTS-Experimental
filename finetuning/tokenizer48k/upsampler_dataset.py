# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler 学習用データセット

データ形式:
- audio_codes: エンコード済みの音声コード (12Hz, 16 quantizers)
- audio: 元の音声ファイルパス（48kHzまたはリサンプリング対象）
"""

from typing import List

import librosa
import numpy as np
import torch


def collate_fn(batch: List[dict]) -> dict:
    """
    バッチをコレートする関数

    異なる長さのオーディオをパディングして揃える
    """
    # 最大長を取得
    max_codes = max(b["audio_codes"].shape[0] for b in batch)
    max_samples_48k = max(b["audio_48k"].shape[0] for b in batch)
    max_samples_24k = max(b["audio_24k"].shape[0] for b in batch)

    batch_size = len(batch)

    # バッチテンソルを初期化
    audio_codes = torch.zeros(batch_size, max_codes, 16, dtype=torch.long)
    audio_48k = torch.zeros(batch_size, max_samples_48k)
    audio_24k = torch.zeros(batch_size, max_samples_24k)
    code_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_48k_lengths = torch.zeros(batch_size, dtype=torch.long)
    audio_24k_lengths = torch.zeros(batch_size, dtype=torch.long)

    for i, b in enumerate(batch):
        codes_len = b["audio_codes"].shape[0]
        samples_48k = b["audio_48k"].shape[0]
        samples_24k = b["audio_24k"].shape[0]

        audio_codes[i, :codes_len] = b["audio_codes"]
        audio_48k[i, :samples_48k] = b["audio_48k"]
        audio_24k[i, :samples_24k] = b["audio_24k"]

        code_lengths[i] = codes_len
        audio_48k_lengths[i] = samples_48k
        audio_24k_lengths[i] = samples_24k

    return {
        "audio_codes": audio_codes,           # (batch, max_codes, 16)
        "audio_48k": audio_48k,               # (batch, max_samples_48k)
        "audio_24k": audio_24k,               # (batch, max_samples_24k)
        "code_lengths": code_lengths,         # (batch,)
        "audio_48k_lengths": audio_48k_lengths,  # (batch,)
        "audio_24k_lengths": audio_24k_lengths,  # (batch,)
    }


def create_webdataset_loader(
    shard_pattern: str,
    target_sample_rate: int = 48000,
    max_audio_length: float = 10.0,
    min_audio_length: float = 1.0,
    batch_size: int = 8,
    num_workers: int = 4,
    shuffle_buffer: int = 1000,
):
    """
    WebDataset形式のデータローダーを作成

    Args:
        shard_pattern: tarファイルのパターン (例: "output/shards-{000000..000010}.tar")
        target_sample_rate: ターゲットのサンプルレート
        max_audio_length: 最大オーディオ長（秒）
        min_audio_length: 最小オーディオ長（秒）
        batch_size: バッチサイズ
        num_workers: ワーカー数
        shuffle_buffer: シャッフルバッファサイズ

    Returns:
        DataLoader
    """
    import io
    import webdataset as wds
    from torch.utils.data import DataLoader

    def _process_sample(sample):
        """WebDatasetのサンプルを処理"""
        # audio_codes を numpy から tensor に変換
        audio_codes = sample["npy"]  # (seq_len, 16) の numpy 配列
        assert isinstance(audio_codes, np.ndarray), "audio_codes must be a numpy array"
        assert audio_codes.ndim == 1
        audio_codes = audio_codes.reshape(-1, 16) # (seq_len, 16)
        audio_codes = torch.from_numpy(audio_codes).long()
        num_codes = audio_codes.shape[0]

        # 最小長チェック
        min_codes = int(12 * min_audio_length)
        if num_codes < min_codes:
            return None

        # 音声データを取得（複数のフォーマットに対応）
        audio_data = None
        for ext in ["wav", "mp3", "flac", "ogg", "opus"]:
            if ext in sample:
                audio_data = sample[ext]
                break

        if audio_data is None:
            return None

        # librosa で読み込み
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)

        # 最大長でクロップ（必要な場合）
        max_codes = int(12 * max_audio_length)
        if num_codes > max_codes:
            # ランダムな開始位置を選択
            start_code = torch.randint(0, num_codes - max_codes, (1,)).item()
            end_code = start_code + max_codes

            audio_codes = audio_codes[start_code:end_code]
            num_codes = max_codes

            # オーディオも対応する範囲でクロップ
            samples_per_code = sr / 12
            start_sample = int(start_code * samples_per_code)
            end_sample = int(end_code * samples_per_code)
            audio = audio[start_sample:end_sample]

        # ターゲット（48kHz）にリサンプリング
        if sr != target_sample_rate:
            audio_48k = librosa.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        else:
            audio_48k = audio

        # 24kHzにもリサンプリング（参照用）
        if sr != 24000:
            audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        else:
            audio_24k = audio

        # tensor に変換
        audio_48k = torch.from_numpy(audio_48k).float()
        audio_24k = torch.from_numpy(audio_24k).float()

        return {
            "audio_codes": audio_codes,  # (seq_len, 16)
            "audio_48k": audio_48k,      # (samples_48k,)
            "audio_24k": audio_24k,      # (samples_24k,)
        }

    # WebDataset を構築
    dataset = (
        wds.WebDataset(shard_pattern, shardshuffle=1000)
        .shuffle(shuffle_buffer if shuffle_buffer > 0 else 0)
        .decode("rgb")  # 画像以外はそのまま
        .map(_process_sample)
        .select(lambda x: x is not None)  # None をフィルタ
    )

    # WebLoader でバッチング
    loader = wds.WebLoader(
        dataset, batch_size=None, num_workers=num_workers
    ).batched(batch_size, collation_fn=collate_fn)

    return loader


if __name__ == "__main__":
    # テスト用
    import glob
    import sys

    if len(sys.argv) < 2:
        print("Usage: python upsampler_dataset.py <webdataset_pattern>")
        sys.exit(1)

    path = sys.argv[1]

    print("Testing WebDataset loader...")

    # glob パターンの場合は展開
    if "*" in path and "{" not in path:
        expanded_files = sorted(glob.glob(path))
        if not expanded_files:
            print(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print(f"Found {len(expanded_files)} tar files")
        shard_pattern = expanded_files
    else:
        shard_pattern = path

    loader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        batch_size=8,
        num_workers=0,
    )
    for i, batch in enumerate(loader):
        print(f"Batch {i}:")
        print(f"  audio_codes: {batch['audio_codes'].shape}")
        print(f"  audio_48k: {batch['audio_48k'].shape}")
        print(f"  audio_24k: {batch['audio_24k'].shape}")
        if i >= 2:
            break
