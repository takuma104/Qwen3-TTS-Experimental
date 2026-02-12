# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler 学習用損失関数

- Multi-resolution STFT Loss
- L1 Loss
- Mel Spectrogram Loss
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class STFTLoss(nn.Module):
    """
    Single-resolution STFT Loss

    Args:
        fft_size: FFT サイズ
        hop_size: ホップサイズ
        win_size: 窓サイズ
        window: 窓関数の種類 ("hann", "hamming", etc.)
    """

    def __init__(
        self,
        fft_size: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        window: str = "hann",
    ):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size

        # 窓関数を登録
        if window == "hann":
            self.register_buffer("window", torch.hann_window(win_size))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(win_size))
        else:
            self.register_buffer("window", torch.ones(win_size))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """STFT を計算して magnitude を返す"""
        # x: (batch, samples)
        x_stft = torch.stft(
            x,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(x.device),
            return_complex=True,
            pad_mode="reflect",
        )
        # magnitude を計算
        magnitude = torch.abs(x_stft)
        return magnitude

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: 予測波形 (batch, samples)
            target: ターゲット波形 (batch, samples)

        Returns:
            spectral_convergence_loss: スペクトル収束損失
            log_magnitude_loss: 対数マグニチュード損失
        """
        pred_mag = self.stft(pred)
        target_mag = self.stft(target)

        # スペクトル収束損失
        spectral_convergence_loss = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # 対数マグニチュード損失
        log_pred_mag = torch.log(pred_mag + 1e-8)
        log_target_mag = torch.log(target_mag + 1e-8)
        log_magnitude_loss = F.l1_loss(log_pred_mag, log_target_mag)

        return spectral_convergence_loss, log_magnitude_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT Loss

    複数の解像度でSTFT損失を計算し、平均を返す

    Args:
        fft_sizes: FFT サイズのリスト
        hop_sizes: ホップサイズのリスト
        win_sizes: 窓サイズのリスト
        window: 窓関数の種類
    """

    def __init__(
        self,
        fft_sizes: List[int] = [512, 1024, 2048],
        hop_sizes: List[int] = [50, 120, 240],
        win_sizes: List[int] = [240, 600, 1200],
        window: str = "hann",
    ):
        super().__init__()

        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)

        self.stft_losses = nn.ModuleList()
        for fft_size, hop_size, win_size in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(
                STFTLoss(
                    fft_size=fft_size,
                    hop_size=hop_size,
                    win_size=win_size,
                    window=window,
                )
            )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: 予測波形 (batch, samples)
            target: ターゲット波形 (batch, samples)

        Returns:
            sc_loss: 平均スペクトル収束損失
            mag_loss: 平均対数マグニチュード損失
        """
        sc_loss = 0.0
        mag_loss = 0.0

        for stft_loss in self.stft_losses:
            sc, mag = stft_loss(pred, target)
            sc_loss += sc
            mag_loss += mag

        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss


class MelSpectrogramLoss(nn.Module):
    """
    Mel Spectrogram Loss

    Args:
        sample_rate: サンプルレート
        n_fft: FFT サイズ
        hop_length: ホップ長
        win_length: 窓長
        n_mels: メル周波数ビンの数
        fmin: 最小周波数
        fmax: 最大周波数
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 2048,
        hop_length: int = 480,
        win_length: int = 2048,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2

        # メルフィルターバンクを事前計算
        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank(),
        )
        self.register_buffer("window", torch.hann_window(win_length))

    def _create_mel_filterbank(self) -> torch.Tensor:
        """メルフィルターバンクを作成"""
        # Hz to Mel
        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)

        # Mel to Hz
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # メル周波数の範囲
        mel_min = hz_to_mel(torch.tensor(self.fmin))
        mel_max = hz_to_mel(torch.tensor(self.fmax))

        # メル周波数を等間隔に配置
        mels = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        freqs = mel_to_hz(mels)

        # FFT ビンに対応する周波数
        fft_freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)

        # フィルターバンクを作成
        mel_basis = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for i in range(self.n_mels):
            lower = freqs[i]
            center = freqs[i + 1]
            upper = freqs[i + 2]

            # 上昇スロープ
            lower_slope = (fft_freqs - lower) / (center - lower + 1e-8)
            # 下降スロープ
            upper_slope = (upper - fft_freqs) / (upper - center + 1e-8)

            mel_basis[i] = torch.maximum(
                torch.zeros_like(fft_freqs),
                torch.minimum(lower_slope, upper_slope),
            )

        return mel_basis

    def mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """メルスペクトログラムを計算"""
        # STFT
        x_stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(x.device),
            return_complex=True,
            pad_mode="reflect",
        )
        # パワースペクトログラム
        power = torch.abs(x_stft) ** 2
        # メルスペクトログラム
        mel = torch.matmul(self.mel_basis.to(x.device), power)
        # 対数スケール
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測波形 (batch, samples)
            target: ターゲット波形 (batch, samples)

        Returns:
            mel_loss: メルスペクトログラム損失
        """
        pred_mel = self.mel_spectrogram(pred)
        target_mel = self.mel_spectrogram(target)
        return F.l1_loss(pred_mel, target_mel)


class RMSLoss(nn.Module):
    """
    RMS (Root Mean Square) Loss

    フレームごとのRMSエネルギーを比較する損失関数。
    波形の振幅エンベロープを捉えるのに有効。

    Args:
        frame_size: RMS計算のフレームサイズ（サンプル数）
        hop_size: フレーム間のホップサイズ
    """

    def __init__(
        self,
        frame_size: int = 2048,
        hop_size: int = 512,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.hop_size = hop_size

    def compute_rms(self, x: torch.Tensor) -> torch.Tensor:
        """
        フレームごとのRMSを計算

        Args:
            x: 入力波形 (batch, samples)

        Returns:
            rms: RMS値 (batch, num_frames)
        """
        # パディング
        pad_size = self.frame_size // 2
        x_padded = F.pad(x, (pad_size, pad_size), mode="reflect")

        # フレームに分割
        # unfold: (batch, samples) -> (batch, num_frames, frame_size)
        frames = x_padded.unfold(dimension=-1, size=self.frame_size, step=self.hop_size)

        # RMS計算: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(frames ** 2, dim=-1) + 1e-8)

        return rms

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 予測波形 (batch, samples)
            target: ターゲット波形 (batch, samples)

        Returns:
            rms_loss: RMS損失
        """
        pred_rms = self.compute_rms(pred)
        target_rms = self.compute_rms(target)

        # L1損失
        return F.l1_loss(pred_rms, target_rms)


class UpsamplerLoss(nn.Module):
    """
    48kHz アップサンプラー用の統合損失関数

    Args:
        sample_rate: サンプルレート（デフォルト: 48000）
        l1_weight: L1 損失の重み
        stft_weight: STFT 損失の重み
        mel_weight: メルスペクトログラム損失の重み
        rms_weight: RMS 損失の重み
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        l1_weight: float = 1.0,
        stft_weight: float = 1.0,
        mel_weight: float = 1.0,
        rms_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.rms_weight = rms_weight

        # 48kHz 用の STFT 設定
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[50, 120, 240, 480],
            win_sizes=[240, 600, 1200, 2400],
        )

        # RMS損失（複数の解像度）
        self.rms_losses = nn.ModuleList([
            RMSLoss(frame_size=512, hop_size=128),
            RMSLoss(frame_size=2048, hop_size=512),
            RMSLoss(frame_size=8192, hop_size=2048),
        ])

        # メルスペクトログラム損失
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=480,
            win_length=2048,
            n_mels=128,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            pred: 予測波形 (batch, samples) or (batch, 1, samples)
            target: ターゲット波形 (batch, samples)
            lengths: 各サンプルの実際の長さ (batch,)

        Returns:
            dict: 各損失値を含む辞書
        """
        # 形状を揃える
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # 長さを揃える（短い方に合わせる）
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # 長さでマスキング（オプション）
        if lengths is not None:
            # バッチ内の最大長さまでマスク
            mask = torch.arange(min_len, device=pred.device)[None, :] < lengths[:, None]
            pred = pred * mask
            target = target * mask

        # ゼロテンソル（スキップされた損失用）
        zero = torch.tensor(0.0, device=pred.device)
        total_loss = zero.clone()

        # L1 損失（重みが0でなければ計算）
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(pred, target)
            total_loss = total_loss + self.l1_weight * l1_loss
        else:
            l1_loss = zero

        # Multi-resolution STFT 損失（重みが0でなければ計算）
        if self.stft_weight > 0:
            sc_loss, mag_loss = self.stft_loss(pred, target)
            stft_loss = sc_loss + mag_loss
            total_loss = total_loss + self.stft_weight * stft_loss
        else:
            sc_loss = zero
            mag_loss = zero
            stft_loss = zero

        # メルスペクトログラム損失（重みが0でなければ計算）
        if self.mel_weight > 0:
            mel_loss = self.mel_loss(pred, target)
            total_loss = total_loss + self.mel_weight * mel_loss
        else:
            mel_loss = zero

        # RMS損失（重みが0でなければ計算）
        if self.rms_weight > 0:
            rms_loss = zero.clone()
            for rms_loss_fn in self.rms_losses:
                rms_loss = rms_loss + rms_loss_fn(pred, target)
            rms_loss = rms_loss / len(self.rms_losses)
            total_loss = total_loss + self.rms_weight * rms_loss
        else:
            rms_loss = zero

        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "stft_loss": stft_loss,
            "sc_loss": sc_loss,
            "mag_loss": mag_loss,
            "mel_loss": mel_loss,
            "rms_loss": rms_loss,
        }


if __name__ == "__main__":
    import time

    # テスト
    print("=" * 50)
    print("Testing UpsamplerLoss (all weights = 1.0)...")
    print("=" * 50)

    loss_fn = UpsamplerLoss()

    # ダミーデータ
    pred = torch.randn(2, 48000)
    target = torch.randn(2, 48000)

    start = time.time()
    losses = loss_fn(pred, target)
    elapsed_all = time.time() - start

    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"  Time: {elapsed_all*1000:.2f}ms")

    # 重みが0のテスト
    print("\n" + "=" * 50)
    print("Testing UpsamplerLoss (stft_weight=0, mel_weight=0)...")
    print("=" * 50)

    loss_fn_partial = UpsamplerLoss(
        l1_weight=1.0,
        stft_weight=0.0,  # スキップ
        mel_weight=0.0,   # スキップ
        rms_weight=1.0,
    )

    start = time.time()
    losses_partial = loss_fn_partial(pred, target)
    elapsed_partial = time.time() - start

    print("Losses:")
    for k, v in losses_partial.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"  Time: {elapsed_partial*1000:.2f}ms")

    # 検証: スキップされた損失は0であるべき
    assert losses_partial["stft_loss"].item() == 0.0, "stft_loss should be 0"
    assert losses_partial["sc_loss"].item() == 0.0, "sc_loss should be 0"
    assert losses_partial["mag_loss"].item() == 0.0, "mag_loss should be 0"
    assert losses_partial["mel_loss"].item() == 0.0, "mel_loss should be 0"
    assert losses_partial["l1_loss"].item() > 0.0, "l1_loss should be > 0"
    assert losses_partial["rms_loss"].item() > 0.0, "rms_loss should be > 0"

    print(f"\nSpeedup: {elapsed_all/elapsed_partial:.2f}x faster when skipping STFT & Mel")
    print("\nAll tests passed!")
