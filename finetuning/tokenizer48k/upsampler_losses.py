# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
Loss Functions for 48kHz Upsampler Training

- Multi-resolution STFT Loss
- L1 Loss
- Mel Spectrogram Loss
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


class STFTLoss(nn.Module):
    """
    Single-resolution STFT Loss

    Args:
        fft_size: FFT size
        hop_size: Hop size
        win_size: Window size
        window: Window function type ("hann", "hamming", etc.)
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

        # Register window function
        if window == "hann":
            self.register_buffer("window", torch.hann_window(win_size))
        elif window == "hamming":
            self.register_buffer("window", torch.hamming_window(win_size))
        else:
            self.register_buffer("window", torch.ones(win_size))

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT and return magnitude"""
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
        # Compute magnitude
        magnitude = torch.abs(x_stft)
        return magnitude

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            spectral_convergence_loss: Spectral convergence loss
            log_magnitude_loss: Log magnitude loss
        """
        pred_mag = self.stft(pred)
        target_mag = self.stft(target)

        # Spectral convergence loss
        spectral_convergence_loss = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log magnitude loss
        log_pred_mag = torch.log(pred_mag + 1e-8)
        log_target_mag = torch.log(target_mag + 1e-8)
        log_magnitude_loss = F.l1_loss(log_pred_mag, log_target_mag)

        return spectral_convergence_loss, log_magnitude_loss


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT Loss

    Computes STFT loss at multiple resolutions and returns the average

    Args:
        fft_sizes: List of FFT sizes
        hop_sizes: List of hop sizes
        win_sizes: List of window sizes
        window: Window function type
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
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            sc_loss: Average spectral convergence loss
            mag_loss: Average log magnitude loss
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
        sample_rate: Sample rate
        n_fft: FFT size
        hop_length: Hop length
        win_length: Window length
        n_mels: Number of mel frequency bins
        fmin: Minimum frequency
        fmax: Maximum frequency
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

        # Pre-compute mel filterbank
        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank(),
        )
        self.register_buffer("window", torch.hann_window(win_length))

    def _create_mel_filterbank(self) -> torch.Tensor:
        """Create mel filterbank"""

        # Hz to Mel
        def hz_to_mel(hz):
            return 2595 * torch.log10(1 + hz / 700)

        # Mel to Hz
        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        # Mel frequency range
        mel_min = hz_to_mel(torch.tensor(self.fmin))
        mel_max = hz_to_mel(torch.tensor(self.fmax))

        # Distribute mel frequencies evenly
        mels = torch.linspace(mel_min, mel_max, self.n_mels + 2)
        freqs = mel_to_hz(mels)

        # Frequencies corresponding to FFT bins
        fft_freqs = torch.linspace(0, self.sample_rate / 2, self.n_fft // 2 + 1)

        # Create filterbank
        mel_basis = torch.zeros(self.n_mels, self.n_fft // 2 + 1)
        for i in range(self.n_mels):
            lower = freqs[i]
            center = freqs[i + 1]
            upper = freqs[i + 2]

            # Rising slope
            lower_slope = (fft_freqs - lower) / (center - lower + 1e-8)
            # Falling slope
            upper_slope = (upper - fft_freqs) / (upper - center + 1e-8)

            mel_basis[i] = torch.maximum(
                torch.zeros_like(fft_freqs),
                torch.minimum(lower_slope, upper_slope),
            )

        return mel_basis

    def mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram"""
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
        # Power spectrogram
        power = torch.abs(x_stft) ** 2
        # Mel spectrogram
        mel = torch.matmul(self.mel_basis.to(x.device), power)
        # Log scale
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            mel_loss: Mel spectrogram loss
        """
        pred_mel = self.mel_spectrogram(pred)
        target_mel = self.mel_spectrogram(target)
        return F.l1_loss(pred_mel, target_mel)


class MultiResolutionMelSpectrogramLoss(nn.Module):
    """
    Multi-resolution Mel Spectrogram Loss

    Computes mel spectrogram loss at 7 different resolutions (window sizes)
    and sums the L1 loss on log10-scaled mel spectrograms.
    Ported from inworld-ai/tts criterion.py.

    Args:
        sample_rate: Sample rate of the audio
        n_mels: Number of mel bins per resolution
        window_lengths: FFT/window size per resolution
        clamp_eps: Epsilon for clamping before log
        pow: Power applied before log (1.0 = amplitude, 2.0 = power)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: List[int] = None,
        window_lengths: List[int] = None,
        clamp_eps: float = 1e-5,
        pow: float = 1.0,
    ):
        super().__init__()
        if n_mels is None:
            n_mels = [5, 10, 20, 40, 80, 160, 320]
        if window_lengths is None:
            window_lengths = [32, 64, 128, 256, 512, 1024, 2048]

        self.mel_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=n_mel,
                    power=1.0,
                    center=True,
                    norm="slaney",
                    mel_scale="slaney",
                )
                for n_mel, window_length in zip(n_mels, window_lengths)
            ]
        )
        self.clamp_eps = clamp_eps
        self.pow = pow
        self.loss_fn = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            loss: Sum of L1 losses on log mel spectrograms across all resolutions
        """
        loss = 0.0
        for mel_transform in self.mel_transforms:
            pred_mel = mel_transform(pred)
            target_mel = mel_transform(target)
            log_pred = pred_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            log_target = target_mel.clamp(self.clamp_eps).pow(self.pow).log10()
            loss = loss + self.loss_fn(log_pred, log_target)
        return loss


class GlobalRMSLoss(nn.Module):
    """
    Global RMS Energy Loss in dB

    Computes per-track global RMS, converts to dB, and takes MSE of the
    dB difference between predicted and target. This captures overall loudness
    matching rather than frame-level energy.
    Ported from inworld-ai/tts decoder.py (compute_generator_loss).
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            loss: MSE of the per-track dB RMS difference
        """
        pred_rms = torch.sqrt(torch.mean(pred**2, dim=-1))
        target_rms = torch.sqrt(torch.mean(target**2, dim=-1))
        pred_rms_db = 20 * torch.log10(pred_rms + 1e-10)
        target_rms_db = 20 * torch.log10(target_rms + 1e-10)
        return torch.mean((pred_rms_db - target_rms_db) ** 2)


class RMSLoss(nn.Module):
    """
    RMS (Root Mean Square) Loss

    Loss function that compares frame-wise RMS energy.
    Effective for capturing amplitude envelope of waveform.

    Args:
        frame_size: Frame size for RMS calculation (number of samples)
        hop_size: Hop size between frames
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
        Compute frame-wise RMS

        Args:
            x: Input waveform (batch, samples)

        Returns:
            rms: RMS values (batch, num_frames)
        """
        # Padding
        pad_size = self.frame_size // 2
        x_padded = F.pad(x, (pad_size, pad_size), mode="reflect")

        # Split into frames
        # unfold: (batch, samples) -> (batch, num_frames, frame_size)
        frames = x_padded.unfold(dimension=-1, size=self.frame_size, step=self.hop_size)

        # RMS computation: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(frames**2, dim=-1) + 1e-8)

        return rms

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted waveform (batch, samples)
            target: Target waveform (batch, samples)

        Returns:
            rms_loss: RMS loss
        """
        pred_rms = self.compute_rms(pred)
        target_rms = self.compute_rms(target)

        # L1 loss
        return F.l1_loss(pred_rms, target_rms)


class UpsamplerLoss(nn.Module):
    """
    Unified loss function for 48kHz upsampler

    Args:
        sample_rate: Sample rate (default: 48000)
        l1_weight: L1 loss weight
        stft_weight: STFT loss weight
        mel_weight: Single-resolution mel spectrogram loss weight
        rms_weight: Frame-based RMS loss weight
        multi_res_mel_weight: Multi-resolution mel spectrogram loss weight (inworld-ai style)
        global_rms_weight: Global dB RMS loss weight (inworld-ai style)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        l1_weight: float = 1.0,
        stft_weight: float = 1.0,
        mel_weight: float = 1.0,
        rms_weight: float = 1.0,
        multi_res_mel_weight: float = 0.0,
        global_rms_weight: float = 0.0,
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.stft_weight = stft_weight
        self.mel_weight = mel_weight
        self.rms_weight = rms_weight
        self.multi_res_mel_weight = multi_res_mel_weight
        self.global_rms_weight = global_rms_weight

        # STFT settings for 48kHz
        self.stft_loss = MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048, 4096],
            hop_sizes=[50, 120, 240, 480],
            win_sizes=[240, 600, 1200, 2400],
        )

        # Single-resolution mel spectrogram loss (original)
        self.mel_loss = MelSpectrogramLoss(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=480,
            win_length=2048,
            n_mels=128,
        )

        # Frame-based RMS loss (original, multiple resolutions)
        self.rms_losses = nn.ModuleList(
            [
                RMSLoss(frame_size=512, hop_size=128),
                RMSLoss(frame_size=2048, hop_size=512),
                RMSLoss(frame_size=8192, hop_size=2048),
            ]
        )

        # Multi-resolution mel spectrogram loss (inworld-ai style: 7 resolutions, torchaudio)
        self.multi_res_mel_loss = MultiResolutionMelSpectrogramLoss(
            sample_rate=sample_rate
        )

        # Global dB RMS loss (inworld-ai style: per-track MSE in dB)
        self.global_rms_loss = GlobalRMSLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            pred: Predicted waveform (batch, samples) or (batch, 1, samples)
            target: Target waveform (batch, samples)
            lengths: Actual length of each sample (batch,)

        Returns:
            dict: Dictionary containing each loss value
        """
        # Align shapes
        if pred.dim() == 3:
            pred = pred.squeeze(1)
        if target.dim() == 3:
            target = target.squeeze(1)

        # Align lengths (to the shorter one)
        min_len = min(pred.shape[-1], target.shape[-1])
        pred = pred[..., :min_len]
        target = target[..., :min_len]

        # Masking by length (optional)
        if lengths is not None:
            # Mask up to maximum length in batch
            mask = torch.arange(min_len, device=pred.device)[None, :] < lengths[:, None]
            pred = pred * mask
            target = target * mask

        # Zero tensor (for skipped losses)
        zero = torch.tensor(0.0, device=pred.device)
        total_loss = zero.clone()

        # L1 loss (compute if weight is non-zero)
        if self.l1_weight > 0:
            l1_loss = F.l1_loss(pred, target)
            total_loss = total_loss + self.l1_weight * l1_loss
        else:
            l1_loss = zero

        # Multi-resolution STFT loss (compute if weight is non-zero)
        if self.stft_weight > 0:
            sc_loss, mag_loss = self.stft_loss(pred, target)
            stft_loss = sc_loss + mag_loss
            total_loss = total_loss + self.stft_weight * stft_loss
        else:
            sc_loss = zero
            mag_loss = zero
            stft_loss = zero

        # Single-resolution mel spectrogram loss (compute if weight is non-zero)
        if self.mel_weight > 0:
            mel_loss = self.mel_loss(pred, target)
            total_loss = total_loss + self.mel_weight * mel_loss
        else:
            mel_loss = zero

        # Frame-based RMS loss (compute if weight is non-zero)
        if self.rms_weight > 0:
            rms_loss = zero.clone()
            for rms_loss_fn in self.rms_losses:
                rms_loss = rms_loss + rms_loss_fn(pred, target)
            rms_loss = rms_loss / len(self.rms_losses)
            total_loss = total_loss + self.rms_weight * rms_loss
        else:
            rms_loss = zero

        # Multi-resolution mel spectrogram loss (inworld-ai style)
        if self.multi_res_mel_weight > 0:
            multi_res_mel_loss = self.multi_res_mel_loss(pred, target)
            total_loss = total_loss + self.multi_res_mel_weight * multi_res_mel_loss
        else:
            multi_res_mel_loss = zero

        # Global dB RMS loss (inworld-ai style)
        if self.global_rms_weight > 0:
            global_rms_loss = self.global_rms_loss(pred, target)
            total_loss = total_loss + self.global_rms_weight * global_rms_loss
        else:
            global_rms_loss = zero

        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "stft_loss": stft_loss,
            "sc_loss": sc_loss,
            "mag_loss": mag_loss,
            "mel_loss": mel_loss,
            "rms_loss": rms_loss,
            "multi_res_mel_loss": multi_res_mel_loss,
            "global_rms_loss": global_rms_loss,
        }


if __name__ == "__main__":
    import time

    # Test
    print("=" * 50)
    print("Testing UpsamplerLoss (all weights = 1.0)...")
    print("=" * 50)

    loss_fn = UpsamplerLoss()

    # Dummy data
    pred = torch.randn(2, 48000)
    target = torch.randn(2, 48000)

    start = time.time()
    losses = loss_fn(pred, target)
    elapsed_all = time.time() - start

    print("Losses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"  Time: {elapsed_all*1000:.2f}ms")

    # Test with zero weights
    print("\n" + "=" * 50)
    print("Testing UpsamplerLoss (stft_weight=0, mel_weight=0)...")
    print("=" * 50)

    loss_fn_partial = UpsamplerLoss(
        l1_weight=1.0,
        stft_weight=0.0,  # Skip
        mel_weight=0.0,  # Skip
        rms_weight=1.0,
    )

    start = time.time()
    losses_partial = loss_fn_partial(pred, target)
    elapsed_partial = time.time() - start

    print("Losses:")
    for k, v in losses_partial.items():
        print(f"  {k}: {v.item():.4f}")
    print(f"  Time: {elapsed_partial*1000:.2f}ms")

    # Verify: skipped losses should be 0
    assert losses_partial["stft_loss"].item() == 0.0, "stft_loss should be 0"
    assert losses_partial["sc_loss"].item() == 0.0, "sc_loss should be 0"
    assert losses_partial["mag_loss"].item() == 0.0, "mag_loss should be 0"
    assert losses_partial["mel_loss"].item() == 0.0, "mel_loss should be 0"
    assert losses_partial["l1_loss"].item() > 0.0, "l1_loss should be > 0"
    assert losses_partial["rms_loss"].item() > 0.0, "rms_loss should be > 0"

    print(
        f"\nSpeedup: {elapsed_all/elapsed_partial:.2f}x faster when skipping STFT & Mel"
    )
    print("\nAll tests passed!")
