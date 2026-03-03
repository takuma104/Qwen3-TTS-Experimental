# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
Discriminators for GAN-style training of decoder_block_48k.

Implements:
- Multi-Period Discriminator (MPD): ported from inworld-ai/tts (X-Codec-2.0, MIT),
  exponentially growing channels (16→64→256→512), 5-layer with stride-1 final layer
- Multi-Scale Discriminator (MSD): HiFi-GAN style, waveform-based 3 scales
- Spec Discriminator (SpecDiscriminator): STFT-based multi-resolution, ported from
  inworld-ai/tts (originally from X-Codec-2.0, MIT License). Uses 8 STFT scales
  optimized for 48kHz audio.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


# Type aliases
DiscriminatorOutput = Tuple[List[torch.Tensor], List[List[torch.Tensor]]]


class HiFiGANPeriodDiscriminator(nn.Module):
    """Period sub-discriminator ported from inworld-ai/tts (X-Codec-2.0, MIT).

    Channels grow exponentially via channel_increasing_factor and are capped at
    max_downsample_channels.  The final stride-1 layer in the default
    downsample_scales=[3,3,3,3,1] broadens the receptive field without further
    temporal downsampling.

    Returns (output_flat, fmap) to match the DiscriminatorOutput interface.
    """

    def __init__(
        self,
        period: int,
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        downsample_scales: List[int] = [3, 3, 3, 3, 1],
        channel_increasing_factor: int = 4,
        max_downsample_channels: int = 512,
        negative_slope: float = 0.1,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "kernel_sizes[0] must be odd"
        assert kernel_sizes[1] % 2 == 1, "kernel_sizes[1] must be odd"

        self.period = period
        self.convs = nn.ModuleList()
        in_chs = 1
        out_chs = channels
        for scale in downsample_scales:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_chs, out_chs,
                        kernel_size=(kernel_sizes[0], 1),
                        stride=(scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                    ),
                    nn.LeakyReLU(negative_slope),
                )
            )
            in_chs = out_chs
            out_chs = min(out_chs * channel_increasing_factor, max_downsample_channels)

        # kernel_sizes[1] - 1 follows the inworld-ai/tts convention (e.g. 3→2)
        self.output_conv = nn.Conv2d(
            in_chs, 1,
            kernel_size=(kernel_sizes[1] - 1, 1),
            stride=1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
        )

        if use_weight_norm:
            self._apply_weight_norm()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            output: flattened discriminator output
            fmap: intermediate feature maps + unflattened output_conv output
        """
        fmap = []

        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.output_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap

    def _apply_weight_norm(self):
        def _wn(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
        self.apply(_wn)


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) ported from inworld-ai/tts (X-Codec-2.0, MIT).

    Uses multiple HiFiGANPeriodDiscriminator sub-discriminators at different periods
    to capture periodic patterns in the waveform at various temporal scales.
    """

    def __init__(
        self,
        periods: List[int] = [2, 3, 5, 7, 11],
        channels: int = 16,
        channel_increasing_factor: int = 4,
        max_downsample_channels: int = 512,
        **kwargs,
    ):
        super().__init__()
        self.discriminators = nn.ModuleList([
            HiFiGANPeriodDiscriminator(
                period=p,
                channels=channels,
                channel_increasing_factor=channel_increasing_factor,
                max_downsample_channels=max_downsample_channels,
                **kwargs,
            )
            for p in periods
        ])

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            outputs: list of discriminator outputs (one per period)
            fmaps: list of feature map lists (one per period)
        """
        outputs = []
        fmaps = []
        for d in self.discriminators:
            out, fmap = d(x)
            outputs.append(out)
            fmaps.append(fmap)
        return outputs, fmaps


class ScaleSubDiscriminator(nn.Module):
    """Sub-discriminator for a single scale in MSD.

    Applies 1D convolutions with grouped convolutions for efficiency.
    """

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm_fn = spectral_norm if use_spectral_norm else weight_norm

        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
            norm_fn(nn.Conv1d(16, 32, kernel_size=41, stride=4, padding=20, groups=4)),
            norm_fn(nn.Conv1d(32, 64, kernel_size=41, stride=4, padding=20, groups=16)),
            norm_fn(nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2)),
        ])
        self.output_conv = norm_fn(nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            output: discriminator output
            fmap: list of intermediate feature maps
        """
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.output_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator (MSD) from HiFi-GAN.

    Uses multiple sub-discriminators at different temporal scales
    (original, 2x downsampled, 4x downsampled).
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleSubDiscriminator(use_spectral_norm=True),   # scale 0: spectral norm
            ScaleSubDiscriminator(use_spectral_norm=False),  # scale 1: weight norm
            ScaleSubDiscriminator(use_spectral_norm=False),  # scale 2: weight norm
        ])
        self.downsamplers = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2),
        ])

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            outputs: list of discriminator outputs (one per scale)
            fmaps: list of feature map lists (one per scale)
        """
        outputs = []
        fmaps = []

        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.downsamplers[i - 1](x)
            out, fmap = d(x)
            outputs.append(out)
            fmaps.append(fmap)

        return outputs, fmaps


# ---------------------------------------------------------------------------
# STFT-based Spec Discriminator (ported from inworld-ai/tts / X-Codec-2.0)
# ---------------------------------------------------------------------------

# Default STFT parameters for 48kHz audio (8 scales, geometrically spaced).
_STFT_PARAMS_48K: Dict = {
    "fft_sizes":  [78,  126,  206,  334,  542,  876, 1418, 2296],
    "hop_sizes":  [39,   63,  103,  167,  271,  438,  709, 1148],
    "win_lengths": [78, 126,  206,  334,  542,  876, 1418, 2296],
    "window": "hann_window",
}


def _stft_magnitude(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Compute STFT magnitude spectrogram.

    Args:
        x: (B, T) waveform
        window: pre-built window tensor (must be on the same device as x)

    Returns:
        (B, T_frames, F) magnitude spectrogram
    """
    # MPS backend has a broken STFT backward that causes Metal GPU crashes
    # when gradients flow through torch.stft. Run STFT on CPU and move the
    # complex result back to the original device so autograd still works.
    if x.device.type == "mps":
        x_stft = torch.stft(
            x.cpu(), fft_size, hop_size, win_length, window.cpu(), return_complex=True
        ).to(x.device)
    else:
        x_stft = torch.stft(
            x, fft_size, hop_size, win_length, window.to(x.device), return_complex=True
        )
    magnitude = torch.sqrt(
        torch.clamp(x_stft.real ** 2 + x_stft.imag ** 2, min=1e-7, max=1e3)
    )
    return magnitude.transpose(2, 1)  # (B, T_frames, F)


class NLayerSpecDiscriminator(nn.Module):
    """Single-scale STFT spectrogram discriminator using Conv2d layers.

    Operates on a (B, 1, F, T) spectrogram tensor and returns feature maps
    from each layer, with the final layer being the discriminator output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Tuple[int, int] = (5, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Tuple[int, ...] = (2, 2, 2),
    ):
        super().__init__()
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        layers = nn.ModuleDict()

        layers["layer_0"] = nn.Sequential(
            nn.Conv2d(
                in_channels, channels,
                kernel_size=kernel_sizes[0],
                stride=2,
                padding=kernel_sizes[0] // 2,
            ),
            nn.LeakyReLU(0.2, True),
        )

        in_chs = channels
        for i, scale in enumerate(downsample_scales):
            out_chs = min(in_chs * scale, max_downsample_channels)
            layers[f"layer_{i + 1}"] = nn.Sequential(
                nn.Conv2d(
                    in_chs, out_chs,
                    kernel_size=scale * 2 + 1,
                    stride=scale,
                    padding=scale,
                ),
                nn.LeakyReLU(0.2, True),
            )
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        layers[f"layer_{len(downsample_scales) + 1}"] = nn.Sequential(
            nn.Conv2d(in_chs, out_chs, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2),
            nn.LeakyReLU(0.2, True),
        )
        layers[f"layer_{len(downsample_scales) + 2}"] = nn.Conv2d(
            out_chs, out_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1] // 2
        )

        self.layers = layers

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (B, 1, F, T) spectrogram

        Returns:
            List of tensors for each layer (including the final output layer).
        """
        results = []
        for layer in self.layers.values():
            x = layer(x)
            results.append(x)
        return results


class SpecDiscriminator(nn.Module):
    """Multi-resolution STFT spectrogram discriminator.

    Ported from inworld-ai/tts (based on X-Codec-2.0, MIT License).
    Uses multiple STFT resolutions to capture spectral patterns at different
    time-frequency scales.  Provides the same (outputs, fmaps) interface as
    MultiScaleDiscriminator so it can be used as a drop-in replacement.

    Args:
        stft_params: Dict with keys fft_sizes, hop_sizes, win_lengths, window.
            Defaults to 8-scale params optimized for 48 kHz.
        in_channels: Input channels for each sub-discriminator.
        out_channels: Output channels for each sub-discriminator.
        kernel_sizes: (first_kernel, later_kernel) for Conv2d layers.
        channels: Base number of channels.
        max_downsample_channels: Channel cap for downsample layers.
        downsample_scales: Stride multipliers for downsampling layers.
        use_weight_norm: Apply weight norm to all Conv layers.
    """

    def __init__(
        self,
        stft_params: Optional[Dict] = None,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes: Tuple[int, int] = (7, 3),
        channels: int = 32,
        max_downsample_channels: int = 512,
        downsample_scales: Tuple[int, ...] = (2, 2, 2),
        use_weight_norm: bool = True,
    ):
        super().__init__()

        if stft_params is None:
            stft_params = _STFT_PARAMS_48K

        self.stft_params = stft_params
        self.sub_discs = nn.ModuleList(
            [
                NLayerSpecDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    downsample_scales=downsample_scales,
                )
                for _ in range(len(stft_params["fft_sizes"]))
            ]
        )

        if use_weight_norm:
            self._apply_weight_norm()
        self._reset_parameters()

    # ------------------------------------------------------------------
    # forward: same (outputs, fmaps) interface as MultiScaleDiscriminator
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            outputs: List[Tensor] — one flattened discriminator output per STFT scale
            fmaps:   List[List[Tensor]] — intermediate feature maps per STFT scale
        """
        outputs = []
        fmaps = []

        x_mono = x.squeeze(1)  # (B, T)
        params = self.stft_params

        for i, disc in enumerate(self.sub_discs):
            window = getattr(torch, params["window"])(params["win_lengths"][i])
            spec = _stft_magnitude(
                x_mono,
                fft_size=params["fft_sizes"][i],
                hop_size=params["hop_sizes"][i],
                win_length=params["win_lengths"][i],
                window=window,
            )  # (B, T_frames, F)
            spec = spec.transpose(1, 2).unsqueeze(1)  # (B, 1, F, T_frames)

            layer_results = disc(spec)  # List[Tensor]
            # Intermediate layers → fmap; final Conv2d output → discriminator output
            fmaps.append(layer_results[:-1])
            outputs.append(layer_results[-1].flatten(1, -1))

        return outputs, fmaps

    def _apply_weight_norm(self):
        def _wn(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                torch.nn.utils.weight_norm(m)
        self.apply(_wn)

    def _reset_parameters(self):
        def _init(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(0.0, 0.02)
        self.apply(_init)


if __name__ == "__main__":
    # Quick test
    print("Testing discriminators...")

    x = torch.randn(2, 1, 48000)

    mpd = MultiPeriodDiscriminator()
    mpd_outputs, mpd_fmaps = mpd(x)
    mpd_params = sum(p.numel() for p in mpd.parameters())
    print(f"MPD: {len(mpd_outputs)} outputs, {mpd_params:,} params")
    for i, (out, fmap) in enumerate(zip(mpd_outputs, mpd_fmaps)):
        print(f"  Period {[2,3,5,7,11][i]}: output {out.shape}, {len(fmap)} feature maps")

    msd = MultiScaleDiscriminator()
    msd_outputs, msd_fmaps = msd(x)
    msd_params = sum(p.numel() for p in msd.parameters())
    print(f"MSD: {len(msd_outputs)} outputs, {msd_params:,} params")
    for i, (out, fmap) in enumerate(zip(msd_outputs, msd_fmaps)):
        print(f"  Scale {i}: output {out.shape}, {len(fmap)} feature maps")

    spec_disc = SpecDiscriminator()
    spec_outputs, spec_fmaps = spec_disc(x)
    spec_params = sum(p.numel() for p in spec_disc.parameters())
    print(f"SpecDisc: {len(spec_outputs)} outputs, {spec_params:,} params")
    for i, (out, fmap) in enumerate(zip(spec_outputs, spec_fmaps)):
        print(f"  STFT scale {i}: output {out.shape}, {len(fmap)} feature maps")

    total_params = mpd_params + spec_params
    print(f"\nTotal discriminator params (MPD + SpecDisc): {total_params:,}")
    print("All tests passed!")
