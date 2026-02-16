# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
Discriminators for GAN-style training of decoder_block_48k.

Implements lightweight Multi-Period Discriminator (MPD) and Multi-Scale Discriminator (MSD)
following HiFi-GAN, with reduced channel counts to match the small generator (~95K params).
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm


# Type aliases
DiscriminatorOutput = Tuple[List[torch.Tensor], List[List[torch.Tensor]]]


class PeriodSubDiscriminator(nn.Module):
    """Sub-discriminator for a single period in MPD.

    Reshapes 1D waveform to 2D (T//p, p) and applies 2D convolutions.
    """

    def __init__(self, period: int, channels: List[int] = [16, 32, 64, 128]):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList()
        in_ch = 1
        for out_ch in channels:
            self.convs.append(
                weight_norm(
                    nn.Conv2d(
                        in_ch, out_ch,
                        kernel_size=(5, 1), stride=(3, 1), padding=(2, 0),
                    )
                )
            )
            in_ch = out_ch

        self.output_conv = weight_norm(
            nn.Conv2d(channels[-1], 1, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, 1, T) waveform

        Returns:
            output: discriminator output
            fmap: list of intermediate feature maps
        """
        fmap = []

        # Reshape to 2D: (B, 1, T) -> (B, 1, T//p, p)
        b, c, t = x.shape
        if t % self.period != 0:
            pad_len = self.period - (t % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            t = x.shape[-1]
        x = x.view(b, c, t // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.output_conv(x)
        fmap.append(x)
        x = x.flatten(1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator (MPD) from HiFi-GAN.

    Uses multiple sub-discriminators with different periods to capture
    periodic patterns in the waveform at various scales.
    """

    def __init__(self, periods: List[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [PeriodSubDiscriminator(p) for p in periods]
        )

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

    total_params = mpd_params + msd_params
    print(f"\nTotal discriminator params: {total_params:,}")
    print("All tests passed!")
