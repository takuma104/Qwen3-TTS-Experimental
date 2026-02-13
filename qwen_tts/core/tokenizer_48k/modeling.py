# coding=utf-8
# Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen3TTSTokenizer48k model (48kHz upsampler extension of 12Hz tokenizer)."""

import numpy as np
import torch
from torch import nn
from transformers.utils import logging

from ..tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2Model,
    Qwen3TTSTokenizerV2PreTrainedModel,
    Qwen3TTSTokenizerV2Encoder,
    Qwen3TTSTokenizerV2CausalConvNet,
    Qwen3TTSTokenizerV2CausalTransConvNet,
    SnakeBeta,
)
from .configuration import Qwen3TTSTokenizer48kConfig, Qwen3TTSTokenizer48kDecoderConfig

logger = logging.get_logger(__name__)


class UpSamplerBlock(nn.Module):
    """
    24kHz -> 48kHz upsampling block.
    TransposedConv + residual block architecture inspired by XCodec2's 44.1kHz implementation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_dim: int = 32,
        kernel_size: int = 4,
        upsample_factor: int = 2,
    ):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.upsample_conv = Qwen3TTSTokenizerV2CausalTransConvNet(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=upsample_factor,
        )

        self.residual_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SnakeBeta(hidden_dim),
                    Qwen3TTSTokenizerV2CausalConvNet(
                        hidden_dim, hidden_dim, kernel_size=7, dilation=1
                    ),
                    SnakeBeta(hidden_dim),
                    Qwen3TTSTokenizerV2CausalConvNet(
                        hidden_dim, hidden_dim, kernel_size=1
                    ),
                ),
                nn.Sequential(
                    SnakeBeta(hidden_dim),
                    Qwen3TTSTokenizerV2CausalConvNet(
                        hidden_dim, hidden_dim, kernel_size=7, dilation=3
                    ),
                    SnakeBeta(hidden_dim),
                    Qwen3TTSTokenizerV2CausalConvNet(
                        hidden_dim, hidden_dim, kernel_size=1
                    ),
                ),
            ]
        )

        self.output_act = SnakeBeta(hidden_dim)
        self.output_conv = Qwen3TTSTokenizerV2CausalConvNet(
            hidden_dim, in_channels, kernel_size=7
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, samples_24k]
        Returns:
            [batch, channels, samples_48k]
        """
        x = self.upsample_conv(x)

        for block in self.residual_blocks:
            x = x + block(x)

        x = self.output_act(x)
        x = self.output_conv(x)

        return x


class Qwen3TTSTokenizer48kDecoder(Qwen3TTSTokenizerV2Decoder):
    """48kHz decoder extending the base 12Hz/24kHz decoder with an upsampling block."""

    def __init__(self, config: Qwen3TTSTokenizer48kDecoderConfig):
        super().__init__(config)

        self.upsampler = None
        if getattr(config, "enable_48khz_upsampler", False):
            self.upsampler = UpSamplerBlock(
                in_channels=1,
                hidden_dim=config.upsampler_hidden_dim,
                kernel_size=config.upsampler_kernel_size,
                upsample_factor=config.upsampler_factor,
            )
            self.total_upsample = int(
                np.prod(config.upsample_rates + config.upsampling_ratios)
                * config.upsampler_factor
            )

    def forward(self, codes):
        wav = super().forward(codes)

        if self.upsampler is not None:
            wav = self.upsampler(wav)
            wav = wav.clamp(min=-1, max=1)

        return wav


class Qwen3TTSTokenizer48kPreTrainedModel(Qwen3TTSTokenizerV2PreTrainedModel):
    config: Qwen3TTSTokenizer48kConfig


class Qwen3TTSTokenizer48kModel(Qwen3TTSTokenizerV2Model):
    """48kHz tokenizer model extending the base 12Hz tokenizer with upsampling capability."""

    config_class = Qwen3TTSTokenizer48kConfig

    def __init__(self, config: Qwen3TTSTokenizer48kConfig):
        super().__init__(config)
        # Replace decoder with 48k version
        self.decoder = Qwen3TTSTokenizer48kDecoder._from_config(
            self.config.decoder_config
        )


__all__ = [
    "Qwen3TTSTokenizer48kModel",
    "Qwen3TTSTokenizer48kPreTrainedModel",
    "UpSamplerBlock",
]
