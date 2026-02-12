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
"""Qwen3TTSTokenizer48k model configuration (48kHz upsampler extension of 12Hz tokenizer)"""

from transformers import MimiConfig
from transformers.utils import logging

from ..tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2DecoderConfig,
)

logger = logging.get_logger(__name__)


class Qwen3TTSTokenizer48kDecoderConfig(Qwen3TTSTokenizerV2DecoderConfig):
    r"""
    Configuration class for the 48kHz upsampler decoder, extending [`Qwen3TTSTokenizerV2DecoderConfig`].

    Adds parameters for the 24kHz to 48kHz upsampling block.

    Args:
        enable_48khz_upsampler (`bool`, *optional*, defaults to `True`):
            Whether to enable the 48kHz upsampler block for 24kHz to 48kHz upsampling.
        upsampler_hidden_dim (`int`, *optional*, defaults to 32):
            Hidden dimension for the upsampler block.
        upsampler_kernel_size (`int`, *optional*, defaults to 4):
            Kernel size for the upsampler transposed convolution.
        upsampler_factor (`int`, *optional*, defaults to 2):
            Upsampling factor (2 for 24kHz to 48kHz).
    """

    def __init__(
        self,
        enable_48khz_upsampler=True,
        upsampler_hidden_dim=32,
        upsampler_kernel_size=4,
        upsampler_factor=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_48khz_upsampler = enable_48khz_upsampler
        self.upsampler_hidden_dim = upsampler_hidden_dim
        self.upsampler_kernel_size = upsampler_kernel_size
        self.upsampler_factor = upsampler_factor


class Qwen3TTSTokenizer48kConfig(Qwen3TTSTokenizerV2Config):
    """
    Configuration class for the 48kHz tokenizer model, extending [`Qwen3TTSTokenizerV2Config`].

    Uses [`Qwen3TTSTokenizer48kDecoderConfig`] as the decoder config and automatically
    adjusts output_sample_rate and decode_upsample_rate based on the upsampler factor.
    """

    model_type = "qwen3_tts_tokenizer_48k"
    sub_configs = {
        "encoder_config": MimiConfig,
        "decoder_config": Qwen3TTSTokenizer48kDecoderConfig,
    }

    def __init__(
        self,
        encoder_config=None,
        decoder_config=None,
        encoder_valid_num_quantizers=16,
        input_sample_rate=24000,
        output_sample_rate=24000,
        decode_upsample_rate=1920,
        encode_downsample_rate=1920,
        **kwargs,
    ):
        if decoder_config is None:
            decoder_config = {}
        if isinstance(decoder_config, dict):
            decoder_config = Qwen3TTSTokenizer48kDecoderConfig(**decoder_config)

        # Temporarily store the 48k decoder config, then pass a plain dict to super()
        # so super().__init__ doesn't try to instantiate it as Qwen3TTSTokenizerV2DecoderConfig
        decoder_dict = decoder_config.to_dict()
        self._48k_decoder_config = decoder_config

        # Call grandparent __init__ to skip Qwen3TTSTokenizerV2Config's decoder instantiation
        from transformers.configuration_utils import PretrainedConfig
        PretrainedConfig.__init__(self, **kwargs)

        if encoder_config is None:
            encoder_config = {}
            logger.info("encoder_config is None. Initializing encoder with default values")

        self.encoder_config = MimiConfig(**encoder_config) if isinstance(encoder_config, dict) else encoder_config
        self.decoder_config = self._48k_decoder_config

        self.encoder_valid_num_quantizers = encoder_valid_num_quantizers
        self.input_sample_rate = input_sample_rate
        self.encode_downsample_rate = encode_downsample_rate

        # Auto-adjust output rates when 48kHz upsampler is enabled
        if self.decoder_config.enable_48khz_upsampler:
            upsampler_factor = self.decoder_config.upsampler_factor
            self.output_sample_rate = output_sample_rate * upsampler_factor
            self.decode_upsample_rate = decode_upsample_rate * upsampler_factor
        else:
            self.output_sample_rate = output_sample_rate
            self.decode_upsample_rate = decode_upsample_rate


__all__ = ["Qwen3TTSTokenizer48kConfig", "Qwen3TTSTokenizer48kDecoderConfig"]
