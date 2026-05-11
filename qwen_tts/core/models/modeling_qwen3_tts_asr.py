# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
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
"""Experimental Qwen3-TTS speech-to-text wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from .modeling_qwen3_tts import Qwen3TTSForConditionalGeneration


@dataclass
class Qwen3TTSASRLoadInfo:
    """Summary of Qwen3 text weights copied into the ASR wrapper."""

    text_embedding_key: str
    text_head_key: str


class Qwen3TTSForSpeechRecognition(nn.Module):
    """
    Thin ASR wrapper around the Qwen3-TTS Talker.

    Two text-embedding paths are supported for the decoder input:

    * ``use_tts_text_embedding=False`` (default): adds a new
      ``asr_text_embedding`` (shape ``[vocab, hidden]``) initialized from
      Qwen3-0.6B ``embed_tokens``.  This keeps the TTS side untouched.

    * ``use_tts_text_embedding=True``: reuses the TTS Talker's existing
      ``text_embedding`` (2048-dim) + ``text_projection`` (2048→1024 MLP)
      that were jointly trained with the Talker body during TTS pre-training.
      When ``freeze_tts_text_embedding=True`` those weights are frozen;
      otherwise they remain trainable.
    """

    def __init__(
        self,
        tts_model: Qwen3TTSForConditionalGeneration,
        text_vocab_size: Optional[int] = None,
        use_acoustic_codebooks: bool = False,
        asr_bos_token_id: Optional[int] = None,
        asr_eos_token_id: Optional[int] = None,
        asr_pad_token_id: Optional[int] = None,
        use_tts_text_embedding: bool = False,
        freeze_tts_text_embedding: bool = False,
    ):
        super().__init__()
        self.tts_model = tts_model
        self.config = tts_model.config
        self.talker_config = tts_model.config.talker_config
        self.use_acoustic_codebooks = use_acoustic_codebooks
        self.use_tts_text_embedding = use_tts_text_embedding

        self.hidden_size = self.talker_config.hidden_size
        self.text_vocab_size = text_vocab_size or self.talker_config.text_vocab_size

        self.asr_bos_token_id = asr_bos_token_id
        self.asr_eos_token_id = asr_eos_token_id or getattr(self.config, "im_end_token_id", None)
        self.asr_pad_token_id = asr_pad_token_id

        talker_parameter = next(self.talker.parameters())

        if not use_tts_text_embedding:
            self.asr_text_embedding = nn.Embedding(self.text_vocab_size, self.hidden_size)
            self.asr_text_embedding.to(device=talker_parameter.device, dtype=talker_parameter.dtype)
        else:
            self.asr_text_embedding = None
            if freeze_tts_text_embedding:
                for p in self.talker.model.text_embedding.parameters():
                    p.requires_grad = False
                for p in self.talker.text_projection.parameters():
                    p.requires_grad = False

        self.text_head = nn.Linear(self.hidden_size, self.text_vocab_size, bias=False)
        self.text_head.to(device=talker_parameter.device, dtype=talker_parameter.dtype)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def talker(self):
        return self.tts_model.talker

    def freeze_talker(self):
        for parameter in self.talker.parameters():
            parameter.requires_grad = False

    def freeze_tts_side_modules(self):
        """Freeze modules that are not used directly by the ASR text loss."""
        if getattr(self.tts_model, "speaker_encoder", None) is not None:
            for parameter in self.tts_model.speaker_encoder.parameters():
                parameter.requires_grad = False
        if getattr(self.talker, "code_predictor", None) is not None and not self.use_acoustic_codebooks:
            for parameter in self.talker.code_predictor.parameters():
                parameter.requires_grad = False
        if getattr(self.talker, "codec_head", None) is not None:
            for parameter in self.talker.codec_head.parameters():
                parameter.requires_grad = False

    def tie_text_weights(self):
        if self.use_tts_text_embedding:
            return
        self.text_head.weight = self.asr_text_embedding.weight

    @torch.no_grad()
    def load_qwen3_text_weights(
        self,
        qwen3_model_name_or_path: str,
        *,
        tie_text_weights: bool = True,
        **from_pretrained_kwargs,
    ) -> Qwen3TTSASRLoadInfo:
        """
        Initialize ASR text head (and embedding when not using TTS path) from a
        Qwen3 CausalLM checkpoint.

        When ``use_tts_text_embedding=True`` only ``text_head`` is loaded;
        ``asr_text_embedding`` is skipped because the TTS ``text_embedding +
        text_projection`` path is used instead.

        Qwen3-0.6B usually ties ``lm_head.weight`` to
        ``model.embed_tokens.weight``.  If ``lm_head.weight`` is absent the
        embedding weight is used as a fallback for ``text_head``.
        """
        qwen3 = AutoModelForCausalLM.from_pretrained(qwen3_model_name_or_path, **from_pretrained_kwargs)
        state_dict = qwen3.state_dict()

        embedding_key = "model.embed_tokens.weight"
        embedding_weight = state_dict.get(embedding_key)

        if not self.use_tts_text_embedding:
            if embedding_weight is None:
                raise KeyError(f"{embedding_key} not found in Qwen3 checkpoint")
            expected_embedding_shape = self.asr_text_embedding.weight.shape
            if embedding_weight.shape != expected_embedding_shape:
                raise ValueError(
                    f"Qwen3 text embedding shape {tuple(embedding_weight.shape)} does not match "
                    f"ASR embedding shape {tuple(expected_embedding_shape)}"
                )
            self.asr_text_embedding.weight.copy_(embedding_weight.to(self.asr_text_embedding.weight.device))

        head_key = "lm_head.weight"
        head_weight = state_dict.get(head_key)
        if head_weight is None:
            if embedding_weight is None:
                raise KeyError(f"Neither {head_key} nor {embedding_key} found in Qwen3 checkpoint")
            head_weight = embedding_weight
            head_key = embedding_key

        expected_head_shape = self.text_head.weight.shape
        if head_weight.shape != expected_head_shape:
            raise ValueError(
                f"Qwen3 text head shape {tuple(head_weight.shape)} does not match "
                f"ASR text head shape {tuple(expected_head_shape)}"
            )
        self.text_head.weight.copy_(head_weight.to(self.text_head.weight.device))

        if tie_text_weights and not self.use_tts_text_embedding:
            self.tie_text_weights()

        del qwen3
        return Qwen3TTSASRLoadInfo(
            text_embedding_key="tts_text_embedding+text_projection" if self.use_tts_text_embedding else embedding_key,
            text_head_key=head_key,
        )

    def _normalize_audio_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        if audio_codes.ndim == 2:
            audio_codes = audio_codes.unsqueeze(0)
        if audio_codes.ndim != 3:
            raise ValueError(f"audio_codes must have shape [B, T, Q] or [T, Q], got {tuple(audio_codes.shape)}")
        if audio_codes.shape[-1] != self.talker_config.num_code_groups:
            raise ValueError(
                f"Expected {self.talker_config.num_code_groups} codebooks, got {audio_codes.shape[-1]}"
            )
        return audio_codes.long()

    def build_speech_embeddings(self, audio_codes: torch.Tensor) -> torch.Tensor:
        audio_codes = self._normalize_audio_codes(audio_codes)
        codebook0 = audio_codes[..., 0]
        speech_embeddings = self.talker.model.codec_embedding(codebook0)

        if self.use_acoustic_codebooks:
            code_embeddings = self.talker.code_predictor.get_input_embeddings()
            for codebook_idx in range(1, self.talker_config.num_code_groups):
                speech_embeddings = speech_embeddings + code_embeddings[codebook_idx - 1](audio_codes[..., codebook_idx])

        return speech_embeddings

    def build_inputs_embeds(
        self,
        audio_codes: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        speech_embeddings = self.build_speech_embeddings(audio_codes)
        ids = decoder_input_ids.long()
        if self.use_tts_text_embedding:
            text_embeddings = self.talker.text_projection(self.talker.get_text_embeddings()(ids))
        else:
            text_embeddings = self.asr_text_embedding(ids)
        return torch.cat([speech_embeddings, text_embeddings], dim=1)

    def forward(
        self,
        audio_codes: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        audio_codes = self._normalize_audio_codes(audio_codes)
        if decoder_input_ids.ndim == 1:
            decoder_input_ids = decoder_input_ids.unsqueeze(0)
        decoder_input_ids = decoder_input_ids.long()

        inputs_embeds = self.build_inputs_embeds(audio_codes, decoder_input_ids)
        batch_size, speech_len = audio_codes.shape[:2]
        text_len = decoder_input_ids.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size,
                speech_len + text_len,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

        outputs = self.talker.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        text_hidden_states = outputs.last_hidden_state[:, speech_len:, :]
        logits = self.text_head(text_hidden_states)

        loss = None
        if labels is not None:
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)
            labels = labels.long().to(logits.device)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.inference_mode()
    def generate(
        self,
        audio_codes: torch.Tensor,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_new_tokens: int = 256,
    ) -> torch.Tensor:
        """Greedy ASR text generation without KV-cache optimization."""
        audio_codes = self._normalize_audio_codes(audio_codes).to(self.device)
        batch_size = audio_codes.shape[0]
        bos_token_id = bos_token_id if bos_token_id is not None else self.asr_bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.asr_eos_token_id
        if bos_token_id is None:
            raise ValueError("bos_token_id must be provided or configured as asr_bos_token_id")

        generated = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_new_tokens):
            outputs = self(audio_codes=audio_codes, decoder_input_ids=generated)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            if eos_token_id is not None:
                finished |= next_token.eq(eos_token_id)
                if finished.all():
                    break

        return generated[:, 1:]
