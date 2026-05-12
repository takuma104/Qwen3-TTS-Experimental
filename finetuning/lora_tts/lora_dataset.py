# coding=utf-8
# Copyright 2026 The Qwen team.
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
"""Dataset utilities for Qwen3-TTS LoRA fine-tuning on the
``extract_audio_tokens_hf.py`` WebDataset shards (see
``docs/extract_audio_tokens_hf_output_format.md``).

Speaker conditioning is disabled in this pipeline: the per-sample 1024-dim
speaker embedding slot is removed entirely, mirroring the
``speaker_embed is None`` branch in ``Qwen3TTSModel`` inference.
"""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class TTSShard:
    tar_path: str
    jsonl_path: str
    sample_count: int
    total_duration: float


def read_data_lst(data_lst: str | Path) -> List[TTSShard]:
    shards: List[TTSShard] = []
    with open(data_lst, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{data_lst}:{line_no}: expected 4 columns, got {len(parts)}")
            tar_path, jsonl_path, sample_count, total_duration = parts
            shards.append(
                TTSShard(
                    tar_path=tar_path,
                    jsonl_path=jsonl_path,
                    sample_count=int(sample_count),
                    total_duration=float(total_duration),
                )
            )
    return shards


def total_sample_count(shards: List[TTSShard]) -> int:
    return sum(shard.sample_count for shard in shards)


def total_duration_sec(shards: List[TTSShard]) -> float:
    return sum(shard.total_duration for shard in shards)


class TTSLoRAWebDataset(IterableDataset):
    """Stream Qwen3-TTS-Tokenizer-12Hz token shards for TTS LoRA fine-tuning.

    Each shard contributes ``(audio_codes, text)`` pairs; text is tokenized on
    the fly using the model processor, wrapped in the same assistant-role
    template as inference (``<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n``)
    with the trailing 5 prompt tokens stripped so only the
    ``<|im_start|>assistant\\n{text}`` prefix remains.
    """

    def __init__(
        self,
        data_lst: str | Path,
        processor,
        *,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_dnsmos: Optional[float] = None,
        languages: Optional[Iterable[str]] = None,
        max_audio_codes_len: Optional[int] = None,
    ):
        self.shards = read_data_lst(data_lst)
        self.processor = processor
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_dnsmos = min_dnsmos
        self.languages = {lang.lower() for lang in languages} if languages is not None else None
        self.max_audio_codes_len = max_audio_codes_len

    def _iter_worker_shards(self) -> Iterator[TTSShard]:
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        rank = 0
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        for idx, shard in enumerate(self.shards):
            if idx % global_num_workers == global_worker_id:
                yield shard

    def _load_metadata(self, jsonl_path: str) -> Dict[str, Dict[str, Any]]:
        metadata: Dict[str, Dict[str, Any]] = {}
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                sample_id = item.get("id")
                if not sample_id:
                    raise ValueError(f"{jsonl_path}:{line_no}: missing `id`")
                metadata[sample_id] = item
        return metadata

    def _keep_metadata(self, metadata: Dict[str, Any]) -> bool:
        duration = metadata.get("audio_duration", metadata.get("duration"))
        if duration is not None:
            duration = float(duration)
            if self.min_duration is not None and duration < self.min_duration:
                return False
            if self.max_duration is not None and duration > self.max_duration:
                return False

        dnsmos = metadata.get("dnsmos")
        if self.min_dnsmos is not None and dnsmos is not None and float(dnsmos) < self.min_dnsmos:
            return False

        if self.languages is not None:
            language = metadata.get("language_id", metadata.get("language"))
            if language is None or str(language).lower() not in self.languages:
                return False

        return True

    @staticmethod
    def _load_npy_from_tar_member(tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"Unable to extract {member.name}")
        with extracted:
            data = extracted.read()
        return np.load(io.BytesIO(data))

    def _tokenize_text(self, text: str) -> torch.Tensor:
        wrapped = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        encoded = self.processor(text=wrapped, return_tensors="pt", padding=True)["input_ids"]
        if encoded.dim() == 1:
            encoded = encoded.unsqueeze(0)
        # Strip the trailing 5 prompt tokens (`<|im_end|>\n<|im_start|>assistant\n`).
        return encoded[:, :-5].long()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for shard in self._iter_worker_shards():
            metadata_by_id = self._load_metadata(shard.jsonl_path)
            with tarfile.open(shard.tar_path, "r:*") as tar:
                for member in tar:
                    if not member.isfile() or not member.name.endswith(".npy"):
                        continue
                    sample_id = Path(member.name).stem
                    metadata = metadata_by_id.get(sample_id)
                    if metadata is None or not self._keep_metadata(metadata):
                        continue

                    codes_np = self._load_npy_from_tar_member(tar, member)
                    if codes_np.ndim != 2 or codes_np.shape[0] != 16:
                        raise ValueError(
                            f"{member.name}: expected [16, T] codes, got {codes_np.shape}"
                        )

                    expected_num_tokens = metadata.get("num_tokens")
                    if expected_num_tokens is not None and int(expected_num_tokens) != int(codes_np.shape[1]):
                        raise ValueError(
                            f"{member.name}: num_tokens={expected_num_tokens} does not match codes T={codes_np.shape[1]}"
                        )

                    if self.max_audio_codes_len is not None and codes_np.shape[1] > self.max_audio_codes_len:
                        continue

                    # (16, T) int16 -> (T, 16) int64
                    audio_codes = torch.from_numpy(
                        codes_np.astype(np.int64, copy=False)
                    ).transpose(0, 1).contiguous()
                    text_ids = self._tokenize_text(str(metadata["text"]))

                    yield {
                        "id": sample_id,
                        "text_ids": text_ids,       # (1, text_len)
                        "audio_codes": audio_codes, # (codec_len, 16)
                    }


class TokenBudgetBatchDataset(IterableDataset):
    """Group iterable TTS LoRA samples into batches bounded by padded token count.

    Token budget approximates the collated tensor size: each sample contributes
    ``(max_text_len + max_codec_len + 7)`` tokens after padding, where ``+7``
    is the fixed prefix (3 role + 4 codec metadata, no speaker slot). The
    constant is included so the budget matches the actual tensor footprint.
    """

    PREFIX_TOKENS = 7

    def __init__(
        self,
        dataset: IterableDataset,
        *,
        max_batch_tokens: int,
        max_batch_samples: Optional[int] = None,
    ):
        if max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        self.dataset = dataset
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_batch_samples = (
            int(max_batch_samples) if max_batch_samples and max_batch_samples > 0 else None
        )

    @staticmethod
    def _sample_lengths(sample: Dict[str, Any]) -> tuple[int, int]:
        text_len = int(sample["text_ids"].shape[1])
        codec_len = int(sample["audio_codes"].shape[0])
        return text_len, codec_len

    @classmethod
    def _padded_token_count(cls, batch_size: int, max_text_len: int, max_codec_len: int) -> int:
        return batch_size * (max_text_len + max_codec_len + cls.PREFIX_TOKENS)

    def __iter__(self) -> Iterator[List[Dict[str, Any]]]:
        batch: List[Dict[str, Any]] = []
        max_text_len = 0
        max_codec_len = 0

        for sample in self.dataset:
            text_len, codec_len = self._sample_lengths(sample)
            next_batch_size = len(batch) + 1
            next_max_text_len = max(max_text_len, text_len)
            next_max_codec_len = max(max_codec_len, codec_len)
            next_token_count = self._padded_token_count(
                next_batch_size, next_max_text_len, next_max_codec_len
            )
            exceeds_token_budget = next_token_count > self.max_batch_tokens
            exceeds_sample_budget = (
                self.max_batch_samples is not None and next_batch_size > self.max_batch_samples
            )

            if batch and (exceeds_token_budget or exceeds_sample_budget):
                yield batch
                batch = []
                max_text_len = 0
                max_codec_len = 0

            batch.append(sample)
            max_text_len = max(max_text_len, text_len)
            max_codec_len = max(max_codec_len, codec_len)

        if batch:
            yield batch


def collate_fn_lora(
    batch: List[Dict[str, Any]],
    config: Qwen3TTSConfig,
) -> Dict[str, torch.Tensor]:
    """Collate a batch for TTS LoRA fine-tuning.

    Layout (no speaker conditioning — the original position 6 speaker slot is
    dropped so everything shifts left by 1 relative to the speaker-enabled
    path)::

        text  [role_0, role_1, role_2, pad, pad, pad, tts_bos, <text...>, tts_eos, pad, ...]
        codec [   _ ,    _ ,    _ , nothink, think_bos, think_eos, codec_pad, codec_pad..., codec_bos, <audio_codec_0...>, codec_eos]

    Index ranges (per sample, with ``T = text_ids_len``, ``C = codec_ids_len``):

      - role:                   positions 0..2
      - codec prefix:           positions 3..5 (nothink, think_bos, think_eos)
      - boundary:               position 6 (codec_pad / tts_bos)
      - text body:              positions 7..7+T-4
      - text eos:               position 7+T-3
      - codec_bos:              position 7+T-2
      - audio codec_0:          positions 7+T-1..7+T+C-2
      - codec eos:              position 7+T+C-1

    Total active length = ``7 + T + C``.
    """
    item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
    max_length = max(item_length) + 7
    b_size, t = len(batch), max_length

    input_ids = torch.zeros((b_size, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b_size, t, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((b_size, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b_size, t), dtype=torch.bool)
    codec_mask = torch.zeros((b_size, t), dtype=torch.bool)
    attention_mask = torch.zeros((b_size, t), dtype=torch.long)
    codec_0_labels = torch.full((b_size, t), -100, dtype=torch.long)

    for i, data in enumerate(batch):
        text_ids = data["text_ids"]
        audio_codecs = data["audio_codes"]
        audio_codec_0 = audio_codecs[:, 0]

        text_ids_len = text_ids.shape[1]
        codec_ids_len = audio_codec_0.shape[0]
        active_len = 7 + text_ids_len + codec_ids_len

        # Text channel
        input_ids[i, :3, 0] = text_ids[0, :3]
        input_ids[i, 3:6, 0] = config.tts_pad_token_id
        input_ids[i, 6, 0] = config.tts_bos_token_id
        input_ids[i, 7:7 + text_ids_len - 3, 0] = text_ids[0, 3:]
        input_ids[i, 7 + text_ids_len - 3, 0] = config.tts_eos_token_id
        input_ids[i, 7 + text_ids_len - 2:active_len, 0] = config.tts_pad_token_id

        # Codec channel
        input_ids[i, 3:7, 1] = torch.tensor([
            config.talker_config.codec_nothink_id,
            config.talker_config.codec_think_bos_id,
            config.talker_config.codec_think_eos_id,
            config.talker_config.codec_pad_id,
        ])
        input_ids[i, 7:7 + text_ids_len - 2, 1] = config.talker_config.codec_pad_id
        input_ids[i, 7 + text_ids_len - 2, 1] = config.talker_config.codec_bos_id
        input_ids[i, 7 + text_ids_len - 1:7 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
        input_ids[i, 7 + text_ids_len - 1 + codec_ids_len, 1] = config.talker_config.codec_eos_token_id

        # Labels for codec_0
        codec_0_labels[i, 7 + text_ids_len - 1:7 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
        codec_0_labels[i, 7 + text_ids_len - 1 + codec_ids_len] = config.talker_config.codec_eos_token_id

        # Full codec ids (all 16 codebooks) live alongside codec_0.
        codec_ids[i, 7 + text_ids_len - 1:7 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

        # Masks
        text_embedding_mask[i, :active_len] = True
        codec_embedding_mask[i, 3:active_len] = True
        codec_mask[i, 7 + text_ids_len - 1:7 + text_ids_len - 1 + codec_ids_len] = True
        attention_mask[i, :active_len] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
        "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
        "codec_0_labels": codec_0_labels,
        "codec_ids": codec_ids,
        "codec_mask": codec_mask,
    }
