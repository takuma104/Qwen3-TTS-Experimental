# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""ASR dataset utilities for Qwen3-TTS token shards."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset, get_worker_info


@dataclass(frozen=True)
class ASRShard:
    tar_path: str
    jsonl_path: str
    sample_count: int
    total_duration: float


@dataclass(frozen=True)
class ASRSpecialTokenIds:
    bos_token_id: int
    eos_token_id: Optional[int]
    pad_token_id: int


def read_data_lst(data_lst: str | Path) -> List[ASRShard]:
    shards: List[ASRShard] = []
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
                ASRShard(
                    tar_path=tar_path,
                    jsonl_path=jsonl_path,
                    sample_count=int(sample_count),
                    total_duration=float(total_duration),
                )
            )
    return shards


def _resolve_token_id(tokenizer, model_config, *, attr_names: Iterable[str], token_strings: Iterable[str]) -> Optional[int]:
    for obj in (tokenizer, model_config):
        if obj is None:
            continue
        for attr_name in attr_names:
            value = getattr(obj, attr_name, None)
            if value is not None:
                return int(value)

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(convert_tokens_to_ids):
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        for token in token_strings:
            try:
                token_id = convert_tokens_to_ids(token)
            except Exception:
                continue
            if token_id is None:
                continue
            token_id = int(token_id)
            if unk_token_id is not None and token_id == int(unk_token_id):
                continue
            return token_id

    return None


def resolve_asr_special_token_ids(processor, model_config=None) -> ASRSpecialTokenIds:
    tokenizer = getattr(processor, "tokenizer", processor)

    bos_token_id = _resolve_token_id(
        tokenizer,
        model_config,
        attr_names=("bos_token_id", "im_start_token_id"),
        token_strings=("<|im_start|>",),
    )
    if bos_token_id is None:
        raise ValueError("Unable to resolve ASR BOS token id from tokenizer or model config.")

    eos_token_id = _resolve_token_id(
        tokenizer,
        model_config,
        attr_names=("eos_token_id", "im_end_token_id"),
        token_strings=("<|im_end|>",),
    )

    pad_token_id = _resolve_token_id(
        tokenizer,
        model_config,
        attr_names=("pad_token_id", "tts_pad_token_id"),
        token_strings=(),
    )
    if pad_token_id is None:
        pad_token_id = eos_token_id if eos_token_id is not None else bos_token_id

    return ASRSpecialTokenIds(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )


class Qwen3TTSASRWebDataset(IterableDataset):
    """
    Stream Qwen3-TTS-Tokenizer-12Hz token shards.

    The expected format is documented in
    `docs/extract_audio_tokens_hf_output_format.md`: each `.npy` in the tar
    stores int16 codes shaped [16, T], and each JSONL metadata row contains at
    least `id`, `text`, and `num_tokens`.
    """

    def __init__(
        self,
        data_lst: str | Path,
        processor,
        *,
        model_config=None,
        special_token_ids: Optional[ASRSpecialTokenIds] = None,
        add_eos_token: bool = True,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        min_dnsmos: Optional[float] = None,
        languages: Optional[Iterable[str]] = None,
    ):
        self.shards = read_data_lst(data_lst)
        self.processor = processor
        self.add_eos_token = add_eos_token
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_dnsmos = min_dnsmos
        self.languages = {lang.lower() for lang in languages} if languages is not None else None

        tokenizer = getattr(processor, "tokenizer", processor)
        self.tokenizer = tokenizer
        resolved_special_tokens = special_token_ids or resolve_asr_special_token_ids(
            processor,
            model_config=model_config,
        )
        self.bos_token_id = resolved_special_tokens.bos_token_id
        self.eos_token_id = resolved_special_tokens.eos_token_id
        self.pad_token_id = resolved_special_tokens.pad_token_id

    def _iter_worker_shards(self) -> Iterator[ASRShard]:
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

    def _load_npy_from_tar_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
        extracted = tar.extractfile(member)
        if extracted is None:
            raise ValueError(f"Unable to extract {member.name}")
        with extracted:
            data = extracted.read()
        return np.load(io.BytesIO(data))

    def _tokenize_text(self, text: str) -> torch.Tensor:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        if self.add_eos_token and self.eos_token_id is not None:
            encoded = torch.cat([encoded, torch.tensor([self.eos_token_id], dtype=torch.long)])
        return encoded.long()

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
                        raise ValueError(f"{member.name}: expected [16, T] codes, got {codes_np.shape}")

                    expected_num_tokens = metadata.get("num_tokens")
                    if expected_num_tokens is not None and int(expected_num_tokens) != int(codes_np.shape[1]):
                        raise ValueError(
                            f"{member.name}: num_tokens={expected_num_tokens} does not match codes T={codes_np.shape[1]}"
                        )

                    audio_codes = torch.from_numpy(codes_np.astype(np.int64, copy=False)).transpose(0, 1).contiguous()
                    text_ids = self._tokenize_text(str(metadata["text"]))

                    yield {
                        "id": sample_id,
                        "audio_codes": audio_codes,
                        "text_ids": text_ids,
                        "text": metadata["text"],
                        "language_id": metadata.get("language_id", metadata.get("language")),
                        "speaker": metadata.get("speaker"),
                        "audio_duration": metadata.get("audio_duration", metadata.get("duration")),
                        "dnsmos": metadata.get("dnsmos"),
                    }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            raise ValueError("Cannot collate an empty batch")

        audio_codes = pad_sequence(
            [item["audio_codes"] for item in batch],
            batch_first=True,
            padding_value=0,
        )
        audio_lengths = torch.tensor([item["audio_codes"].shape[0] for item in batch], dtype=torch.long)

        decoder_inputs: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        for item in batch:
            text_ids = item["text_ids"]
            bos = torch.tensor([self.bos_token_id], dtype=torch.long)
            decoder_inputs.append(torch.cat([bos, text_ids[:-1]]))
            labels.append(text_ids)

        decoder_input_ids = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
        text_lengths = torch.tensor([item["text_ids"].shape[0] for item in batch], dtype=torch.long)

        max_audio_len = audio_codes.shape[1]
        max_text_len = decoder_input_ids.shape[1]
        audio_mask = torch.arange(max_audio_len).unsqueeze(0) < audio_lengths.unsqueeze(1)
        text_mask = torch.arange(max_text_len).unsqueeze(0) < text_lengths.unsqueeze(1)
        attention_mask = torch.cat([audio_mask, text_mask], dim=1).long()

        return {
            "audio_codes": audio_codes,
            "audio_lengths": audio_lengths,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_padded,
            "text_lengths": text_lengths,
            "attention_mask": attention_mask,
        }
