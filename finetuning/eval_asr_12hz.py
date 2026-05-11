# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate an experimental Qwen3-TTS 12Hz ASR checkpoint on token shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from asr_dataset import Qwen3TTSASRWebDataset, resolve_asr_special_token_ids
from peft import PeftModel
from qwen_tts.core.models.modeling_qwen3_tts_asr import Qwen3TTSForSpeechRecognition
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import load_file


def parse_dtype(dtype: str):
    if dtype == "auto":
        return "auto"
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_training_config(checkpoint_dir: Path) -> Dict[str, Any]:
    config_path = checkpoint_dir / "asr_training_config.json"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def choose_arg(explicit_value: Optional[str], config: Dict[str, Any], key: str, default: str) -> str:
    if explicit_value is not None:
        return explicit_value
    value = config.get(key)
    if value is not None:
        return str(value)
    return default


def strip_after_eos(token_ids: torch.Tensor, eos_token_id: Optional[int]) -> torch.Tensor:
    token_ids = token_ids.detach().cpu()
    if eos_token_id is None:
        return token_ids
    eos_positions = (token_ids == int(eos_token_id)).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        return token_ids
    return token_ids[: int(eos_positions[0].item())]


def find_eos_index(token_ids: torch.Tensor, eos_token_id: Optional[int]) -> Optional[int]:
    if eos_token_id is None:
        return None
    eos_positions = (token_ids.detach().cpu() == int(eos_token_id)).nonzero(as_tuple=False)
    if eos_positions.numel() == 0:
        return None
    return int(eos_positions[0].item())


def normalize_for_match(text: str) -> str:
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_lst", type=str, required=True)
    parser.add_argument("--init_tts_model_path", type=str, default=None)
    parser.add_argument("--qwen3_model_path", type=str, default=None,
                        help="Qwen3 text model path for loading asr_text_embedding/text_head. "
                             "Required for LoRA checkpoints trained with --lora_modules_to_save none. "
                             "Falls back to qwen3_model_path in asr_training_config.json.")
    parser.add_argument("--output_jsonl", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_acoustic_codebooks", action="store_true")
    parser.add_argument("--print_all", action="store_true")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    is_lora = (checkpoint_dir / "adapter_config.json").exists()

    if not is_lora:
        checkpoint_path = checkpoint_dir / "asr_model.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint weights not found: {checkpoint_path}")

    training_config = load_training_config(checkpoint_dir)
    init_tts_model_path = choose_arg(
        args.init_tts_model_path,
        training_config,
        "init_tts_model_path",
        "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    )
    use_acoustic_codebooks = bool(args.use_acoustic_codebooks or training_config.get("use_acoustic_codebooks", False))

    dtype = parse_dtype(args.dtype)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        init_tts_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    special_token_ids = resolve_asr_special_token_ids(
        qwen3tts.processor,
        model_config=qwen3tts.model.config,
    )
    asr_model = Qwen3TTSForSpeechRecognition(
        qwen3tts.model,
        use_acoustic_codebooks=use_acoustic_codebooks,
        asr_bos_token_id=special_token_ids.bos_token_id,
        asr_eos_token_id=special_token_ids.eos_token_id,
        asr_pad_token_id=special_token_ids.pad_token_id,
    )

    if is_lora:
        qwen3_model_path = args.qwen3_model_path or training_config.get("qwen3_model_path")
        if qwen3_model_path:
            load_info = asr_model.load_qwen3_text_weights(qwen3_model_path, torch_dtype=dtype)
            print(f"Loaded Qwen3 text weights from {qwen3_model_path}: {load_info}")
        else:
            print(
                "WARNING: qwen3_model_path not specified and not found in training config. "
                "asr_text_embedding and text_head will use TTS-side initialization, "
                "which is incorrect if they were not saved in the LoRA adapter (modules_to_save=None)."
            )
        asr_model = PeftModel.from_pretrained(asr_model, str(checkpoint_dir))
        print(f"Loaded LoRA checkpoint: {checkpoint_dir}")
    else:
        state_dict = load_file(str(checkpoint_path), device="cpu")
        load_result = asr_model.load_state_dict(state_dict, strict=True)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Loaded state dict: missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}")
    print(
        "Resolved ASR special tokens: "
        f"bos={special_token_ids.bos_token_id} "
        f"eos={special_token_ids.eos_token_id} "
        f"pad={special_token_ids.pad_token_id}"
    )

    device = torch.device(args.device)
    asr_model.to(device)
    asr_model.eval()

    dataset = Qwen3TTSASRWebDataset(
        args.data_lst,
        qwen3tts.processor,
        model_config=qwen3tts.model.config,
        special_token_ids=special_token_ids,
    )
    tokenizer = qwen3tts.processor.tokenizer

    output_file = None
    if args.output_jsonl is not None:
        output_path = Path(args.output_jsonl)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_file = open(output_path, "w", encoding="utf-8")

    total = 0
    exact = 0
    teacher_forced_exact = 0
    try:
        for sample in dataset:
            audio_codes = sample["audio_codes"].unsqueeze(0).to(device)
            generated_ids = asr_model.generate(
                audio_codes,
                max_new_tokens=args.max_new_tokens,
            )[0]
            eos_index = find_eos_index(generated_ids, special_token_ids.eos_token_id)
            visible_generated_ids = strip_after_eos(generated_ids, special_token_ids.eos_token_id)
            prediction = tokenizer.decode(visible_generated_ids.tolist(), skip_special_tokens=True)
            reference = str(sample["text"])
            is_exact = normalize_for_match(prediction) == normalize_for_match(reference)

            reference_ids = sample["text_ids"].unsqueeze(0).to(device)
            bos = torch.tensor([[special_token_ids.bos_token_id]], dtype=torch.long, device=device)
            decoder_input_ids = torch.cat([bos, reference_ids[:, :-1]], dim=1)
            attention_mask = torch.ones(
                1,
                audio_codes.shape[1] + decoder_input_ids.shape[1],
                dtype=torch.long,
                device=device,
            )
            outputs = asr_model(
                audio_codes=audio_codes,
                decoder_input_ids=decoder_input_ids,
                attention_mask=attention_mask,
            )
            teacher_forced_ids = outputs.logits.argmax(dim=-1)[0].detach().cpu()
            reference_ids_cpu = sample["text_ids"].detach().cpu()
            is_teacher_forced_exact = torch.equal(teacher_forced_ids, reference_ids_cpu)
            teacher_forced_eos_correct = bool(
                special_token_ids.eos_token_id is not None
                and reference_ids_cpu.numel() > 0
                and int(reference_ids_cpu[-1].item()) == int(special_token_ids.eos_token_id)
                and int(teacher_forced_ids[-1].item()) == int(special_token_ids.eos_token_id)
            )

            total += 1
            exact += int(is_exact)
            teacher_forced_exact += int(is_teacher_forced_exact)

            record = {
                "id": sample["id"],
                "reference": reference,
                "prediction": prediction,
                "exact_match": is_exact,
                "eos_generated": eos_index is not None,
                "eos_index": eos_index,
                "generated_token_count": int(generated_ids.numel()),
                "visible_token_count": int(visible_generated_ids.numel()),
                "reference_token_count": int(reference_ids_cpu.numel()),
                "teacher_forced_exact": is_teacher_forced_exact,
                "teacher_forced_eos_correct": teacher_forced_eos_correct,
            }
            if output_file is not None:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.print_all or not is_exact:
                marker = "OK" if is_exact else "NG"
                print(f"[{marker}] {sample['id']}")
                print(f"REF: {reference}")
                print(f"HYP: {prediction}")
                print(
                    "EOS: "
                    f"generated={eos_index is not None} "
                    f"index={eos_index} "
                    f"teacher_forced_eos_correct={teacher_forced_eos_correct}"
                )

        accuracy = exact / total if total else 0.0
        teacher_forced_accuracy = teacher_forced_exact / total if total else 0.0
        print(
            "Summary: "
            f"total={total} "
            f"exact={exact} "
            f"exact_match={accuracy:.4f} "
            f"teacher_forced_exact={teacher_forced_exact} "
            f"teacher_forced_exact_match={teacher_forced_accuracy:.4f}"
        )
    finally:
        if output_file is not None:
            output_file.close()


if __name__ == "__main__":
    main()
