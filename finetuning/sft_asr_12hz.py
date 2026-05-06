# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""SFT entrypoint for experimental Qwen3-TTS 12Hz ASR."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from asr_dataset import Qwen3TTSASRWebDataset, resolve_asr_special_token_ids
from qwen_tts.core.models.modeling_qwen3_tts_asr import Qwen3TTSForSpeechRecognition
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader


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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_tts_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--qwen3_model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_lst", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="asr_output")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_acoustic_codebooks", action="store_true")
    parser.add_argument("--freeze_talker", action="store_true")
    parser.add_argument("--min_duration", type=float, default=None)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--min_dnsmos", type=float, default=None)
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language filter, e.g. ja,en")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    dtype = parse_dtype(args.dtype)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_tts_model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    )

    special_token_ids = resolve_asr_special_token_ids(
        qwen3tts.processor,
        model_config=qwen3tts.model.config,
    )
    accelerator.print(
        "Resolved ASR special tokens: "
        f"bos={special_token_ids.bos_token_id} "
        f"eos={special_token_ids.eos_token_id} "
        f"pad={special_token_ids.pad_token_id}"
    )
    asr_model = Qwen3TTSForSpeechRecognition(
        qwen3tts.model,
        use_acoustic_codebooks=args.use_acoustic_codebooks,
        asr_bos_token_id=special_token_ids.bos_token_id,
        asr_eos_token_id=special_token_ids.eos_token_id,
        asr_pad_token_id=special_token_ids.pad_token_id,
    )
    load_info = asr_model.load_qwen3_text_weights(
        args.qwen3_model_path,
        torch_dtype=dtype,
    )
    accelerator.print(f"Loaded Qwen3 text weights: {load_info}")

    asr_model.freeze_tts_side_modules()
    if args.freeze_talker:
        asr_model.freeze_talker()

    languages = None
    if args.languages:
        languages = [item.strip() for item in args.languages.split(",") if item.strip()]

    dataset = Qwen3TTSASRWebDataset(
        args.data_lst,
        qwen3tts.processor,
        model_config=qwen3tts.model.config,
        special_token_ids=special_token_ids,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_dnsmos=args.min_dnsmos,
        languages=languages,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )

    trainable_parameters = [parameter for parameter in asr_model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    asr_model, optimizer, dataloader = accelerator.prepare(asr_model, optimizer, dataloader)
    asr_model.train()

    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(asr_model):
                outputs = asr_model(
                    audio_codes=batch["audio_codes"],
                    decoder_input_ids=batch["decoder_input_ids"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(asr_model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if global_step % 10 == 0:
                accelerator.print(f"epoch={epoch} step={global_step} loss={loss.item():.4f}")

            if args.save_every_steps > 0 and global_step > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(accelerator, asr_model, args.output_dir, f"checkpoint-step-{global_step}", args)

            global_step += 1
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        save_checkpoint(accelerator, asr_model, args.output_dir, f"checkpoint-epoch-{epoch}", args)
        if args.max_steps > 0 and global_step >= args.max_steps:
            break


def save_checkpoint(accelerator: Accelerator, model, output_dir: str, name: str, args):
    if not accelerator.is_main_process:
        return

    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    state_dict = {key: value.detach().cpu() for key, value in unwrapped.state_dict().items()}
    save_file(state_dict, checkpoint_dir / "asr_model.safetensors")

    config = {
        "init_tts_model_path": args.init_tts_model_path,
        "qwen3_model_path": args.qwen3_model_path,
        "use_acoustic_codebooks": args.use_acoustic_codebooks,
        "freeze_talker": args.freeze_talker,
    }
    with open(checkpoint_dir / "asr_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
