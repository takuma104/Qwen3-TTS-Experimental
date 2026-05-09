# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""LoRA SFT entrypoint for experimental Qwen3-TTS 12Hz ASR.

Mirrors `finetuning/sft_asr_12hz.py` but applies a PEFT LoRA adapter to the
Talker body. The newly added `asr_text_embedding` and `text_head` are
registered as `modules_to_save` so they remain fully trainable from their
Qwen3 initialization.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import torch
from accelerate import Accelerator
from asr_dataset import Qwen3TTSASRWebDataset, TokenBudgetBatchDataset, resolve_asr_special_token_ids
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_tts.core.models.modeling_qwen3_tts_asr import Qwen3TTSForSpeechRecognition
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# LoRA targets the Talker body only — code_predictor layers share the same
# projection names but live under a different parent and must be excluded.
DEFAULT_LORA_TARGET_REGEX = (
    r"^tts_model\.talker\.model\.layers\.\d+\.(self_attn|mlp)\."
    r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
)
DEFAULT_MODULES_TO_SAVE = ("asr_text_embedding", "text_head")


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


def parse_report_to(report_to: str):
    trackers = [item.strip() for item in report_to.split(",") if item.strip()]
    if not trackers or trackers == ["none"]:
        return None
    return trackers


def parse_modules_to_save(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return list(DEFAULT_MODULES_TO_SAVE)
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items or items == ["none"]:
        return None
    return items


def build_lora_config(args) -> LoraConfig:
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=args.lora_target_regex,
        modules_to_save=parse_modules_to_save(args.lora_modules_to_save),
        task_type=None,
    )


def build_asr_dataset(args, data_lst: str, processor, model_config, special_token_ids):
    return Qwen3TTSASRWebDataset(
        data_lst,
        processor,
        model_config=model_config,
        special_token_ids=special_token_ids,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_dnsmos=args.min_dnsmos,
        languages=args.languages,
    )


def build_dataloader(dataset, batch_size: int, max_batch_tokens: Optional[int] = None):
    if max_batch_tokens is not None and max_batch_tokens > 0:
        batched_dataset = TokenBudgetBatchDataset(
            dataset,
            max_batch_tokens=max_batch_tokens,
            max_batch_samples=batch_size,
        )
        return DataLoader(
            batched_dataset,
            batch_size=None,
            collate_fn=dataset.collate_fn,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.collate_fn,
    )


def get_lr(optimizer, default_lr: float) -> float:
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups:
        return float(param_groups[0]["lr"])
    wrapped_optimizer = getattr(optimizer, "optimizer", None)
    param_groups = getattr(wrapped_optimizer, "param_groups", None)
    if param_groups:
        return float(param_groups[0]["lr"])
    return float(default_lr)


@torch.no_grad()
def evaluate(asr_model, dataloader, accelerator: Accelerator, max_eval_batches: Optional[int] = None):
    was_training = asr_model.training
    asr_model.eval()

    device = accelerator.device
    loss_sum = torch.tensor(0.0, device=device)
    text_token_count = torch.tensor(0, device=device, dtype=torch.long)
    audio_token_count = torch.tensor(0, device=device, dtype=torch.long)
    sample_count = torch.tensor(0, device=device, dtype=torch.long)

    for batch_idx, batch in enumerate(dataloader):
        if max_eval_batches is not None and max_eval_batches > 0 and batch_idx >= max_eval_batches:
            break

        outputs = asr_model(
            audio_codes=batch["audio_codes"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
            attention_mask=batch["attention_mask"],
        )
        batch_text_tokens = (batch["labels"] != -100).sum()
        batch_audio_tokens = batch["audio_lengths"].sum()
        loss_sum += outputs.loss.detach() * batch_text_tokens
        text_token_count += batch_text_tokens
        audio_token_count += batch_audio_tokens
        sample_count += torch.tensor(batch["audio_codes"].shape[0], device=device, dtype=torch.long)

    loss_sum = accelerator.reduce(loss_sum, reduction="sum")
    text_token_count = accelerator.reduce(text_token_count, reduction="sum")
    audio_token_count = accelerator.reduce(audio_token_count, reduction="sum")
    sample_count = accelerator.reduce(sample_count, reduction="sum")

    eval_loss = loss_sum / text_token_count.clamp_min(1)
    metrics = {
        "eval/loss": eval_loss.item(),
        "eval/text_tokens": text_token_count.item(),
        "eval/audio_tokens": audio_token_count.item(),
        "eval/samples": sample_count.item(),
    }

    if was_training:
        asr_model.train()
    return metrics


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_tts_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    parser.add_argument("--qwen3_model_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--data_lst", type=str, required=True)
    parser.add_argument("--eval_data_lst", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="asr_lora_output")
    parser.add_argument("--resume_from_adapter", type=str, default=None,
                        help="Path to a previously saved LoRA adapter directory to resume from.")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=0,
        help="If >0, dynamically batches by padded token budget: batch_size * (max_audio_len + max_text_len).",
    )
    parser.add_argument(
        "--eval_max_batch_tokens",
        type=int,
        default=None,
        help="Eval padded token budget. Defaults to --max_batch_tokens when omitted.",
    )
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="LoRA defaults to a higher LR than full fine-tune (2e-5 in sft_asr_12hz.py).")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--use_acoustic_codebooks", action="store_true")
    parser.add_argument("--min_duration", type=float, default=None)
    parser.add_argument("--max_duration", type=float, default=None)
    parser.add_argument("--min_dnsmos", type=float, default=None)
    parser.add_argument("--languages", type=str, default=None, help="Comma-separated language filter, e.g. ja,en")
    parser.add_argument("--save_every_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_every_steps", type=int, default=0)
    parser.add_argument("--eval_every_epochs", type=int, default=1)
    parser.add_argument("--eval_at_start", action="store_true")
    parser.add_argument("--max_eval_batches", type=int, default=-1)
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Comma-separated trackers, e.g. tensorboard,wandb or none")
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-asr")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--use_8bit_optimizer", action="store_true")

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument(
        "--lora_target_regex",
        type=str,
        default=DEFAULT_LORA_TARGET_REGEX,
        help="Regex matched against fully-qualified module names. Default targets Talker body only.",
    )
    parser.add_argument(
        "--lora_modules_to_save",
        type=str,
        default=",".join(DEFAULT_MODULES_TO_SAVE),
        help="Comma-separated module name suffixes kept fully trainable. Set to 'none' to LoRA them too.",
    )
    args = parser.parse_args()

    if args.languages:
        args.languages = [item.strip() for item in args.languages.split(",") if item.strip()]
    else:
        args.languages = None
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.eval_max_batch_tokens is None:
        args.eval_max_batch_tokens = args.max_batch_tokens
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=parse_report_to(args.report_to),
        project_dir=args.output_dir,
    )
    tracker_init_kwargs = {}
    if args.wandb_run_name:
        tracker_init_kwargs["wandb"] = {"name": args.wandb_run_name}
    if parse_report_to(args.report_to) is not None:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs=tracker_init_kwargs,
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

    if args.resume_from_adapter:
        accelerator.print(f"Resuming LoRA adapter from {args.resume_from_adapter}")
        asr_model = PeftModel.from_pretrained(asr_model, args.resume_from_adapter, is_trainable=True)
    else:
        lora_config = build_lora_config(args)
        accelerator.print(
            "LoRA config: "
            f"r={lora_config.r} alpha={lora_config.lora_alpha} dropout={lora_config.lora_dropout} "
            f"bias={lora_config.bias} modules_to_save={lora_config.modules_to_save}"
        )
        asr_model = get_peft_model(asr_model, lora_config)

    if accelerator.is_main_process:
        asr_model.print_trainable_parameters()

    dataset = build_asr_dataset(
        args,
        args.data_lst,
        qwen3tts.processor,
        qwen3tts.model.config,
        special_token_ids,
    )
    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    eval_dataloader = None
    if args.eval_data_lst:
        eval_dataset = build_asr_dataset(
            args,
            args.eval_data_lst,
            qwen3tts.processor,
            qwen3tts.model.config,
            special_token_ids,
        )
        eval_dataloader = build_dataloader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            max_batch_tokens=args.eval_max_batch_tokens,
        )

    trainable_parameters = [parameter for parameter in asr_model.parameters() if parameter.requires_grad]
    if accelerator.is_main_process:
        accelerator.print("Trainable parameters:")
        for name, param in asr_model.named_parameters():
            if param.requires_grad:
                accelerator.print(f"  {name}: {param.numel() / 1e6:.2f}M parameters")
        total_trainable_params = sum(p.numel() for p in trainable_parameters)
        accelerator.print(f"Total trainable parameters: {total_trainable_params / 1e6:.2f}M")

    if args.use_8bit_optimizer:
        import bitsandbytes as bnb
        accelerator.print("Using 8-bit AdamW optimizer from bitsandbytes.")
        optimizer = bnb.optim.AdamW8bit(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    if eval_dataloader is not None:
        asr_model, optimizer, dataloader, eval_dataloader = accelerator.prepare(
            asr_model,
            optimizer,
            dataloader,
            eval_dataloader,
        )
    else:
        asr_model, optimizer, dataloader = accelerator.prepare(asr_model, optimizer, dataloader)
    asr_model.train()

    global_step = 0
    train_audio_tokens = 0
    train_text_tokens = 0
    train_samples = 0

    if eval_dataloader is not None and args.eval_at_start:
        metrics = evaluate(
            asr_model,
            eval_dataloader,
            accelerator,
            max_eval_batches=args.max_eval_batches,
        )
        accelerator.log(metrics, step=global_step)
        accelerator.print(f"eval step={global_step} loss={metrics['eval/loss']:.4f}")
        asr_model.train()

    if accelerator.is_main_process:
        progress_bar = tqdm(desc="Training", unit="step")

    last_grad_norm = 0.0
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
                    grad_norm = accelerator.clip_grad_norm_(asr_model.parameters(), 1.0)
                    last_grad_norm = grad_norm.item() if grad_norm is not None else 0.0
                    optimizer.step()
                    optimizer.zero_grad()

            batch_audio_tokens = accelerator.reduce(batch["audio_lengths"].sum(), reduction="sum").item()
            batch_text_tokens = accelerator.reduce((batch["labels"] != -100).sum(), reduction="sum").item()
            batch_padded_tokens = accelerator.reduce(
                torch.tensor(
                    batch["audio_codes"].shape[0]
                    * (batch["audio_codes"].shape[1] + batch["decoder_input_ids"].shape[1]),
                    device=accelerator.device,
                    dtype=torch.long,
                ),
                reduction="sum",
            ).item()
            batch_samples = accelerator.reduce(
                torch.tensor(batch["audio_codes"].shape[0], device=accelerator.device, dtype=torch.long),
                reduction="sum",
            ).item()
            train_loss = accelerator.reduce(loss.detach(), reduction="mean").item()
            train_audio_tokens += batch_audio_tokens
            train_text_tokens += batch_text_tokens
            train_samples += batch_samples

            if not accelerator.sync_gradients:
                continue

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                train_metrics = {
                    "train/loss": train_loss,
                    "train/audio_tokens": train_audio_tokens,
                    "train/text_tokens": train_text_tokens,
                    "train/samples": train_samples,
                    "train/batch_audio_tokens": batch_audio_tokens,
                    "train/batch_text_tokens": batch_text_tokens,
                    "train/batch_padded_tokens": batch_padded_tokens,
                    "train/batch_samples": batch_samples,
                    "train/epoch": epoch,
                    "train/lr": get_lr(optimizer, args.lr),
                    "train/grad_norm": last_grad_norm,
                }
                accelerator.log(train_metrics, step=global_step)

            if (
                eval_dataloader is not None
                and args.eval_every_steps > 0
                and global_step > 0
                and global_step % args.eval_every_steps == 0
            ):
                metrics = evaluate(
                    asr_model,
                    eval_dataloader,
                    accelerator,
                    max_eval_batches=args.max_eval_batches,
                )
                accelerator.log(metrics, step=global_step)
                if accelerator.is_main_process:
                    progress_bar.write(f"eval step={global_step} loss={metrics['eval/loss']:.4f}")
                asr_model.train()

            if args.save_every_steps > 0 and global_step > 0 and global_step % args.save_every_steps == 0:
                save_checkpoint(accelerator, asr_model, args.output_dir, f"checkpoint-step-{global_step}", args)

            global_step += 1
            if accelerator.is_main_process:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{train_loss:.4f}", 
                                        "lr": f"{get_lr(optimizer, args.lr):.2e}", 
                                        "grad_norm": f"{last_grad_norm:.2f}",
                                        "audid_dur": f"{train_audio_tokens / 12.5 / 3600.0:.1f}h"})
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

        save_checkpoint(accelerator, asr_model, args.output_dir, f"checkpoint-epoch-{epoch}", args)
        if (
            eval_dataloader is not None
            and args.eval_every_epochs > 0
            and (epoch + 1) % args.eval_every_epochs == 0
        ):
            metrics = evaluate(
                asr_model,
                eval_dataloader,
                accelerator,
                max_eval_batches=args.max_eval_batches,
            )
            accelerator.log(metrics, step=global_step)
            if accelerator.is_main_process:
                progress_bar.write(f"eval epoch={epoch} step={global_step} loss={metrics['eval/loss']:.4f}")
            asr_model.train()
        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    accelerator.end_training()


def save_checkpoint(accelerator: Accelerator, model, output_dir: str, name: str, args):
    if not accelerator.is_main_process:
        return

    checkpoint_dir = Path(output_dir) / name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(str(checkpoint_dir))

    config = {
        "init_tts_model_path": args.init_tts_model_path,
        "qwen3_model_path": args.qwen3_model_path,
        "use_acoustic_codebooks": args.use_acoustic_codebooks,
        "eval_data_lst": args.eval_data_lst,
        "max_batch_tokens": args.max_batch_tokens,
        "eval_max_batch_tokens": args.eval_max_batch_tokens,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "bias": args.lora_bias,
            "target_regex": args.lora_target_regex,
            "modules_to_save": parse_modules_to_save(args.lora_modules_to_save),
        },
    }
    with open(checkpoint_dir / "asr_training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    train()
