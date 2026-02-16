# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
DecoderBlock Addition Method - 48kHz Training Script

Adds a DecoderBlock with upsample_rate=2 to the existing decoder's upsample_rates,
enabling 48kHz output without modifying upstream code.

Usage:
    # Single GPU
    python finetuning/decoder_block_48k/train.py \
        --train_shards "data/train-{000000..000010}.tar" \
        --val_shards "data/val-{000000..000002}.tar" \
        --output_dir output/decoder_block_48k

    # Multi-GPU (accelerate)
    accelerate launch finetuning/decoder_block_48k/train.py \
        --train_shards "data/train-*.tar" \
        --val_shards "data/val-*.tar" \
        --output_dir output/decoder_block_48k
"""

import argparse
import gc
import glob
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.tokenizer48k.upsampler_dataset import create_webdataset_loader
from finetuning.tokenizer48k.upsampler_losses import UpsamplerLoss
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 48kHz decoder (DecoderBlock addition method)"
    )

    # Data
    parser.add_argument(
        "--train_shards",
        type=str,
        required=True,
        help="WebDataset shard pattern for training data",
    )
    parser.add_argument(
        "--val_shards",
        type=str,
        default=None,
        help="WebDataset shard pattern for validation data",
    )

    # Model
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz decoder model path",
    )
    parser.add_argument(
        "--extra_upsample_rate",
        type=int,
        default=2,
        help="Additional upsample rate to append (default: 2 for 48kHz)",
    )

    # Training settings
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Warmup steps"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Loss function weights
    parser.add_argument(
        "--l1_weight", type=float, default=1.0, help="L1 loss weight"
    )
    parser.add_argument(
        "--stft_weight", type=float, default=1.0, help="STFT loss weight"
    )
    parser.add_argument(
        "--mel_weight", type=float, default=1.0, help="Mel loss weight"
    )
    parser.add_argument(
        "--rms_weight", type=float, default=1.0, help="RMS loss weight"
    )

    # Data settings
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=10.0,
        help="Maximum audio length (seconds)",
    )
    parser.add_argument(
        "--min_audio_length",
        type=float,
        default=1.0,
        help="Minimum audio length (seconds)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/decoder_block_48k",
        help="Output directory",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Checkpoint save interval (steps)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=500,
        help="Evaluation interval (steps)",
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log output interval (steps)"
    )

    # Logging settings
    parser.add_argument(
        "--log_with",
        type=str,
        default="wandb",
        help="Logging method (e.g., wandb, tensorboard)",
    )

    # WandB settings
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3-tts-decoder-block-48k",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name (default: auto-generated)",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (organization/username)",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum training steps (recommended for WebDataset)",
    )

    return parser.parse_args()


class DecoderTrainingWrapper(nn.Module):
    """Wraps Qwen3TTSTokenizerV2Decoder for efficient training.

    Runs frozen layers under torch.no_grad() to save VRAM,
    and only computes gradients for the new decoder blocks.
    """

    def __init__(self, decoder: Qwen3TTSTokenizerV2Decoder, num_frozen_decoder_modules: int):
        super().__init__()
        self.decoder = decoder
        self.num_frozen = num_frozen_decoder_modules

    def forward(self, codes):
        # Validate input
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} layers of codes, "
                f"got {codes.shape[1]}"
            )

        # Frozen part: no_grad for VRAM savings
        with torch.no_grad():
            hidden = self.decoder.quantizer.decode(codes)
            hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
            hidden = self.decoder.pre_transformer(
                inputs_embeds=hidden
            ).last_hidden_state
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in self.decoder.decoder[: self.num_frozen]:
                wav = block(wav)
        wav = wav.detach()

        # Trainable part: gradients enabled
        for block in self.decoder.decoder[self.num_frozen :]:
            wav = block(wav)

        return wav.clamp(min=-1, max=1)


def create_model(args, accelerator):
    """Create 48kHz decoder model with frozen base weights."""
    accelerator.print(f"Loading base decoder from {args.decoder_model_path}...")

    # Load base 24kHz model to CPU
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.decoder_model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_decoder = tokenizer.model.decoder
    base_state_dict = base_decoder.state_dict()
    base_num_decoder_modules = len(base_decoder.decoder)

    accelerator.print(
        f"Base decoder: upsample_rates={list(base_decoder.config.upsample_rates)}, "
        f"decoder modules={base_num_decoder_modules}"
    )

    # Create 48kHz config by extending upsample_rates
    config_dict = base_decoder.config.to_dict()
    base_upsample_rates = list(config_dict["upsample_rates"])
    new_upsample_rates = base_upsample_rates + [args.extra_upsample_rate]
    config_dict["upsample_rates"] = new_upsample_rates

    # Remove keys that shouldn't be passed to constructor
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
    decoder_config._attn_implementation = "flash_attention_2"

    accelerator.print(f"New upsample_rates: {new_upsample_rates}")

    # Create new decoder with extended upsample_rates
    decoder = Qwen3TTSTokenizerV2Decoder(decoder_config).to(torch.bfloat16)

    # Load base weights (strict=False: new blocks will be missing, old final layers unexpected)
    missing_keys, unexpected_keys = decoder.load_state_dict(
        base_state_dict, strict=False
    )
    accelerator.print(
        f"Weight loading: {len(missing_keys)} missing keys (new blocks), "
        f"{len(unexpected_keys)} unexpected keys (old final layers)"
    )
    if accelerator.is_main_process:
        accelerator.print(f"  Missing: {missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}")
        accelerator.print(f"  Unexpected: {unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}")

    # Free base model
    del tokenizer, base_decoder, base_state_dict
    gc.collect()

    # Freeze base parameters, train only new decoder blocks
    # base_num_decoder_modules = len([pre_conv] + base_DecoderBlocks + [SnakeBeta, OutputConv])
    # We freeze decoder[0:base_num_decoder_modules-2] (pre_conv + all base DecoderBlocks)
    # Note: the old SnakeBeta and OutputConv (last 2) are gone; new ones are at different indices
    num_frozen = base_num_decoder_modules - 2  # freeze pre_conv + base DecoderBlocks

    for param in decoder.parameters():
        param.requires_grad = False

    for i in range(num_frozen, len(decoder.decoder)):
        for param in decoder.decoder[i].parameters():
            param.requires_grad = True

    # Display trainable parameters
    trainable_params = sum(
        p.numel() for p in decoder.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in decoder.parameters())
    accelerator.print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params * 100:.4f}%)"
    )
    accelerator.print(
        f"Frozen decoder modules: decoder[0:{num_frozen}]"
    )
    accelerator.print(
        f"Trainable decoder modules: decoder[{num_frozen}:{len(decoder.decoder)}]"
    )

    if accelerator.is_main_process:
        for name, param in decoder.named_parameters():
            if param.requires_grad:
                accelerator.print(f"  Trainable: {name} {list(param.shape)}")

    # Wrap for efficient training
    wrapper = DecoderTrainingWrapper(decoder, num_frozen)

    return wrapper, num_frozen, base_upsample_rates, new_upsample_rates


def train_step(
    model: nn.Module,
    batch: dict,
    loss_fn: UpsamplerLoss,
    accelerator: Accelerator,
) -> dict:
    """Single training step."""
    audio_codes = batch["audio_codes"].to(accelerator.device)
    target_48k = batch["audio_48k"].to(accelerator.device)
    lengths_48k = batch["audio_48k_lengths"].to(accelerator.device)

    batch_size, seq_len, _ = audio_codes.shape
    total_seq_len = batch_size * seq_len

    # (batch, seq_len, 16) → (batch, 16, seq_len)
    audio_codes = audio_codes.transpose(1, 2)

    # Forward
    pred_48k = model(audio_codes)

    # Loss
    losses = loss_fn(pred_48k, target_48k, lengths_48k)
    losses["seq_len"] = torch.tensor(
        total_seq_len, dtype=torch.float32, device=accelerator.device
    )

    return losses


@torch.no_grad()
def eval_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: UpsamplerLoss,
    accelerator: Accelerator,
    max_batches: int = 50,
) -> dict:
    """Evaluation."""
    model.eval()

    total_losses = {}
    num_batches = 0

    for batch in dataloader:
        if num_batches >= max_batches:
            break

        audio_codes = batch["audio_codes"].to(accelerator.device)
        target_48k = batch["audio_48k"].to(accelerator.device)
        lengths_48k = batch["audio_48k_lengths"].to(accelerator.device)

        audio_codes = audio_codes.transpose(1, 2)
        pred_48k = model(audio_codes)
        losses = loss_fn(pred_48k, target_48k, lengths_48k)

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()

        num_batches += 1

    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    model.train()
    return avg_losses


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    step: int,
    epoch: int,
    args,
    accelerator: Accelerator,
    num_frozen: int,
    base_upsample_rates: list,
    new_upsample_rates: list,
    is_best: bool = False,
):
    """Save checkpoint (trainable decoder block weights only)."""
    if not accelerator.is_main_process:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = "checkpoint-best" if is_best else f"checkpoint-step-{step}"
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Extract trainable weights only
    unwrapped_model = accelerator.unwrap_model(model)
    decoder = unwrapped_model.decoder

    trainable_state_dict = {}
    for i in range(num_frozen, len(decoder.decoder)):
        prefix = f"decoder.{i}."
        for k, v in decoder.state_dict().items():
            if k.startswith(prefix):
                trainable_state_dict[k] = v.cpu()

    save_file(trainable_state_dict, str(checkpoint_dir / "decoder_block.safetensors"))

    # Save config
    config_dict = {
        "base_upsample_rates": base_upsample_rates,
        "new_upsample_rates": new_upsample_rates,
        "extra_upsample_rate": args.extra_upsample_rate,
        "num_frozen_decoder_modules": num_frozen,
        "step": step,
        "epoch": epoch,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save optimizer and scheduler state
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "step": step,
            "epoch": epoch,
        },
        checkpoint_dir / "training_state.pt",
    )

    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_dir=args.output_dir,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    model, num_frozen, base_upsample_rates, new_upsample_rates = create_model(
        args, accelerator
    )

    # Loss function
    target_sample_rate = 24000 * args.extra_upsample_rate
    loss_fn = UpsamplerLoss(
        sample_rate=target_sample_rate,
        l1_weight=args.l1_weight,
        stft_weight=args.stft_weight,
        mel_weight=args.mel_weight,
        rms_weight=args.rms_weight,
    )

    # Training data
    accelerator.print(f"Loading training data: {args.train_shards}...")
    path = args.train_shards
    if "*" in path and "{" not in path:
        expanded_files = sorted(glob.glob(path))
        if not expanded_files:
            print(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print(f"Found {len(expanded_files)} tar files")
        shard_pattern = expanded_files
    else:
        shard_pattern = path

    train_dataloader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        target_sample_rate=target_sample_rate,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=1000,
    )

    # Validation data (optional)
    val_dataloader = None
    if args.val_shards:
        path = args.val_shards
        if "*" in path and "{" not in path:
            expanded_files = sorted(glob.glob(path))
            if not expanded_files:
                print(f"Error: No files found matching pattern: {path}")
                sys.exit(1)
            print(f"Found {len(expanded_files)} tar files")
            shard_pattern = expanded_files
        else:
            shard_pattern = path

        val_dataloader = create_webdataset_loader(
            shard_pattern=shard_pattern,
            target_sample_rate=target_sample_rate,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=0,
        )

    # Optimizer (only trainable parameters)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    if args.max_train_steps:
        total_steps = args.max_train_steps
    else:
        try:
            total_steps = (
                len(train_dataloader)
                * args.num_epochs
                // args.gradient_accumulation_steps
            )
        except TypeError:
            accelerator.print(
                "WARNING: Cannot determine dataset length (WebDataset). "
                "Please specify --max_train_steps for proper learning rate scheduling."
            )
            total_steps = 100000

    scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.1
    )
    accelerator.print(f"Total training steps: {total_steps}")

    # Prepare with Accelerate
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Initialize tracker
    if args.log_with:
        tracker_config = {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "extra_upsample_rate": args.extra_upsample_rate,
            "base_upsample_rates": base_upsample_rates,
            "new_upsample_rates": new_upsample_rates,
            "l1_weight": args.l1_weight,
            "stft_weight": args.stft_weight,
            "mel_weight": args.mel_weight,
            "rms_weight": args.rms_weight,
            "max_audio_length": args.max_audio_length,
            "decoder_model_path": args.decoder_model_path,
        }

        if accelerator.is_main_process:
            if args.log_with == "wandb":
                accelerator.init_trackers(
                    project_name=args.wandb_project,
                    config=tracker_config,
                    init_kwargs={
                        "wandb": {
                            "name": args.wandb_run_name,
                            "entity": args.wandb_entity,
                            "dir": args.output_dir,
                        }
                    },
                )
            else:
                accelerator.init_trackers(
                    project_name=args.wandb_project,
                    config=tracker_config,
                )
        else:
            if args.log_with == "wandb":
                accelerator.init_trackers(project_name=args.wandb_project)
            else:
                accelerator.init_trackers(project_name=args.wandb_project)

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming from {args.resume_from}...")
        resume_dir = Path(args.resume_from)

        # Load trainable decoder block weights
        weights_path = resume_dir / "decoder_block.safetensors"
        if weights_path.exists():
            saved_weights = load_file(str(weights_path))
            unwrapped_model = accelerator.unwrap_model(model)
            missing, unexpected = unwrapped_model.decoder.load_state_dict(
                saved_weights, strict=False
            )
            accelerator.print(
                f"Resumed model weights: {len(saved_weights)} tensors loaded, "
                f"{len(missing)} missing, {len(unexpected)} unexpected"
            )
        else:
            accelerator.print(
                f"WARNING: {weights_path} not found, model weights not restored!"
            )

        # Load optimizer/scheduler/step/epoch
        training_state = torch.load(
            resume_dir / "training_state.pt", map_location="cpu"
        )
        optimizer.load_state_dict(training_state["optimizer"])
        if training_state["scheduler"] and scheduler:
            scheduler.load_state_dict(training_state["scheduler"])
        start_step = training_state["step"]
        start_epoch = training_state["epoch"]

    # Training loop
    global_step = start_step
    best_val_loss = float("inf")
    total_seq_len_accumulated = 0

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\n{'=' * 50}")
        accelerator.print(f"Epoch {epoch + 1}/{args.num_epochs}")
        accelerator.print(f"{'=' * 50}")

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                losses = train_step(model, batch, loss_fn, accelerator)
                loss = losses["total_loss"]

                total_seq_len_accumulated += losses["seq_len"].item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), args.max_grad_norm
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log
            if global_step % args.log_every == 0:
                log_dict = {k: v.item() for k, v in losses.items()}
                log_dict["lr"] = scheduler.get_last_lr()[0]
                log_dict["total_seq_len_accumulated"] = total_seq_len_accumulated
                accelerator.log(log_dict, step=global_step)

                progress_bar.set_postfix(
                    loss=losses["total_loss"].item(),
                    l1=losses["l1_loss"].item(),
                    stft=losses["stft_loss"].item(),
                )

            # Evaluation
            if (
                val_dataloader
                and global_step % args.eval_every == 0
                and global_step > 0
            ):
                val_losses = eval_step(
                    model, val_dataloader, loss_fn, accelerator
                )
                accelerator.print(f"\nStep {global_step} - Validation losses:")
                for k, v in val_losses.items():
                    accelerator.print(f"  {k}: {v:.4f}")
                accelerator.log(
                    {f"val_{k}": v for k, v in val_losses.items()},
                    step=global_step,
                )

                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        global_step,
                        epoch,
                        args,
                        accelerator,
                        num_frozen,
                        base_upsample_rates,
                        new_upsample_rates,
                        is_best=True,
                    )

            # Save checkpoint
            if global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    global_step,
                    epoch,
                    args,
                    accelerator,
                    num_frozen,
                    base_upsample_rates,
                    new_upsample_rates,
                )

            global_step += 1

            # Check max_train_steps
            if args.max_train_steps and global_step >= args.max_train_steps:
                break

        # Save at end of epoch
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            global_step,
            epoch,
            args,
            accelerator,
            num_frozen,
            base_upsample_rates,
            new_upsample_rates,
        )

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    # Final checkpoint
    save_checkpoint(
        model,
        optimizer,
        scheduler,
        global_step,
        args.num_epochs,
        args,
        accelerator,
        num_frozen,
        base_upsample_rates,
        new_upsample_rates,
    )

    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()
