# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler Training Script

Usage:
    # WebDataset format (single GPU)
    python finetuning/tokenizer48k/train_upsampler.py \
        --train_shards "data/train-{000000..000010}.tar" \
        --val_shards "data/val-{000000..000002}.tar" \
        --output_dir output/upsampler

    # マルチGPU (accelerate)
    accelerate launch finetuning/tokenizer48k/train_upsampler.py \
        --train_shards "data/train-*.tar" \
        --val_shards "data/val-*.tar" \
        --output_dir output/upsampler
"""

import argparse
import json
import os
import sys
from pathlib import Path
import glob

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.tokenizer48k.upsampler_dataset import create_webdataset_loader
from finetuning.tokenizer48k.upsampler_losses import UpsamplerLoss
from qwen_tts.core.tokenizer_48k.configuration import Qwen3TTSTokenizer48kDecoderConfig
from qwen_tts.core.tokenizer_48k.modeling import Qwen3TTSTokenizer48kDecoder

from qwen_tts import Qwen3TTSTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train 48kHz Upsampler")

    # Data
    parser.add_argument("--train_shards", type=str, required=True, help="WebDataset shard pattern for training data")
    parser.add_argument("--val_shards", type=str, default=None, help="WebDataset shard pattern for validation data")

    # Model
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz decoder model path",
    )
    parser.add_argument("--upsampler_hidden_dim", type=int, default=32, help="Upsampler hidden dimension")
    parser.add_argument("--upsampler_kernel_size", type=int, default=4, help="Upsampler kernel size")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm for clipping")

    # Loss function weights
    parser.add_argument("--l1_weight", type=float, default=1.0, help="L1 loss weight")
    parser.add_argument("--stft_weight", type=float, default=1.0, help="STFT loss weight")
    parser.add_argument("--mel_weight", type=float, default=1.0, help="Mel loss weight")
    parser.add_argument("--rms_weight", type=float, default=1.0, help="RMS loss weight")

    # Data settings
    parser.add_argument("--max_audio_length", type=float, default=10.0, help="Maximum audio length (seconds)")
    parser.add_argument("--min_audio_length", type=float, default=1.0, help="Minimum audio length (seconds)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of DataLoader workers")

    # Output
    parser.add_argument("--output_dir", type=str, default="output/upsampler", help="Output directory")
    parser.add_argument("--save_every", type=int, default=1000, help="Checkpoint save interval (steps)")
    parser.add_argument("--eval_every", type=int, default=500, help="Evaluation interval (steps)")
    parser.add_argument("--log_every", type=int, default=10, help="Log output interval (steps)")

    # Logging settings
    parser.add_argument("--log_with", type=str, default="wandb", help="Logging method (e.g., wandb)")

    # WandB settings
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-upsampler", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: auto-generated)")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity (organization/username)")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Maximum training steps (for WebDataset)")

    return parser.parse_args()


def create_model(args, accelerator):
    """Create model"""
    accelerator.print(f"Loading base decoder from {args.decoder_model_path}...")

    # Load 24kHz decoder
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.decoder_model_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_decoder = tokenizer.model.decoder

    # Create 48kHz decoder
    config_dict = base_decoder.config.to_dict()
    config_dict.update({
        "enable_48khz_upsampler": True,
        "upsampler_hidden_dim": args.upsampler_hidden_dim,
        "upsampler_kernel_size": args.upsampler_kernel_size,
        "upsampler_factor": 2,
    })
    decoder_config = Qwen3TTSTokenizer48kDecoderConfig(
        **config_dict,
    )
    decoder = Qwen3TTSTokenizer48kDecoder(decoder_config)

    # Copy 24kHz part weights
    missing_keys, unexpected_keys = decoder.load_state_dict(
        base_decoder.state_dict(), strict=False
    )
    accelerator.print(f"Missing keys (expected for upsampler): {missing_keys}")
    accelerator.print(f"Unexpected keys: {unexpected_keys}")

    # Freeze 24kHz part, train only upsampler
    for name, param in decoder.named_parameters():
        if 'upsampler' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            accelerator.print(f"Trainable: {name}")

    # Display trainable parameter count
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder.parameters())
    accelerator.print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    return decoder


def train_step(
    model: nn.Module,
    batch: dict,
    loss_fn: UpsamplerLoss,
    accelerator: Accelerator,
) -> dict:
    """Single training step"""
    audio_codes = batch["audio_codes"]  # (batch, seq_len, 16)
    target_48k = batch["audio_48k"]     # (batch, samples)
    lengths_48k = batch["audio_48k_lengths"]

    # Move tensors to device
    audio_codes = audio_codes.to(accelerator.device)
    target_48k = target_48k.to(accelerator.device)
    lengths_48k = lengths_48k.to(accelerator.device)

    # Calculate seq_len (before transpose)
    batch_size, seq_len, _ = audio_codes.shape
    total_seq_len = batch_size * seq_len

    # Convert codes shape to (batch, 16, seq_len)
    audio_codes = audio_codes.transpose(1, 2)

    # Generate 48kHz waveform with decoder
    pred_48k = model(audio_codes)  # (batch, 1, samples)

    # Calculate loss
    losses = loss_fn(pred_48k, target_48k, lengths_48k)

    # Add seq_len information
    losses["seq_len"] = torch.tensor(total_seq_len, dtype=torch.float32, device=accelerator.device)

    return losses


@torch.no_grad()
def eval_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: UpsamplerLoss,
    accelerator: Accelerator,
    max_batches: int = 50,
) -> dict:
    """Evaluation"""
    model.eval()

    total_losses = {}
    num_batches = 0

    for batch in dataloader:
        if num_batches >= max_batches:
            break

        audio_codes = batch["audio_codes"]
        target_48k = batch["audio_48k"]
        lengths_48k = batch["audio_48k_lengths"]

        # Move tensors to device
        audio_codes = audio_codes.to(accelerator.device)
        target_48k = target_48k.to(accelerator.device)
        lengths_48k = lengths_48k.to(accelerator.device)

        audio_codes = audio_codes.transpose(1, 2)

        pred_48k = model(audio_codes)
        losses = loss_fn(pred_48k, target_48k, lengths_48k)

        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0.0
            total_losses[k] += v.item()

        num_batches += 1

    # Calculate average
    avg_losses = {k: v / num_batches for k, v in total_losses.items()}

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
    is_best: bool = False,
):
    """Save checkpoint"""
    if not accelerator.is_main_process:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save only upsampler weights
    unwrapped_model = accelerator.unwrap_model(model)
    upsampler_state_dict = {
        k: v.cpu() for k, v in unwrapped_model.state_dict().items()
        if 'upsampler' in k
    }

    # Checkpoint name
    checkpoint_name = f"checkpoint-step-{step}"
    if is_best:
        checkpoint_name = "checkpoint-best"

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    save_file(upsampler_state_dict, str(checkpoint_dir / "upsampler.safetensors"))

    # Save config
    config_dict = {
        "upsampler_hidden_dim": args.upsampler_hidden_dim,
        "upsampler_kernel_size": args.upsampler_kernel_size,
        "upsampler_factor": 2,
        "step": step,
        "epoch": epoch,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save optimizer and scheduler state
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
    }, checkpoint_dir / "training_state.pt")

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

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create model
    model = create_model(args, accelerator)

    # Loss function
    loss_fn = UpsamplerLoss(
        sample_rate=48000,
        l1_weight=args.l1_weight,
        stft_weight=args.stft_weight,
        mel_weight=args.mel_weight,
        rms_weight=args.rms_weight,
    )

    # Create dataset (WebDataset)
    accelerator.print(f"Loading training data from WebDataset: {args.train_shards}...")

    # Expand glob pattern if applicable
    path = args.train_shards
    if "*" in path and "{" not in path:
        expanded_files = sorted(glob.glob(path))
        if not expanded_files:
            print(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print(f"Found {len(expanded_files)} tar files")
        # Convert list to WebDataset format
        shard_pattern = expanded_files
    else:
        shard_pattern = path

    train_dataloader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        target_sample_rate=48000,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=1000,
    )
    accelerator.print("Training dataloader created (WebDataset)")

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
            # Convert list to WebDataset format
            shard_pattern = expanded_files
        else:
            shard_pattern = path

        accelerator.print(f"Loading validation data from WebDataset: {args.val_shards}...")
        val_dataloader = create_webdataset_loader(
            shard_pattern=shard_pattern,
            target_sample_rate=48000,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=0,  # No shuffle needed for validation data
        )
        accelerator.print("Validation dataloader created (WebDataset)")

    # Optimizer
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
            total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        except TypeError:
            # For WebDataset, length cannot be obtained, so issue a warning
            accelerator.print(
                "WARNING: Cannot determine dataset length (WebDataset). "
                "Please specify --max_train_steps for proper learning rate scheduling."
            )
            total_steps = 100000  # Default value

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    accelerator.print(f"Total training steps: {total_steps}")

    # Prepare with Accelerate
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Initialize tracker
    if args.log_with:
        # Common settings
        tracker_config = {
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "upsampler_hidden_dim": args.upsampler_hidden_dim,
            "upsampler_kernel_size": args.upsampler_kernel_size,
            "l1_weight": args.l1_weight,
            "stft_weight": args.stft_weight,
            "mel_weight": args.mel_weight,
            "rms_weight": args.rms_weight,
            "max_audio_length": args.max_audio_length,
            "decoder_model_path": args.decoder_model_path,
        }

        if accelerator.is_main_process:
            if args.log_with == "wandb":
                # WandB-specific settings
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
            elif args.log_with == "tensorboard":
                # TensorBoard initialization
                accelerator.init_trackers(
                    project_name="qwen3-tts-upsampler",
                    config=tracker_config,
                )
            else:
                # Other trackers
                accelerator.init_trackers(
                    project_name="qwen3-tts-upsampler",
                    config=tracker_config,
                )
        else:
            # Minimal initialization for non-main processes
            if args.log_with == "wandb":
                accelerator.init_trackers(project_name=args.wandb_project)
            else:
                accelerator.init_trackers(project_name="qwen3-tts-upsampler")

    # Resume from checkpoint
    start_step = 0
    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming from {args.resume_from}...")
        training_state = torch.load(Path(args.resume_from) / "training_state.pt")
        optimizer.load_state_dict(training_state["optimizer"])
        if training_state["scheduler"] and scheduler:
            scheduler.load_state_dict(training_state["scheduler"])
        start_step = training_state["step"]
        start_epoch = training_state["epoch"]

    # Training loop
    global_step = start_step
    best_val_loss = float("inf")
    total_seq_len_accumulated = 0  # Accumulated seq_len

    model.train()

    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\n{'='*50}")
        accelerator.print(f"Epoch {epoch + 1}/{args.num_epochs}")
        accelerator.print(f"{'='*50}")

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Training step
                losses = train_step(model, batch, loss_fn, accelerator)
                loss = losses["total_loss"]

                # Accumulate seq_len
                total_seq_len_accumulated += losses["seq_len"].item()

                # Backward
                accelerator.backward(loss)

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log output
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
            if val_dataloader and global_step % args.eval_every == 0 and global_step > 0:
                val_losses = eval_step(model, val_dataloader, loss_fn, accelerator)
                accelerator.print(f"\nStep {global_step} - Validation losses:")
                for k, v in val_losses.items():
                    accelerator.print(f"  {k}: {v:.4f}")
                accelerator.log({f"val_{k}": v for k, v in val_losses.items()}, step=global_step)

                # Save best model
                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, epoch,
                        args, accelerator, is_best=True
                    )

            # Save checkpoint
            if global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    args, accelerator
                )

            global_step += 1

        # Save checkpoint at end of epoch
        save_checkpoint(
            model, optimizer, scheduler, global_step, epoch,
            args, accelerator
        )

    # Save final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, global_step, args.num_epochs,
        args, accelerator
    )

    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()
