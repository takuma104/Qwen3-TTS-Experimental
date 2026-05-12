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
"""
LoRA fine-tuning script for Qwen3-TTS models.

Usage:
    # Single GPU
    python finetuning/lora_tts/train_lora.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --data_lst /path/to/train/data.lst \
        --output_dir output/lora \
        --batch_size 2 \
        --lr 2e-5 \
        --num_epochs 3

    # Multi-GPU with accelerate
    accelerate launch finetuning/lora_tts/train_lora.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --data_lst /path/to/train/data.lst \
        --output_dir output/lora \
        --batch_size 2 \
        --lr 2e-5 \
        --num_epochs 3

    # With torch.compile for faster training (10-30% speedup expected)
    python finetuning/lora_tts/train_lora.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --data_lst /path/to/train/data.lst \
        --output_dir output/lora \
        --use_torch_compile \
        --torch_compile_mode reduce-overhead

    # Resume from a checkpoint
    python finetuning/lora_tts/train_lora.py \
        --base_model_path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --data_lst /path/to/train/data.lst \
        --output_dir output/lora \
        --resume_from_checkpoint output/lora/checkpoint-1000

Note on torch.compile:
    - First few iterations will be slower due to compilation overhead
    - Compiles only talker.model and code_predictor.model to avoid dynamic shape issues
    - Modes: "default" (balanced), "reduce-overhead" (faster, recommended),
             "max-autotune" (slowest compile, fastest runtime)
    - Requires PyTorch 2.0+
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Optional

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import AutoConfig

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from finetuning.lora_tts.lora_dataset import (
    TokenBudgetBatchDataset,
    TTSLoRAWebDataset,
    collate_fn_lora,
    read_data_lst,
    total_duration_sec,
    total_sample_count,
)

torch.backends.cuda.matmul.allow_tf32 = True

def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-TTS")

    # Model arguments
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        help="Path to base Qwen3-TTS model",
    )

    # Data arguments
    parser.add_argument(
        "--data_lst",
        type=str,
        required=True,
        help="Path to data.lst produced by extract_audio_tokens_hf.py for training",
    )
    parser.add_argument(
        "--eval_data_lst",
        type=str,
        default=None,
        help="Optional path to validation data.lst",
    )
    parser.add_argument(
        "--max_audio_codes_len",
        type=int,
        default=None,
        help="Maximum audio codes sequence length (frames). Longer sequences will be skipped.",
    )
    parser.add_argument("--min_duration", type=float, default=None, help="Filter samples shorter than this (seconds)")
    parser.add_argument("--max_duration", type=float, default=None, help="Filter samples longer than this (seconds)")
    parser.add_argument("--min_dnsmos", type=float, default=None, help="Filter samples with dnsmos below this")
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated language filter applied to language_id/language, e.g. ja,en",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/lora",
        help="Output directory for checkpoints",
    )

    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=None,
        help="Target modules for LoRA. If not specified, uses default modules.",
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per device")
    parser.add_argument(
        "--max_batch_tokens",
        type=int,
        default=0,
        help=(
            "If >0, dynamically batches by padded token budget "
            "(batch_size * (max_text_len + max_codec_len + 7)), capped at "
            "--batch_size samples per batch. Disables fixed --batch_size when set."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides num_epochs)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")

    # Validation arguments
    parser.add_argument(
        "--val_every_n_steps",
        type=int,
        default=500,
        help="Run validation every N steps",
    )
    parser.add_argument(
        "--val_steps",
        type=int,
        default=50,
        help="Number of validation steps per validation run",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--save_every_n_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep",
    )

    # Resume arguments
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from (e.g., output/lora/checkpoint-1000)",
    )

    # Logging arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3-tts-lora",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="WandB run name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="WandB entity (team or username)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )

    # torch.compile arguments
    parser.add_argument(
        "--use_torch_compile",
        action="store_true",
        help="Enable torch.compile for faster training",
    )
    parser.add_argument(
        "--torch_compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (default: reduce-overhead)",
    )

    return parser.parse_args()


def get_default_target_modules():
    """Get default LoRA target modules for Qwen3-TTS."""
    # return [
    #     "q_proj",
    #     "k_proj",
    #     "v_proj",
    #     "o_proj",
    # ]
    DEFAULT_LORA_TARGET_REGEX = (
        r"^talker\.model\.layers\.\d+\.(self_attn|mlp)\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
    )
    return DEFAULT_LORA_TARGET_REGEX

def setup_lora(model, args):
    """
    Setup LoRA for the Qwen3-TTS model.

    Args:
        model: Qwen3TTSForConditionalGeneration model
        args: Training arguments

    Returns:
        Model with LoRA adapters
    """
    target_modules = args.lora_target_modules or get_default_target_modules()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    # Print trainable parameters
    model.print_trainable_parameters()

    return model


def freeze_non_lora_modules(model):
    """
    Freeze all modules except LoRA parameters.

    This ensures that speaker_encoder and speech_tokenizer are frozen,
    and only LoRA parameters in the talker are trainable.
    """
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False


def forward_step(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Perform a single forward step and compute loss.

    Args:
        model: The model (with LoRA)
        batch: Batch of data
        device: Device to use
        dtype: Data type

    Returns:
        Total loss
    """
    input_ids = batch['input_ids'].to(device)
    codec_ids = batch['codec_ids'].to(device)
    text_embedding_mask = batch['text_embedding_mask'].to(device)
    codec_embedding_mask = batch['codec_embedding_mask'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    codec_0_labels = batch['codec_0_labels'].to(device)
    codec_mask = batch['codec_mask'].to(device)

    # Get the underlying model (unwrap PEFT if needed)
    base_model = model.base_model if hasattr(model, 'base_model') else model
    if hasattr(base_model, 'model'):
        base_model = base_model.model

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    # Get embeddings
    # Note: text_embedding outputs 2048 dim, codec_embedding outputs 1024 dim
    # text_projection converts 2048 -> 1024 so they can be added together
    input_text_embedding = base_model.talker.text_projection(
        base_model.talker.model.text_embedding(input_text_ids)
    ) * text_embedding_mask
    input_codec_embedding = base_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

    input_embeddings = input_text_embedding + input_codec_embedding

    # Add codec embeddings for layers 1-15
    for i in range(1, 16):
        codec_i_embedding = base_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    # Forward pass through talker
    outputs = base_model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )

    # Get hidden states for sub-talker
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    # Sub-talker loss
    _, sub_talker_loss = base_model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )

    # Total loss
    loss = outputs.loss + sub_talker_loss

    return loss


@torch.no_grad()
def validate(
    model,
    val_dataloader,
    accelerator: Accelerator,
    device: torch.device,
    dtype: torch.dtype,
    val_steps: int,
) -> Dict[str, float]:
    """
    Run validation and return metrics.

    Args:
        model: The model
        val_dataloader: Validation dataloader
        accelerator: Accelerator instance
        device: Device
        dtype: Data type
        val_steps: Number of validation steps

    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    num_steps = 0

    val_iter = iter(val_dataloader)

    for _ in range(val_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        loss = forward_step(model, batch, device, dtype)
        total_loss += accelerator.gather(loss).mean().item()
        num_steps += 1

    model.train()

    if num_steps == 0:
        return {"val_loss": 0.0}

    return {"val_loss": total_loss / num_steps}


def save_checkpoint(
    model,
    accelerator: Accelerator,
    output_dir: str,
    step: int,
    epoch: int,
    best_val_loss: float,
    is_best: bool = False,
):
    """
    Save a checkpoint (LoRA adapter + optimizer/scheduler states).

    Args:
        model: The PEFT model
        accelerator: Accelerator instance
        output_dir: Output directory
        step: Current step
        epoch: Current epoch
        best_val_loss: Best validation loss so far
        is_best: Whether this is the best checkpoint
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save accelerator state (optimizer, scheduler, random states)
    accelerator.save_state(checkpoint_dir)

    if accelerator.is_main_process:
        # Unwrap model and save LoRA adapter
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            "step": step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
        }
        with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
            json.dump(state, f)

        accelerator.print(f"Saved checkpoint to {checkpoint_dir}")

        if is_best:
            best_dir = os.path.join(output_dir, "checkpoint-best")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            # Copy the entire checkpoint directory for best
            shutil.copytree(checkpoint_dir, best_dir)
            accelerator.print(f"Saved best checkpoint to {best_dir}")


def train(args):
    """Main training function."""
    # Set seed
    set_seed(args.seed)

    # Initialize accelerator with wandb logging
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb",
    )

    # Initialize wandb
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={
                "wandb": {
                    "name": args.wandb_run_name,
                    "entity": args.wandb_entity,
                }
            },
        )

    accelerator.print(f"Loading model from {args.base_model_path}")

    # Load model
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(args.base_model_path)

    # Setup LoRA (or load from checkpoint)
    if args.resume_from_checkpoint:
        accelerator.print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        # Load LoRA adapter from checkpoint
        model = PeftModel.from_pretrained(
            qwen3tts.model,
            args.resume_from_checkpoint,
            is_trainable=True,
        )
        accelerator.print("Loaded LoRA adapter from checkpoint")
        model.print_trainable_parameters()
    else:
        accelerator.print("Setting up LoRA...")
        model = setup_lora(qwen3tts.model, args)

    freeze_non_lora_modules(model)

    # Apply torch.compile if enabled
    # Note: We compile the talker module only to avoid issues with dynamic shapes
    # in the forward_step function (boolean mask indexing creates variable-sized tensors).
    # torch.compile is applied AFTER LoRA setup but BEFORE accelerator.prepare().
    if args.use_torch_compile:
        accelerator.print(f"Applying torch.compile (mode={args.torch_compile_mode})...")
        # Get the base model's talker for compilation
        base_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(base_model, 'model'):
            base_model = base_model.model
        # Compile the talker module (main transformer) and code_predictor
        base_model.talker.model = torch.compile(
            base_model.talker.model,
            mode=args.torch_compile_mode,
            fullgraph=False,  # Allow graph breaks for compatibility
        )
        base_model.talker.code_predictor.model = torch.compile(
            base_model.talker.code_predictor.model,
            mode=args.torch_compile_mode,
            fullgraph=False,
        )
        accelerator.print("torch.compile applied to talker.model and code_predictor.model")

    # Create dataloaders
    accelerator.print("Creating dataloaders...")
    languages = [s.strip() for s in args.languages.split(",") if s.strip()] if args.languages else None

    train_shards = read_data_lst(args.data_lst)
    num_train_samples = total_sample_count(train_shards)
    train_duration_sec = total_duration_sec(train_shards)
    accelerator.print(
        f"Train shards: {len(train_shards)} "
        f"(manifest samples: {num_train_samples}, duration: {train_duration_sec/3600:.2f}h)"
    )

    train_dataset = TTSLoRAWebDataset(
        args.data_lst,
        qwen3tts.processor,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        min_dnsmos=args.min_dnsmos,
        languages=languages,
        max_audio_codes_len=args.max_audio_codes_len,
    )

    collate_fn = lambda batch: collate_fn_lora(batch, config)

    def _build_dataloader(dataset, *, num_workers: int, persistent: bool):
        if args.max_batch_tokens > 0:
            batched = TokenBudgetBatchDataset(
                dataset,
                max_batch_tokens=args.max_batch_tokens,
                max_batch_samples=args.batch_size,
            )
            return torch.utils.data.DataLoader(
                batched,
                batch_size=None,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0 and persistent,
                prefetch_factor=2 if num_workers > 0 else None,
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=num_workers > 0,
            persistent_workers=num_workers > 0 and persistent,
            prefetch_factor=2 if num_workers > 0 else None,
        )

    train_dataloader = _build_dataloader(
        train_dataset, num_workers=args.num_workers, persistent=True
    )

    val_dataloader = None
    if args.eval_data_lst is not None:
        val_shards = read_data_lst(args.eval_data_lst)
        accelerator.print(
            f"Val shards: {len(val_shards)} (manifest samples: {total_sample_count(val_shards)})"
        )
        val_dataset = TTSLoRAWebDataset(
            args.eval_data_lst,
            qwen3tts.processor,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_dnsmos=args.min_dnsmos,
            languages=languages,
            max_audio_codes_len=args.max_audio_codes_len,
        )
        val_dataloader = _build_dataloader(
            val_dataset, num_workers=min(args.num_workers, 4), persistent=False
        )

    # Setup optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True
    )

    # Approximate dataloader length from the shard manifest, since the
    # IterableDataset has no len(). Filters (duration / dnsmos / language /
    # max_audio_codes_len) may shrink the actual count.
    num_processes = max(accelerator.num_processes, 1)
    if args.max_batch_tokens > 0:
        # Token-budget batching: estimate batches from codec tokens
        # (audio_frames at 12Hz) + a small text-token overhead, divided by
        # the per-batch token budget and the number of devices.
        FRAMES_PER_SECOND = 12
        TEXT_TOKEN_OVERHEAD = 1.1  # ~10% extra to account for text + prefix
        approx_total_tokens = train_duration_sec * FRAMES_PER_SECOND * TEXT_TOKEN_OVERHEAD
        batches_per_epoch_per_device = max(
            int(approx_total_tokens // args.max_batch_tokens // num_processes),
            1,
        )
    else:
        batches_per_epoch_per_device = max(
            num_train_samples // num_processes // max(args.batch_size, 1),
            1,
        )
    num_update_steps_per_epoch = max(
        batches_per_epoch_per_device // args.gradient_accumulation_steps, 1
    )
    if args.max_train_steps is not None:
        total_steps = args.max_train_steps
        args.num_epochs = (total_steps // num_update_steps_per_epoch) + 1
    else:
        total_steps = num_update_steps_per_epoch * args.num_epochs

    # Setup scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)

    # print trainable vs frozen parameters for verification
    if accelerator.is_main_process:
        trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        accelerator.print("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                accelerator.print(f"  {name}: {param.numel() / 1e6:.2f}M parameters")
        total_trainable_params = sum(p.numel() for p in trainable_parameters)
        accelerator.print(f"Total trainable parameters: {total_trainable_params / 1e6:.2f}M")

        accelerator.print("Frozen parameters:")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                accelerator.print(f"  {name}: {param.numel() / 1e6:.2f}M parameters")
        total_frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        accelerator.print(f"Total frozen parameters: {total_frozen_params / 1e6:.2f}M")

    # Prepare with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )

    if val_dataloader is not None:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Resume from checkpoint if specified
    global_step = 0
    starting_epoch = 0
    best_val_loss = float("inf")

    if args.resume_from_checkpoint:
        # Load accelerator state (optimizer, scheduler, random states)
        accelerator.load_state(args.resume_from_checkpoint)
        accelerator.print("Loaded optimizer and scheduler states from checkpoint")

        # Load training state
        training_state_path = os.path.join(args.resume_from_checkpoint, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
            global_step = training_state.get("step", 0)
            starting_epoch = training_state.get("epoch", 0)
            best_val_loss = training_state.get("best_val_loss", float("inf"))
            accelerator.print(f"Resumed from step {global_step}, epoch {starting_epoch}, best_val_loss {best_val_loss:.4f}")

    # Training loop
    accelerator.print("Starting training...")
    accelerator.print(f"  Num epochs: {args.num_epochs}")
    accelerator.print(f"  Total steps: {total_steps}")
    accelerator.print(f"  Batch size per device: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    if args.resume_from_checkpoint:
        accelerator.print(f"  Resuming from step: {global_step}")

    model.train()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    for epoch in range(starting_epoch, args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        # Calculate steps to skip in this epoch (for resuming mid-epoch).
        # The IterableDataset has no len(), so skipping is approximate.
        skip_steps = 0
        if epoch == starting_epoch and global_step > 0:
            steps_in_epoch = batches_per_epoch_per_device
            completed_steps_in_epoch = global_step * args.gradient_accumulation_steps
            if completed_steps_in_epoch >= steps_in_epoch:
                # Skip entire epoch (already completed)
                accelerator.print(f"Skipping epoch {epoch} (already completed)")
                continue
            skip_steps = completed_steps_in_epoch

        if skip_steps > 0:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, num_batches=skip_steps)
            accelerator.print(f"Skipping {skip_steps} batches in epoch {epoch} via skip_first_batches")
        else:
            active_dataloader = train_dataloader

        progress_bar = tqdm(
            active_dataloader,
            desc=f"Epoch {epoch}",
            disable=not accelerator.is_main_process,
            total=batches_per_epoch_per_device,
            initial=skip_steps,
        )

        for step, batch in enumerate(progress_bar):

            with accelerator.accumulate(model):
                loss = forward_step(
                    model,
                    batch,
                    accelerator.device,
                    torch.bfloat16,
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_steps += 1

            if accelerator.sync_gradients:
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # Log metrics
                if global_step % args.log_every_n_steps == 0:
                    accelerator.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )

                # Validation
                if (
                    val_dataloader is not None
                    and global_step % args.val_every_n_steps == 0
                ):
                    val_metrics = validate(
                        model,
                        val_dataloader,
                        accelerator,
                        accelerator.device,
                        torch.bfloat16,
                        args.val_steps,
                    )
                    accelerator.log(
                        {"val/loss": val_metrics["val_loss"]},
                        step=global_step,
                    )
                    accelerator.print(
                        f"Step {global_step} | Val Loss: {val_metrics['val_loss']:.4f}"
                    )

                    # Save best checkpoint
                    if val_metrics["val_loss"] < best_val_loss:
                        best_val_loss = val_metrics["val_loss"]
                        save_checkpoint(
                            model, accelerator, args.output_dir, global_step,
                            epoch, best_val_loss, is_best=True
                        )

                # Save checkpoint
                if global_step % args.save_every_n_steps == 0:
                    save_checkpoint(
                        model, accelerator, args.output_dir, global_step,
                        epoch, best_val_loss
                    )

                # Check if we've reached max steps
                if args.max_train_steps is not None and global_step >= args.max_train_steps:
                    break

        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        accelerator.print(f"Epoch {epoch} | Avg Loss: {avg_epoch_loss:.4f}")

        if args.max_train_steps is not None and global_step >= args.max_train_steps:
            break

    # Save final checkpoint
    save_checkpoint(model, accelerator, args.output_dir, global_step, epoch, best_val_loss)

    accelerator.print("Training complete!")
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    train(args)
