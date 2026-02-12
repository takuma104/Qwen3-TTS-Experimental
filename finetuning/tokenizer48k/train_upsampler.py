# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
48kHz Upsampler 学習スクリプト

Usage:
    # JSONL形式（単一GPU）
    python finetuning/tokenizer48k/train_upsampler.py \
        --train_jsonl data/train.jsonl \
        --val_jsonl data/val.jsonl \
        --output_dir output/upsampler

    # WebDataset形式（単一GPU）
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

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from finetuning.tokenizer48k.upsampler_dataset import (
    UpsamplerDataset,
    collate_fn,
    load_data_from_jsonl,
    create_webdataset_loader,
)
from finetuning.tokenizer48k.upsampler_losses import UpsamplerLoss
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
    UpSamplerBlock,
)

from qwen_tts import Qwen3TTSTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Train 48kHz Upsampler")

    # データ
    parser.add_argument("--train_jsonl", type=str, default=None, help="訓練データのJSONLファイル")
    parser.add_argument("--val_jsonl", type=str, default=None, help="検証データのJSONLファイル")
    parser.add_argument("--train_shards", type=str, default=None, help="訓練データのWebDatasetシャードパターン")
    parser.add_argument("--val_shards", type=str, default=None, help="検証データのWebDatasetシャードパターン")
    parser.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "jsonl", "webdataset"],
                        help="データセットタイプ（auto: 自動判定）")

    # モデル
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="ベースとなる24kHzデコーダーモデルのパス",
    )
    parser.add_argument("--upsampler_hidden_dim", type=int, default=32, help="アップサンプラーの隠れ層次元")
    parser.add_argument("--upsampler_kernel_size", type=int, default=4, help="アップサンプラーのカーネルサイズ")

    # 学習設定
    parser.add_argument("--batch_size", type=int, default=8, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=1e-4, help="学習率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="エポック数")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="ウォームアップステップ数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="勾配累積ステップ数")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="勾配クリッピングの最大ノルム")

    # 損失関数の重み
    parser.add_argument("--l1_weight", type=float, default=1.0, help="L1損失の重み")
    parser.add_argument("--stft_weight", type=float, default=1.0, help="STFT損失の重み")
    parser.add_argument("--mel_weight", type=float, default=1.0, help="メル損失の重み")
    parser.add_argument("--rms_weight", type=float, default=1.0, help="RMS損失の重み")

    # データ設定
    parser.add_argument("--max_audio_length", type=float, default=10.0, help="最大オーディオ長（秒）")
    parser.add_argument("--min_audio_length", type=float, default=1.0, help="最小オーディオ長（秒）")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoaderのワーカー数")

    # 出力
    parser.add_argument("--output_dir", type=str, default="output/upsampler", help="出力ディレクトリ")
    parser.add_argument("--save_every", type=int, default=1000, help="チェックポイント保存間隔（ステップ）")
    parser.add_argument("--eval_every", type=int, default=500, help="評価間隔（ステップ）")
    parser.add_argument("--log_every", type=int, default=10, help="ログ出力間隔（ステップ）")

    # ログ設定
    parser.add_argument("--log_with", type=str, default="wandb", help="ログ出力方法（例: wandb）")

    # WandB設定
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-upsampler", help="WandBプロジェクト名")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run名（デフォルト: 自動生成）")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity（組織/ユーザー名）")

    # その他
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--resume_from", type=str, default=None, help="チェックポイントから再開")
    parser.add_argument("--max_train_steps", type=int, default=None, help="最大学習ステップ数（WebDataset用）")

    return parser.parse_args()


def create_model(args, accelerator):
    """モデルを作成"""
    accelerator.print(f"Loading base decoder from {args.decoder_model_path}...")

    # 24kHzデコーダーをロード
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.decoder_model_path,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_decoder = tokenizer.model.decoder

    # 48kHzデコーダーを作成
    config_dict = base_decoder.config.to_dict()
    config_dict.update({
        "enable_48khz_upsampler": True,
        "upsampler_hidden_dim": args.upsampler_hidden_dim,
        "upsampler_kernel_size": args.upsampler_kernel_size,
        "upsampler_factor": 2,
    })
    decoder_config = Qwen3TTSTokenizerV2DecoderConfig(
        **config_dict,
    )
    decoder = Qwen3TTSTokenizerV2Decoder(decoder_config)

    # 24kHz部分の重みをコピー
    missing_keys, unexpected_keys = decoder.load_state_dict(
        base_decoder.state_dict(), strict=False
    )
    accelerator.print(f"Missing keys (expected for upsampler): {missing_keys}")
    accelerator.print(f"Unexpected keys: {unexpected_keys}")

    # 24kHz部分を凍結、アップサンプラーのみ学習
    for name, param in decoder.named_parameters():
        if 'upsampler' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
            accelerator.print(f"Trainable: {name}")

    # 学習可能なパラメータ数を表示
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
    """1ステップの学習"""
    audio_codes = batch["audio_codes"]  # (batch, seq_len, 16)
    target_48k = batch["audio_48k"]     # (batch, samples)
    lengths_48k = batch["audio_48k_lengths"]

    # Move tensors to device
    audio_codes = audio_codes.to(accelerator.device)
    target_48k = target_48k.to(accelerator.device)
    lengths_48k = lengths_48k.to(accelerator.device)

    # seq_lenを計算（transpose前）
    batch_size, seq_len, _ = audio_codes.shape
    total_seq_len = batch_size * seq_len

    # codes の形状を (batch, 16, seq_len) に変換
    audio_codes = audio_codes.transpose(1, 2)

    # デコーダーで48kHz波形を生成
    pred_48k = model(audio_codes)  # (batch, 1, samples)

    # 損失計算
    losses = loss_fn(pred_48k, target_48k, lengths_48k)

    # seq_len情報を追加
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
    """評価"""
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

    # 平均を計算
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
    """チェックポイントを保存"""
    if not accelerator.is_main_process:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # アップサンプラーの重みのみ保存
    unwrapped_model = accelerator.unwrap_model(model)
    upsampler_state_dict = {
        k: v.cpu() for k, v in unwrapped_model.state_dict().items()
        if 'upsampler' in k
    }

    # チェックポイント名
    checkpoint_name = f"checkpoint-step-{step}"
    if is_best:
        checkpoint_name = "checkpoint-best"

    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 重みを保存
    save_file(upsampler_state_dict, str(checkpoint_dir / "upsampler.safetensors"))

    # 設定を保存
    config_dict = {
        "upsampler_hidden_dim": args.upsampler_hidden_dim,
        "upsampler_kernel_size": args.upsampler_kernel_size,
        "upsampler_factor": 2,
        "step": step,
        "epoch": epoch,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # オプティマイザとスケジューラの状態を保存
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "step": step,
        "epoch": epoch,
    }, checkpoint_dir / "training_state.pt")

    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


def main():
    args = parse_args()

    # Accelerator の初期化
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_dir=args.output_dir,
    )

    # 乱数シードを設定
    set_seed(args.seed)

    # 出力ディレクトリを作成
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # モデルを作成
    model = create_model(args, accelerator)

    # 損失関数
    loss_fn = UpsamplerLoss(
        sample_rate=48000,
        l1_weight=args.l1_weight,
        stft_weight=args.stft_weight,
        mel_weight=args.mel_weight,
        rms_weight=args.rms_weight,
    )

    # データセットタイプを判定
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        if args.train_shards:
            dataset_type = "webdataset"
        elif args.train_jsonl:
            dataset_type = "jsonl"
        else:
            raise ValueError("Either --train_jsonl or --train_shards must be specified")

    # データセット作成
    if dataset_type == "webdataset":
        accelerator.print(f"Loading training data from WebDataset: {args.train_shards}...")

        # glob パターンの場合は展開
        path = args.train_shards
        if "*" in path and "{" not in path:
            expanded_files = sorted(glob.glob(path))
            if not expanded_files:
                print(f"Error: No files found matching pattern: {path}")
                sys.exit(1)
            print(f"Found {len(expanded_files)} tar files")
            # リストを WebDataset 形式に変換
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

        # 検証データ（オプション）
        val_dataloader = None
        if args.val_shards:
            path = args.val_shards
            if "*" in path and "{" not in path:
                expanded_files = sorted(glob.glob(path))
                if not expanded_files:
                    print(f"Error: No files found matching pattern: {path}")
                    sys.exit(1)
                print(f"Found {len(expanded_files)} tar files")
                # リストを WebDataset 形式に変換
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
                shuffle_buffer=0,  # 検証データはシャッフル不要
            )
            accelerator.print("Validation dataloader created (WebDataset)")

    else:  # jsonl
        accelerator.print(f"Loading training data from {args.train_jsonl}...")
        train_data = load_data_from_jsonl(args.train_jsonl)
        train_dataset = UpsamplerDataset(
            train_data,
            target_sample_rate=48000,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
        )
        accelerator.print(f"Training samples: {len(train_dataset)}")

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        # 検証データ（オプション）
        val_dataloader = None
        if args.val_jsonl:
            accelerator.print(f"Loading validation data from {args.val_jsonl}...")
            val_data = load_data_from_jsonl(args.val_jsonl)
            val_dataset = UpsamplerDataset(
                val_data,
                target_sample_rate=48000,
                max_audio_length=args.max_audio_length,
                min_audio_length=args.min_audio_length,
            )
            accelerator.print(f"Validation samples: {len(val_dataset)}")

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=args.num_workers,
                pin_memory=True,
            )

    # オプティマイザ
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # スケジューラ
    if args.max_train_steps:
        total_steps = args.max_train_steps
    else:
        try:
            total_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        except TypeError:
            # WebDataset の場合、長さが取得できないので警告を出す
            accelerator.print(
                "WARNING: Cannot determine dataset length (WebDataset). "
                "Please specify --max_train_steps for proper learning rate scheduling."
            )
            total_steps = 100000  # デフォルト値

    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.1)
    accelerator.print(f"Total training steps: {total_steps}")

    # Accelerate で準備
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # トラッカーを初期化
    if args.log_with:
        # 共通の設定
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
                # WandB固有の設定
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
                # TensorBoard用の初期化
                accelerator.init_trackers(
                    project_name="qwen3-tts-upsampler",
                    config=tracker_config,
                )
            else:
                # その他のトラッカー
                accelerator.init_trackers(
                    project_name="qwen3-tts-upsampler",
                    config=tracker_config,
                )
        else:
            # 非メインプロセスでは最小限の初期化
            if args.log_with == "wandb":
                accelerator.init_trackers(project_name=args.wandb_project)
            else:
                accelerator.init_trackers(project_name="qwen3-tts-upsampler")

    # チェックポイントから再開
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

    # 学習ループ
    global_step = start_step
    best_val_loss = float("inf")
    total_seq_len_accumulated = 0  # 累積seq_len

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
                # 学習ステップ
                losses = train_step(model, batch, loss_fn, accelerator)
                loss = losses["total_loss"]

                # seq_lenを累積
                total_seq_len_accumulated += losses["seq_len"].item()

                # バックワード
                accelerator.backward(loss)

                # 勾配クリッピング
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ログ出力
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

            # 評価
            if val_dataloader and global_step % args.eval_every == 0 and global_step > 0:
                val_losses = eval_step(model, val_dataloader, loss_fn, accelerator)
                accelerator.print(f"\nStep {global_step} - Validation losses:")
                for k, v in val_losses.items():
                    accelerator.print(f"  {k}: {v:.4f}")
                accelerator.log({f"val_{k}": v for k, v in val_losses.items()}, step=global_step)

                # ベストモデルを保存
                if val_losses["total_loss"] < best_val_loss:
                    best_val_loss = val_losses["total_loss"]
                    save_checkpoint(
                        model, optimizer, scheduler, global_step, epoch,
                        args, accelerator, is_best=True
                    )

            # チェックポイント保存
            if global_step % args.save_every == 0 and global_step > 0:
                save_checkpoint(
                    model, optimizer, scheduler, global_step, epoch,
                    args, accelerator
                )

            global_step += 1

        # エポック終了時にチェックポイント保存
        save_checkpoint(
            model, optimizer, scheduler, global_step, epoch,
            args, accelerator
        )

    # 最終チェックポイントを保存
    save_checkpoint(
        model, optimizer, scheduler, global_step, args.num_epochs,
        args, accelerator
    )

    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()
