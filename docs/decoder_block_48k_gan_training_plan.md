# decoder_block_48k GAN-Style Training 追加プラン

## Context

現在のdecoder_block_48kトレーニング（run1, 110K+ steps）はMel Spectrogram L1ロスのみで学習しており、広いスペクトル形状は捉えているが、再構成ロスの限界として波形がover-smoothedになり高周波のディテールが不足する傾向がある。HiFi-GAN/BigVGANで実証済みのGAN学習を追加することで、知覚的にリアルな48kHz波形生成を目指す。

## 新規ファイル構成

```
finetuning/decoder_block_48k/
  discriminators.py     # NEW: MPD + MSD 実装
  gan_losses.py         # NEW: Adversarial / Feature Matching / Combined Loss
  train_gan.py          # NEW: GAN学習スクリプト（train.pyベース）
  train_gan.sh          # NEW: 実行シェルスクリプト
```

既存ファイル（train.py, merge.py, inference.py）は変更しない。

---

## Step 1: discriminators.py — Discriminator実装

### Multi-Period Discriminator (MPD)
- Periods: `[2, 3, 5, 7, 11]`
- 各sub-discriminator: 1D→2D reshape後、Conv2d 4層（channels: 16→32→64→128→1）
- Weight normalization、LeakyReLU(0.1)
- forward()は`(output, feature_maps)`を返す（feature matching loss用）

### Multi-Scale Discriminator (MSD)
- 3スケール: 原音、AvgPool1d(2)、AvgPool1d(4)
- 各sub-discriminator: Conv1d 4層（channels: 16→32→64→128→1）
- Scale 0はSpectral normalization、他はWeight normalization
- forward()は`(output, feature_maps)`を返す

総パラメータ: ~900K（Generator ~95Kの約10倍、HiFi-GANの標準的な比率）

## Step 2: gan_losses.py — Loss関数実装

- **LSGAN Loss**: `generator_loss = Σ mean((1 - D(fake))²)`, `discriminator_loss = Σ mean((1 - D(real))² + D(fake)²)`
- **Feature Matching Loss**: discriminatorの中間層特徴量のL1距離 `Σ |fmap_real.detach() - fmap_fake|`
- 既存の`MelSpectrogramLoss`（upsampler_losses.py）を再利用

## Step 3: train_gan.py — GAN学習スクリプト

train.pyをベースに以下を変更:

### Optimizer
- Generator: AdamW(lr=1e-4, betas=(0.8, 0.99))
- Discriminator: AdamW(lr=2e-4, betas=(0.8, 0.99))

### 学習ループ（各ステップ）
1. Generator forward: `pred = G(codes)`
2. **Discriminator更新**: `D(real)` vs `D(pred.detach())` → LSGAN loss → backward → step
3. **Generator更新**: adversarial loss + feature matching loss + mel loss → backward → step

### Loss重み（HiFi-GAN準拠）
| Loss | Weight |
|------|--------|
| Adversarial (λ_adv) | 1.0 |
| Feature Matching (λ_fm) | 2.0 |
| Mel Reconstruction (λ_mel) | 45.0 |

### Warm-start
- run1の`checkpoint-best`からGenerator重みをロード（`--resume_generator_from`）
- Discriminatorは新規初期化

### Checkpoint保存
- Generator: `decoder_block.safetensors`（既存形式、merge.py互換）
- Discriminator: `discriminator.pt`（学習時のみ使用）
- Training state: `training_state.pt`（optimizer_g, optimizer_d, scheduler, step）

### WandB Logging
- `d/loss_total`, `d/loss_mpd`, `d/loss_msd`
- `g/loss_total`, `g/loss_adv`, `g/loss_fm`, `g/loss_mel`
- `lr/generator`, `lr/discriminator`

## Step 4: train_gan.sh — 実行スクリプト

主要ハイパーパラメータ:
- batch_size: 16, gradient_accumulation_steps: 2（GANでは小さめに）
- lr_g: 1e-4, lr_d: 2e-4
- lambda_adv: 1.0, lambda_fm: 2.0, lambda_mel: 45.0
- max_train_steps: 500000
- mixed_precision: bf16

---

## 重要な参照ファイル

- `finetuning/decoder_block_48k/train.py` — ベース学習スクリプト（DecoderTrainingWrapper, create_model, save_checkpoint）
- `finetuning/tokenizer48k/upsampler_losses.py` — MelSpectrogramLoss再利用
- `qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` — モデルアーキテクチャ参照
- `finetuning/decoder_block_48k/train.sh` — 現行設定参照

## 検証方法

1. `python -c "from discriminators import *; ..."` でDiscriminatorのforward確認（ランダムテンソル）
2. train_gan.pyを短いステップ数で実行し、D/G lossが期待通り変動するか確認
3. WandBでd/loss, g/loss_adv, g/loss_mel の推移を監視
4. 数千ステップ後の生成音声をinference.pyで確認（既存merge.pyと互換）
