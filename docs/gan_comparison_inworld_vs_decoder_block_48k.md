# GAN比較: inworld_ai_tts vs decoder_block_48k

`inworld_ai_tts/tts/training/codec/train_codec.py` と `finetuning/decoder_block_48k/train_gan.py` のGANアルゴリズム・パラメータの差異をまとめる。

---

## 1. 識別器アーキテクチャ

| 項目 | inworld_ai_tts | decoder_block_48k |
|------|---------------|-------------------|
| **MPD** | periods=[2,3,5,7,11], max 512ch | periods=[2,3,5,7,11], ch=[16,32,64,128] |
| **MSD** | **STFTベース**のSpecDiscriminator（8スケール） | **波形ベース**のMultiScaleDiscriminator（3スケール） |

最大の差異はMSDの実装方針。inworld_ai_ttsはSTFT変換後のスペクトログラムを識別するSpecDiscriminator（8解像度）を採用。decoder_block_48kは従来のHiFi-GAN流のAvgPoolダウンサンプリングによる波形ベースMSD（3スケール）。

**inworld_ai_tts の48kHz向けSTFTパラメータ** (`inworld_ai_tts/tts/core/codec/discriminator.py`):
```
fft_sizes:  [78, 126, 206, 334, 542, 876, 1418, 2296]
hop_sizes:  [39,  63, 103, 167, 271, 438,  709, 1148]
```

---

## 2. 損失関数の構成

| 損失 | inworld_ai_tts (λ) | decoder_block_48k (λ) |
|------|-------------------|----------------------|
| Adversarial (LSGAN) | 1.0 | 1.0 |
| Feature Matching (L1) | 1.0 | **2.0** |
| Mel スペクトログラム | **15.0**（**7解像度**） | **45.0**（**1解像度**） |
| Multi-Resolution STFT | あり（spectral convergence + log mag） | **なし** |
| **RMS Loss** | **あり（1.0）** | **なし** |

### 主な差異

- **inworld_ai_tts はRMSロスを追加搭載**
  生成音声の音量がターゲットと一致するよう制約（dBスケールのMSE）。GANだけでは音量が発散しやすい問題への対応。

- **Melロスの解像度**
  inworldは7スケール（`n_mels=[5,10,20,40,80,160,320]`）vs decoder_block_48kは1スケール（`n_mels=128`）。

- **STFTロス**
  inworldのみ追加（3解像度：spectral convergence + log magnitude）。

- **重みバランス**
  Melロスの絶対値は異なる（15 vs 45）が、他損失との相対比は近い。

---

## 3. オプティマイザ・学習率

| 項目 | inworld_ai_tts | decoder_block_48k |
|------|---------------|-------------------|
| Generator LR | 1e-4 | 1e-4 |
| Discriminator LR | **1e-4（Gと同じ）** | **2e-4（Gの2倍）** |
| Beta1 / Beta2 | 0.8 / **0.9** | 0.8 / **0.99** |
| Weight Decay | **0.1** | **0.01** |
| Scheduler | CosineAnnealing + Warmup (10%) | CosineAnnealing（eta_min=lr×0.1） |

---

## 4. 学習ループ構造

| 項目 | inworld_ai_tts | decoder_block_48k |
|------|---------------|-------------------|
| G/D 更新比率 | 1:1（毎ステップ両方） | 1:1（毎ステップ両方） |
| Gradient Accumulation | 1ステップ | **16ステップ** |
| 混合精度 | あり（Lightning Fabric経由） | **bf16** |
| VQ Quantizer | **Frozen（最適化対象外）** | - |
| ベースDecoder | 全体を学習対象 | **前段ブロックをFreezeして後半のみ学習** |

---

## 5. decoder_block_48k のみの特徴

- **ウォームスタート**: `--resume_generator_from` で再構成専用学習（train.py）からGAN学習へ移行可能
- **部分Freeze**: VRAMを節約するためベースモデルの前段ブロックを固定、追加したDecoderブロックのみ学習
- **G/D で異なるLR**: Discriminatorを2倍速く学習（`lr_d=2e-4`）でGANトレーニングを安定化

---

## 6. inworld_ai_tts のみの特徴

- **RMSロス**: 音量一致の明示的制約。dBスケールでのMSE損失（`inworld_ai_tts/tts/core/codec/decoder.py`）
- **STFT Loss**: spectral convergence + log magnitudeによる周波数ドメインの追加制約（3解像度）
- **多解像度Mel（7スケール）**: 細粒度〜粗粒度まで幅広くカバー（`n_mels=[5,10,20,40,80,160,320]`）
- **STFTベースMSD**: 波形ではなくスペクトログラムで識別（位相の影響を受けにくい、48kHz向けに8スケール最適化）
- **VQ Quantizer Frozen**: Codebookを安定させたままDecoderのみfinetuning

---

## 参照ファイル

| ファイル | 役割 |
|---------|------|
| `inworld_ai_tts/tts/training/codec/train_codec.py` | メインエントリポイント |
| `inworld_ai_tts/tts/training/codec/gan_training_loop.py` | GANループ（1:1 G/D更新） |
| `inworld_ai_tts/tts/core/codec/decoder.py` | TrainableDecoder（損失計算含む） |
| `inworld_ai_tts/tts/core/codec/discriminator.py` | HiFiGAN MPD & SpecDiscriminator |
| `inworld_ai_tts/tts/core/codec/criterion.py` | GANLoss, MelLoss, STFTLoss |
| `inworld_ai_tts/example/configs/codec_training_config.json` | 学習設定 |
| `finetuning/decoder_block_48k/train_gan.py` | GAN学習メインスクリプト |
| `finetuning/decoder_block_48k/gan_losses.py` | LSGAN + Feature Matching損失 |
| `finetuning/decoder_block_48k/discriminators.py` | 軽量MPD + 波形MSD |
| `finetuning/decoder_block_48k/train_gan.sh` | 学習起動スクリプト |
