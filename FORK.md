# Qwen3-TTS 48kHz Upsampler Implementation

## 概要

Qwen3-TTS-Tokenizer-12Hz のデコーダーを拡張し、48kHzアップサンプリング機能を追加しました。XCodec2の44.1kHz実装を参考に、既存の24kHz出力の後段に`UpSamplerBlock`を追加する方式を採用しています。

48kHz関連のコードは `qwen_tts/core/tokenizer_48k/` に独立して配置し、12Hzトークナイザーのクラスをサブクラスで拡張する構成です。これにより upstream の12Hzトークナイザー更新とのコンフリクトを回避しています。

## アーキテクチャ

```
[既存24kHzデコーダー] → [UpSamplerBlock (×2)] → [48kHz出力]
```

### UpSamplerBlock 構造

```
UpSamplerBlock
├── CausalTransConvNet (1 → hidden_dim, stride=2)  # 2倍アップサンプリング
├── ResidualBlock × 2
│   ├── SnakeBeta + CausalConvNet (dilation=1)
│   └── SnakeBeta + CausalConvNet (dilation=3)
├── SnakeBeta
└── CausalConvNet (hidden_dim → 1)  # 出力層
```

## ファイル構成

### 48kHz コアモジュール（新規）

| ファイル | 内容 |
|----------|------|
| `qwen_tts/core/tokenizer_48k/configuration.py` | `Qwen3TTSTokenizer48kConfig`, `Qwen3TTSTokenizer48kDecoderConfig`（12Hz のサブクラス） |
| `qwen_tts/core/tokenizer_48k/modeling.py` | `UpSamplerBlock`, `Qwen3TTSTokenizer48kDecoder`, `Qwen3TTSTokenizer48kModel` |

### 学習・推論スクリプト

| ファイル | 説明 |
|----------|------|
| `finetuning/tokenizer48k/train_upsampler.py` | 学習スクリプト（JSONL/WebDataset対応） |
| `finetuning/tokenizer48k/upsampler_dataset.py` | 学習用データセットクラス |
| `finetuning/tokenizer48k/upsampler_losses.py` | 損失関数（L1 + Multi-resolution STFT + Mel + RMS） |
| `finetuning/tokenizer48k/merge_upsampler.py` | 学習済みアップサンプラーをマージするユーティリティ |
| `finetuning/tokenizer48k/inference_upsampler.py` | 推論スクリプト |
| `tests/test_48khz_upsampler.py` | テストスクリプト |

### クラス継承構造

```
Qwen3TTSTokenizerV2DecoderConfig  →  Qwen3TTSTokenizer48kDecoderConfig
Qwen3TTSTokenizerV2Config         →  Qwen3TTSTokenizer48kConfig
Qwen3TTSTokenizerV2Decoder        →  Qwen3TTSTokenizer48kDecoder (+ UpSamplerBlock)
Qwen3TTSTokenizerV2Model          →  Qwen3TTSTokenizer48kModel
```

## 追加された設定パラメータ

### Qwen3TTSTokenizer48kDecoderConfig

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `enable_48khz_upsampler` | `True` | 48kHzアップサンプラーを有効化 |
| `upsampler_hidden_dim` | `32` | アップサンプラーの隠れ層次元 |
| `upsampler_kernel_size` | `4` | 転置畳み込みのカーネルサイズ |
| `upsampler_factor` | `2` | アップサンプリング倍率 |

### 自動調整される値（48kHz有効時）

| パラメータ | 24kHz | 48kHz |
|-----------|-------|-------|
| `output_sample_rate` | 24000 | 48000 |
| `decode_upsample_rate` | 1920 | 3840 |

## 使用方法

### 48kHzモードでモデルを初期化

```python
from qwen_tts.core.tokenizer_48k.configuration import Qwen3TTSTokenizer48kConfig
from qwen_tts.core.tokenizer_48k.modeling import Qwen3TTSTokenizer48kModel

# 48kHz設定でコンフィグ作成
config = Qwen3TTSTokenizer48kConfig(
    decoder_config={
        "enable_48khz_upsampler": True,
        "upsampler_hidden_dim": 32,
    }
)

# モデル初期化
model = Qwen3TTSTokenizer48kModel(config)
```

### マージ済み48kHzモデルをロード

```python
from qwen_tts import Qwen3TTSTokenizer

# model_type: "qwen3_tts_tokenizer_48k" の config.json を持つモデルを自動検出
tokenizer = Qwen3TTSTokenizer.from_pretrained("output/Qwen3-TTS-Tokenizer-12Hz-48kHz")
```

### 学習時の凍結設定

```python
# 既存の24kHz部分を凍結し、アップサンプラーのみ学習
for name, param in model.named_parameters():
    if 'upsampler' not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

## テスト

```bash
uv run python tests/test_48khz_upsampler.py
```

### テスト結果

```
==================================================
48kHz Upsampler Implementation Tests
==================================================
  [PASS] UpSamplerBlock test
  [PASS] 24kHz config test
  [PASS] 48kHz config test
  [PASS] 24kHz decoder test
  [PASS] 48kHz decoder test
  [PASS] Parameter count test
    - 24kHz decoder: 187,118,273 params
    - 48kHz decoder: 187,135,490 params
    - Upsampler:     17,217 params (0.01% overhead)
==================================================
All tests passed!
```

## パラメータ数

| モデル | パラメータ数 | 増加率 |
|--------|-------------|--------|
| 24kHz decoder | 187,118,273 | - |
| 48kHz decoder | 187,135,490 | +0.01% |
| Upsampler only | 17,217 | - |

## 学習について

48kHz品質を得るには、アップサンプラー部分の追加学習が必要です。

### 推奨学習設定

| パラメータ | 値 |
|-----------|-----|
| バッチサイズ | 32 |
| 学習率 | 1e-4 |
| ステップ数 | 100k-200k |
| オプティマイザ | AdamW |
| 損失関数 | L1 + Multi-resolution STFT |

### データセット要件

- 48kHz音声データ（または24kHz→48kHzペアデータ）
- 学習時は48kHz音声を24kHzにダウンサンプリングしてペア作成可能

---

## アップサンプラー学習

### データ形式

#### JSONL形式

JSONLファイルで、各行は以下の形式:

```json
{
  "audio": "path/to/audio.wav",
  "audio_codes": [[c0, c1, ..., c15], [c0, c1, ..., c15], ...]
}
```

- `audio`: 元の音声ファイルパス（任意のサンプルレート、48kHzにリサンプリングされる）
- `audio_codes`: エンコード済みの音声コード (shape: [seq_len, 16])

#### WebDataset形式

`dataset/parquet_to_webdataset.py` で Parquet から変換した tar アーカイブ形式。
大規模データセットに推奨。

各サンプルは以下のキーを持つ:
- `{filetype}`: 音声データのバイナリ（wav, mp3 など）
- `npy`: audio_codes の numpy 配列（uint16、shape: [seq_len, 16]）
- `json`: メタデータ（row_id, speaker, transcribe など）
- `txt`: transcribe のテキスト

変換方法:
```bash
python dataset/parquet_to_webdataset.py \
    audio_parquet_dir/ \
    codes_parquet_dir/ \
    output_wds_dir/
```

### 学習の実行

#### JSONL形式を使う場合

```bash
# 単一GPU
python finetuning/tokenizer48k/train_upsampler.py \
    --train_jsonl data/train.jsonl \
    --val_jsonl data/val.jsonl \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 100

# マルチGPU (accelerate)
accelerate launch finetuning/tokenizer48k/train_upsampler.py \
    --train_jsonl data/train.jsonl \
    --val_jsonl data/val.jsonl \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --num_epochs 100
```

#### WebDataset形式を使う場合（大規模データ推奨）

```bash
# 単一GPU
python finetuning/tokenizer48k/train_upsampler.py \
    --train_shards "data/train-{000000..000100}.tar" \
    --val_shards "data/val-{000000..000010}.tar" \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --max_train_steps 100000

# マルチGPU (accelerate)
accelerate launch finetuning/tokenizer48k/train_upsampler.py \
    --train_shards "data/train-*.tar" \
    --val_shards "data/val-*.tar" \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --max_train_steps 100000
```

**注意**: WebDataset を使う場合は `--max_train_steps` の指定を推奨（データセット長が不定のため）

### 学習パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--batch_size` | 8 | バッチサイズ |
| `--lr` | 1e-4 | 学習率 |
| `--num_epochs` | 100 | エポック数 |
| `--gradient_accumulation_steps` | 4 | 勾配累積ステップ数 |
| `--l1_weight` | 1.0 | L1損失の重み |
| `--stft_weight` | 1.0 | STFT損失の重み |
| `--mel_weight` | 1.0 | メル損失の重み |
| `--rms_weight` | 1.0 | RMS損失の重み |
| `--max_audio_length` | 10.0 | 最大オーディオ長（秒） |
| `--upsampler_hidden_dim` | 32 | アップサンプラーの隠れ層次元 |

### WandB設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--wandb_project` | `qwen3-tts-upsampler` | WandBプロジェクト名 |
| `--wandb_run_name` | (自動生成) | WandB run名 |
| `--wandb_entity` | (なし) | WandB entity（組織/ユーザー名） |

```bash
# WandB設定を指定して学習
python finetuning/tokenizer48k/train_upsampler.py \
    --train_jsonl data/train.jsonl \
    --wandb_project my-upsampler-project \
    --wandb_run_name experiment-1 \
    --wandb_entity my-team
```

### 学習済みモデルのマージ

学習完了後、24kHzモデルとアップサンプラーをマージして48kHzモデルを作成:

```bash
python finetuning/tokenizer48k/merge_upsampler.py \
    --base_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --output_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz
```

### 損失関数

アップサンプラーの学習には以下の損失関数を使用:

1. **L1 Loss**: 波形の直接比較
2. **Multi-resolution STFT Loss**: 複数の解像度でスペクトル比較
   - FFT sizes: [512, 1024, 2048, 4096]
   - Spectral convergence loss + Log magnitude loss
3. **Mel Spectrogram Loss**: メルスペクトログラムの比較
4. **RMS Loss**: 複数解像度のRMSエネルギー比較
   - Frame sizes: [512, 2048, 8192]
   - 振幅エンベロープの一致を促進

合計損失 = L1 × l1_weight + STFT × stft_weight + Mel × mel_weight + RMS × rms_weight

---

## 推論

学習済みのアップサンプラーを使用して48kHz音声を生成する方法です。

### 方法1: チェックポイントから直接推論

学習済みのupsampler.safetensorsとconfig.jsonから48kHzモデルを復元して推論を行います。

```bash
# 音声ファイルをエンコード→48kHzデコード
python finetuning/tokenizer48k/inference_upsampler.py \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --input_audio input.wav \
    --output_audio output_48k.wav

# audio_codesファイル（.npy）から48kHzデコード
python finetuning/tokenizer48k/inference_upsampler.py \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --input_codes input_codes.npy \
    --output_audio output_48k.wav
```

### 方法2: マージ済みモデルで推論

`merge_upsampler.py`でマージした48kHzモデルを使用:

```bash
python finetuning/tokenizer48k/inference_upsampler.py \
    --model_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz \
    --input_audio input.wav \
    --output_audio output_48k.wav
```

### 推論パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--model_path` | なし | マージ済み48kHzモデルのパス |
| `--base_model_path` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | ベース24kHzモデルのパス |
| `--upsampler_checkpoint` | なし | アップサンプラーチェックポイントのパス |
| `--input_audio` | なし | 入力音声ファイル |
| `--input_codes` | なし | 入力audio_codes（.npy形式） |
| `--output_audio` | `output_48k.wav` | 出力音声ファイル |
| `--device` | `auto` | デバイス（auto, cpu, cuda） |
| `--dtype` | `bfloat16` | データ型 |

### Pythonコードから使用

```python
from finetuning.tokenizer48k.inference_upsampler import Qwen3TTSTokenizer48kHz

# チェックポイントから48kHzモデルを復元
tokenizer = Qwen3TTSTokenizer48kHz(
    base_model_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    upsampler_checkpoint="output/upsampler/checkpoint-best",
)

# 音声ファイルをエンコード→48kHzデコード
wav, sr = tokenizer.encode_decode("input.wav")
print(f"Output sample rate: {sr}")  # 48000

# audio_codesから直接デコード
import numpy as np
audio_codes = np.load("input_codes.npy")  # shape: [seq_len, 16]
wavs, sr = tokenizer.decode_from_codes(audio_codes)

# 波形を保存
import soundfile as sf
sf.write("output_48k.wav", wav, sr)
```

---

## 参考リンク

- [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
- [Anime-XCodec2-44.1kHz-v2](https://huggingface.co/NandemoGHS/Anime-XCodec2-44.1kHz-v2)
- [XCodec2 Repository](https://github.com/zhenye234/xcodec)

---

## config.json 例

### 48kHz用 config.json

```json
{
  "model_type": "qwen3_tts_tokenizer_48k",
  "input_sample_rate": 24000,
  "output_sample_rate": 48000,
  "decode_upsample_rate": 3840,
  "encode_downsample_rate": 1920,
  "decoder_config": {
    "enable_48khz_upsampler": true,
    "upsampler_hidden_dim": 32,
    "upsampler_kernel_size": 4,
    "upsampler_factor": 2,
    "codebook_size": 2048,
    "codebook_dim": 128,
    "latent_dim": 1024,
    "decoder_dim": 1536,
    "num_quantizers": 16,
    "upsample_rates": [8, 5, 4, 3],
    "upsampling_ratios": [2, 2]
  }
}
```
