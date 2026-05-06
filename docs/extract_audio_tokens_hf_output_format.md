# extract_audio_tokens_hf.py 出力データ形式

`omnivoice/scripts/extract_audio_tokens_hf.py` は HuggingFace Dataset から音声を読み込み、Qwen3-TTS-Tokenizer でオーディオトークンに変換し、WebDataset 形式でシャードに書き出すスクリプトです。

## ディレクトリ構成

```
output_dir/
├── audios/                  # WebDataset tar シャード
│   ├── shard-000000.tar
│   ├── shard-000001.tar
│   └── ...
├── txts/                    # シャードごとの JSONL メタデータ
│   ├── shard-000000.jsonl
│   ├── shard-000001.jsonl
│   └── ...
├── data.lst                 # マニフェストファイル
└── errors.jsonl             # エラーログ（処理失敗サンプルの詳細）
```

## 各ファイルの詳細

### 1. オーディオトークン tar ファイル (`audios/shard-XXXXXX.tar`)

WebDataset 形式の tar アーカイブ。各サンプルは `.npy` ファイルとして格納されます。

| 項目 | 値 |
|---|---|
| ファイル名 | `{sample_id}.npy` |
| データ型 | `int16` |
| 形状 | `(16, T)` — 16コードブック × T タイムステップ |
| 値の範囲 | `0` ~ `2047` |
| フレームレート | 12 Hz（Qwen3-TTS-Tokenizer-12Hz） |

**例:**
```
JA_B00000_S00534_W000123.npy  — shape=(16, 97), 約 97/12 ≈ 8.1 秒分
JA_B00002_S04294_W000113.npy  — shape=(16, 41), 約 41/12 ≈ 3.4 秒分
```

`T`（タイムステップ数）は音声の長さに依存し、`metadata["num_tokens"]` と一致します。

### 2. メタデータ JSONL ファイル (`txts/shard-XXXXXX.jsonl`)

tar ファイルと 1:1 で対応する JSONL ファイル。1行が1サンプルのメタデータ（JSON オブジェクト）です。

**フィールド:**

| フィールド | 型 | 説明 |
|---|---|---|
| `id` | string | サンプル ID（`.npy` ファイル名と一致） |
| `text` | string | 発話テキスト |
| `language_id` | string | 言語コード（例: `"ja"`） |
| `audio_duration` | float | 音声の長さ（秒） |
| `speaker` | string | 話者 ID |
| `dnsmos` | float | DNSMOS 音声品質スコア |
| `num_tokens` | int | オーディオトークンのタイムステップ数（= `.npy` の列数） |
| `duration` | float | 元データの duration フィールド |
| `language` | string | 元データの language フィールド |

**例:**
```json
{"id": "JA_B00000_S00534_W000123", "text": "これまで何か隠していた、秘密にしていた、こそこそしていた関係性に終止符が売ったれて、", "language_id": "ja", "audio_duration": 7.707, "speaker": "JA_B00000_S00534", "dnsmos": 3.2707, "duration": 7.707, "language": "ja", "num_tokens": 97}
```

> **注:** `language_id` / `language` および `audio_duration` / `duration` は元データのフィールドとスクリプト側で生成したフィールドの両方が含まれるため重複しています。元データに追加のフィールドがある場合はそのまま保持されます。

### 3. マニフェストファイル (`data.lst`)

スペース区切りのテキストファイル。各行が1シャードに対応します。

**形式:**
```
<tar_path> <jsonl_path> <sample_count> <total_duration>
```

| カラム | 説明 |
|---|---|
| `tar_path` | tar ファイルの絶対パス |
| `jsonl_path` | 対応する JSONL ファイルの絶対パス |
| `sample_count` | シャード内のサンプル数 |
| `total_duration` | シャード内の合計音声時間（秒、小数点以下3桁） |

**例:**
```
/mnt/artifacts/.../audios/shard-000000.tar /mnt/artifacts/.../txts/shard-000000.jsonl 10000 69690.110
/mnt/artifacts/.../audios/shard-000001.tar /mnt/artifacts/.../txts/shard-000001.jsonl 10000 69406.873
```

### 4. エラーログ (`errors.jsonl`)

処理に失敗したサンプルの記録。1行1 JSON オブジェクトで `id` と `reason` を含みます。
処理が全て成功した場合は空ファイルになります。

**形式:**
```json
{"id": "SAMPLE_ID", "reason": "エラーメッセージ"}
```
