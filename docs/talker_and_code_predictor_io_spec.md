# Qwen3TTSTalkerModel / Qwen3TTSTalkerCodePredictorModel 入出力仕様

このドキュメントは、`Qwen/Qwen3-TTS-12Hz-0.6B-Base` の
[`config.json`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/raw/main/config.json)
を前提とした、Talker 本体 (`Qwen3TTSTalkerModel`) と
サブ Talker / MTP (`Qwen3TTSTalkerCodePredictorModel`) の
アーキテクチャ寸法および forward の入出力仕様をまとめたものです。

実装は以下を参照:
- `qwen_tts/core/models/modeling_qwen3_tts.py`
- `qwen_tts/core/models/configuration_qwen3_tts.py`

---

## 1. Qwen3TTSTalkerModel (メインの Talker バックボーン)

Transformer デコーダー本体。定義位置: `modeling_qwen3_tts.py:1427`。

### アーキテクチャ (`talker_config` 由来)

| 項目 | 値 |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 28 |
| num_attention_heads / num_key_value_heads | 16 / 8 (GQA, 2 groups) |
| head_dim | 128 |
| intermediate_size | 3072 |
| max_position_embeddings | 32768 |
| rope_theta | 1,000,000 |
| rope_scaling | `mrope_section=[24,20,20]`, `interleaved=True` (3D MRoPE) |
| sliding_window | None (全層 full attention) |
| vocab_size (codec) | 3072 |
| text_vocab_size | 151936 |
| text_hidden_size | 2048 |
| num_code_groups | 16 |
| rms_norm_eps | 1e-6 |

### 内部 Embedding

- `codec_embedding: nn.Embedding(3072, 1024)` — codec トークン (group 0) 用
- `text_embedding: nn.Embedding(151936, 2048)` — テキストトークン用
  - 実運用では外側で `Qwen3TTSTalkerResizeMLP` (`text_projection`) により 2048 → 1024 に
    縮小してから codec 側の埋め込みと加算する

### forward の入力

| 引数 | 形状 / 型 | 備考 |
|---|---|---|
| `input_ids` | `(B, T)` Long | `inputs_embeds` と排他。渡されると `codec_embedding` で埋め込み |
| `inputs_embeds` | `(B, T, 1024)` Float | 通常はこちらを使用 |
| `attention_mask` | `(B, T)` Long (0/1) | left-padding 想定 |
| `position_ids` | 下表参照 | 未指定時は `cache_position` から自動生成 |
| `past_key_values` | `DynamicCache` | 省略時は自動作成 |
| `cache_position` | `(T,)` Long | 省略時は `past_seen_tokens` から自動生成 |
| `use_cache` | bool | デフォルト `config.use_cache=True` |
| `output_attentions` / `output_hidden_states` | bool | - |

`position_ids` の扱い:

| 入力形状 | 解釈 |
|---|---|
| None | `cache_position.view(1,1,-1).expand(3, B, -1)` を生成 |
| `(B, T)` | `(3, B, T)` に expand |
| `(3, B, T)` | MRoPE 用 (temporal / height / width) |
| `(4, B, T)` | 先頭 `[0]` が **text position_ids** (attention mask 用)、`[1:]` が MRoPE 用 |

### 出力 (`BaseModelOutputWithPast`)

- `last_hidden_state`: `(B, T, 1024)`
- `past_key_values`: `DynamicCache` (use_cache 時)
- `hidden_states`: 各層の隠れ状態タプル (オプション)
- `attentions`: 各層の attention weights タプル (オプション)

> 備考: `Qwen3TTSTalkerForConditionalGeneration` がこのモデルをラップし、
> `codec_head: Linear(1024, 3072)` で group 0 の logits を生成し、
> 続けて `code_predictor.generate` を呼んで group 1..15 の codec id を得ます。

---

## 2. Qwen3TTSTalkerCodePredictorModel (サブ Talker / MTP バックボーン)

Talker が各時刻で吐いた hidden と group 0 の codec id を起点に、
残り 15 個の codebook を自己回帰生成する小型 Transformer。
定義位置: `modeling_qwen3_tts.py:1015`。

### アーキテクチャ (`code_predictor_config` 由来)

| 項目 | 値 |
|---|---|
| hidden_size | 1024 |
| num_hidden_layers | 5 |
| num_attention_heads / num_key_value_heads | 16 / 8 (GQA) |
| head_dim | 128 |
| intermediate_size | 3072 |
| max_position_embeddings | 65536 |
| rope_theta | 1,000,000 |
| rope_scaling | None (**通常の 1D RoPE**, MRoPE ではない) |
| layer_types | `["full_attention"] * 5` |
| sliding_window | None |
| vocab_size (1 codebook あたり) | 2048 |
| num_code_groups | 16 |
| rms_norm_eps | 1e-6 |

### 内部 Embedding

- `codec_embedding: ModuleList([nn.Embedding(2048, 1024)] * 15)`
  — 各サブ codebook (group 1..15) 用
- group 0 は Talker 本体の `codec_embedding` を使用

### forward の入力

| 引数 | 形状 / 型 | 備考 |
|---|---|---|
| `input_ids` | — | **必ず `None`** (渡すと `ValueError`) |
| `inputs_embeds` | `(B, T, talker_hidden_size=1024)` | 下記構成 |
| `attention_mask` | `(B, T)` | - |
| `position_ids` | `(B, T)` または None | None なら `cache_position` から生成 |
| `past_key_values` | `DynamicCache` | - |
| `cache_position` | `(T,)` | - |
| `generation_steps` | int | 生成時に何番目の codebook を予測中かを示すインデックス |

`inputs_embeds` の構成 (プレフィル時):
- `t=0`: Talker の最終 hidden state (`past_hidden`)
- `t=1..15`: group 0..14 の埋め込み
  - group 0 は Talker 本体 `codec_embedding`
  - group 1..14 は `code_predictor.codec_embedding[i-1]`

`small_to_mtp_projection`: `talker.hidden_size != code_predictor.hidden_size` のときのみ
`Linear(talker.hidden_size → code_predictor.hidden_size)` が入る。
本 config では両方 1024 なので `Identity`。

### 出力 (`BaseModelOutputWithPast`)

- `last_hidden_state`: `(B, T, 1024)`
- `past_key_values`, `hidden_states`, `attentions`

---

## 3. Qwen3TTSTalkerCodePredictorModelForConditionalGeneration (LM ヘッド付きラッパー)

定義位置: `modeling_qwen3_tts.py:1156`。

- `model`: 上記 `Qwen3TTSTalkerCodePredictorModel`
- `lm_head: ModuleList([Linear(1024, 2048, bias=False)] * 15)` — 各 sub-codebook の logits 用

### forward (生成)

- **プレフィル**: `inputs_embeds.shape[1] > 1` の場合
  - `generation_steps = T - 2` (hidden + group 0 を除いたインデックス起点)
  - `logits = lm_head[generation_steps](hidden)` は最終位置に対応する codebook
- **1 ステップ生成**: `input_ids: (B, 1)` を渡す
  - `inputs_embeds = codec_embedding[generation_steps - 1](input_ids)`
  - モデル通過後 `logits = lm_head[generation_steps](hidden)` → `(B, 1, 2048)`
  - 出力内の `generation_steps` は +1 される

戻り値 `Qwen3TTSTalkerCodePredictorOutputWithPast`:
- `logits`: `(B, T, 2048)`
- `past_key_values`, `hidden_states`, `attentions`
- `generation_steps`: 次ステップ用インデックス
- `loss`: `labels` 指定時

### forward_finetune (学習用)

- `inputs_embeds`: `(B, 16, 1024)` = talker hidden + group 0..14 の埋め込み 計 16 要素
- `labels`: `(B, 15)` (group 1..15 の target codec id)
- 出力 `logits`: `(B, 15, 2048)`

---

## 4. 推論時のフレーム単位 codec 生成フロー

1. **Talker (28 層)** が次フレームの hidden を出力
2. `codec_head(hidden)` → group 0 の logits `(B, 3072)` をサンプリングして group 0 codec id を得る
3. `hidden + group0_embed` を **CodePredictor (5 層)** に渡し、group 1..15 を逐次 `generate`
4. 16 codebook 揃ったら各 codebook の埋め込みを合計 (`.sum(1)`) し、trailing text hidden
   または `tts_pad_embed` を加算して、次フレームの Talker 入力埋め込みとする
5. group 0 が `codec_eos_token_id` に達したら停止

---

## 5. config.json 主要パラメータ早見表

```jsonc
{
  "im_start_token_id": 151644,
  "im_end_token_id":   151645,
  "tts_pad_token_id":  151671,
  "tts_bos_token_id":  151672,
  "tts_eos_token_id":  151673,
  "assistant_token_id": 77091,
  "talker_config": {
    "hidden_size": 1024, "num_hidden_layers": 28,
    "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
    "intermediate_size": 3072, "max_position_embeddings": 32768,
    "rope_theta": 1000000,
    "rope_scaling": {"interleaved": true, "mrope_section": [24, 20, 20], "rope_type": "default"},
    "vocab_size": 3072, "text_vocab_size": 151936, "text_hidden_size": 2048,
    "num_code_groups": 16, "position_id_per_seconds": 13,
    "codec_bos_id": 2149, "codec_eos_token_id": 2150, "codec_pad_id": 2148,
    "codec_think_id": 2154, "codec_nothink_id": 2155,
    "codec_think_bos_id": 2156, "codec_think_eos_id": 2157,
    "code_predictor_config": {
      "hidden_size": 1024, "num_hidden_layers": 5,
      "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
      "intermediate_size": 3072, "max_position_embeddings": 65536,
      "rope_theta": 1000000, "rope_scaling": null,
      "vocab_size": 2048, "num_code_groups": 16,
      "layer_types": ["full_attention", "full_attention", "full_attention",
                      "full_attention", "full_attention"]
    }
  },
  "speaker_encoder_config": {"enc_dim": 1024, "sample_rate": 24000}
}
```
