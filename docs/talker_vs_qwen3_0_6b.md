# Qwen3TTSTalkerModel と Qwen3-0.6B (Text LLM) の差分

本ドキュメントは、Talker バックボーンの元になったと思われる
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) と、
[`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
の `talker_config` を比較し、同一点・相違点を整理したものです。

実装参照:
- `qwen_tts/core/models/modeling_qwen3_tts.py`
- `qwen_tts/core/models/configuration_qwen3_tts.py`

---

## 1. 結論（要旨）

- **Transformer のハイパーパラメータ（層数・次元・ヘッド数・MLP 幅・RMSNorm・GQA・attention_bias 等）は Qwen3-0.6B と完全に一致**しており、Talker のデコーダー本体は Qwen3-0.6B をそのまま流用した構造です。
- 主な改造は以下の 5 点:
  1. **RoPE を 1D → 3D MRoPE** に変更（`mrope_section=[24,20,20]`, `interleaved=True`）
  2. **トークン埋め込みを 2 系統に分離**（text 用 2048 次元 + codec 用 1024 次元）、`tie_word_embeddings` を false 化
  3. **出力頭を text LM head から codec head へ差し替え**（`Linear(1024, 3072)`）
  4. **max_position_embeddings を 40960 → 32768**
  5. **補助モジュールを追加**: `text_projection`, `code_predictor`（MTP 5 層）, `speaker_encoder`, codec 特殊トークン群

---

## 2. 同一の部分（バックボーン諸元）

| 項目 | Qwen3-0.6B | Qwen3TTSTalker | 備考 |
|---|---|---|---|
| hidden_size | 1024 | 1024 | 一致 |
| num_hidden_layers | 28 | 28 | 一致 |
| num_attention_heads | 16 | 16 | 一致 |
| num_key_value_heads | 8 | 8 | GQA 2 groups, 一致 |
| head_dim | 128 | 128 | 一致（Qwen3 系は hidden/heads ではなく明示 128） |
| intermediate_size | 3072 | 3072 | SwiGLU MLP 幅, 一致 |
| hidden_act | silu | silu | 一致 |
| attention_bias | false | false | 一致 |
| attention_dropout | 0 | 0 | 一致 |
| rms_norm_eps | 1e-6 | 1e-6 | 一致 |
| rope_theta | 1,000,000 | 1,000,000 | 一致 |
| use_sliding_window | false | false | 一致（sliding_window = null） |

実装上もデコーダーブロック構造は Qwen3 そのもの:
Pre-Norm → Self-Attn（Q/K 各 head_dim RMSNorm 付き）→ Residual
→ Post-Norm → SwiGLU MLP → Residual。
(`Qwen3TTSTalkerDecoderLayer` @ `modeling_qwen3_tts.py:1348`)

---

## 3. 相違点

### 3.1 RoPE: 1D → 3D MRoPE

| 項目 | Qwen3-0.6B | Qwen3TTSTalker |
|---|---|---|
| rope_scaling | `null` (1D) | `{"mrope_section":[24,20,20], "interleaved":true, "rope_type":"default"}` |
| RoPE 実装 | `apply_rotary_pos_emb` (1D) | `apply_multimodal_rotary_pos_emb` (3D MRoPE) |
| RotaryEmbedding | 標準 | `Qwen3TTSTalkerRotaryEmbedding` が `(3, B, T)` の position_ids を受け取り `inv_freq` を 3 軸に expand |
| position_ids 形状 | `(B, T)` | `(3, B, T)` または `(4, B, T)`（先頭行が text 用 attention mask 位置） |

- `head_dim/2 = 64` を 3 軸に分割: 24 + 20 + 20（Qwen2-VL と同スタイル）
- `interleaved=True` なので各 64 次元をインターリーブして 3 軸に配置
- `get_rope_index()` が attention_mask から `(3, B, T)` の position_ids を生成（`modeling_qwen3_tts.py:1746`）
- config の `position_id_per_seconds: 13` は 12Hz codec の時間軸 pos id ステップ幅（フレーム当たりの position インクリメント）

### 3.2 埋め込み構造

| 項目 | Qwen3-0.6B | Qwen3TTSTalker |
|---|---|---|
| embed_tokens | `Embedding(151936, 1024)`（1 本） | 2 本に分離 |
| codec_embedding | — | `Embedding(3072, 1024)` |
| text_embedding | — | `Embedding(151936, **2048**)` |
| vocab_size (config) | 151936 | 3072（codec 側） |
| text_vocab_size (config) | — | 151936（text 側） |
| text_hidden_size | — | **2048**（Talker 本体の 1024 と異なる） |
| tie_word_embeddings | **true** | **false** |
| text → decoder への整形 | 不要 | `Qwen3TTSTalkerResizeMLP`（2048 → 3072 SiLU → 1024, bias=True）で次元縮小 |

- 151936 というテキスト語彙サイズは Qwen3 トークナイザーと完全一致
- ただし text embedding の次元 2048 は Qwen3-0.6B (1024) ではなく Qwen3-1.7B 系（hidden=2048）の次元であり、テキスト表現をより大きなモデルから持ち込む設計の痕跡
- 入力は通常、text_projection で 1024 化したベクトルと codec_embedding の **和** として渡される（例: `generate_speech` 内）

### 3.3 出力側

| 項目 | Qwen3-0.6B | Qwen3TTSTalker |
|---|---|---|
| lm_head | text vocab 151936（embed と tied） | 廃止 |
| codec_head | — | `Linear(1024, 3072, bias=False)`（group 0 codec 予測） |
| 生成対象 | テキストトークン | 12Hz codec トークン（16 codebook 構成の group 0） |

- 残り 15 codebook は後段の `Qwen3TTSTalkerCodePredictorModel`（5 層の MTP）が逐次生成
- 生成時 `suppress_tokens=[2048..3071]\{codec_eos}` で codec 以外の範囲をマスク

### 3.4 位置長とキャッシュ

| 項目 | Qwen3-0.6B | Qwen3TTSTalker |
|---|---|---|
| max_position_embeddings | 40960 | 32768 |
| max_window_layers | 28 | 未設定（sliding window 自体を使わない） |

### 3.5 付加モジュール（Qwen3-0.6B には存在しない）

| モジュール | 役割 |
|---|---|
| `text_projection` (`Qwen3TTSTalkerResizeMLP`) | text_embedding(2048) → Talker hidden(1024) |
| `code_predictor` (`Qwen3TTSTalkerCodePredictorModelForConditionalGeneration`) | 5 層 Transformer + 15 個の `lm_head(1024→2048)`。残り 15 codebook を MTP 生成 |
| `speaker_encoder` (ECAPA-TDNN, `enc_dim=1024`) | 話者プロンプトから 1024 次元話者ベクトル抽出（24kHz 入力） |
| Speech tokenizer (外部ファイル) | 波形 ↔ codec トークン変換器 |

### 3.6 特殊トークン

Qwen3-0.6B は `bos=151643`, `eos=151645` のみ。Talker は Qwen3 のテキスト側特殊 ID をそのまま使いつつ、codec 側の特殊 ID を大幅に追加:

```
im_start_token_id:   151644   (Qwen3 と共通)
im_end_token_id:     151645   (Qwen3 の eos)
tts_pad_token_id:    151671   (追加)
tts_bos_token_id:    151672   (追加)
tts_eos_token_id:    151673   (追加)
assistant_token_id:   77091

# codec 側 (vocab_size=3072 の末尾領域)
codec_pad_id:         2148
codec_bos_id:         2149
codec_eos_token_id:   2150
codec_think_id:       2154
codec_nothink_id:     2155
codec_think_bos_id:   2156
codec_think_eos_id:   2157
codec_language_id:    2050..2071 (english/chinese/... 10 言語)
```

### 3.7 アーキテクチャクラス

| 項目 | Qwen3-0.6B | Qwen3TTS |
|---|---|---|
| `architectures` | `Qwen3ForCausalLM` | `Qwen3TTSForConditionalGeneration` |
| `model_type` | `qwen3` | `qwen3_tts` |

Qwen3TTS では以下の階層:

```
Qwen3TTSForConditionalGeneration
├── talker: Qwen3TTSTalkerForConditionalGeneration
│   ├── model: Qwen3TTSTalkerModel        ← ここが Qwen3-0.6B バックボーン相当
│   ├── text_projection: Qwen3TTSTalkerResizeMLP
│   ├── codec_head: Linear(1024, 3072)
│   └── code_predictor: Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
└── speaker_encoder: Qwen3TTSSpeakerEncoder (tts_model_type="base" 時のみ)
```

---

## 4. config.json の直接比較

```diff
 "architectures":             Qwen3ForCausalLM   →  Qwen3TTSForConditionalGeneration
 "model_type":                qwen3              →  qwen3_tts
 "tie_word_embeddings":       true               →  false
 "max_position_embeddings":   40960              →  32768
 "vocab_size":                151936             →  3072        (codec 側に定義変更)
-                                                   text_vocab_size: 151936  (新設)
-                                                   text_hidden_size: 2048   (新設)
 "rope_scaling":              null               →  { mrope_section:[24,20,20],
                                                       interleaved:true,
                                                       rope_type:"default" }
 (共通 — 差分なし)
   hidden_size:        1024
   num_hidden_layers:  28
   num_attention_heads:16
   num_key_value_heads:8
   head_dim:           128
   intermediate_size:  3072
   rope_theta:         1_000_000
   rms_norm_eps:       1e-6
   attention_bias:     false
   attention_dropout:  0
   use_sliding_window: false
   hidden_act:         silu
```

---

## 5. まとめ

Qwen3TTSTalkerModel の本体は **Qwen3-0.6B の 28 層デコーダーを丸ごと流用**したものです。
TTS 化のための変更は **(a) 位置情報の 3D 化（MRoPE）**、
**(b) 入力側を text / codec の 2 埋め込みに分離**、
**(c) 出力を codec 語彙 3072 の単一ヘッドに変更**、
**(d) 補助として MTP 用 CodePredictor と Speaker Encoder を付加**、
という 4 本柱であり、Transformer ブロックそのものには手を入れていない、
という設計と読み取れます。
