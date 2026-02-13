# [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 48kHz Upsampler Implementation

## Overview

We have extended the Qwen3-TTS-Tokenizer-12Hz decoder to add 48kHz upsampling functionality. Based on XCodec2's 44.1kHz implementation, we adopted an approach that adds `UpSamplerBlock` after the existing 24kHz output.

The 48kHz-related code is independently placed in `qwen_tts/core/tokenizer_48k/`, extending the 12Hz tokenizer classes with subclasses. This avoids conflicts with upstream updates to the 12Hz tokenizer.

## Architecture

```
[Existing 24kHz Decoder] → [UpSamplerBlock (×2)] → [48kHz Output]
```

### UpSamplerBlock Structure

```
UpSamplerBlock
├── CausalTransConvNet (1 → hidden_dim, stride=2)  # 2× upsampling
├── ResidualBlock × 2
│   ├── SnakeBeta + CausalConvNet (dilation=1)
│   └── SnakeBeta + CausalConvNet (dilation=3)
├── SnakeBeta
└── CausalConvNet (hidden_dim → 1)  # Output layer
```

## File Structure

### 48kHz Core Module (New)

| File | Content |
|------|---------|
| `qwen_tts/core/tokenizer_48k/configuration.py` | `Qwen3TTSTokenizer48kConfig`, `Qwen3TTSTokenizer48kDecoderConfig` (subclass of 12Hz) |
| `qwen_tts/core/tokenizer_48k/modeling.py` | `UpSamplerBlock`, `Qwen3TTSTokenizer48kDecoder`, `Qwen3TTSTokenizer48kModel` |

### Training & Inference Scripts

| File | Description |
|------|-------------|
| `finetuning/tokenizer48k/train_upsampler.py` | Training script (WebDataset support) |
| `finetuning/tokenizer48k/upsampler_dataset.py` | Training dataset class |
| `finetuning/tokenizer48k/upsampler_losses.py` | Loss functions (L1 + Multi-resolution STFT + Mel + RMS) |
| `finetuning/tokenizer48k/merge_upsampler.py` | Utility to merge trained upsampler |
| `finetuning/tokenizer48k/inference_upsampler.py` | Inference script |
| `tests/test_48khz_upsampler.py` | Test script |

### Class Inheritance Structure

```
Qwen3TTSTokenizerV2DecoderConfig  →  Qwen3TTSTokenizer48kDecoderConfig
Qwen3TTSTokenizerV2Config         →  Qwen3TTSTokenizer48kConfig
Qwen3TTSTokenizerV2Decoder        →  Qwen3TTSTokenizer48kDecoder (+ UpSamplerBlock)
Qwen3TTSTokenizerV2Model          →  Qwen3TTSTokenizer48kModel
```

## Added Configuration Parameters

### Qwen3TTSTokenizer48kDecoderConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_48khz_upsampler` | `True` | Enable 48kHz upsampler |
| `upsampler_hidden_dim` | `32` | Upsampler hidden dimension |
| `upsampler_kernel_size` | `4` | Transposed convolution kernel size |
| `upsampler_factor` | `2` | Upsampling factor |

### Auto-adjusted Values (when 48kHz enabled)

| Parameter | 24kHz | 48kHz |
|-----------|-------|-------|
| `output_sample_rate` | 24000 | 48000 |
| `decode_upsample_rate` | 1920 | 3840 |

## Usage

### Initialize Model in 48kHz Mode

```python
from qwen_tts.core.tokenizer_48k.configuration import Qwen3TTSTokenizer48kConfig
from qwen_tts.core.tokenizer_48k.modeling import Qwen3TTSTokenizer48kModel

# Create config with 48kHz settings
config = Qwen3TTSTokenizer48kConfig(
    decoder_config={
        "enable_48khz_upsampler": True,
        "upsampler_hidden_dim": 32,
    }
)

# Initialize model
model = Qwen3TTSTokenizer48kModel(config)
```

### Load Merged 48kHz Model

```python
from qwen_tts import Qwen3TTSTokenizer

# Automatically detects models with model_type: "qwen3_tts_tokenizer_48k" in config.json
tokenizer = Qwen3TTSTokenizer.from_pretrained("output/Qwen3-TTS-Tokenizer-12Hz-48kHz")
```

### Freeze Settings During Training

```python
# Freeze existing 24kHz parts and train only the upsampler
for name, param in model.named_parameters():
    if 'upsampler' not in name:
        param.requires_grad = False

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
```

## Testing

```bash
uv run python tests/test_48khz_upsampler.py
```

### Test Results

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

## Parameter Count

| Model | Parameters | Increase |
|-------|-----------|----------|
| 24kHz decoder | 187,118,273 | - |
| 48kHz decoder | 187,135,490 | +0.01% |
| Upsampler only | 17,217 | - |

## Training

To achieve 48kHz quality, additional training of the upsampler is required.

### Recommended Training Settings

| Parameter | Value |
|-----------|-------|
| Batch size | 32 |
| Learning rate | 1e-4 |
| Steps | 100k-200k |
| Optimizer | AdamW |
| Loss function | L1 + Multi-resolution STFT |

### Dataset Requirements

- 48kHz audio data (or 24kHz→48kHz paired data)
- During training, 48kHz audio can be downsampled to 24kHz to create pairs

---

## Upsampler Training

### Data Format (WebDataset)

Prepare data in [WebDataset](https://github.com/webdataset/webdataset) format. Each sample must have the following keys:
- `{filetype}`: Audio data binary (.flac, .wav, .mp3, etc.)
- `npy`: audio_codes numpy array (uint16, shape: [seq_len, 16])

### Running Training

```bash
# Single GPU
python finetuning/tokenizer48k/train_upsampler.py \
    --train_shards "data/train-{000000..000100}.tar" \
    --val_shards "data/val-{000000..000010}.tar" \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --max_train_steps 100000

# Multi-GPU (accelerate)
accelerate launch finetuning/tokenizer48k/train_upsampler.py \
    --train_shards "data/train-*.tar" \
    --val_shards "data/val-*.tar" \
    --output_dir output/upsampler \
    --batch_size 8 \
    --lr 1e-4 \
    --max_train_steps 100000
```

**Note**: When using WebDataset, specifying `--max_train_steps` is recommended (since dataset length is indeterminate)

The released model was trained using the following script:
```
finetuning/tokenizer48k/train_upsampler.sh
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--num_epochs` | 100 | Number of epochs |
| `--gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `--l1_weight` | 1.0 | L1 loss weight |
| `--stft_weight` | 1.0 | STFT loss weight |
| `--mel_weight` | 1.0 | Mel loss weight |
| `--rms_weight` | 1.0 | RMS loss weight |
| `--max_audio_length` | 10.0 | Maximum audio length (seconds) |
| `--upsampler_hidden_dim` | 32 | Upsampler hidden dimension |

### WandB Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--wandb_project` | `qwen3-tts-upsampler` | WandB project name |
| `--wandb_run_name` | (auto-generated) | WandB run name |
| `--wandb_entity` | (none) | WandB entity (organization/username) |

```bash
# Training with WandB settings
python finetuning/tokenizer48k/train_upsampler.py \
    --train_shards "data/train-{000000..000100}.tar" \
    --wandb_project my-upsampler-project \
    --wandb_run_name experiment-1 \
    --wandb_entity my-team
```

### Merging Trained Model

After training, merge the 24kHz model and upsampler to create a 48kHz model:

```bash
python finetuning/tokenizer48k/merge_upsampler.py \
    --base_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --output_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz
```

### Loss Functions

The upsampler training uses the following loss functions:

1. **L1 Loss**: Direct waveform comparison
2. **Multi-resolution STFT Loss**: Spectral comparison at multiple resolutions
   - FFT sizes: [512, 1024, 2048, 4096]
   - Spectral convergence loss + Log magnitude loss
3. **Mel Spectrogram Loss**: Mel spectrogram comparison
4. **RMS Loss**: Multi-resolution RMS energy comparison
   - Frame sizes: [512, 2048, 8192]
   - Promotes amplitude envelope matching

Total loss = L1 × l1_weight + STFT × stft_weight + Mel × mel_weight + RMS × rms_weight

---

## Inference

How to generate 48kHz audio using a trained upsampler.

### Method 1: Direct Inference from Checkpoint

Restore a 48kHz model from trained upsampler.safetensors and config.json for inference.

```bash
# Encode audio file → decode to 48kHz
python finetuning/tokenizer48k/inference_upsampler.py \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --input_audio input.wav \
    --output_audio output_48k.wav

# Decode from audio_codes file (.npy) to 48kHz
python finetuning/tokenizer48k/inference_upsampler.py \
    --upsampler_checkpoint output/upsampler/checkpoint-best \
    --input_codes input_codes.npy \
    --output_audio output_48k.wav
```

### Method 2: Inference with Merged Model

Use a 48kHz model merged with `merge_upsampler.py`:

```bash
python finetuning/tokenizer48k/inference_upsampler.py \
    --model_path output/Qwen3-TTS-Tokenizer-12Hz-48kHz \
    --input_audio input.wav \
    --output_audio output_48k.wav
```

### Inference Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | None | Path to merged 48kHz model |
| `--base_model_path` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Base 24kHz model path |
| `--upsampler_checkpoint` | None | Upsampler checkpoint path |
| `--input_audio` | None | Input audio file |
| `--input_codes` | None | Input audio_codes (.npy format) |
| `--output_audio` | `output_48k.wav` | Output audio file |
| `--device` | `auto` | Device (auto, cpu, cuda) |
| `--dtype` | `bfloat16` | Data type |

### Using from Python Code

```python
from finetuning.tokenizer48k.inference_upsampler import Qwen3TTSTokenizer48kHz

# Restore 48kHz model from checkpoint
tokenizer = Qwen3TTSTokenizer48kHz(
    base_model_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    upsampler_checkpoint="output/upsampler/checkpoint-best",
)

# Encode audio file → decode to 48kHz
wav, sr = tokenizer.encode_decode("input.wav")
print(f"Output sample rate: {sr}")  # 48000

# Decode directly from audio_codes
import numpy as np
audio_codes = np.load("input_codes.npy")  # shape: [seq_len, 16]
wavs, sr = tokenizer.decode_from_codes(audio_codes)

# Save waveform
import soundfile as sf
sf.write("output_48k.wav", wav, sr)
```

---

## References

- [Qwen3-TTS-Tokenizer-12Hz](https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz)
- [Anime-XCodec2-44.1kHz-v2](https://huggingface.co/NandemoGHS/Anime-XCodec2-44.1kHz-v2)
- [XCodec2 Repository](https://github.com/zhenye234/xcodec)

---

## config.json Example

### 48kHz config.json

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
