"""
48kHz アップサンプラーの動作テスト
"""

import torch
import sys
sys.path.insert(0, '.')

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)
from qwen_tts.core.tokenizer_48k.configuration import (
    Qwen3TTSTokenizer48kConfig,
    Qwen3TTSTokenizer48kDecoderConfig,
)
from qwen_tts.core.tokenizer_48k.modeling import (
    Qwen3TTSTokenizer48kDecoder,
    UpSamplerBlock,
)


def test_upsampler_block():
    """UpSamplerBlock単体のテスト"""
    print("=" * 50)
    print("Testing UpSamplerBlock...")

    upsampler = UpSamplerBlock(
        in_channels=1,
        hidden_dim=32,
        kernel_size=4,
        upsample_factor=2,
    )

    # 入力: [batch=1, channels=1, samples=24000] (1秒の24kHz音声)
    x = torch.randn(1, 1, 24000)

    with torch.no_grad():
        y = upsampler(x)

    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Expected output samples: ~{24000 * 2} (may vary slightly due to causal conv padding)")

    # Allow small tolerance due to causal conv padding
    expected_min = 24000 * 2 - 10
    expected_max = 24000 * 2 + 10
    assert expected_min <= y.shape[2] <= expected_max, f"Expected ~48000, got {y.shape[2]}"
    print("  [PASS] UpSamplerBlock test passed!")
    return True


def test_decoder_config_24khz():
    """24kHz設定（デフォルト）のテスト"""
    print("=" * 50)
    print("Testing 24kHz config (default)...")

    config = Qwen3TTSTokenizerV2Config()

    print(f"  output_sample_rate: {config.output_sample_rate}")
    print(f"  decode_upsample_rate: {config.decode_upsample_rate}")

    assert config.output_sample_rate == 24000
    assert config.decode_upsample_rate == 1920
    print("  [PASS] 24kHz config test passed!")
    return True


def test_decoder_config_48khz():
    """48kHz設定のテスト"""
    print("=" * 50)
    print("Testing 48kHz config...")

    decoder_config = {
        "enable_48khz_upsampler": True,
        "upsampler_hidden_dim": 32,
        "upsampler_kernel_size": 4,
        "upsampler_factor": 2,
    }

    config = Qwen3TTSTokenizer48kConfig(decoder_config=decoder_config)

    print(f"  enable_48khz_upsampler: {config.decoder_config.enable_48khz_upsampler}")
    print(f"  output_sample_rate: {config.output_sample_rate}")
    print(f"  decode_upsample_rate: {config.decode_upsample_rate}")

    assert config.decoder_config.enable_48khz_upsampler == True
    assert config.output_sample_rate == 48000  # 24000 * 2
    assert config.decode_upsample_rate == 3840  # 1920 * 2
    print("  [PASS] 48kHz config test passed!")
    return True


def test_decoder_24khz():
    """24kHzデコーダーのテスト"""
    print("=" * 50)
    print("Testing 24kHz decoder...")

    config = Qwen3TTSTokenizerV2DecoderConfig()
    decoder = Qwen3TTSTokenizerV2Decoder(config)

    print(f"  total_upsample: {decoder.total_upsample}")

    assert decoder.total_upsample == 1920  # 2*2*8*5*4*3 = 1920

    # ダミー入力でforward
    # codes: [batch, num_quantizers, seq_len]
    codes = torch.randint(0, 2048, (1, 16, 10))

    with torch.no_grad():
        wav = decoder(codes)

    print(f"  Input codes shape: {codes.shape}")
    print(f"  Output wav shape: {wav.shape}")
    expected_samples = 10 * 1920
    print(f"  Expected samples: {expected_samples}")

    assert wav.shape[0] == 1
    assert wav.shape[1] == 1
    # 出力サンプル数は入力コード長 × total_upsample
    print("  [PASS] 24kHz decoder test passed!")
    return True


def test_decoder_48khz():
    """48kHzデコーダーのテスト"""
    print("=" * 50)
    print("Testing 48kHz decoder...")

    config = Qwen3TTSTokenizer48kDecoderConfig(
        enable_48khz_upsampler=True,
        upsampler_hidden_dim=32,
        upsampler_kernel_size=4,
        upsampler_factor=2,
    )
    decoder = Qwen3TTSTokenizer48kDecoder(config)

    print(f"  total_upsample: {decoder.total_upsample}")
    print(f"  upsampler: {decoder.upsampler is not None}")

    assert decoder.upsampler is not None
    assert decoder.total_upsample == 3840  # 1920 * 2 = 3840

    # ダミー入力でforward
    codes = torch.randint(0, 2048, (1, 16, 10))

    with torch.no_grad():
        wav = decoder(codes)

    print(f"  Input codes shape: {codes.shape}")
    print(f"  Output wav shape: {wav.shape}")
    expected_samples = 10 * 3840
    print(f"  Expected samples: {expected_samples}")

    assert wav.shape[0] == 1
    assert wav.shape[1] == 1
    print("  [PASS] 48kHz decoder test passed!")
    return True


def test_parameter_count():
    """パラメータ数の比較"""
    print("=" * 50)
    print("Comparing parameter counts...")

    config_24k = Qwen3TTSTokenizerV2DecoderConfig()
    decoder_24k = Qwen3TTSTokenizerV2Decoder(config_24k)

    config_48k = Qwen3TTSTokenizer48kDecoderConfig(
        enable_48khz_upsampler=True,
        upsampler_hidden_dim=32,
    )
    decoder_48k = Qwen3TTSTokenizer48kDecoder(config_48k)

    params_24k = sum(p.numel() for p in decoder_24k.parameters())
    params_48k = sum(p.numel() for p in decoder_48k.parameters())
    params_upsampler = sum(p.numel() for p in decoder_48k.upsampler.parameters())

    print(f"  24kHz decoder params: {params_24k:,}")
    print(f"  48kHz decoder params: {params_48k:,}")
    print(f"  Upsampler params:     {params_upsampler:,}")
    print(f"  Overhead:             {(params_48k - params_24k) / params_24k * 100:.2f}%")

    return True


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("48kHz Upsampler Implementation Tests")
    print("=" * 50 + "\n")

    all_passed = True

    try:
        all_passed &= test_upsampler_block()
    except Exception as e:
        print(f"  [FAIL] UpSamplerBlock test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_decoder_config_24khz()
    except Exception as e:
        print(f"  [FAIL] 24kHz config test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_decoder_config_48khz()
    except Exception as e:
        print(f"  [FAIL] 48kHz config test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_decoder_24khz()
    except Exception as e:
        print(f"  [FAIL] 24kHz decoder test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_decoder_48khz()
    except Exception as e:
        print(f"  [FAIL] 48kHz decoder test failed: {e}")
        all_passed = False

    try:
        all_passed &= test_parameter_count()
    except Exception as e:
        print(f"  [FAIL] Parameter count test failed: {e}")
        all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed!")
    print("=" * 50)
