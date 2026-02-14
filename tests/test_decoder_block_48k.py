#!/usr/bin/env python
# coding=utf-8
"""
Test for DecoderBlock addition method (48kHz).

Verifies:
  1. Config creation with extended upsample_rates
  2. Weight loading via load_state_dict(strict=False)
  3. Freeze/trainable parameter split
  4. DecoderTrainingWrapper forward pass
  5. Total upsample factor calculation
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


def test_config_creation():
    """Test creating 48kHz config by extending upsample_rates."""
    print("[TEST] Config creation with extended upsample_rates...")

    # Base config (24kHz)
    base_config = Qwen3TTSTokenizerV2DecoderConfig(
        upsample_rates=[8, 5, 4, 3],
        upsampling_ratios=[2, 2],
        decoder_dim=1536,
        latent_dim=1024,
        codebook_dim=512,
        codebook_size=2048,
        num_quantizers=16,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=16,
        intermediate_size=1024,
    )

    # 48kHz config
    config_dict = base_config.to_dict()
    config_dict["upsample_rates"] = [8, 5, 4, 3, 2]
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)

    assert list(new_config.upsample_rates) == [8, 5, 4, 3, 2]
    assert list(new_config.upsampling_ratios) == [2, 2]

    print("  [PASS] Config created successfully")
    return base_config, new_config


def test_model_creation(base_config, new_config):
    """Test creating models with both configs."""
    print("[TEST] Model creation...")

    base_decoder = Qwen3TTSTokenizerV2Decoder(base_config)
    new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)

    # Check decoder module counts
    base_modules = len(base_decoder.decoder)
    new_modules = len(new_decoder.decoder)

    # base: 1 (pre_conv) + 4 (DecoderBlocks) + 1 (SnakeBeta) + 1 (OutputConv) = 7
    assert base_modules == 7, f"Expected 7 base modules, got {base_modules}"
    # new: 1 (pre_conv) + 5 (DecoderBlocks) + 1 (SnakeBeta) + 1 (OutputConv) = 8
    assert new_modules == 8, f"Expected 8 new modules, got {new_modules}"

    print(f"  Base decoder modules: {base_modules}")
    print(f"  New decoder modules: {new_modules}")

    # Check total_upsample
    base_total = base_decoder.total_upsample
    new_total = new_decoder.total_upsample
    expected_base = int(np.prod([8, 5, 4, 3, 2, 2]))  # 1920
    expected_new = int(np.prod([8, 5, 4, 3, 2, 2, 2]))  # 3840

    assert base_total == expected_base, f"Base total_upsample: {base_total} != {expected_base}"
    assert new_total == expected_new, f"New total_upsample: {new_total} != {expected_new}"

    print(f"  Base total_upsample: {base_total} (24kHz)")
    print(f"  New total_upsample: {new_total} (48kHz)")
    print("  [PASS] Models created successfully")

    return base_decoder, new_decoder


def test_weight_loading(base_decoder, new_decoder):
    """Test weight loading via load_state_dict(strict=False)."""
    print("[TEST] Weight loading (strict=False)...")

    base_state_dict = base_decoder.state_dict()
    missing_keys, unexpected_keys = new_decoder.load_state_dict(
        base_state_dict, strict=False
    )

    # Verify shared weights loaded correctly
    assert len(missing_keys) > 0, "Expected some missing keys for new blocks"
    assert len(unexpected_keys) > 0, "Expected unexpected keys for old final layers"

    print(f"  Missing keys: {len(missing_keys)} (new blocks)")
    print(f"  Unexpected keys: {len(unexpected_keys)} (old final layers)")

    # Check that decoder[0:5] weights match
    for key in base_state_dict:
        if key.startswith("decoder.0.") or any(
            key.startswith(f"decoder.{i}.") for i in range(1, 5)
        ):
            if key in new_decoder.state_dict():
                assert torch.equal(
                    base_state_dict[key], new_decoder.state_dict()[key]
                ), f"Weight mismatch for {key}"

    # Verify unexpected keys are the old final layers (decoder.5.*, decoder.6.*)
    for key in unexpected_keys:
        assert key.startswith("decoder.5.") or key.startswith(
            "decoder.6."
        ), f"Unexpected key not from old final layers: {key}"

    # Verify missing keys are new decoder blocks (decoder.5.*, decoder.6.*, decoder.7.*)
    for key in missing_keys:
        assert key.startswith("decoder.5.") or key.startswith(
            "decoder.6."
        ) or key.startswith("decoder.7."), f"Missing key not from new blocks: {key}"

    print("  [PASS] Weight loading works correctly")


def test_freeze_strategy(new_decoder):
    """Test freeze/trainable parameter split."""
    print("[TEST] Freeze strategy...")

    num_frozen = 5  # pre_conv + 4 DecoderBlocks

    # Freeze all
    for param in new_decoder.parameters():
        param.requires_grad = False

    # Unfreeze new blocks
    for i in range(num_frozen, len(new_decoder.decoder)):
        for param in new_decoder.decoder[i].parameters():
            param.requires_grad = True

    trainable_params = sum(
        p.numel() for p in new_decoder.parameters() if p.requires_grad
    )
    frozen_params = sum(
        p.numel() for p in new_decoder.parameters() if not p.requires_grad
    )
    total_params = trainable_params + frozen_params

    print(f"  Total parameters: {total_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params / total_params * 100:.4f}%)")

    assert trainable_params > 0, "No trainable parameters"
    assert frozen_params > 0, "No frozen parameters"
    assert trainable_params < total_params * 0.01, (
        f"Trainable params ({trainable_params}) should be < 1% of total ({total_params})"
    )

    # Verify specific modules
    for name, param in new_decoder.named_parameters():
        if name.startswith("decoder.5.") or name.startswith("decoder.6.") or name.startswith("decoder.7."):
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"

    print("  [PASS] Freeze strategy correct")
    return num_frozen


def test_training_wrapper(new_decoder, num_frozen):
    """Test DecoderTrainingWrapper forward pass."""
    print("[TEST] DecoderTrainingWrapper forward pass...")

    # Import the wrapper
    from finetuning.decoder_block_48k.train import DecoderTrainingWrapper

    wrapper = DecoderTrainingWrapper(new_decoder, num_frozen)

    # Create dummy input: codes [batch=1, 16 quantizers, T=10]
    codes = torch.randint(0, 2048, (1, 16, 10))

    # Forward pass
    with torch.no_grad():
        wav = wrapper(codes)

    expected_samples = 10 * 3840  # T * total_upsample
    assert wav.shape[0] == 1, f"Batch dim: {wav.shape[0]} != 1"
    assert wav.shape[1] == 1, f"Channel dim: {wav.shape[1]} != 1"
    # Allow some tolerance due to conv padding
    assert abs(wav.shape[2] - expected_samples) < 100, (
        f"Sample dim: {wav.shape[2]}, expected ~{expected_samples}"
    )
    assert wav.min() >= -1.0, f"Min value {wav.min()} < -1.0"
    assert wav.max() <= 1.0, f"Max value {wav.max()} > 1.0"

    print(f"  Input: codes {list(codes.shape)}")
    print(f"  Output: wav {list(wav.shape)}")
    print(f"  Value range: [{wav.min():.4f}, {wav.max():.4f}]")
    print("  [PASS] Forward pass works correctly")


def main():
    print("=" * 60)
    print("DecoderBlock Addition Method (48kHz) Tests")
    print("=" * 60)

    base_config, new_config = test_config_creation()
    base_decoder, new_decoder = test_model_creation(base_config, new_config)
    test_weight_loading(base_decoder, new_decoder)

    # Re-create for clean state
    new_decoder_clean = Qwen3TTSTokenizerV2Decoder(new_config)
    new_decoder_clean.load_state_dict(base_decoder.state_dict(), strict=False)

    num_frozen = test_freeze_strategy(new_decoder_clean)
    test_training_wrapper(new_decoder_clean, num_frozen)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
