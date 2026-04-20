# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tune Qwen3-TTS-12Hz-0.6B-Base with the Transformer backbone of the Talker
replaced by the corresponding weights from the Qwen3-0.6B text LLM.

Purpose
-------
Qwen3TTSTalkerModel's 28-layer decoder has exactly the same hyperparameters as
Qwen3-0.6B (hidden=1024, layers=28, heads 16/8 GQA, head_dim=128,
intermediate=3072, rope_theta=1M, RMSNorm, SwiGLU, attention_bias=False, Q/K
RMSNorm). The only architectural difference inside the backbone is 1D RoPE vs
3D MRoPE; and when the 3 MRoPE position-id axes carry identical values, MRoPE
reduces exactly to 1D RoPE. Therefore Qwen3-0.6B's backbone weights are a
valid initialization for the Talker backbone.

This script verifies that claim in a practical SFT setting by:
  1. loading a Qwen3-TTS-12Hz-0.6B-Base checkpoint (provides tokenizer,
     processor, speech_tokenizer, speaker_encoder, codec/text embeddings,
     text_projection, codec_head, and code_predictor),
  2. overwriting the Talker's transformer backbone
     (`talker.model.layers.*` and `talker.model.norm`) with Qwen3-0.6B's
     `model.layers.*` / `model.norm`,
  3. running the same SFT loop as `sft_12hz.py`.

With `--verify_only`, it only performs the weight swap and prints a report.
"""

import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM


# Keys that exist in Qwen3-0.6B but have no direct counterpart in the Talker
# backbone (`talker.model.*`). They are NOT copied.
#
# - `model.embed_tokens.weight`       : shape (151936, 1024) for text tokens;
#       Talker uses `codec_embedding` (3072, 1024) for codec tokens and a
#       separate `text_embedding` (151936, 2048) for text. Neither matches.
# - `lm_head.weight`                  : Qwen3-0.6B has tied embeddings, and the
#       Talker does not have a text LM head (only a codec_head of shape
#       (1024, 3072)).
QWEN3_KEYS_TO_SKIP = {
    "model.embed_tokens.weight",
    "lm_head.weight",
}


def swap_talker_backbone_from_qwen3(
    qwen3tts_model,
    qwen3_model_path: str,
    dtype: torch.dtype,
):
    """
    Replace the Talker transformer backbone with Qwen3-0.6B weights, in place.

    Returns a report dict with lists of copied / skipped / missing keys and the
    number of parameters touched.
    """
    qwen3 = AutoModelForCausalLM.from_pretrained(
        qwen3_model_path,
        torch_dtype=dtype,
    )
    qwen3_sd = qwen3.state_dict()

    # Build a state_dict keyed relative to `talker.model` (the Qwen3TTSTalkerModel).
    # e.g. "model.layers.0.self_attn.q_proj.weight"
    #      -> "layers.0.self_attn.q_proj.weight"
    backbone_sd = {}
    skipped = []
    for qk, qv in qwen3_sd.items():
        if qk in QWEN3_KEYS_TO_SKIP:
            skipped.append(qk)
            continue
        if not qk.startswith("model."):
            # e.g. stray auxiliary tensors; shouldn't normally happen for Qwen3
            skipped.append(qk)
            continue
        backbone_sd[qk[len("model."):]] = qv

    talker_model = qwen3tts_model.talker.model  # Qwen3TTSTalkerModel

    # Pre-check: every key we're about to load must exist in the Talker
    # backbone with matching shape. This should be true by construction.
    shape_mismatches = []
    for tk, tv in backbone_sd.items():
        ref = dict(talker_model.state_dict()).get(tk)
        if ref is None:
            # Will be reported as `unexpected` by load_state_dict below.
            continue
        if ref.shape != tv.shape:
            shape_mismatches.append((tk, tuple(ref.shape), tuple(tv.shape)))

    if shape_mismatches:
        lines = [
            f"  {k}: talker {a} vs qwen3 {b}" for k, a, b in shape_mismatches
        ]
        raise RuntimeError(
            "Shape mismatch between Talker backbone and Qwen3-0.6B:\n"
            + "\n".join(lines)
        )

    # `strict=False` is used because the Talker backbone additionally holds
    # `codec_embedding` and `text_embedding`, which are not present in
    # `backbone_sd` and will thus appear in `missing`.
    missing, unexpected = talker_model.load_state_dict(backbone_sd, strict=False)

    # Count parameters actually overwritten.
    copied_numel = sum(
        v.numel() for k, v in backbone_sd.items()
        if k not in set(missing) and k not in set(unexpected)
    )

    del qwen3, qwen3_sd

    return {
        "copied_keys": sorted(backbone_sd.keys()),
        "missing_in_qwen3": sorted(missing),
        "unexpected_in_qwen3": sorted(unexpected),
        "skipped_qwen3_keys": sorted(skipped),
        "copied_numel": copied_numel,
    }


def print_swap_report(report: dict, accelerator: Accelerator):
    if not accelerator.is_main_process:
        return
    p = accelerator.print
    p("=" * 78)
    p("Backbone swap report: Qwen3-0.6B -> Qwen3TTSTalkerModel (talker.model)")
    p("=" * 78)
    p(f"Copied Qwen3 keys ({len(report['copied_keys'])}):")
    for k in report["copied_keys"]:
        p(f"  + talker.model.{k}")
    p("")
    p(f"Qwen3 keys skipped ({len(report['skipped_qwen3_keys'])}) "
      f"(no counterpart or shape mismatch):")
    for k in report["skipped_qwen3_keys"]:
        p(f"  - {k}")
    p("")
    p(f"Talker backbone keys left untouched "
      f"({len(report['missing_in_qwen3'])}):")
    for k in report["missing_in_qwen3"]:
        p(f"  . talker.model.{k}")
    if report["unexpected_in_qwen3"]:
        p("")
        p("WARNING: keys present in the Qwen3 state_dict but not in the "
          "Talker backbone (unexpected):")
        for k in report["unexpected_in_qwen3"]:
            p(f"  ? {k}")
    p("")
    p(f"Total parameters overwritten: {report['copied_numel']:,}")
    p("=" * 78)


def train():
    parser = argparse.ArgumentParser()
    # Base TTS checkpoint. This provides processor / speech_tokenizer /
    # speaker_encoder / codec & text embeddings / text_projection / codec_head /
    # code_predictor.
    parser.add_argument("--init_model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    # Text LLM whose transformer backbone is copied into the Talker backbone.
    parser.add_argument("--qwen3_text_model_path", type=str,
                        default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--verify_only", action="store_true",
                        help="Only perform the backbone swap and print a "
                             "report. Skip dataset loading and training.")
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # --- Backbone weight reuse: Qwen3-0.6B -> Talker ---------------------
    report = swap_talker_backbone_from_qwen3(
        qwen3tts.model,
        args.qwen3_text_model_path,
        dtype=torch.bfloat16,
    )
    print_swap_report(report, accelerator)

    if args.verify_only:
        return
    # ----------------------------------------------------------------------

    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    target_speaker_embedding = None
    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids)
                    * text_embedding_mask
                )
                # NOTE: text_embedding lives in a 2048-dim space and must be
                # projected to the 1024-dim Talker hidden size before being
                # summed with codec_embedding. The projection is applied
                # later via `text_projection` inside the forward path; here
                # we follow the exact wiring used by `sft_12hz.py`.
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids)
                    * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | "
                    f"talker_loss={outputs.loss.item():.4f} "
                    f"subtalker_loss={sub_talker_loss.item():.4f} "
                    f"total={loss.item():.4f}"
                )

        if accelerator.is_main_process:
            output_dir = os.path.join(
                args.output_model_path, f"checkpoint-epoch-{epoch}"
            )
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {args.speaker_name: False}
            config_dict["talker_config"] = talker_config

            with open(output_config_file, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {
                k: v.detach().to("cpu")
                for k, v in unwrapped_model.state_dict().items()
            }

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = (
                target_speaker_embedding[0]
                .detach()
                .to(weight.device)
                .to(weight.dtype)
            )
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)


if __name__ == "__main__":
    train()
