# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
GAN Loss Functions for decoder_block_48k training.

- LSGAN adversarial loss (generator and discriminator)
- Feature matching loss
"""

from typing import List

import torch


def generator_adversarial_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
    """LSGAN generator loss.

    Args:
        disc_outputs: List of discriminator outputs for fake samples.

    Returns:
        Generator adversarial loss.
    """
    loss = torch.tensor(0.0, device=disc_outputs[0].device)
    for dg in disc_outputs:
        loss = loss + torch.mean((1 - dg) ** 2)
    return loss


def discriminator_loss(
    disc_real_outputs: List[torch.Tensor],
    disc_fake_outputs: List[torch.Tensor],
) -> torch.Tensor:
    """LSGAN discriminator loss.

    Args:
        disc_real_outputs: List of discriminator outputs for real samples.
        disc_fake_outputs: List of discriminator outputs for fake samples.

    Returns:
        Discriminator loss.
    """
    loss = torch.tensor(0.0, device=disc_real_outputs[0].device)
    for dr, dg in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss
    return loss


def feature_matching_loss(
    fmap_real: List[List[torch.Tensor]],
    fmap_fake: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching loss between real and fake feature maps.

    Args:
        fmap_real: List of feature map lists from discriminator on real samples.
        fmap_fake: List of feature map lists from discriminator on fake samples.

    Returns:
        Feature matching loss.
    """
    loss = torch.tensor(0.0, device=fmap_real[0][0].device)
    for fr_list, fg_list in zip(fmap_real, fmap_fake):
        for fr, fg in zip(fr_list, fg_list):
            loss = loss + torch.mean(torch.abs(fr.detach() - fg))
    return loss
