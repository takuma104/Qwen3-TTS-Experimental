# coding=utf-8
# Copyright 2026 The Alibaba Qwen team & Takuma Mori.
# SPDX-License-Identifier: Apache-2.0
"""
GAN Loss Functions for decoder_block_48k training.

- LSGAN adversarial loss (generator and discriminator)
- Feature matching loss
"""

from typing import List, Tuple

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """LSGAN discriminator loss.

    Args:
        disc_real_outputs: List of discriminator outputs for real samples.
        disc_fake_outputs: List of discriminator outputs for fake samples.

    Returns:
        (total_loss, r_loss_mean, g_loss_mean, dr_mean, dg_mean)
        - total_loss:   sum of r_loss + g_loss across all sub-discriminators
        - r_loss_mean:  mean r_loss per sub-discriminator (for logging)
        - g_loss_mean:  mean g_loss per sub-discriminator (for logging)
        - dr_mean:      mean discriminator output for real samples (for logging)
        - dg_mean:      mean discriminator output for fake samples (for logging)
    """
    device = disc_real_outputs[0].device
    loss = torch.tensor(0.0, device=device)
    r_loss_total = torch.tensor(0.0, device=device)
    g_loss_total = torch.tensor(0.0, device=device)
    dr_total = torch.tensor(0.0, device=device)
    dg_total = torch.tensor(0.0, device=device)
    n = len(disc_real_outputs)
    for dr, dg in zip(disc_real_outputs, disc_fake_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss = loss + r_loss + g_loss
        r_loss_total = r_loss_total + r_loss
        g_loss_total = g_loss_total + g_loss
        dr_total = dr_total + torch.mean(dr.float())
        dg_total = dg_total + torch.mean(dg.float())
    return loss, r_loss_total / n, g_loss_total / n, dr_total / n, dg_total / n


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
