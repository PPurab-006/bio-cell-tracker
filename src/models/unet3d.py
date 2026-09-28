"""Compact 3D U-Net architecture for cell detection in anisotropic microscopy volumes.

Problem Solved:
---------------
Live fluorescence microscopy of zebrafish development exhibits 4:1 axial anisotropy
(Z=1.625 um/voxel vs XY=0.40625 um/voxel). Standard isotropic 3x3x3 convolutions blur
spatial context excessively across axial planes in early stages.
This module provides a lightweight, parameter-efficient 3D U-Net with:
1. Anisotropy-aware early convolutional blocks (1, 3, 3) to match physical receptive fields.
2. Selective axial downsampling: stage 1 downsamples only in lateral dimensions (1, 2, 2)
   to restore an isotropic physical aspect ratio before subsequent (2, 2, 2) pooling.
3. Instance normalization and stable LeakyReLU activations for robust small-batch training.
4. Guaranteed spatial dimension preservation through matching transpose convolutions.
"""

from __future__ import annotations

from typing import Sequence, Tuple
import torch
import torch.nn as nn


class AnisotropicConvBlock(nn.Module):
    """Two-layer 3D convolutional block with custom kernel dimensions."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int] = (1, 3, 3),
        padding: Tuple[int, int, int] = (0, 1, 1),
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StandardConvBlock(nn.Module):
    """Two-layer isotropic 3D convolutional block (3, 3, 3)."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Compact3DUNet(nn.Module):
    """Compact, anisotropic-aware 3D U-Net for cell-center heatmap regression."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        final_bias_init: float | None = None,
    ) -> None:
        """Initialize Compact3DUNet.

        Parameters
        ----------
        in_channels : int, default=1
            Number of input volume channels (typically 1 for raw fluorescence).
        out_channels : int, default=1
            Number of output channels (1 for cell-center heatmap).
        base_channels : int, default=16
            Number of feature channels in the initial stage.
        final_bias_init : float | None, default=None
            If provided, initialize the bias of the final conv layer to this constant
            (e.g. -4.0 for sigmoid output floor of ~0.018).
        """
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        # Encoder Stage 1: Anisotropic early convolutions (1, 3, 3)
        self.enc1 = AnisotropicConvBlock(
            in_channels, c1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )
        # Downsample only XY in first stage to balance 4:1 Z anisotropy
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Encoder Stage 2: Standard 3D convolutions (3, 3, 3)
        self.enc2 = StandardConvBlock(c1, c2)
        # Downsample Z, Y, X symmetrically
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # Bottleneck Stage
        self.bottleneck = StandardConvBlock(c2, c3)

        # Decoder Stage 2
        self.up2 = nn.ConvTranspose3d(
            c3, c2, kernel_size=(2, 2, 2), stride=(2, 2, 2)
        )
        self.dec2 = StandardConvBlock(c2 + c2, c2)

        # Decoder Stage 1
        self.up1 = nn.ConvTranspose3d(
            c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )
        self.dec1 = AnisotropicConvBlock(
            c1 + c1, c1, kernel_size=(1, 3, 3), padding=(0, 1, 1)
        )

        # Output Projection Head
        self.head = nn.Conv3d(c1, out_channels, kernel_size=1)
        if final_bias_init is not None:
            nn.init.constant_(self.head.bias, float(final_bias_init))

    @property
    def num_parameters(self) -> int:
        """Total trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, D, H, W).
            D should be a multiple of 2; H, W multiples of 4.

        Returns
        -------
        torch.Tensor
            Predicted cell center heatmap in [0.0, 1.0] of shape (B, out_channels, D, H, W).
        """
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        d2 = self.up2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.head(d1)
        return torch.sigmoid(logits)


def masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute batch-pooled weighted masked L1 loss between prediction and target.

    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction tensor in [0.0, 1.0].
    target : torch.Tensor
        Ground-truth heatmap target in [0.0, 1.0].
    mask : torch.Tensor
        Non-negative spatial weight mask (0.0 for ignored/unlabeled voxels).
    eps : float, default=1e-6
        Small constant to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Scalar masked loss value.
    """
    diff = torch.abs(prediction - target)
    weighted_diff = mask * diff
    loss = torch.sum(weighted_diff) / (torch.sum(mask) + eps)
    return loss


def per_patch_masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute per-patch normalized masked L1 loss.

    Each patch in the batch is normalized by its own supervised mask weight sum.
    Patches with sum(mask) == 0 contribute zero loss and zero gradient.

    Parameters
    ----------
    prediction : torch.Tensor
        Model prediction tensor of shape (B, 1, D, H, W) in [0.0, 1.0].
    target : torch.Tensor
        Target tensor of shape (B, 1, D, H, W) in [0.0, 1.0].
    mask : torch.Tensor
        Mask tensor of shape (B, 1, D, H, W).
    eps : float, default=1e-6
        Numerical stabilizer.

    Returns
    -------
    torch.Tensor
        Scalar average loss across valid supervised patches.
    """
    diff = torch.abs(prediction - target)
    weighted_diff = mask * diff
    # Sum over spatial dimensions (D, H, W): shape (B, 1) or (B,)
    sample_weighted_diff = torch.sum(weighted_diff, dim=(-3, -2, -1))
    sample_mask_sum = torch.sum(mask, dim=(-3, -2, -1))

    valid_samples = (sample_mask_sum > 0).float()
    sample_losses = sample_weighted_diff / (sample_mask_sum + eps)

    total_valid = torch.sum(valid_samples)
    if total_valid > 0:
        return torch.sum(sample_losses * valid_samples) / total_valid
    return torch.tensor(0.0, device=prediction.device, dtype=prediction.dtype)

