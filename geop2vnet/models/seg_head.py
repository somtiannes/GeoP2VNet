# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0
#
# Segmentation Head Module

"""
Segmentation Head for Aneurysm Segmentation

Architecture (from diagram):
    Conv3d → Norm → ReLU → Conv3d
    
    Input:  [B, 48, 96, 96, 96] decoder features
    Output: [B, 2, 96, 96, 96] aneurysm mask logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class SegmentationHead(nn.Module):
    """
    Segmentation Head for dense prediction.
    
    Architecture:
        Conv3d(in, mid, 3) → InstanceNorm → ReLU → Conv3d(mid, out, 1)
    
    Args:
        in_channels: Input feature channels (default 48)
        mid_channels: Intermediate channels (default 48)
        out_channels: Output classes (default 2: background + aneurysm)
        dropout: Dropout probability (default 0.0)
    
    Input:
        features: [B, C, D, H, W] decoder output
    
    Output:
        logits: [B, num_classes, D, H, W] segmentation logits
    """
    
    def __init__(
        self,
        in_channels: int = 48,
        mid_channels: int = 48,
        out_channels: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [B, C, D, H, W] decoder features
        
        Returns:
            logits: [B, num_classes, D, H, W]
        """
        return self.head(x)


class DeepSupervisionHead(nn.Module):
    """
    Segmentation Head with Deep Supervision.
    
    Outputs predictions at multiple scales for deep supervision loss.
    Each aux head has non-linear capacity (InstanceNorm + ReLU) to handle
    high-dimensional intermediate features more gracefully.
    
    Args:
        in_channels: List of input channels at each scale
        out_channels: Output classes (default 2)
    """
    
    def __init__(
        self,
        in_channels: list[int] = [48, 96, 192, 384],
        out_channels: int = 2,
    ) -> None:
        super().__init__()
        
        # Each aux head: InstanceNorm → ReLU → Conv1x1
        # This adds non-linear capacity to handle high-dim features (192, 384)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.InstanceNorm3d(c),
                nn.ReLU(inplace=True),
                nn.Conv3d(c, out_channels, kernel_size=1),
            )
            for c in in_channels
        ])
    
    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass with multi-scale outputs.
        
        Args:
            features: List of [B, C_i, D_i, H_i, W_i] at different scales
        
        Returns:
            logits: List of [B, num_classes, D_i, H_i, W_i]
        
        Note:
            Output resolutions are multi-scale (e.g., 96³, 48³, 24³).
            Loss computation should either:
              - Downsample GT to match each scale (recommended), or
              - Upsample logits to full resolution before computing loss.
        """
        return [head(f) for head, f in zip(self.heads, features)]


def build_seg_head(cfg: dict | None = None) -> SegmentationHead:
    """Factory function for Segmentation Head."""
    if cfg is None:
        cfg = {}
    
    return SegmentationHead(
        in_channels=cfg.get("in_channels", 48),
        mid_channels=cfg.get("mid_channels", 48),
        out_channels=cfg.get("out_channels", 2),
        dropout=cfg.get("dropout", 0.0),
    )


def build_deep_supervision_head(cfg: dict | None = None) -> DeepSupervisionHead:
    """
    Factory function for Deep Supervision Head.
    
    Args:
        cfg: Configuration dict with optional keys:
            - in_channels: List[int], channels at each scale (default [48, 96, 192, 384])
            - out_channels: int, number of classes (default 2)
    
    Returns:
        DeepSupervisionHead instance
    """
    if cfg is None:
        cfg = {}
    
    return DeepSupervisionHead(
        in_channels=cfg.get("in_channels", [48, 96, 192, 384]),
        out_channels=cfg.get("out_channels", 2),
    )


