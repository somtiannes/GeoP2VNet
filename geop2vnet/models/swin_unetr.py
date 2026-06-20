# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0
#
# Swin-UNETR Backbone Module (using MONAI's official implementation)
#
# Reference:
#   - Hatamizadeh et al., "Swin UNETR: Swin Transformers for Semantic 
#     Segmentation of Brain Tumors in MRI Images", MICCAI BrainLes 2022

"""
Swin-UNETR Backbone for 3D Medical Image Segmentation

This module wraps MONAI's official SwinUNETR implementation, which includes:
    - Proper Relative Position Bias in Window Attention
    - Correct Attention Mask for Shifted Windows
    - Validated architecture and hyperparameters

Architecture:
    Encoder (Swin Transformer):
        Stage 1: C=48,  resolution=48³  (2× downsample from patch embed)
        Stage 2: C=96,  resolution=24³  (4× downsample)
        Stage 3: C=192, resolution=12³  (8× downsample)
        Stage 4: C=384, resolution=6³   (16× downsample)
        
    Decoder (CNN with skip connections):
        Up-Stage 1: 384→192, 12³
        Up-Stage 2: 192→96,  24³
        Up-Stage 3: 96→48,   48³
        Final: 48→48, 96³
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Sequence, Optional

from monai.networks.nets import SwinUNETR as MonaiSwinUNETR


class SwinUNETR(nn.Module):
    """
    Swin-UNETR backbone wrapping MONAI's official implementation.
    
    This wrapper extracts the decoder features (before the final segmentation head)
    so that we can apply our own SegmentationHead.
    
    Args:
        in_channels: Number of input channels (1 for CTA, 48 for fused)
        embed_dim: Base embedding dimension (default 48)
        depths: Number of Swin blocks per stage (default (2, 2, 2, 2))
        num_heads: Number of attention heads per stage (default (3, 6, 12, 24))
        feature_size: Base feature size (same as embed_dim)
        spatial_dims: Number of spatial dimensions (3 for 3D)
        use_v2: Use SwinUNETR V2 (improved version)
    
    Input:
        x: [B, C, D, H, W] input volume (typically 96³)
    
    Output:
        features: [B, embed_dim, D, H, W] decoded features
    """
    
    def __init__(
        self,
        in_channels: int = 48,
        embed_dim: int = 48,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Use MONAI's official SwinUNETR
        # Note: MONAI's out_channels is for final segmentation, 
        # but we'll extract features before that
        self.swin_unetr = MonaiSwinUNETR(
            in_channels=in_channels,
            out_channels=2,  # Will be ignored - we extract features before this
            feature_size=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            use_checkpoint=use_checkpoint,
            spatial_dims=3,
            normalize=True,
        )
        
        # Remove MONAI's final output layer (we use our own SegmentationHead)
        # The decoder5 output is what we want
        # self.swin_unetr.out is the final conv - we'll bypass it
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass - extract decoder features.
        
        Args:
            x: [B, C, D, H, W] input volume
        
        Returns:
            features: [B, embed_dim, D, H, W] decoder features
        """
        # MONAI SwinUNETR internal structure:
        # swinViT -> encoder stages
        # encoder1-10 -> CNN encoders
        # decoder1-5 -> decoders with skip connections
        # out -> final segmentation layer (we skip this)
        
        hidden_states = self.swin_unetr.swinViT(x, self.swin_unetr.normalize)
        enc0 = self.swin_unetr.encoder1(x)
        enc1 = self.swin_unetr.encoder2(hidden_states[0])
        enc2 = self.swin_unetr.encoder3(hidden_states[1])
        enc3 = self.swin_unetr.encoder4(hidden_states[2])
        dec4 = self.swin_unetr.encoder10(hidden_states[4])
        
        dec3 = self.swin_unetr.decoder5(dec4, hidden_states[3])
        dec2 = self.swin_unetr.decoder4(dec3, enc3)
        dec1 = self.swin_unetr.decoder3(dec2, enc2)
        dec0 = self.swin_unetr.decoder2(dec1, enc1)
        features = self.swin_unetr.decoder1(dec0, enc0)
        
        # features is [B, embed_dim, D, H, W]
        return features
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def build_swin_unetr(cfg: dict | None = None) -> SwinUNETR:
    """
    Factory function for Swin-UNETR backbone.
    
    Args:
        cfg: Configuration dict with optional keys:
            - in_channels: int (default 48 for fused features)
            - embed_dim: int (default 48)
            - depths: tuple (default (2, 2, 2, 2))
            - num_heads: tuple (default (3, 6, 12, 24))
            - drop_rate: float (default 0.0)
            - attn_drop_rate: float (default 0.0)
            - use_checkpoint: bool (default False)
    
    Returns:
        SwinUNETR model instance
    """
    if cfg is None:
        cfg = {}
    
    return SwinUNETR(
        in_channels=cfg.get("in_channels", 48),
        embed_dim=cfg.get("embed_dim", 48),
        depths=cfg.get("depths", (2, 2, 2, 2)),
        num_heads=cfg.get("num_heads", (3, 6, 12, 24)),
        drop_rate=cfg.get("drop_rate", 0.0),
        attn_drop_rate=cfg.get("attn_drop_rate", 0.0),
        use_checkpoint=cfg.get("use_checkpoint", False),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("Swin-UNETR (MONAI) Unit Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model with fused input (48 channels from 1x1 conv)
    model = SwinUNETR(in_channels=48, embed_dim=48).to(device)
    
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel parameters: {n_params:.2f}M")
    
    # Test input (48 channels = fused CTA + geometry)
    B = 1
    x = torch.randn(B, 48, 96, 96, 96, device=device)
    print(f"Input: {x.shape}")
    
    # Forward pass
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    
    with torch.no_grad():
        out = model(x)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.time() - t0) * 1000
    
    print(f"Output: {out.shape}")
    print(f"Forward time: {elapsed:.2f} ms")
    
    # Verify output shape
    assert out.shape == (B, 48, 96, 96, 96), f"Expected (1, 48, 96, 96, 96), got {out.shape}"
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)
