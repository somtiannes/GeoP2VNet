# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
GeoP2VNet: Point-to-Voxel Projection with Geometric Feature Preservation
for Intracranial Aneurysm Segmentation

Architecture:
    CTA Image [B, 1, 96³] + Point Cloud [B, 2048, 3]
        → GFE (10D geometric features)
        → P2V (Gaussian splatting)
        → DGRF (CTA-guided residual fusion)
        → Swin-UNETR (encoder-decoder)
        → Segmentation Head
        → Aneurysm Mask [B, 2, 96³]
"""

from .models import GeoP2VNet, build_model

__version__ = "1.0.0"
__all__ = ["GeoP2VNet", "build_model"]
