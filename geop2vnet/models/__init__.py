# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
GeoP2VNet Models

Components:
    - GFEModule: Geometric Feature Extraction (10D handcrafted features)
    - P2VModule: Point-to-Voxel Projection (Gaussian Splatting)
    - SwinUNETR: Swin Transformer Encoder-Decoder backbone
    - SegmentationHead: Final segmentation head
    - GeoP2VNet: Complete model
"""

from .gfe import GFEModule, build_gfe
from .p2v import P2VModule, P2VModuleFast, build_p2v
from .swin_unetr import SwinUNETR, build_swin_unetr
from .seg_head import SegmentationHead, build_seg_head
from .geop2vnet import GeoP2VNet, GeoP2VNetOutput, build_model

__all__ = [
    # GFE
    "GFEModule",
    "build_gfe",
    # P2V
    "P2VModule",
    "P2VModuleFast", 
    "build_p2v",
    # Backbone
    "SwinUNETR",
    "build_swin_unetr",
    # Head
    "SegmentationHead",
    "build_seg_head",
    # Complete model
    "GeoP2VNet",
    "GeoP2VNetOutput",
    "build_model",
]
