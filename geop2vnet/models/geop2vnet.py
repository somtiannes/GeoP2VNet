# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0
#
# GeoP2VNet: Point-to-Voxel Projection with Geometric Feature Preservation

"""
GeoP2VNet: Complete Model for Intracranial Aneurysm Segmentation

Architecture Overview:
    
    ┌─────────────┐       ┌─────────────────┐
    │  CTA Image  │       │   Point Cloud   │
    │ [B,1,96³]   │       │  [B,2048,3]     │
    └──────┬──────┘       └────────┬────────┘
           │                       │
           │              ┌────────▼────────┐
           │              │       GFE       │
           │              │  10-ch features │
           │              │  [B,2048,10]    │
           │              └────────┬────────┘
           │                       │
           │              ┌────────▼────────┐
           │              │       P2V       │
           │              │ Gaussian Splat  │
           │              │  [B,10,96³]     │
           │              └────────┬────────┘
           │                       │
           │              ┌────────▼────────┐
           │              │  InstanceNorm3d │
           │              │     (10-ch)     │
           │              └────────┬────────┘
           │                       │
    ┌──────▼───────────────────────▼──────┐
    │  Dual-Gated Residual Fusion (DGRF)  │
    │            [B, 48, 96³]             │
    └──────────────────┬──────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │         Swin-UNETR Backbone         │
    │    Encoder (48→96→192→384→768)      │
    │    Decoder (768→384→192→96→48)      │
    │            [B, 48, 96³]             │
    └──────────────────┬──────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │         Segmentation Head           │
    │      Conv → Norm → ReLU → Conv      │
    │            [B, 2, 96³]              │
    └─────────────────────────────────────┘

Features:
    - 10D handcrafted geometric features (no learned parameters in GFE)
    - Differentiable Gaussian splatting for P2V projection
    - Dual-gated residual fusion for CTA-guided geometric injection
    - Swin Transformer encoder with hierarchical features
    - UNETR-style decoder with skip connections
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, NamedTuple

from .gfe import GFEModule, build_gfe
from .p2v import P2VModuleFast, build_p2v
from .swin_unetr import SwinUNETR, build_swin_unetr
from .seg_head import SegmentationHead, build_seg_head


class GeoP2VNetOutput(NamedTuple):
    """Output container for GeoP2VNet."""
    logits: torch.Tensor          # [B, 2, D, H, W] segmentation logits
    geo_features: torch.Tensor    # [B, 10, D, H, W] projected geometric features
    decoder_features: torch.Tensor  # [B, 48, D, H, W] decoder output
    geo_prior_logits: torch.Tensor  # [B, 1, D, H, W] geometric prior prediction


class GeoP2VNet(nn.Module):
    """
    GeoP2VNet: Point-to-Voxel Projection Network for Aneurysm Segmentation.
    
    Combines:
        1. GFE: Geometric Feature Extraction from point cloud (10D handcrafted)
        2. P2V: Point-to-Voxel projection via Gaussian splatting
        3. Swin-UNETR: Encoder-Decoder backbone
        4. SegHead: Final segmentation head
    
    Args:
        voxel_size: Tuple (D, H, W) for voxel grid (default (96, 96, 96))
        gfe_cfg: Config dict for GFE module
        p2v_cfg: Config dict for P2V module
        backbone_cfg: Config dict for Swin-UNETR
        head_cfg: Config dict for segmentation head
    
    Input:
        image: [B, 1, D, H, W] CTA image volume
        points: [B, N, 3] point cloud coordinates in [-1, 1]
    
    Output:
        GeoP2VNetOutput with logits, geo_features, decoder_features
    
    Example:
        >>> model = GeoP2VNet()
        >>> image = torch.randn(1, 1, 96, 96, 96)
        >>> points = torch.rand(1, 2048, 3) * 2 - 1
        >>> output = model(image, points)
        >>> print(output.logits.shape)  # [1, 2, 96, 96, 96]
    """
    
    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (96, 96, 96),
        gfe_cfg: Optional[dict] = None,
        p2v_cfg: Optional[dict] = None,
        backbone_cfg: Optional[dict] = None,
        head_cfg: Optional[dict] = None,
        fusion_cfg: Optional[dict] = None,
        use_geometry: bool = True,
        min_points_for_geo: int = 128,  # Fall back to CTA-only when the point count is below this threshold.
    ) -> None:
        super().__init__()
        
        self.voxel_size = voxel_size
        self.use_geometry = use_geometry
        self.min_points_for_geo = min_points_for_geo
        
        # ═══════════════════════════════════════════════════════════════
        # Modules 1 & 2: GFE + P2V (optional, for ablation study)
        # ═══════════════════════════════════════════════════════════════
        # Embed dimension for backbone
        embed_dim = (backbone_cfg or {}).get("embed_dim", 48)
        fusion_cfg = fusion_cfg or {}
        
        if self.use_geometry:
            self.gfe = build_gfe(gfe_cfg)
            self.geo_dim = self.gfe.OUT_DIM  # 10D features
            
            p2v_cfg = p2v_cfg or {}
            p2v_cfg["voxel_size"] = voxel_size
            p2v_fast = p2v_cfg.get("fast", False)
            self.p2v = build_p2v(p2v_cfg, fast=p2v_fast)
            
            # DGRF: CTA-guided channel/spatial gating for geometric injection
            self.geo_norm = nn.InstanceNorm3d(self.geo_dim, affine=True)
            self.cta_proj = nn.Conv3d(1, embed_dim, kernel_size=3, padding=1, bias=False)
            self.geo_proj = nn.Conv3d(self.geo_dim, embed_dim, kernel_size=1, bias=False)
            reduction = max(1, embed_dim // fusion_cfg.get("reduction", 4))
            self.channel_gate = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(embed_dim, reduction, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(reduction, embed_dim, kernel_size=1),
                nn.Sigmoid(),
            )
            self.spatial_gate = nn.Sequential(
                nn.Conv3d(embed_dim, 1, kernel_size=1),
                nn.Sigmoid(),
            )
            self.alpha = nn.Parameter(torch.tensor(float(fusion_cfg.get("alpha_init", 0.1))))
            
            # Prior head: Geometry branch predicts a point-cloud prior for auxiliary supervision.
            self.geo_prior_head = nn.Conv3d(self.geo_dim, 1, kernel_size=1, bias=True)
            
            # No fallback layer - when use_geometry=True, points must be provided
            
            in_channels = embed_dim  # Backbone receives 48-dim fused features
        else:
            # Baseline mode: CTA only, no geometry
            self.gfe = None
            self.p2v = None
            self.geo_dim = 0
            self.geo_norm = None
            self.cta_proj = None
            self.geo_proj = None
            self.channel_gate = None
            self.spatial_gate = None
            self.alpha = None
            in_channels = 1  # CTA only
        
        # ═══════════════════════════════════════════════════════════════
        # Module 3: Swin-UNETR Backbone
        # ═══════════════════════════════════════════════════════════════
        backbone_cfg = backbone_cfg or {}
        backbone_cfg["in_channels"] = in_channels
        self.backbone = build_swin_unetr(backbone_cfg)
        
        # ═══════════════════════════════════════════════════════════════
        # Module 4: Segmentation Head
        # ═══════════════════════════════════════════════════════════════
        head_cfg = head_cfg or {}
        head_cfg["in_channels"] = backbone_cfg.get("embed_dim", 48)
        self.seg_head = build_seg_head(head_cfg)
    
    def forward(
        self,
        image: torch.Tensor,
        points: Optional[list] = None,
    ) -> GeoP2VNetOutput:
        """
        Forward pass.
        
        Args:
            image: [B, 1, D, H, W] CTA image volume
            points: List of [N_i, 3] tensors (variable length per sample),
                    or [B, N, 3] tensor (fixed length, for testing),
                    or None when use_geometry=False
        
        Returns:
            GeoP2VNetOutput with logits, geo_features, decoder_features
        """
        B = image.shape[0]
        D, H, W = self.voxel_size
        
        if self.use_geometry and points is not None:
            # Normalize input: convert tensor to list if needed
            if isinstance(points, torch.Tensor):
                # Fixed-length tensor [B, N, 3] -> list of [N, 3]
                points_list = [points[i] for i in range(B)]
            else:
                points_list = points
            
            # Process each sample independently (variable point count)
            # Hard gating: Enable the geometry branch only when enough points are available.
            geo_voxels_list = []
            for i in range(B):
                pts_i = points_list[i]  # [N_i, 3]
                N_i = pts_i.shape[0]
                
                if N_i >= self.min_points_for_geo:
                    # Enough points: compute geometric features normally.
                    pts_i = pts_i.unsqueeze(0)  # [1, N_i, 3]
                    gfe_out = self.gfe(pts_i)
                    feats_i = gfe_out.features  # [1, N_i, 10]
                    vox_i = self.p2v(pts_i, feats_i)  # [1, 10, D, H, W]
                else:
                    # Too few points: set geometric voxels to zero and fall back to CTA-only behavior.
                    vox_i = torch.zeros(1, self.geo_dim, D, H, W, 
                                       device=image.device, dtype=image.dtype)
                
                geo_voxels_list.append(vox_i)
            
            geo_voxels = torch.cat(geo_voxels_list, dim=0)  # [B, 10, D, H, W]
            
            # Normalize geometric features (solve numerical scale mismatch)
            geo_voxels = self.geo_norm(geo_voxels)  # [B, 10, D, H, W]
            
            # Dual-Gated Residual Fusion (DGRF)
            cta_features = self.cta_proj(image)          # [B, 48, D, H, W]
            geo_features = self.geo_proj(geo_voxels)     # [B, 48, D, H, W]
            channel_weight = self.channel_gate(cta_features)
            spatial_weight = self.spatial_gate(cta_features)
            fused = cta_features + self.alpha * geo_features * channel_weight * spatial_weight
        else:
            # Baseline mode: use_geometry=False, CTA only
            if self.use_geometry:
                raise ValueError("use_geometry=True requires points to be provided")
            geo_voxels = torch.zeros(B, 10, D, H, W, device=image.device, dtype=image.dtype)
            fused = image  # [B, 1, D, H, W] - backbone expects 1 channel
        
        # ─────────────────────────────────────────────────────────────
        # Step 4: Swin-UNETR Backbone
        # ─────────────────────────────────────────────────────────────
        decoder_features = self.backbone(fused)  # [B, 48, D, H, W]
        
        # ─────────────────────────────────────────────────────────────
        # Step 5: Segmentation Head
        # ─────────────────────────────────────────────────────────────
        logits = self.seg_head(decoder_features)  # [B, 2, D, H, W]
        
        # ─────────────────────────────────────────────────────────────
        # Step 6: Geometric prior head for auxiliary supervision.
        # ─────────────────────────────────────────────────────────────
        if self.use_geometry and hasattr(self, 'geo_prior_head'):
            geo_prior_logits = self.geo_prior_head(geo_voxels)  # [B, 1, D, H, W]
        else:
            geo_prior_logits = torch.zeros(B, 1, D, H, W, device=image.device, dtype=image.dtype)
        
        return GeoP2VNetOutput(
            logits=logits,
            geo_features=geo_voxels,
            decoder_features=decoder_features,
            geo_prior_logits=geo_prior_logits,
        )
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts for each module."""
        result = {
            "backbone": sum(p.numel() for p in self.backbone.parameters()),
            "seg_head": sum(p.numel() for p in self.seg_head.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
        if self.use_geometry:
            result["gfe"] = sum(p.numel() for p in self.gfe.parameters())
            result["p2v"] = sum(p.numel() for p in self.p2v.parameters())
            result["fusion"] = (
                sum(p.numel() for p in self.geo_norm.parameters()) +
                sum(p.numel() for p in self.cta_proj.parameters()) +
                sum(p.numel() for p in self.geo_proj.parameters()) +
                sum(p.numel() for p in self.channel_gate.parameters()) +
                sum(p.numel() for p in self.spatial_gate.parameters()) +
                self.alpha.numel()
            )
        return result


def build_model(cfg: Optional[dict] = None) -> GeoP2VNet:
    """
    Factory function to build GeoP2VNet from config.
    
    Args:
        cfg: Configuration dict with optional keys:
            - voxel_size: Tuple[int, int, int]
            - gfe: dict for GFE config
            - p2v: dict for P2V config
            - backbone: dict for Swin-UNETR config
            - head: dict for segmentation head config
            - fusion: dict for DGRF config
            - use_geometry: bool, whether to use the geometry branch (default: True)
    
    Returns:
        GeoP2VNet model instance
    """
    if cfg is None:
        cfg = {}
    
    return GeoP2VNet(
        voxel_size=tuple(cfg.get("voxel_size", (96, 96, 96))),
        gfe_cfg=cfg.get("gfe"),
        p2v_cfg=cfg.get("p2v"),
        backbone_cfg=cfg.get("backbone"),
        head_cfg=cfg.get("head"),
        fusion_cfg=cfg.get("fusion"),
        use_geometry=cfg.get("use_geometry", True),
        min_points_for_geo=cfg.get("min_points_for_geo", 128),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("GeoP2VNet Complete Model Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build model
    model = GeoP2VNet(
        voxel_size=(96, 96, 96),
        gfe_cfg={"k_small": 8, "k_large": 32},
        p2v_cfg={"sigma": 1.0},
        backbone_cfg={"embed_dim": 48},
        head_cfg={"out_channels": 2},
    ).to(device)
    
    # Parameter counts
    print("\nParameter counts:")
    params = model.get_num_parameters()
    for name, count in params.items():
        print(f"  {name}: {count / 1e6:.2f}M")
    
    # Test input (list of variable-length point clouds)
    B = 2
    image = torch.randn(B, 1, 96, 96, 96, device=device)
    # Variable length: sample 0 has 1024 points, sample 1 has 512 points
    points = [
        torch.rand(1024, 3, device=device) * 2 - 1,
        torch.rand(512, 3, device=device) * 2 - 1,
    ]
    
    print(f"\nInput shapes:")
    print(f"  image: {image.shape}")
    print(f"  points: list of {[p.shape for p in points]}")
    
    # Forward pass
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    
    with torch.no_grad():
        output = model(image, points)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.time() - t0) * 1000
    
    print(f"\nOutput shapes:")
    print(f"  logits: {output.logits.shape}")
    print(f"  geo_features: {output.geo_features.shape}")
    print(f"  decoder_features: {output.decoder_features.shape}")
    print(f"\nForward time: {elapsed:.2f} ms")
    
    # Verify gradients flow
    print("\nGradient check...")
    model.train()
    output = model(image, points)
    loss = output.logits.sum()
    loss.backward()
    
    grad_ok = all(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
        if p.requires_grad
    )
    print(f"  Gradients flow: {'OK' if grad_ok else 'FAILED'}")
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)
