# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
Geometry-aware Loss Functions for GeoP2VNet

Implements:
1. PointWeightedDiceCELoss: optional point-cloud-region weighted DiceCE
2. GeometricFeatureConsistencyLoss: optional geometric feature consistency regularization
3. GeoP2VCompositeLoss: composite loss wrapper

Composite objective:
    L = L_seg + λ_prior * L_prior + λ_boundary * L_boundary
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict

from .seg_loss import DiceCELoss


class PointWeightedDiceCELoss(nn.Module):
    """
    Point-cloud-region weighted DiceCE.
    
    Voxels near the vessel point cloud are assigned larger loss weights.
    
    Args:
        base_dicece: base DiceCELoss instance
        base_weight: default region weight
        point_weight: point-cloud-region weight
        radius: point-cloud influence radius in voxels
    """
    
    def __init__(
        self,
        base_dicece: DiceCELoss,
        base_weight: float = 1.0,
        point_weight: float = 3.0,
        radius: int = 3,
    ) -> None:
        super().__init__()
        self.base_dicece = base_dicece
        self.base_weight = base_weight
        self.point_weight = point_weight
        self.radius = radius
    
    def _points_to_voxel_coords(
        self,
        pts: torch.Tensor,
        D: int, H: int, W: int,
    ) -> torch.Tensor:
        """
        Convert point-cloud coordinates to voxel indices.
        
        Supports two input conventions:
        - [-1, 1] normalized coordinates used by P2V
        - voxel indices
        
        Args:
            pts: [N, 3] point-cloud coordinates
            D, H, W: voxel grid size
            
        Returns:
            voxel_coords: [N, 3] voxel indices in ranges [0, D-1], [0, H-1], [0, W-1]
        """
        pts_f = pts.float()
        
        # Treat as normalized coordinates if values are negative or max <= 1.5.
        if (pts_f.min() < 0) or (pts_f.max() <= 1.5):
            # [-1, 1] -> [0, 1] -> voxel index
            z = (pts_f[:, 0] + 1) * 0.5 * (D - 1)
            y = (pts_f[:, 1] + 1) * 0.5 * (H - 1)
            x = (pts_f[:, 2] + 1) * 0.5 * (W - 1)
            pts_f = torch.stack([z, y, x], dim=-1)
        
        return pts_f
    
    def _create_prior(
        self,
        points_list: List[torch.Tensor],
        vol_shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a point-cloud prior: [B, 1, D, H, W] in [0, 1]
        
        Approximate distance decay with GPU Gaussian convolution.
        
        Args:
            points_list: List of [N, 3] point-cloud coordinates in [-1, 1] or voxel space
            vol_shape: (B, C, D, H, W)
            device: target device
            
        Returns:
            prior: [B, 1, D, H, W] point-cloud prior map
        """
        B, _, D, H, W = vol_shape
        sigma = float(self.radius)
        
        # Build a 3D Gaussian kernel.
        k = int(2 * np.ceil(2 * sigma) + 1)
        k = max(k, 3)
        pad = k // 2
        
        zz, yy, xx = torch.meshgrid(
            torch.arange(k, device=device, dtype=torch.float32),
            torch.arange(k, device=device, dtype=torch.float32),
            torch.arange(k, device=device, dtype=torch.float32),
            indexing="ij",
        )
        c = (k - 1) / 2.0
        dist2 = (zz - c) ** 2 + (yy - c) ** 2 + (xx - c) ** 2
        kernel = torch.exp(-dist2 / (2 * sigma ** 2 + 1e-8))
        kernel = kernel / (kernel.sum() + 1e-8)
        kernel = kernel.view(1, 1, k, k, k)
        
        prior = torch.zeros((B, 1, D, H, W), device=device, dtype=torch.float32)
        
        for b, pts in enumerate(points_list):
            # Empty point cloud: keep prior at zero and fall back to CTA-only behavior.
            if pts is None or pts.numel() == 0:
                continue
            
            # Coordinate conversion: supports [-1, 1] or voxel coordinates.
            voxel_coords = self._points_to_voxel_coords(pts, D, H, W)
            coords = voxel_coords.long()
            coords[:, 0] = coords[:, 0].clamp(0, D - 1)
            coords[:, 1] = coords[:, 1].clamp(0, H - 1)
            coords[:, 2] = coords[:, 2].clamp(0, W - 1)
            
            seed = torch.zeros((1, 1, D, H, W), device=device, dtype=torch.float32)
            seed[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            
            p = F.conv3d(seed, kernel, padding=pad).clamp(0.0, 1.0)
            prior[b:b+1] = p
        
        return prior  # Empty point-cloud samples have an all-zero prior.
    
    def _create_weight_map(
        self,
        points_list: List[torch.Tensor],
        vol_shape: tuple,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a weight map by converting the prior into loss weights.
        
        Args:
            points_list: List of [N, 3] point-cloud coordinates
            vol_shape: (B, C, D, H, W)
            device: target device
            
        Returns:
            weight_map: [B, 1, D, H, W]
        """
        prior = self._create_prior(points_list, vol_shape, device)
        
        # weight = base + (point - base) * prior
        weight = self.base_weight + (self.point_weight - self.base_weight) * prior
        
        # Normalize the weight map to keep the mean weight at 1 for stable BCE scale.
        weight = weight / (weight.mean(dim=(2, 3, 4), keepdim=True) + 1e-6)
        
        return weight
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        points_list: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute point-cloud weighted DiceCE loss.
        
        Args:
            logits: [B, 2, D, H, W]
            targets: [B, D, H, W] or [B, 1, D, H, W]
            points_list: List of [N, 3] point-cloud coordinates
            
        Returns:
            dict with 'loss' key
        """
        device = logits.device
        
        if targets.dim() == 4:
            targets = targets.unsqueeze(1)
        
        # Compute weight map.
        if points_list is not None:
            weight_map = self._create_weight_map(points_list, logits.shape, device)
        else:
            weight_map = None
        
        # Use the base DiceCE; weight-map support can be added later.
        # Current simplified form: use the weight map for an additional CE term.
        loss_dict = self.base_dicece(logits, targets)
        base_loss = loss_dict["loss"]
        
        # Add a weighted BCE term only when point-region weighting is enabled.
        use_point_weighting = weight_map is not None and abs(self.point_weight - self.base_weight) > 1e-6
        if use_point_weighting:
            # Use foreground logits; AMP-safe.
            fg_logits = logits[:, 1:2, ...]  # [B, 1, D, H, W]
            fg_target = targets.float()      # [B, 1, D, H, W]
            
            # Weighted BCE with logits; AMP-safe.
            # Manually implement weighted reduction='mean'.
            bce_unreduced = F.binary_cross_entropy_with_logits(
                fg_logits, fg_target, reduction='none'
            )
            weighted_bce = (bce_unreduced * weight_map).sum() / (weight_map.sum() + 1e-6)
            
            # Combine base loss with 0.1 * weighted BCE.
            loss = base_loss + 0.1 * weighted_bce
        else:
            loss = base_loss
        
        return {"loss": loss}


class GeometricFeatureConsistencyLoss(nn.Module):
    """
    Geometric feature consistency loss.
    
    Foreground geometric features should be compact with low variance.
    
    This encourages discriminative geometric features:
    - foreground aneurysm features should be consistent
    - background features should remain distinct
    """
    
    def __init__(self, min_fg_voxels: int = 10) -> None:
        super().__init__()
        self.min_fg_voxels = min_fg_voxels
    
    def forward(
        self,
        geo_features: torch.Tensor,
        pred_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intra-class variance of foreground geometric features.
        
        Args:
            geo_features: [B, Cg, D, H, W] geometric features output by P2V
            pred_mask: [B, D, H, W] predicted segmentation (0/1)
            
        Returns:
            scalar loss
        """
        B, C, D, H, W = geo_features.shape
        fg_mask = pred_mask > 0.5  # [B, D, H, W] bool
        
        loss = 0.0
        valid_batches = 0
        
        for b in range(B):
            mask_b = fg_mask[b]  # [D, H, W]
            
            if mask_b.sum() < self.min_fg_voxels:
                continue  # Skip samples with too few foreground voxels.
            
            feat_b = geo_features[b, :, mask_b]  # [C, N_fg]
            
            # Compute variance per channel and average.
            var_b = feat_b.var(dim=-1).mean()
            loss += var_b
            valid_batches += 1
        
        if valid_batches == 0:
            return geo_features.new_tensor(0.0)
        
        return loss / valid_batches


class BoundaryConsistencyLoss(nn.Module):
    """
    Optional boundary consistency loss.
    
    Predicted segmentation boundaries should lie near vessel-surface points.
    
    Simplified implementation: boundary voxels should be close to point-cloud locations.
    """
    
    def __init__(self, sigma: float = 2.0) -> None:
        super().__init__()
        self.sigma = sigma
    
    def _points_to_voxel_coords(
        self,
        pts: torch.Tensor,
        D: int, H: int, W: int,
    ) -> torch.Tensor:
        """Convert point-cloud coordinates to voxel indices."""
        pts_f = pts.float()
        if (pts_f.min() < 0) or (pts_f.max() <= 1.5):
            z = (pts_f[:, 0] + 1) * 0.5 * (D - 1)
            y = (pts_f[:, 1] + 1) * 0.5 * (H - 1)
            x = (pts_f[:, 2] + 1) * 0.5 * (W - 1)
            pts_f = torch.stack([z, y, x], dim=-1)
        return pts_f
    
    def forward(
        self,
        logits: torch.Tensor,
        points_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute boundary consistency loss.
        
        Args:
            logits: [B, 2, D, H, W]
            points_list: List of [N, 3] point-cloud coordinates in [-1, 1] or voxel space
            
        Returns:
            scalar loss
        """
        probs = F.softmax(logits, dim=1)[:, 1]  # [B, D, H, W]
        
        loss = 0.0
        valid = 0
        
        for b, points in enumerate(points_list):
            if points is None or points.numel() == 0:
                continue
            
            prob_b = probs[b]  # [D, H, W]
            D, H, W = prob_b.shape
            
            # 1. Extract predicted boundaries from large probability gradients.
            grad_z = torch.abs(prob_b[1:, :, :] - prob_b[:-1, :, :])
            grad_y = torch.abs(prob_b[:, 1:, :] - prob_b[:, :-1, :])
            grad_x = torch.abs(prob_b[:, :, 1:] - prob_b[:, :, :-1])
            
            # Simple boundary probability map.
            pred_boundary = torch.zeros_like(prob_b)
            pred_boundary[:-1, :, :] += grad_z
            pred_boundary[1:, :, :] += grad_z
            pred_boundary[:, :-1, :] += grad_y
            pred_boundary[:, 1:, :] += grad_y
            pred_boundary[:, :, :-1] += grad_x
            pred_boundary[:, :, 1:] += grad_x
            pred_boundary = pred_boundary / 6.0
            
            # 2. Generate target boundary map from point-cloud locations.
            target_boundary = torch.zeros_like(prob_b)
            voxel_coords = self._points_to_voxel_coords(points, D, H, W)
            coords = voxel_coords.long()
            coords[:, 0] = coords[:, 0].clamp(0, D - 1)
            coords[:, 1] = coords[:, 1].clamp(0, H - 1)
            coords[:, 2] = coords[:, 2].clamp(0, W - 1)
            target_boundary[coords[:, 0], coords[:, 1], coords[:, 2]] = 1.0
            
            # 3. Align predicted and target boundaries with MSE.
            batch_loss = F.mse_loss(pred_boundary, target_boundary)
            loss += batch_loss
            valid += 1
        
        if valid == 0:
            return logits.new_tensor(0.0)
        
        return loss / valid


class GeoP2VCompositeLoss(nn.Module):
    """
    GeoP2VNet composite loss
    
    L = L_seg + λ_prior * L_prior + λ_boundary * L_boundary
    
    Args:
        dice_weight: Dice loss weight
        ce_weight: CE loss weight
        lambda_prior: geometric prior BCE weight; start from 0 for stable Dice training
        lambda_boundary: boundary consistency weight; start from 0
        point_weight: point-cloud-region loss weight; 1.0 disables extra point weighting
        radius: point-cloud influence radius in voxels
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 0.5,
        lambda_prior: float = 0.0,  # disabled by default for stable Dice training
        lambda_boundary: float = 0.0,
        point_weight: float = 1.0,
        radius: int = 3,
    ) -> None:
        super().__init__()
        
        # Base DiceCE
        base_dicece = DiceCELoss(
            dice_weight=dice_weight,
            ce_weight=ce_weight,
            include_background=False,
            batch=True,
        )
        
        # Main DiceCE segmentation loss; optional point weighting is disabled when point_weight=1.0.
        self.seg_loss = PointWeightedDiceCELoss(
            base_dicece=base_dicece,
            base_weight=1.0,
            point_weight=point_weight,
            radius=radius,
        )
        
        # Geometric prior BCE loss
        self.lambda_prior = lambda_prior
        
        # Optional boundary consistency loss
        self.boundary_loss = BoundaryConsistencyLoss()
        self.lambda_boundary = lambda_boundary
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        points_list: Optional[List[torch.Tensor]] = None,
        geo_features: Optional[torch.Tensor] = None,
        geo_prior_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the composite loss.
        
        Args:
            logits: [B, 2, D, H, W] segmentation logits
            targets: [B, D, H, W] segmentation targets
            points_list: List of [N, 3] point-cloud coordinates (optional)
            geo_features: [B, Cg, D, H, W] geometric features (optional)
            geo_prior_logits: [B, 1, D, H, W] geometric prior logits (optional)
            
        Returns:
            dict with 'loss', 'loss_seg', 'loss_geo_prior', 'loss_boundary'
        """
        result = {}
        
        # 1. Main segmentation loss.
        seg_dict = self.seg_loss(logits, targets, points_list=points_list)
        loss = seg_dict["loss"]
        result["loss_seg"] = seg_dict["loss"].detach()
        
        # 2. Geometric prior BCE loss.
        # This encourages the geometry branch to explain the point-cloud distribution and avoids collapse.
        loss_geo_prior = logits.new_tensor(0.0)
        if geo_prior_logits is not None and points_list is not None and self.lambda_prior > 0:
            # Generate point-cloud prior as supervision target.
            prior_target = self.seg_loss._create_prior(points_list, logits.shape, logits.device)
            
            # BCE with logits: geo_prior_logits -> prior_target
            loss_geo_prior = F.binary_cross_entropy_with_logits(
                geo_prior_logits,
                prior_target,
                reduction='mean',
            )
            loss = loss + self.lambda_prior * loss_geo_prior
        result["loss_geo_prior"] = loss_geo_prior.detach()
        
        # 3. Optional boundary consistency.
        loss_boundary = logits.new_tensor(0.0)
        if points_list is not None and self.lambda_boundary > 0:
            loss_boundary = self.boundary_loss(logits, points_list)
            loss = loss + self.lambda_boundary * loss_boundary
        result["loss_boundary"] = loss_boundary.detach()
        
        result["loss"] = loss
        return result


def build_geo_loss(cfg: dict | None = None) -> GeoP2VCompositeLoss:
    """
    Factory function for GeoP2VCompositeLoss.
    
    Args:
        cfg: Config dict with optional keys:
            - dice_weight: float (default 1.0)
            - ce_weight: float (default 0.5)
            - lambda_prior: float (default 0.0; disabled by default for stable Dice training)
            - lambda_boundary: float (default 0.0)
            - point_weight: float (default 1.0; disables extra point weighting)
            - radius: int (default 3)
    
    Returns:
        GeoP2VCompositeLoss instance
    """
    if cfg is None:
        cfg = {}
    
    return GeoP2VCompositeLoss(
        dice_weight=cfg.get("dice_weight", 1.0),
        ce_weight=cfg.get("ce_weight", 0.5),
        lambda_prior=cfg.get("lambda_prior", 0.0),
        lambda_boundary=cfg.get("lambda_boundary", 0.0),
        point_weight=cfg.get("point_weight", 1.0),
        radius=cfg.get("radius", 3),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("GeoP2VCompositeLoss Unit Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build loss
    loss_fn = GeoP2VCompositeLoss(
        lambda_prior=0.05,
        lambda_boundary=0.01,
        point_weight=3.0,
        radius=3,
    ).to(device)
    
    # Test input
    B, C, D, H, W = 2, 2, 32, 32, 32
    Cg = 10  # Geometric feature channels
    
    logits = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    targets = torch.randint(0, C, (B, D, H, W), device=device)
    geo_features = torch.randn(B, Cg, D, H, W, device=device)
    points_list = [
        torch.rand(100, 3, device=device) * 31,
        torch.rand(80, 3, device=device) * 31,
    ]
    
    print(f"Logits: {logits.shape}")
    print(f"Targets: {targets.shape}")
    print(f"Geo features: {geo_features.shape}")
    print(f"Points: {[p.shape for p in points_list]}")
    
    # Forward
    result = loss_fn(
        logits=logits,
        targets=targets,
        points_list=points_list,
        geo_features=geo_features,
    )
    
    print(f"\nLoss breakdown:")
    print(f"  Total: {result['loss'].item():.4f}")
    print(f"  Seg: {result['loss_seg'].item():.4f}")
    print(f"  Geo feat: {result['loss_geo_feat'].item():.4f}")
    print(f"  Boundary: {result['loss_boundary'].item():.4f}")
    
    # Gradient check
    result["loss"].backward()
    print("\nGradient flow: OK")
    
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)
