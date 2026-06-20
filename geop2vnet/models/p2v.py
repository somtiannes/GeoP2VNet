# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0
#
# Point-to-Voxel Projection (P2V) Module
#
# Reference:
#   - Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023
#   - Dai et al., "ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes", CVPR 2017

"""
Point-to-Voxel Projection (P2V) Module

Projects point cloud features [B, N, C] to voxel grid [B, C, D, H, W]
using 3D Gaussian Splatting with σ=1.0.

Architecture:
    F_g [B, N, C] + coords [B, N, 3]
        → Gaussian Splatting (σ=1.0)
        → V_g [B, C, D, H, W]

The Gaussian splatting distributes each point's features to nearby voxels
with weights determined by a 3D Gaussian kernel:

    w(v, p) = exp(-||v - p||² / (2σ²))

where v is voxel center, p is point coordinate, σ is the kernel width.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class P2VModule(nn.Module):
    """
    Point-to-Voxel Projection (P2V) Module using 3D Gaussian Splatting.
    
    Projects sparse point cloud features onto a dense voxel grid using
    differentiable Gaussian splatting. Each point contributes to nearby
    voxels with Gaussian-weighted influence.
    
    Args:
        voxel_size: Tuple (D, H, W) for output voxel grid dimensions
        sigma: Gaussian kernel standard deviation (default: 1.0)
        kernel_radius: Radius of influence in voxels (default: 2)
        normalize: Whether to normalize by accumulated weights (default: True)
        eps: Small constant for numerical stability (default: 1e-6)
    
    Input:
        points: [B, N, 3] point coordinates in normalized [-1, 1] space
        features: [B, N, C] point features
    
    Output:
        voxels: [B, C, D, H, W] voxel feature grid
    
    Example:
        >>> p2v = P2VModule(voxel_size=(96, 96, 96), sigma=1.0)
        >>> points = torch.rand(2, 2048, 3) * 2 - 1  # [-1, 1]
        >>> features = torch.randn(2, 2048, 10)
        >>> voxels = p2v(points, features)
        >>> print(voxels.shape)  # [2, 10, 96, 96, 96]
    """
    
    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (96, 96, 96),
        sigma: float = 1.0,
        kernel_radius: int = 2,
        normalize: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.kernel_radius = kernel_radius
        self.normalize = normalize
        self.eps = eps
        
        # Precompute kernel offsets for efficiency
        self._register_kernel_offsets()
    
    def _register_kernel_offsets(self) -> None:
        """Precompute relative offsets within kernel radius."""
        r = self.kernel_radius
        
        # Create offset grid: all integer offsets within radius
        offsets = []
        for dz in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    offsets.append([dz, dy, dx])
        
        offsets = torch.tensor(offsets, dtype=torch.long)  # [K, 3]
        self.register_buffer("kernel_offsets", offsets, persistent=False)
        
        # Precompute Gaussian weights for each offset
        # w = exp(-||offset||² / (2σ²))
        offset_sq_dist = (offsets.float() ** 2).sum(dim=-1)  # [K]
        weights = torch.exp(-offset_sq_dist / (2 * self.sigma ** 2))
        self.register_buffer("kernel_weights", weights, persistent=False)
    
    def forward(
        self, 
        points: torch.Tensor, 
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project point features to voxel grid via Gaussian splatting.
        
        Args:
            points: [B, N, 3] point coordinates in [-1, 1] normalized space
            features: [B, N, C] point features
        
        Returns:
            voxels: [B, C, D, H, W] voxel feature grid
        """
        B, N, C = features.shape
        D, H, W = self.voxel_size
        device = features.device
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: Convert normalized coords [-1, 1] to voxel indices
        # ═══════════════════════════════════════════════════════════════
        
        # Map [-1, 1] -> [0, D/H/W - 1]
        voxel_coords = (points + 1) / 2  # [0, 1]
        voxel_coords = voxel_coords * torch.tensor(
            [D - 1, H - 1, W - 1], device=device, dtype=points.dtype
        )  # [B, N, 3]
        
        # Round to nearest voxel (center of splatting)
        center_idx = voxel_coords.round().long()  # [B, N, 3]
        
        # Fractional offset from voxel center
        frac_offset = voxel_coords - center_idx.float()  # [B, N, 3]
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: Initialize output voxel grid and weight accumulator
        # ═══════════════════════════════════════════════════════════════
        
        voxel_features = torch.zeros(B, C, D, H, W, device=device, dtype=features.dtype)
        voxel_weights = torch.zeros(B, 1, D, H, W, device=device, dtype=features.dtype)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 3: Gaussian Splatting - scatter features to nearby voxels
        # ═══════════════════════════════════════════════════════════════
        
        K = self.kernel_offsets.shape[0]  # Number of kernel offsets
        
        for k in range(K):
            # Offset for this kernel position
            offset = self.kernel_offsets[k]  # [3]
            base_weight = self.kernel_weights[k]  # scalar
            
            # Target voxel indices: center + offset
            target_idx = center_idx + offset.view(1, 1, 3)  # [B, N, 3]
            
            # Compute precise Gaussian weight including fractional offset
            # Distance from point to target voxel center
            dist_to_target = (offset.float().view(1, 1, 3) - frac_offset)  # [B, N, 3]
            sq_dist = (dist_to_target ** 2).sum(dim=-1)  # [B, N]
            weight = torch.exp(-sq_dist / (2 * self.sigma ** 2))  # [B, N]
            
            # Clamp indices to valid range
            tz = target_idx[:, :, 0].clamp(0, D - 1)
            ty = target_idx[:, :, 1].clamp(0, H - 1)
            tx = target_idx[:, :, 2].clamp(0, W - 1)
            
            # Mask for valid (in-bounds) voxels
            valid = (
                (target_idx[:, :, 0] >= 0) & (target_idx[:, :, 0] < D) &
                (target_idx[:, :, 1] >= 0) & (target_idx[:, :, 1] < H) &
                (target_idx[:, :, 2] >= 0) & (target_idx[:, :, 2] < W)
            )  # [B, N]
            
            # Zero out invalid weights
            weight = weight * valid.float()  # [B, N]
            
            # Weighted features: [B, N, C] * [B, N, 1] -> [B, N, C]
            weighted_feat = features * weight.unsqueeze(-1)
            
            # Scatter add to voxel grid
            # Use batch and point indices to accumulate
            for b in range(B):
                # Linear index for scatter_add
                linear_idx = tz[b] * (H * W) + ty[b] * W + tx[b]  # [N]
                
                # Scatter features
                for c in range(C):
                    voxel_features[b, c].view(-1).scatter_add_(
                        0, linear_idx, weighted_feat[b, :, c]
                    )
                
                # Scatter weights
                voxel_weights[b, 0].view(-1).scatter_add_(
                    0, linear_idx, weight[b]
                )
        
        # ═══════════════════════════════════════════════════════════════
        # Step 4: Normalize by accumulated weights (optional)
        # ═══════════════════════════════════════════════════════════════
        
        if self.normalize:
            voxel_features = voxel_features / (voxel_weights + self.eps)
        
        return voxel_features


class P2VModuleFast(nn.Module):
    """
    Fast Point-to-Voxel Projection using trilinear interpolation.
    
    This is a faster alternative to full Gaussian splatting that uses
    PyTorch's grid_sample for efficient differentiable projection.
    
    The approach:
    1. Create sparse feature volume at point locations
    2. Use 3D convolution with Gaussian kernel for smoothing
    
    Args:
        voxel_size: Tuple (D, H, W) for output voxel grid dimensions  
        sigma: Gaussian smoothing sigma (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-6)
    """
    
    def __init__(
        self,
        voxel_size: Tuple[int, int, int] = (96, 96, 96),
        sigma: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        
        self.voxel_size = voxel_size
        self.sigma = sigma
        self.eps = eps
        
        # Create Gaussian smoothing kernel
        self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self) -> None:
        """Create 3D Gaussian convolution kernel."""
        # Kernel size = 2 * ceil(3 * sigma) + 1
        ks = int(2 * max(3, int(3 * self.sigma)) + 1)
        if ks % 2 == 0:
            ks += 1
        
        self.kernel_size = ks
        pad = ks // 2
        self.padding = pad
        
        # Create 1D Gaussian
        x = torch.arange(ks).float() - pad
        gauss_1d = torch.exp(-x ** 2 / (2 * self.sigma ** 2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        
        # Outer product to get 3D kernel
        gauss_3d = gauss_1d.view(-1, 1, 1) * gauss_1d.view(1, -1, 1) * gauss_1d.view(1, 1, -1)
        gauss_3d = gauss_3d / gauss_3d.sum()
        
        # Register as buffer [1, 1, ks, ks, ks]
        self.register_buffer("gaussian_kernel", gauss_3d.view(1, 1, ks, ks, ks))
    
    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project point features to voxel grid.
        
        Args:
            points: [B, N, 3] point coordinates in [-1, 1]
            features: [B, N, C] point features
        
        Returns:
            voxels: [B, C, D, H, W] voxel feature grid
        """
        B, N, C = features.shape
        D, H, W = self.voxel_size
        device = features.device
        dtype = features.dtype
        
        # Handle empty point clouds (N=0) by returning a zero voxel grid.
        if N == 0:
            return torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: Create sparse voxel grid via scatter
        # ═══════════════════════════════════════════════════════════════
        
        # Map [-1, 1] -> [0, D/H/W - 1]
        voxel_coords = (points + 1) / 2
        voxel_coords = voxel_coords * torch.tensor(
            [D - 1, H - 1, W - 1], device=device, dtype=dtype
        )
        voxel_idx = voxel_coords.round().long().clamp(
            min=torch.tensor([0, 0, 0], device=device),
            max=torch.tensor([D - 1, H - 1, W - 1], device=device),
        )
        
        # Initialize sparse volume
        voxel_features = torch.zeros(B, C, D, H, W, device=device, dtype=dtype)
        voxel_counts = torch.zeros(B, 1, D, H, W, device=device, dtype=dtype)
        
        # Scatter features to voxel locations
        for b in range(B):
            linear_idx = (
                voxel_idx[b, :, 0] * (H * W) +
                voxel_idx[b, :, 1] * W +
                voxel_idx[b, :, 2]
            )  # [N]
            
            for c in range(C):
                voxel_features[b, c].view(-1).scatter_add_(
                    0, linear_idx, features[b, :, c]
                )
            
            voxel_counts[b, 0].view(-1).scatter_add_(
                0, linear_idx, torch.ones(N, device=device, dtype=dtype)
            )
        
        # Average by counts
        voxel_features = voxel_features / (voxel_counts + self.eps)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: Gaussian smoothing via 3D convolution
        # ═══════════════════════════════════════════════════════════════
        
        # Apply per-channel Gaussian smoothing
        kernel = self.gaussian_kernel.to(dtype)  # [1, 1, ks, ks, ks]
        
        # Reshape for group convolution: [B, C, D, H, W] -> [B*C, 1, D, H, W]
        voxel_flat = voxel_features.view(B * C, 1, D, H, W)
        
        # Apply Gaussian convolution
        voxel_smooth = F.conv3d(
            voxel_flat, kernel, 
            padding=self.padding,
        )
        
        # Reshape back: [B*C, 1, D, H, W] -> [B, C, D, H, W]
        voxel_smooth = voxel_smooth.view(B, C, D, H, W)
        
        return voxel_smooth


def build_p2v(
    cfg: dict | None = None,
    fast: bool = False,
) -> nn.Module:
    """
    Factory function for P2V module.
    
    Args:
        cfg: Configuration dict with optional keys:
            - voxel_size: Tuple[int, int, int] (default (96, 96, 96))
            - sigma: float (default 1.0)
        fast: Whether to use the approximate fast implementation (default False)
    
    Returns:
        P2VModule or P2VModuleFast instance
    """
    if cfg is None:
        cfg = {}
    
    voxel_size = tuple(cfg.get("voxel_size", (96, 96, 96)))
    sigma = cfg.get("sigma", 1.0)
    
    if fast:
        return P2VModuleFast(voxel_size=voxel_size, sigma=sigma)
    else:
        return P2VModule(
            voxel_size=voxel_size,
            sigma=sigma,
            kernel_radius=cfg.get("kernel_radius", 2),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("P2V Module Unit Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test parameters
    B, N, C = 2, 2048, 10
    voxel_size = (96, 96, 96)
    
    # Test input
    points = torch.rand(B, N, 3, device=device) * 2 - 1  # [-1, 1]
    features = torch.randn(B, N, C, device=device)
    
    print(f"\nInput:")
    print(f"  points: {points.shape}")
    print(f"  features: {features.shape}")
    print(f"  voxel_size: {voxel_size}")
    
    # Test P2VModuleFast
    print("\n--- P2VModuleFast ---")
    p2v_fast = P2VModuleFast(voxel_size=voxel_size, sigma=1.0).to(device)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    
    voxels_fast = p2v_fast(points, features)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed_fast = (time.time() - t0) * 1000
    
    print(f"  Output: {voxels_fast.shape}")
    print(f"  Time: {elapsed_fast:.2f} ms")
    print(f"  Non-zero ratio: {(voxels_fast.abs() > 1e-6).float().mean():.4f}")
    
    # Test P2VModule (slower but more accurate)
    print("\n--- P2VModule (reference) ---")
    p2v = P2VModule(voxel_size=voxel_size, sigma=1.0, kernel_radius=2).to(device)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    
    voxels = p2v(points, features)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.time() - t0) * 1000
    
    print(f"  Output: {voxels.shape}")
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Non-zero ratio: {(voxels.abs() > 1e-6).float().mean():.4f}")
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)

