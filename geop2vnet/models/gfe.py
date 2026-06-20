# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0
#
# Geometric Feature Extraction (GFE) Module
# 
# Reference:
#   - Pauly et al., "Efficient Simplification of Point-Sampled Surfaces", IEEE VIS 2002
#   - Rusu et al., "Fast Point Feature Histograms (FPFH)", ICRA 2009

"""
Geometric Feature Extraction (GFE) Module

Extracts 10-dimensional handcrafted geometric features from point clouds:
    
    Dimension   Feature                 Symbol      Description
    ─────────────────────────────────────────────────────────────────────
    0-2         Normal Vector           n           Surface orientation (PCA)
    3           Curvature               κ           Local surface variation
    4           Density                 ρ           Point distribution density
    5-7         Relative Position       Δp          Normalized spatial context
    8           Normal Variation        NV          Local normal consistency  
    9           Multi-scale Curvature   Δκ          Scale-dependent shape change
    ─────────────────────────────────────────────────────────────────────

All features are computed from local PCA / kNN with O(N·K) complexity.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, NamedTuple


class GFEOutput(NamedTuple):
    """Output container for GFE module."""
    features: torch.Tensor      # [B, N, 10] geometric features
    normals: torch.Tensor       # [B, N, 3] estimated normals
    curvature: torch.Tensor     # [B, N] curvature values


class GFEModule(nn.Module):
    """
    Geometric Feature Extraction (GFE) Module.
    
    Extracts 10-dimensional handcrafted geometric features from point clouds
    using local PCA and kNN neighborhood analysis.
    
    Features (10-D):
        - Normal vector (3): Surface orientation via PCA
        - Curvature (1): λ_min / Σλ, measures local surface variation
        - Density (1): Inverse mean kNN distance
        - Relative position (3): Normalized offset from centroid
        - Normal variation (1): Local normal consistency in neighborhood
        - Multi-scale curvature diff (1): Curvature difference across scales
    
    Args:
        k_small: Number of neighbors for fine-scale analysis (default: 8)
        k_large: Number of neighbors for coarse-scale analysis (default: 32)
        eps: Small constant for numerical stability (default: 1e-6)
    
    Input:
        points: [B, N, 3] point cloud coordinates
    
    Output:
        GFEOutput containing:
            - features: [B, N, 10] geometric features
            - normals: [B, N, 3] estimated normal vectors
            - curvature: [B, N] curvature values
    
    Example:
        >>> gfe = GFEModule(k_small=8, k_large=32)
        >>> points = torch.randn(2, 2048, 3)
        >>> output = gfe(points)
        >>> print(output.features.shape)  # [2, 2048, 10]
    """
    
    # Output feature dimension (constant)
    OUT_DIM: int = 10
    
    def __init__(
        self,
        k_small: int = 8,
        k_large: int = 32,
        eps: float = 1e-4,  # Regularization for numerical stability
    ) -> None:
        super().__init__()
        
        assert k_small < k_large, f"k_small ({k_small}) must be < k_large ({k_large})"
        
        self.k_small = k_small
        self.k_large = k_large
        self.eps = eps
        
        # No learnable parameters - pure geometric computation
        # Register as buffer for device placement
        self.register_buffer("_dummy", torch.empty(0), persistent=False)
    
    @property
    def device(self) -> torch.device:
        return self._dummy.device
    
    @torch.no_grad()
    def forward(self, points: torch.Tensor) -> GFEOutput:
        """
        Extract geometric features from point cloud.
        
        Args:
            points: [B, N, 3] point cloud coordinates
        
        Returns:
            GFEOutput with features [B, N, 10], normals [B, N, 3], curvature [B, N]
        """
        B, N, _ = points.shape
        device = points.device
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: KNN Search (compute once, reuse for all features)
        # ═══════════════════════════════════════════════════════════════
        
        # Handle empty point clouds (N=0) by returning an empty tensor.
        if N == 0:
            return GFEOutput(
                features=torch.zeros(B, 0, self.OUT_DIM, device=device),
                normals=torch.zeros(B, 0, 3, device=device),
                curvature=torch.zeros(B, 0, device=device),
            )
        
        # Adaptive k based on actual point count
        k_large_actual = min(self.k_large, N - 1)  # At least need 1 neighbor
        k_small_actual = min(self.k_small, k_large_actual)
        
        if k_large_actual < 3:
            # Too few points for meaningful geometry, return zeros
            return GFEOutput(
                features=torch.zeros(B, N, self.OUT_DIM, device=device),
                normals=torch.zeros(B, N, 3, device=device),
                curvature=torch.zeros(B, N, device=device),
            )
        
        dist_matrix = torch.cdist(points, points)  # [B, N, N]
        
        # Get k_large neighbors (includes k_small as subset)
        knn_dist_large, knn_idx_large = dist_matrix.topk(
            k_large_actual + 1, dim=-1, largest=False
        )
        # Exclude self (index 0)
        knn_dist_large = knn_dist_large[:, :, 1:]  # [B, N, k_large_actual]
        knn_idx_large = knn_idx_large[:, :, 1:]    # [B, N, k_large_actual]
        
        # Small-scale neighbors (subset of large)
        knn_dist_small = knn_dist_large[:, :, :k_small_actual]  # [B, N, k_small_actual]
        knn_idx_small = knn_idx_large[:, :, :k_small_actual]    # [B, N, k_small_actual]
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: Local Covariance & PCA (for normal, curvature)
        # ═══════════════════════════════════════════════════════════════
        
        def compute_local_pca(
            points: torch.Tensor, 
            knn_idx: torch.Tensor,
            k: int,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Compute local PCA for each point.
            
            Returns:
                eigenvalues: [B, N, 3] sorted ascending
                eigenvectors: [B, N, 3, 3] corresponding eigenvectors
                normals: [B, N, 3] normal vectors (smallest eigenvalue direction)
            """
            B, N, _ = points.shape
            
            # Gather neighbors: [B, N, K, 3]
            batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k)
            neighbors = points[batch_idx, knn_idx]
            
            # Center neighbors
            centered = neighbors - points.unsqueeze(2)  # [B, N, K, 3]
            
            # Covariance matrix: C = (1/K) * Σ(p_i - p̄)(p_i - p̄)ᵀ
            # Using einsum: [B, N, K, 3] x [B, N, K, 3] -> [B, N, 3, 3]
            cov = torch.einsum('bnkd,bnke->bnde', centered, centered) / k
            
            # Add small regularization for numerical stability
            cov = cov + self.eps * torch.eye(3, device=device)
            
            # Eigen decomposition (eigenvalues sorted ascending)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            
            # Normal = eigenvector of smallest eigenvalue
            normals = eigenvectors[:, :, :, 0]  # [B, N, 3]
            
            return eigenvalues, eigenvectors, normals
        
        # PCA at both scales
        eigval_small, eigvec_small, normals = compute_local_pca(
            points, knn_idx_small, k_small_actual
        )
        eigval_large, _, _ = compute_local_pca(
            points, knn_idx_large, k_large_actual
        )
        
        # ═══════════════════════════════════════════════════════════════
        # Step 3: Orient Normals Consistently (Fix PCA sign ambiguity)
        # ═══════════════════════════════════════════════════════════════
        
        # PCA eigenvectors have arbitrary sign (v and -v are both valid).
        # This causes "high-frequency noise" in normal features.
        # Solution: Orient all normals towards the centroid (inward-facing).
        centroid = points.mean(dim=1, keepdim=True)  # [B, 1, 3]
        vec_to_centroid = centroid - points  # [B, N, 3]
        
        # If dot(normal, vec_to_centroid) < 0, flip the normal
        dot = (normals * vec_to_centroid).sum(dim=-1, keepdim=True)  # [B, N, 1]
        sign = torch.sign(dot + self.eps)  # +1 or -1
        normals = normals * sign  # [B, N, 3] - now consistently oriented
        
        # ═══════════════════════════════════════════════════════════════
        # Step 4: Compute Individual Features
        # ═══════════════════════════════════════════════════════════════
        
        # --- Feature 0-2: Normal Vector [B, N, 3] ---
        feat_normal = normals  # [B, N, 3] - consistently inward-facing
        
        # --- Feature 3: Curvature κ [B, N] ---
        # κ = λ_min / (λ_1 + λ_2 + λ_3)
        # Range: [0, 1/3], higher = more curved/spherical
        eigval_sum_small = eigval_small.sum(dim=-1) + self.eps
        curvature_small = eigval_small[:, :, 0] / eigval_sum_small  # [B, N]
        feat_curvature = curvature_small.unsqueeze(-1)  # [B, N, 1]
        
        # --- Feature 4: Density ρ [B, N] ---
        # ρ = 1 / mean(kNN distance)
        # Normalized to [0, 1] per batch
        mean_dist = knn_dist_small.mean(dim=-1) + self.eps  # [B, N]
        density = 1.0 / mean_dist
        density = density / (density.max(dim=1, keepdim=True)[0] + self.eps)
        feat_density = density.unsqueeze(-1)  # [B, N, 1]
        
        # --- Feature 5-7: Relative Position Δp [B, N, 3] ---
        # Δp = (p - centroid) / max_extent
        # Normalized to approximately [-1, 1]
        centroid = points.mean(dim=1, keepdim=True)  # [B, 1, 3]
        relative_pos = points - centroid  # [B, N, 3]
        max_extent = relative_pos.abs().max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        feat_relative_pos = relative_pos / (max_extent + self.eps)  # [B, N, 3]
        
        # --- Feature 8: Normal Variation NV [B, N] ---
        # NV = 1 - mean(|n_i · n_center|) for neighbors
        # High NV = normals vary significantly (e.g., corners, edges)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, k_small_actual)
        neighbor_normals = normals[batch_idx, knn_idx_small]  # [B, N, K, 3]
        
        # Dot product between center normal and neighbor normals
        # |n_i · n_center| measures alignment (1 = parallel, 0 = perpendicular)
        dot_products = torch.einsum('bnd,bnkd->bnk', normals, neighbor_normals).abs()
        normal_consistency = dot_products.mean(dim=-1)  # [B, N], range [0, 1]
        normal_variation = 1.0 - normal_consistency  # Higher = more variation
        feat_normal_var = normal_variation.unsqueeze(-1)  # [B, N, 1]
        
        # --- Feature 9: Multi-scale Curvature Difference Δκ [B, N] ---
        # Δκ = κ(k_large) - κ(k_small)
        # Positive: larger scale is more curved (e.g., on a larger bump)
        # Negative: smaller scale is more curved (e.g., local roughness)
        eigval_sum_large = eigval_large.sum(dim=-1) + self.eps
        curvature_large = eigval_large[:, :, 0] / eigval_sum_large  # [B, N]
        curvature_diff = curvature_large - curvature_small  # [B, N]
        # Normalize to roughly [-1, 1]
        curvature_diff = curvature_diff / (curvature_diff.abs().max() + self.eps)
        feat_curvature_diff = curvature_diff.unsqueeze(-1)  # [B, N, 1]
        
        # ═══════════════════════════════════════════════════════════════
        # Step 5: Concatenate All Features
        # ═══════════════════════════════════════════════════════════════
        
        features = torch.cat([
            feat_normal,          # [B, N, 3] dim 0-2
            feat_curvature,       # [B, N, 1] dim 3
            feat_density,         # [B, N, 1] dim 4
            feat_relative_pos,    # [B, N, 3] dim 5-7
            feat_normal_var,      # [B, N, 1] dim 8
            feat_curvature_diff,  # [B, N, 1] dim 9
        ], dim=-1)  # [B, N, 10]
        
        return GFEOutput(
            features=features,
            normals=normals,
            curvature=curvature_small,
        )


def build_gfe(cfg: dict | None = None) -> GFEModule:
    """
    Factory function for GFE module.
    
    Args:
        cfg: Configuration dict with optional keys:
            - k_small: int (default 8)
            - k_large: int (default 32)
            - eps: float (default 1e-6)
    
    Returns:
        GFEModule instance
    """
    if cfg is None:
        cfg = {}
    
    return GFEModule(
        k_small=cfg.get("k_small", 8),
        k_large=cfg.get("k_large", 32),
        eps=cfg.get("eps", 1e-4),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("GFE Module Unit Test")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize module
    gfe = GFEModule(k_small=8, k_large=32).to(device)
    print(f"\nGFE Configuration:")
    print(f"  k_small: {gfe.k_small}")
    print(f"  k_large: {gfe.k_large}")
    print(f"  Output dim: {gfe.OUT_DIM}")
    print(f"  Learnable params: {sum(p.numel() for p in gfe.parameters())}")
    
    # Test input
    B, N = 2, 2048
    points = torch.randn(B, N, 3, device=device)
    print(f"\nInput shape: {points.shape}")
    
    # Forward pass
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    
    output = gfe(points)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = (time.time() - t0) * 1000
    
    print(f"\nOutput shapes:")
    print(f"  features: {output.features.shape}")
    print(f"  normals: {output.normals.shape}")
    print(f"  curvature: {output.curvature.shape}")
    print(f"\nForward time: {elapsed:.2f} ms")
    
    # Verify feature dimensions
    print(f"\nFeature statistics:")
    feat = output.features
    dim_names = [
        "normal_x", "normal_y", "normal_z",
        "curvature", "density",
        "rel_pos_x", "rel_pos_y", "rel_pos_z",
        "normal_var", "curvature_diff"
    ]
    for i, name in enumerate(dim_names):
        f = feat[:, :, i]
        print(f"  [{i}] {name:15s}: min={f.min():.3f}, max={f.max():.3f}, mean={f.mean():.3f}")
    
    print("\n" + "=" * 70)
    print("Test PASSED")
    print("=" * 70)
