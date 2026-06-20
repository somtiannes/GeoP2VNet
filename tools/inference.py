#!/usr/bin/env python
# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
GeoP2VNet Inference on Single Case

Usage:
    python tools/inference.py \
        --config configs/default.yaml \
        --checkpoint logs/best.pth \
        --image /path/to/image.nii.gz \
        --pointcloud /path/to/points.npy \
        --output /path/to/output.nii.gz
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from geop2vnet.models import build_model
from geop2vnet.utils import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="GeoP2VNet Inference")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Input CTA image (.nii.gz or .npy)")
    parser.add_argument("--pointcloud", type=str, required=True, help="Input point cloud (.npy)")
    parser.add_argument("--output", type=str, required=True, help="Output segmentation")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--threshold", type=float, default=0.3, help="Probability threshold (0.3-0.5)")
    parser.add_argument("--window_level", type=float, default=450)
    parser.add_argument("--window_width", type=float, default=900)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_image(path: str) -> tuple:
    """Load image and return array + affine."""
    if path.endswith(".npy"):
        return np.load(path), np.eye(4)
    else:
        nii = nib.load(path)
        return nii.get_fdata(), nii.affine


def window_normalize(image: np.ndarray, level: float, width: float) -> np.ndarray:
    """Apply CT windowing and normalize to [0, 1]."""
    lower = level - width / 2
    upper = level + width / 2
    image = np.clip(image, lower, upper)
    image = (image - lower) / (upper - lower)
    return image.astype(np.float32)


def normalize_points(points: np.ndarray, patch_size: tuple) -> np.ndarray:
    """Normalize points to [-1, 1] based on patch size."""
    points = points.copy().astype(np.float32)
    for i in range(3):
        points[:, i] = (points[:, i] / (patch_size[i] - 1)) * 2 - 1
    return np.clip(points, -1, 1)


def sample_points(points: np.ndarray, num_points: int) -> np.ndarray:
    """
    Sample point clouds consistently with training.
    - If num_points is None, return all points.
    - If the point count exceeds num_points, downsample.
    - If the point count is below num_points, return all points without padding or duplication.
    """
    N = points.shape[0]
    if num_points is None or num_points <= 0:
        return points
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
        return points[idx]
    else:
        # No padding; return all points as in training.
        return points


def get_patch_coords(volume_shape: tuple, patch_size: tuple, stride: tuple) -> list:
    """Generate all patch coordinates for sliding window."""
    D, H, W = volume_shape
    Pd, Ph, Pw = patch_size
    Sd, Sh, Sw = stride
    
    coords = []
    for d in range(0, max(1, D - Pd + 1), Sd):
        for h in range(0, max(1, H - Ph + 1), Sh):
            for w in range(0, max(1, W - Pw + 1), Sw):
                coords.append((d, h, w))
    
    # Ensure last patches cover the end
    if coords and coords[-1][0] + Pd < D:
        for h in range(0, max(1, H - Ph + 1), Sh):
            for w in range(0, max(1, W - Pw + 1), Sw):
                coords.append((D - Pd, h, w))
    
    return list(set(coords))  # Remove duplicates


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    image: np.ndarray,
    points: np.ndarray,
    patch_size: tuple = (96, 96, 96),
    stride: tuple = (48, 48, 48),
    num_points: int = 2048,
    batch_size: int = 4,
    threshold: float = 0.30,
) -> np.ndarray:
    """
    Sliding window inference with probability accumulation.
    
    DLIA-style probability accumulation:
    - Accumulate softmax probabilities instead of argmax labels.
    - Overlapping regions accumulate probabilities.
    - Apply a threshold to obtain the binary mask.
    
    Args:
        model: GeoP2VNet model
        image: [D, H, W] preprocessed CTA image
        points: [N, 3] point cloud (voxel coordinates)
        patch_size: Patch size for inference
        stride: Stride for sliding window
        num_points: Number of points per patch
        batch_size: Batch size for efficient inference
        threshold: Probability threshold (default: 0.30, overlapped regions receive implicit voting)
    
    Returns:
        segmentation: [D, H, W] binary mask
    """
    model.eval()
    device = next(model.parameters()).device
    
    D, H, W = image.shape
    Pd, Ph, Pw = patch_size
    
    # Accumulated probability map without count normalization.
    # Overlapping regions accumulate probability, acting as implicit voting.
    seg_prob = torch.zeros((D, H, W), dtype=torch.float32, device=device)
    
    # Get all patch coordinates
    coords = get_patch_coords((D, H, W), patch_size, stride)
    print(f"Total patches: {len(coords)}")
    
    # Process in batches
    batch_imgs, batch_pts, batch_coords = [], [], []
    
    from tqdm import tqdm
    for coord in tqdm(coords, desc="Inference"):
        d, h, w = coord
        
        # Extract image patch
        img_patch = image[d:d+Pd, h:h+Ph, w:w+Pw]
        
        # Pad if at boundary
        if img_patch.shape != (Pd, Ph, Pw):
            padded = np.zeros((Pd, Ph, Pw), dtype=np.float32)
            ad, ah, aw = img_patch.shape
            padded[:ad, :ah, :aw] = img_patch
            img_patch = padded
        
        # Filter points to patch ROI
        in_patch = (
            (points[:, 0] >= d) & (points[:, 0] < d + Pd) &
            (points[:, 1] >= h) & (points[:, 1] < h + Ph) &
            (points[:, 2] >= w) & (points[:, 2] < w + Pw)
        )
        pts_patch = points[in_patch].copy()
        
        if len(pts_patch) > 0:
            # Transform to local coordinates and normalize
            pts_patch[:, 0] -= d
            pts_patch[:, 1] -= h
            pts_patch[:, 2] -= w
            pts_patch = normalize_points(pts_patch, patch_size)
        else:
            pts_patch = np.zeros((1, 3), dtype=np.float32)
        
        pts_patch = sample_points(pts_patch, num_points)
        
        batch_imgs.append(img_patch)
        batch_pts.append(pts_patch)
        batch_coords.append(coord)
        
        # Process batch
        if len(batch_imgs) == batch_size or coord == coords[-1]:
            # Stack images to GPU
            img_tensor = torch.from_numpy(np.stack(batch_imgs)).unsqueeze(1).float().to(device)
            
            # Variable-length points are passed as a list, matching training.
            pts_list = [torch.from_numpy(p).float().to(device) for p in batch_pts]
            
            # Forward pass
            out = model(img_tensor, pts_list)
            # Use softmax to obtain foreground probability.
            pred = F.softmax(out.logits, dim=1)  # [B, C, D, H, W]
            fg_prob = pred[:, 1]  # [B, D, H, W] foreground probability
            
            # Accumulate probabilities over overlapping regions.
            for i, (cd, ch, cw) in enumerate(batch_coords):
                ad = min(Pd, D - cd)
                ah = min(Ph, H - ch)
                aw = min(Pw, W - cw)
                seg_prob[cd:cd+ad, ch:ch+ah, cw:cw+aw] += fg_prob[i, :ad, :ah, :aw]
            
            # Clear batch
            batch_imgs, batch_pts, batch_coords = [], [], []
    
    torch.cuda.empty_cache()
    
    # Threshold probabilities to obtain the binary mask.
    # With accumulation, a region covered N times has an effective threshold of threshold/N.
    # The default 0.30 threshold works well when stride is half the patch size.
    seg = (seg_prob >= threshold).cpu().numpy().astype(np.uint8)
    
    return seg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(cfg.get("seed", 37))
    
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")
    print(f"Point cloud: {args.pointcloud}")
    print(f"Threshold: {args.threshold}")
    
    # Build model
    print("Building model...")
    model = build_model(cfg.get("model")).cuda()
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    # Remove DDP prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}, dice={ckpt.get('dice', 'N/A')}")
    
    # Load data
    print("Loading data...")
    image, affine = load_image(args.image)
    points = np.load(args.pointcloud)
    
    print(f"Image shape: {image.shape}")
    print(f"Points shape: {points.shape}")
    
    # Preprocess
    image = window_normalize(image, args.window_level, args.window_width)
    
    # Inference
    import time
    t0 = time.time()
    print("Running sliding window inference...")
    
    patch_size = tuple(cfg["data"].get("patch_size", [96, 96, 96]))
    num_points = cfg["data"].get("num_points") or 2048  # Default 2048 if None
    stride = tuple(s // 2 for s in patch_size)
    
    segmentation = inference(
        model, image, points,
        patch_size=patch_size,
        stride=stride,
        num_points=num_points,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )
    
    t1 = time.time()
    print(f"Inference time: {t1-t0:.2f}s")
    print(f"Foreground voxels: {segmentation.sum()}")
    
    # Save
    print(f"Saving to {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    if args.output.endswith(".npy"):
        np.save(args.output, segmentation)
    else:
        nib.save(nib.Nifti1Image(segmentation, affine), args.output)
    
    print("Done!")


if __name__ == "__main__":
    main()

