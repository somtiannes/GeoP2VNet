# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
Aneurysm Dataset with Aligned Patch Sampling

Key features:
1. Pre-load all cases into memory for fast training
2. Sample multiple patches per case per epoch (8 positive + 8 negative)
3. Crop image & mask patch from the same ROI
4. Filter point cloud to patch ROI and transform to local [-1, 1] coordinates
5. Random flip augmentation (CTA and point cloud synchronized)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AneurysmDataset(Dataset):
    """
    Dataset for aneurysm segmentation with aligned patch sampling.
    
    Features:
        - Pre-load all cases into memory
        - Multiple patches per case per epoch
        - Aligned image & point cloud patch sampling
        - Random flip augmentation
    
    Args:
        data_dirs: List of directories containing case folders
        pointcloud_dir: Directory containing pre-generated point clouds
        patch_size: Tuple (D, H, W) for patch extraction
        num_points: Number of points to sample from point cloud
        patches_per_case: Number of patches to sample per case per epoch
        pos_patches: Number of positive (aneurysm-centered) patches per case
        neg_patches: Number of negative (random) patches per case
        train: Whether this is training set
        window_level: CT window level (HU)
        window_width: CT window width (HU)
        min_points_in_patch: Minimum points required in patch
        preload: Whether to preload all data into memory
    """
    
    def __init__(
        self,
        data_dirs: List[str] = None,
        pointcloud_dir: str = None,
        patch_size: Tuple[int, int, int] = (96, 96, 96),
        num_points: int = 2048,
        pos_patches: int = 8,
        neg_patches: int = 8,
        train: bool = True,
        window_level: float = 450,
        window_width: float = 900,
        min_points_in_patch: int = 0,  # 不限制，与推理一致
        preload: bool = True,
        # Flat directory structure (alternative to data_dirs)
        image_dir: str = None,
        mask_dir: str = None,
        # Case list filter (for train/val split)
        case_list: List[str] = None,
    ) -> None:
        super().__init__()
        
        self.data_dirs = data_dirs or []
        self.pointcloud_dir = Path(pointcloud_dir) if pointcloud_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.patch_size = np.array(patch_size)
        self.num_points = num_points
        self.pos_patches = pos_patches
        self.neg_patches = neg_patches
        self.patches_per_case = pos_patches + neg_patches
        self.train = train
        self.window_level = window_level
        self.window_width = window_width
        self.min_points_in_patch = min_points_in_patch
        self.preload = preload
        self.case_list = set(case_list) if case_list else None
        
        # Collect cases
        self.cases = self._collect_cases()
        logger.info(f"Found {len(self.cases)} cases")
        
        # Memory cache for preloaded data
        self.cache: Dict[str, Dict] = {}
        
        # Preload all data into memory
        if self.preload:
            self._preload_all_cases()
    
    def _collect_cases(self) -> List[Dict]:
        """Collect all valid cases from data directories."""
        cases = []
        
        # Mode 1: Flat directory structure (image_dir + mask_dir + pointcloud_dir)
        if self.image_dir and self.mask_dir and self.pointcloud_dir:
            cases = self._collect_cases_flat()
        
        # Mode 2: Nested directory structure (data_dirs with case folders)
        else:
            for data_dir in self.data_dirs:
                data_path = Path(data_dir)
                if not data_path.exists():
                    logger.warning(f"Data directory not found: {data_dir}")
                    continue
                
                for case_dir in data_path.iterdir():
                    if not case_dir.is_dir():
                        continue
                    
                    case_id = case_dir.name
                    
                    # Check required files
                    image_path = self._find_file(case_dir, ["image.npy", "image.nii.gz"])
                    mask_path = self._find_file(case_dir, ["aneurysm.npy", "mask.npy", "aneurysm.nii.gz"])
                    pc_path = self.pointcloud_dir / f"{case_id}.npy"
                    
                    if image_path and mask_path and pc_path.exists():
                        cases.append({
                            "case_id": case_id,
                            "image_path": str(image_path),
                            "mask_path": str(mask_path),
                            "pointcloud_path": str(pc_path),
                        })
        
        return cases
    
    def _collect_cases_flat(self) -> List[Dict]:
        """Collect cases from flat directory structure."""
        cases = []
        
        # Find all image files
        for img_path in self.image_dir.glob("*.nii.gz"):
            case_id = img_path.name.replace(".nii.gz", "")
            
            # Filter by case_list if provided
            if self.case_list is not None and case_id not in self.case_list:
                continue
            
            # Find corresponding mask
            mask_path = self._find_file(self.mask_dir, [
                f"{case_id}_mask.nii.gz",
                f"{case_id}.nii.gz",
                f"{case_id}_mask.npy",
            ])
            
            # Find corresponding point cloud
            pc_path = self._find_file(self.pointcloud_dir, [
                f"{case_id}_points.npy",
                f"{case_id}.npy",
            ])
            
            if mask_path and pc_path:
                cases.append({
                    "case_id": case_id,
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "pointcloud_path": str(pc_path),
                })
            else:
                logger.warning(f"Missing files for case {case_id}: mask={mask_path}, pc={pc_path}")
        
        return cases
    
    def _find_file(self, directory: Path, candidates: List[str]) -> Optional[Path]:
        """Find first existing file from candidates."""
        for name in candidates:
            path = directory / name
            if path.exists():
                return path
        return None
    
    def _load_npy_or_nifti(self, path: str) -> np.ndarray:
        """Load numpy or NIfTI file."""
        if path.endswith(".npy"):
            return np.load(path)
        else:
            import nibabel as nib
            return nib.load(path).get_fdata()
    
    def _preload_all_cases(self) -> None:
        """Preload all cases into memory."""
        logger.info("Preloading all cases into memory...")
        
        for case in tqdm(self.cases, desc="Loading cases"):
            case_id = case["case_id"]
            
            image = self._load_npy_or_nifti(case["image_path"])
            mask = self._load_npy_or_nifti(case["mask_path"])
            points = np.load(case["pointcloud_path"])
            
            # Compute aneurysm center once
            aneurysm_center = self._get_aneurysm_center(mask)
            
            self.cache[case_id] = {
                "image": image,
                "mask": mask,
                "points": points,
                "aneurysm_center": aneurysm_center,
            }
        
        logger.info(f"Loaded {len(self.cache)} cases into memory")
    
    def _get_case_data(self, case: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Get case data from cache or load from disk."""
        case_id = case["case_id"]
        
        if case_id in self.cache:
            data = self.cache[case_id]
            return data["image"], data["mask"], data["points"], data["aneurysm_center"]
        else:
            image = self._load_npy_or_nifti(case["image_path"])
            mask = self._load_npy_or_nifti(case["mask_path"])
            points = np.load(case["pointcloud_path"])
            aneurysm_center = self._get_aneurysm_center(mask)
            return image, mask, points, aneurysm_center
    
    def _window_normalize(self, image: np.ndarray) -> np.ndarray:
        """Apply CT windowing and normalize to [0, 1]."""
        lower = self.window_level - self.window_width / 2
        upper = self.window_level + self.window_width / 2
        image = np.clip(image, lower, upper)
        image = (image - lower) / (upper - lower)
        return image.astype(np.float32)
    
    def _get_aneurysm_center(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Get center of mass of aneurysm mask."""
        if mask.sum() < 10:
            return None
        
        center = ndimage.center_of_mass(mask > 0)
        return np.array(center).astype(int)
    
    def _sample_positive_center(
        self,
        image_shape: np.ndarray,
        aneurysm_center: np.ndarray,
    ) -> np.ndarray:
        """Sample patch center around aneurysm."""
        half_patch = self.patch_size // 2
        
        # Valid range for center
        min_center = half_patch
        max_center = image_shape - half_patch - 1
        max_center = np.maximum(max_center, min_center + 1)
        
        # Random offset around aneurysm
        offset_range = self.patch_size // 4
        offset = np.random.randint(-offset_range, offset_range + 1, size=3)
        center = aneurysm_center + offset
        
        # Clamp to valid range
        center = np.clip(center, min_center, max_center)
        
        return center
    
    def _sample_negative_center(
        self,
        image_shape: np.ndarray,
        aneurysm_center: Optional[np.ndarray],
    ) -> np.ndarray:
        """Sample random patch center (avoiding aneurysm if possible)."""
        half_patch = self.patch_size // 2
        
        min_center = half_patch
        max_center = image_shape - half_patch - 1
        max_center = np.maximum(max_center, min_center + 1)
        
        # Random center
        center = np.array([
            np.random.randint(min_center[i], max_center[i] + 1)
            for i in range(3)
        ])
        
        return center
    
    def _crop_patch(
        self,
        volume: np.ndarray,
        center: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop patch from volume."""
        half_patch = self.patch_size // 2
        
        start = center - half_patch
        end = start + self.patch_size
        
        # Handle boundary
        pad_before = np.maximum(-start, 0)
        pad_after = np.maximum(end - np.array(volume.shape), 0)
        
        start_clipped = np.maximum(start, 0)
        end_clipped = np.minimum(end, np.array(volume.shape))
        
        # Crop
        patch = volume[
            start_clipped[0]:end_clipped[0],
            start_clipped[1]:end_clipped[1],
            start_clipped[2]:end_clipped[2],
        ]
        
        # Pad if necessary
        if np.any(pad_before > 0) or np.any(pad_after > 0):
            patch = np.pad(
                patch,
                [(pad_before[i], pad_after[i]) for i in range(3)],
                mode='constant',
                constant_values=0,
            )
        
        return patch, start, end
    
    def _filter_points_to_patch(
        self,
        points: np.ndarray,
        start: np.ndarray,
        end: np.ndarray,
    ) -> np.ndarray:
        """Filter points to only those within patch ROI."""
        in_roi = (
            (points[:, 0] >= start[0]) & (points[:, 0] < end[0]) &
            (points[:, 1] >= start[1]) & (points[:, 1] < end[1]) &
            (points[:, 2] >= start[2]) & (points[:, 2] < end[2])
        )
        return points[in_roi]
    
    def _transform_points_to_local(
        self,
        points: np.ndarray,
        start: np.ndarray,
    ) -> np.ndarray:
        """Transform points to patch local coordinates and normalize to [-1, 1]."""
        points_local = points.copy().astype(np.float32)
        points_local[:, 0] -= start[0]
        points_local[:, 1] -= start[1]
        points_local[:, 2] -= start[2]
        
        # Normalize to [-1, 1]
        points_norm = points_local.copy()
        points_norm[:, 0] = (points_local[:, 0] / (self.patch_size[0] - 1)) * 2 - 1
        points_norm[:, 1] = (points_local[:, 1] / (self.patch_size[1] - 1)) * 2 - 1
        points_norm[:, 2] = (points_local[:, 2] / (self.patch_size[2] - 1)) * 2 - 1
        
        points_norm = np.clip(points_norm, -1, 1)
        
        return points_norm
    
    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """
        Sample points: use all available points, optionally downsample if limit set.
        No replacement, no padding - variable length output.
        
        Args:
            points: [M, 3] input points
            
        Returns:
            sampled: [N, 3] where N = M (or min(M, num_points) if limit set)
        """
        M = points.shape[0]
        
        if M == 0:
            # 返回真正的空 tensor [0, 3]，让模型对该样本退化为 CTA-only
            return np.zeros((0, 3), dtype=np.float32)
        
        # If num_points is set and we have more, downsample
        if self.num_points is not None and self.num_points > 0 and M > self.num_points:
            idx = np.random.choice(M, self.num_points, replace=False)
            return points[idx]
        else:
            # Use all points as-is (no limit, no repetition, no padding)
            return points
    
    def _random_flip(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Random flip augmentation.
        
        IMPORTANT: CTA and point cloud must flip together to maintain alignment!
        """
        # Make copies to avoid modifying cached data
        image = image.copy()
        mask = mask.copy()
        points = points.copy()
        
        # Random flip z (up-down)
        if random.random() < 0.5:
            image = image[::-1, :, :].copy()
            mask = mask[::-1, :, :].copy()
            points[:, 0] *= -1  # z flip
        
        # Random flip y (front-back)
        if random.random() < 0.5:
            image = image[:, ::-1, :].copy()
            mask = mask[:, ::-1, :].copy()
            points[:, 1] *= -1  # y flip
        
        # Random flip x (left-right)
        if random.random() < 0.5:
            image = image[:, :, ::-1].copy()
            mask = mask[:, :, ::-1].copy()
            points[:, 2] *= -1  # x flip
        
        return image, mask, points
    
    def __len__(self) -> int:
        """Each case samples multiple patches per epoch."""
        return len(self.cases) * self.patches_per_case
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # ═══════════════════════════════════════════════════════════════
        # Step 0: Determine case and patch type
        # ═══════════════════════════════════════════════════════════════
        case_idx = idx // self.patches_per_case
        patch_idx = idx % self.patches_per_case
        is_positive = patch_idx < self.pos_patches
        
        case = self.cases[case_idx]
        
        # ═══════════════════════════════════════════════════════════════
        # Step 1: Load data (from cache or disk)
        # ═══════════════════════════════════════════════════════════════
        image_full, mask_full, points_full, aneurysm_center = self._get_case_data(case)
        image_shape = np.array(image_full.shape)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 2: Sample patch center
        # ═══════════════════════════════════════════════════════════════
        max_attempts = 10
        
        for attempt in range(max_attempts):
            if is_positive and aneurysm_center is not None:
                center = self._sample_positive_center(image_shape, aneurysm_center)
            else:
                center = self._sample_negative_center(image_shape, aneurysm_center)
            
            # ═══════════════════════════════════════════════════════════
            # Step 3: Crop image & mask patch
            # ═══════════════════════════════════════════════════════════
            image_patch, start, end = self._crop_patch(image_full, center)
            mask_patch, _, _ = self._crop_patch(mask_full, center)
            
            # ═══════════════════════════════════════════════════════════
            # Step 4: Filter points to patch ROI
            # ═══════════════════════════════════════════════════════════
            points_in_patch = self._filter_points_to_patch(points_full, start, end)
            
            if len(points_in_patch) >= self.min_points_in_patch or attempt == max_attempts - 1:
                break
        
        # ═══════════════════════════════════════════════════════════════
        # Step 5: Transform points to local coordinates [-1, 1]
        # ═══════════════════════════════════════════════════════════════
        if len(points_in_patch) > 0:
            points_local = self._transform_points_to_local(points_in_patch, start)
        else:
            points_local = np.zeros((0, 3), dtype=np.float32)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 6: Sample points (variable length, no replacement)
        # ═══════════════════════════════════════════════════════════════
        points_sampled = self._sample_points(points_local)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 7: Preprocess image
        # ═══════════════════════════════════════════════════════════════
        image_patch = self._window_normalize(image_patch)
        
        # ═══════════════════════════════════════════════════════════════
        # Step 8: Data augmentation (flip)
        # ═══════════════════════════════════════════════════════════════
        if self.train:
            image_patch, mask_patch, points_sampled = self._random_flip(
                image_patch, mask_patch, points_sampled
            )
        
        # ═══════════════════════════════════════════════════════════════
        # Step 9: Convert to tensors
        # ═══════════════════════════════════════════════════════════════
        image = torch.from_numpy(np.ascontiguousarray(image_patch)).float().unsqueeze(0)
        mask = torch.from_numpy(np.ascontiguousarray(mask_patch)).long()
        points = torch.from_numpy(np.ascontiguousarray(points_sampled)).float()
        
        return {
            "image": image,
            "mask": mask,
            "points": points,  # Variable length [N, 3]
            "num_points": points.shape[0],
            "case_id": case["case_id"],
            "is_positive": is_positive,
        }


def collate_variable_points(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-length point clouds.
    
    Images and masks are stacked normally.
    Points are kept as a list of tensors (variable length per sample).
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    masks = torch.stack([b["mask"] for b in batch], dim=0)
    points_list = [b["points"] for b in batch]  # List of [N_i, 3]
    num_points = [b["num_points"] for b in batch]
    
    return {
        "image": images,        # [B, 1, D, H, W]
        "mask": masks,          # [B, D, H, W]
        "points": points_list,  # List of [N_i, 3] tensors
        "num_points": num_points,
        "case_id": [b["case_id"] for b in batch],
        "is_positive": [b["is_positive"] for b in batch],
    }


def build_dataloader(
    cfg: dict,
    train: bool = True,
) -> DataLoader:
    """Build dataloader from config."""
    dataset = AneurysmDataset(
        data_dirs=cfg.get("data_dirs", []),
        pointcloud_dir=cfg.get("pointcloud_dir", "data/pointclouds"),
        patch_size=tuple(cfg.get("patch_size", (96, 96, 96))),
        num_points=cfg.get("num_points", 2048),
        pos_patches=cfg.get("pos_patches", 8),
        neg_patches=cfg.get("neg_patches", 8),
        train=train,
        window_level=cfg.get("window_level", 450),
        window_width=cfg.get("window_width", 900),
        min_points_in_patch=cfg.get("min_points_in_patch", 0),
        preload=cfg.get("preload", True),
        # Flat directory structure
        image_dir=cfg.get("image_dir"),
        mask_dir=cfg.get("mask_dir"),
    )
    
    return DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 2),
        shuffle=train,
        num_workers=cfg.get("num_workers", 0) if cfg.get("preload", True) else cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=train,
        collate_fn=collate_variable_points,
    )
