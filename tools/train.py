#!/usr/bin/env python
# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
GeoP2VNet Training Script (Single GPU / DDP)

Usage:
    # Single GPU
    python tools/train.py --config configs/default.yaml
    
    # Multi-GPU DDP
    torchrun --nproc_per_node=2 tools/train.py --config configs/default.yaml
    
    # With wandb
    python tools/train.py --config configs/default.yaml --wandb
"""

from __future__ import annotations

import gc
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from scipy import ndimage
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

from geop2vnet.models import build_model
from geop2vnet.losses import build_loss, build_geo_loss
from geop2vnet.data import AneurysmDataset
from geop2vnet.data.dataset import collate_variable_points
from geop2vnet.utils import setup_logger, set_seed, AverageMeter

# Optional wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="GeoP2VNet Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES for single-process training")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="GeoP2VNet")
    parser.add_argument("--wandb_run", type=str, default=None, help="Wandb run name")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# DDP Helpers
# ═══════════════════════════════════════════════════════════════════════════

def is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def setup_ddp() -> int:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def is_main() -> bool:
    return not is_ddp() or dist.get_rank() == 0


def world_size() -> int:
    return dist.get_world_size() if is_ddp() else 1


def reduce_value(value: float) -> float:
    if not is_ddp():
        return value
    tensor = torch.tensor(value, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item() / world_size()


# ═══════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════

def load_case_list(list_file: str) -> List[str]:
    """Load case list from txt file (one case per line)."""
    cases = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                cases.append(line)
    return cases


def build_dataloader(cfg: dict, train: bool = True):
    """Build dataloader with case list support."""
    # Use shared directories
    image_dir = cfg.get("image_dir")
    mask_dir = cfg.get("mask_dir")
    pointcloud_dir = cfg.get("pointcloud_dir")
    
    # Load case list from txt file
    case_list = None
    list_file = cfg.get("train_list") if train else cfg.get("val_list")
    if list_file and os.path.exists(list_file):
        case_list = load_case_list(list_file)
        logging.info(f"Loaded {len(case_list)} {'train' if train else 'val'} cases from {list_file}")
    
    dataset = AneurysmDataset(
        data_dirs=cfg.get("data_dirs", []),
        pointcloud_dir=pointcloud_dir,
        patch_size=tuple(cfg.get("patch_size", (96, 96, 96))),
        num_points=cfg.get("num_points", 2048),
        pos_patches=cfg.get("pos_patches", 8),
        neg_patches=cfg.get("neg_patches", 8),
        train=train,
        window_level=cfg.get("window_level", 450),
        window_width=cfg.get("window_width", 900),
        min_points_in_patch=cfg.get("min_points_in_patch", 100),
        preload=cfg.get("preload", True),
        image_dir=image_dir,
        mask_dir=mask_dir,
        case_list=case_list,
    )
    
    sampler = DistributedSampler(dataset, shuffle=train) if is_ddp() else None
    
    # Use fewer workers for validation (HD95 metric is memory-intensive)
    if cfg.get("preload", True):
        num_workers = 0
    else:
        num_workers = cfg.get("num_workers", 4) if train else min(4, cfg.get("num_workers", 4))
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.get("batch_size", 2),
        shuffle=(train and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        collate_fn=collate_variable_points,
    )
    return loader, sampler


# ═══════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, sampler, criterion, optimizer, scaler, epoch, cfg):
    """Single epoch training loop."""
    model.train()
    if sampler:
        sampler.set_epoch(epoch)
    
    # Check if model uses geometry branch
    use_geo = getattr(model.module if hasattr(model, 'module') else model, 'use_geometry', True)
    
    # Gradient accumulation
    grad_accum = cfg["train"].get("grad_accum", 1)
    
    loss_m = AverageMeter()
    geo_loss_m = AverageMeter()  # Track geo feature loss
    
    # Point-cloud statistics monitor using thresholds from config.
    MIN_POINTS_FOR_GEO = cfg["model"].get("min_points_for_geo", 128)
    total_samples = 0
    samples_with_geo = 0
    pos_samples = 0
    pos_with_geo = 0
    all_point_counts = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main())
    
    optimizer.zero_grad(set_to_none=True)  # Zero grad at epoch start
    
    for step, batch in enumerate(pbar):
        image = batch["image"].cuda(non_blocking=True)
        mask = batch["mask"].cuda(non_blocking=True)
        # Points is a list of tensors (variable length per sample)
        points = [p.cuda(non_blocking=True) for p in batch["points"]] if use_geo else None
        
        # Track point-cloud distribution.
        if use_geo and points is not None:
            for i, pts in enumerate(points):
                n_pts = pts.shape[0]
                all_point_counts.append(n_pts)
                total_samples += 1
                
                has_geo = n_pts >= MIN_POINTS_FOR_GEO
                if has_geo:
                    samples_with_geo += 1
                
                # Positive sample if the mask has foreground voxels.
                is_pos = mask[i].sum() > 0
                if is_pos:
                    pos_samples += 1
                    if has_geo:
                        pos_with_geo += 1
        
        with autocast("cuda", enabled=cfg["train"]["amp"]):
            out = model(image, points)
            
            # Debug: print shapes on first step
            if step == 0 and is_main():
                print(f"[DEBUG] logits: {out.logits.shape}, geo_features: {out.geo_features.shape}")
            
            # GeoP2VCompositeLoss: Pass points, geo_features, and geo_prior_logits.
            loss_dict = criterion(
                logits=out.logits,
                targets=mask,
                points_list=points,
                geo_features=out.geo_features,
                geo_prior_logits=out.geo_prior_logits,
            )
            loss = loss_dict["loss"] / grad_accum  # Scale loss for accumulation
        
        scaler.scale(loss).backward()
        
        # Update weights every grad_accum steps
        if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
            if cfg["train"].get("grad_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        bs = image.size(0)
        loss_m.update(loss.item() * grad_accum, bs)  # Recover original loss scale
        
        # Track geo prior loss for logging
        geo_loss_val = loss_dict.get("loss_geo_prior", torch.tensor(0.0)).item()
        geo_loss_m.update(geo_loss_val, bs)
        
        # Cleanup to prevent memory fragmentation
        del image, mask, points, out, loss, loss_dict
        
        if is_main():
            pbar.set_postfix(loss=f"{loss_m.avg:.4f}", geo=f"{geo_loss_m.avg:.4f}")
    
    # Print point-cloud statistics at the end of each epoch.
    if is_main() and use_geo and total_samples > 0:
        import numpy as np
        pcts = np.percentile(all_point_counts, [10, 50, 90]) if all_point_counts else [0, 0, 0]
        pct_has_geo = 100 * samples_with_geo / total_samples
        pct_pos_has_geo = 100 * pos_with_geo / pos_samples if pos_samples > 0 else 0
        print(f"[PointStats] Epoch {epoch}: "
              f"pct_has_geo={pct_has_geo:.1f}% ({samples_with_geo}/{total_samples}), "
              f"pct_pos_has_geo={pct_pos_has_geo:.1f}% ({pos_with_geo}/{pos_samples}), "
              f"points p10/p50/p90={int(pcts[0])}/{int(pcts[1])}/{int(pcts[2])}")
    
    return {
        "loss": reduce_value(loss_m.avg),
        "loss_geo": reduce_value(geo_loss_m.avg),
    }


def filter_small_components(pred_binary: torch.Tensor, min_voxels: int = 10) -> torch.Tensor:
    """
    Filter out small connected components from binary prediction.
    
    Args:
        pred_binary: Binary prediction tensor [B, 1, D, H, W] or [B, D, H, W]
        min_voxels: Minimum voxel count to keep a component
        
    Returns:
        Filtered binary prediction (same shape as input)
    """
    if min_voxels <= 0:
        return pred_binary
    
    # Handle channel dimension
    has_channel = pred_binary.dim() == 5
    if has_channel:
        pred_np = pred_binary[:, 0].cpu().numpy()  # [B, D, H, W]
    else:
        pred_np = pred_binary.cpu().numpy()  # [B, D, H, W]
    
    filtered = np.zeros_like(pred_np)
    
    for b in range(pred_np.shape[0]):
        binary = pred_np[b].astype(bool)
        if not binary.any():
            continue
            
        # Connected component analysis
        labeled, num_features = ndimage.label(binary, structure=ndimage.generate_binary_structure(3, 1))
        
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            if component_mask.sum() >= min_voxels:
                filtered[b][component_mask] = 1
    
    result = torch.from_numpy(filtered).to(pred_binary.device, dtype=pred_binary.dtype)
    if has_channel:
        result = result.unsqueeze(1)  # [B, 1, D, H, W]
    
    return result


@torch.no_grad()
def validate(model, loader, criterion, min_lesion_voxels: int = 10, compute_hd95: bool = False):
    """Validation loop with memory management, proper Dice calculation, and volume filtering."""
    model.eval()
    use_geo = getattr(model.module if hasattr(model, 'module') else model, 'use_geometry', True)
    
    loss_m = AverageMeter()
    
    # MONAI metrics
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    iou_metric = MeanIoU(include_background=False, reduction="mean", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=False) if compute_hd95 else None
    
    # Manual TP/FP/FN accumulation for consistent sensitivity/precision (global)
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # Per-case metrics (to avoid large cases dominating)
    case_sens_list, case_prec_list = [], []
    
    # Clear cache before validation
    torch.cuda.empty_cache()
    
    for step, batch in enumerate(tqdm(loader, desc="Val", disable=not is_main())):
        image = batch["image"].cuda(non_blocking=True)
        mask = batch["mask"].cuda(non_blocking=True)
        points = [p.cuda(non_blocking=True) for p in batch["points"]] if use_geo else None
        
        out = model(image, points)
        # GeoP2VCompositeLoss: Pass points, geo_features, and geo_prior_logits.
        loss_dict = criterion(
            logits=out.logits,
            targets=mask,
            points_list=points,
            geo_features=out.geo_features,
            geo_prior_logits=out.geo_prior_logits,
        )
        loss = loss_dict["loss"]
        
        bs = image.size(0)
        loss_m.update(loss.item(), bs)
        
        # Ensure mask is [B, 1, D, H, W] for MONAI AsDiscrete
        if mask.dim() == 4:
            # [B, D, H, W] -> [B, 1, D, H, W]
            mask_ch = mask.unsqueeze(1).long()
        elif mask.dim() == 5:
            if mask.shape[1] == 1:
                mask_ch = mask.long()
            else:
                mask_ch = mask[:, 0:1, ...].long()
        else:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        
        # Debug: print shapes on first batch
        if step == 0 and is_main():
            print(f"[DEBUG] mask: {mask.shape}, mask_ch: {mask_ch.shape}, logits: {out.logits.shape}")
        
        # Manual one-hot conversion (more reliable than AsDiscrete)
        # Prediction: argmax + scatter
        pred_idx = out.logits.argmax(dim=1, keepdim=True)  # [B, 1, D, H, W]
        pred_binary = (pred_idx == 1).float()  # [B, 1, D, H, W] foreground only
        
        # Volume filtering: remove small connected components
        if min_lesion_voxels > 0:
            pred_binary = filter_small_components(pred_binary, min_voxels=min_lesion_voxels)
        
        # Convert filtered prediction to one-hot
        pred_onehot = torch.zeros_like(out.logits)
        pred_onehot[:, 0, ...] = 1 - pred_binary[:, 0, ...]  # background
        pred_onehot[:, 1, ...] = pred_binary[:, 0, ...]       # foreground
        
        # Label: scatter (use logits shape to avoid 5D mask issues)
        label_onehot = torch.zeros_like(out.logits)
        label_onehot.scatter_(1, mask_ch, 1)  # [B, 2, D, H, W]
        
        dice_metric(y_pred=pred_onehot, y=label_onehot)
        iou_metric(y_pred=pred_onehot, y=label_onehot)
        
        # Manual TP/FP/FN for consistent sensitivity/precision (foreground class only)
        pred_fg = pred_onehot[:, 1, ...].bool()  # [B, D, H, W]
        label_fg = label_onehot[:, 1, ...].bool()  # [B, D, H, W]
        
        # Global accumulation
        batch_tp = (pred_fg & label_fg).sum().item()
        batch_fp = (pred_fg & ~label_fg).sum().item()
        batch_fn = (~pred_fg & label_fg).sum().item()
        total_tp += batch_tp
        total_fp += batch_fp
        total_fn += batch_fn
        
        # Per-case (batch) metrics - only if there's foreground
        if batch_tp + batch_fn > 0:  # Has ground truth foreground
            case_sens = batch_tp / (batch_tp + batch_fn + 1e-8)
            case_sens_list.append(case_sens)
        if batch_tp + batch_fp > 0:  # Has predicted foreground
            case_prec = batch_tp / (batch_tp + batch_fp + 1e-8)
            case_prec_list.append(case_prec)
        
        # HD95 (optional, slow)
        if hd95_metric is not None:
            try:
                hd95_metric(y_pred=pred_onehot, y=label_onehot)
            except Exception:
                pass
        
        # Explicit cleanup to prevent memory accumulation
        del image, mask, mask_ch, points, out, loss, loss_dict, pred_idx, pred_onehot, label_onehot, pred_fg, label_fg
    
    # Aggregate metrics
    dice_val = dice_metric.aggregate().item()
    iou_val = iou_metric.aggregate().item()
    
    # Global sensitivity/precision from accumulated TP/FP/FN (mathematically consistent with Dice)
    sensitivity_val = total_tp / (total_tp + total_fn + 1e-8)
    precision_val = total_tp / (total_tp + total_fp + 1e-8)
    
    # Per-case mean (avoids large cases dominating)
    case_sens_val = sum(case_sens_list) / len(case_sens_list) if case_sens_list else 0.0
    case_prec_val = sum(case_prec_list) / len(case_prec_list) if case_prec_list else 0.0
    
    # HD95 (optional)
    if hd95_metric is not None:
        try:
            hd95_val = hd95_metric.aggregate().item()
            if hd95_val != hd95_val:  # Check for NaN
                hd95_val = -1.0
        except Exception:
            hd95_val = -1.0
        hd95_metric.reset()
    else:
        hd95_val = -1.0
    
    dice_metric.reset()
    iou_metric.reset()
    
    # Clear cache after validation
    torch.cuda.empty_cache()
    
    return {
        "loss": reduce_value(loss_m.avg),
        "dice": reduce_value(dice_val),
        "iou": reduce_value(iou_val),
        "hd95": reduce_value(hd95_val),
        # Global (pixel-weighted)
        "sensitivity": reduce_value(sensitivity_val),
        "precision": reduce_value(precision_val),
        # Per-case mean (case-weighted)
        "case_sens": reduce_value(case_sens_val),
        "case_prec": reduce_value(case_prec_val),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    if not is_ddp():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    cfg = load_config(args.config)
    
    # DDP setup
    local_rank = 0
    if is_ddp():
        local_rank = setup_ddp()
    
    set_seed(cfg.get("seed", 37) + local_rank)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Directory Setup
    # ═══════════════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = cfg["output"].get("experiment_name", "geop2vnet")
    run_name = f"{exp}_{ts}"
    
    # Logs directory
    log_dir = Path(cfg["output"]["log_dir"]) / run_name
    # Checkpoints directory
    ckpt_dir = Path(cfg["output"].get("ckpt_dir", "./ckpts")) / run_name
    
    logger = None
    use_wandb = args.wandb and WANDB_AVAILABLE and is_main()
    
    if is_main():
        log_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        logger = setup_logger("train", log_dir / "train.log")
        logger.info(f"Config: {args.config}, DDP: {is_ddp()}, World: {world_size()}")
        logger.info(f"Logs: {log_dir}")
        logger.info(f"Checkpoints: {ckpt_dir}")
        
        # Save config copy
        with open(log_dir / "config.yaml", "w") as f:
            yaml.dump(cfg, f)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Wandb Init
    # ═══════════════════════════════════════════════════════════════════════
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or run_name,
            config=cfg,
            dir=str(log_dir),
        )
        logger.info(f"Wandb enabled: {wandb.run.url}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Model
    # ═══════════════════════════════════════════════════════════════════════
    model = build_model(cfg.get("model")).cuda()
    if is_ddp():
        # find_unused_parameters=True: handles conditional branches (e.g., use_geometry switch)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if is_main():
        n = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Parameters: {n:.2f}M")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Data
    # ═══════════════════════════════════════════════════════════════════════
    train_loader, train_sampler = build_dataloader(cfg["data"], train=True)
    val_loader, _ = build_dataloader(cfg["data"], train=False)
    
    # ═══════════════════════════════════════════════════════════════════════
    # Loss & Optimizer
    # ═══════════════════════════════════════════════════════════════════════
    # Use GeoP2VCompositeLoss for geometry-aware training
    criterion = build_geo_loss(cfg.get("loss")).cuda()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"] * world_size(),
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"].get("min_lr", 1e-6)
    )
    scaler = GradScaler("cuda", enabled=cfg["train"]["amp"])
    
    # ═══════════════════════════════════════════════════════════════════════
    # Resume
    # ═══════════════════════════════════════════════════════════════════════
    start_epoch, best_dice = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        state = ckpt["model"]
        if is_ddp() and not any(k.startswith("module.") for k in state):
            state = {f"module.{k}": v for k, v in state.items()}
        model.load_state_dict(state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            start_epoch = ckpt["epoch"] + 1
            best_dice = ckpt.get("best_dice", 0)
        if is_main():
            logger.info(f"Resumed from epoch {ckpt.get('epoch', '?')}")
    
    # ═══════════════════════════════════════════════════════════════════════
    # Training Loop
    # ═══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, cfg["train"]["epochs"]):
        metrics = train_one_epoch(
            model, train_loader, train_sampler, criterion, optimizer, scaler, epoch, cfg
        )
        scheduler.step()
        
        lr = optimizer.param_groups[0]["lr"]
        
        if is_main():
            geo_loss = metrics.get('loss_geo', 0.0)
            logger.info(f"E{epoch}: loss={metrics['loss']:.4f} geo={geo_loss:.4f} lr={lr:.2e}")
            
            # Wandb log training metrics
            if use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/loss": metrics["loss"],
                    "train/loss_geo": metrics.get("loss_geo", 0.0),
                    "lr": lr,
                }, step=epoch)
        
        # ═══════════════════════════════════════════════════════════════════
        # Validation
        # ═══════════════════════════════════════════════════════════════════
        val_interval = cfg.get("val", {}).get("interval", cfg["train"].get("val_interval", 1))
        min_lesion_voxels = cfg.get("val", {}).get("min_lesion_voxels", 10)
        if (epoch + 1) % val_interval == 0:
            compute_hd95 = cfg.get("val", {}).get("compute_hd95", False)
            val = validate(model, val_loader, criterion, min_lesion_voxels=min_lesion_voxels, compute_hd95=compute_hd95)
            
            if is_main():
                hd95_str = f" hd95={val['hd95']:.2f}" if val['hd95'] > 0 else ""
                logger.info(
                    f"Val: loss={val['loss']:.4f} dice={val['dice']:.4f} iou={val['iou']:.4f} "
                    f"sens={val['sensitivity']:.4f} prec={val['precision']:.4f} "
                    f"c_sens={val['case_sens']:.4f} c_prec={val['case_prec']:.4f}{hd95_str}"
                )
                
                # Wandb log validation metrics
                if use_wandb:
                    log_dict = {
                        "val/loss": val["loss"],
                        "val/dice": val["dice"],
                        "val/iou": val["iou"],
                        "val/sensitivity": val["sensitivity"],
                        "val/precision": val["precision"],
                        "val/case_sens": val["case_sens"],
                        "val/case_prec": val["case_prec"],
                    }
                    if val["hd95"] > 0:
                        log_dict["val/hd95"] = val["hd95"]
                    wandb.log(log_dict, step=epoch)
                
                # Save metrics.json
                metrics_dict = {
                    "epoch": epoch,
                    "train_loss": metrics["loss"],
                    "val_loss": val["loss"],
                    "val_dice": val["dice"],
                    "val_iou": val["iou"],
                    "val_hd95": val["hd95"],
                    "val_sensitivity": val["sensitivity"],
                    "val_precision": val["precision"],
                    "val_case_sens": val["case_sens"],
                    "val_case_prec": val["case_prec"],
                    "lr": lr,
                    "best_dice": max(best_dice, val["dice"]),
                }
                with open(log_dir / "metrics.json", "w") as f:
                    json.dump(metrics_dict, f, indent=2)
                
                # Save best model
                if val["dice"] > best_dice:
                    best_dice = val["dice"]
                    state = model.module.state_dict() if is_ddp() else model.state_dict()
                    torch.save({
                        "model": state, 
                        "dice": best_dice, 
                        "epoch": epoch
                    }, ckpt_dir / "best.pth")
                    logger.info(f"Best dice: {best_dice:.4f} -> saved to {ckpt_dir / 'best.pth'}")
            
            # Force memory cleanup after validation (HD95 is memory-intensive)
            gc.collect()
            torch.cuda.empty_cache()
        
        # ═══════════════════════════════════════════════════════════════════
        # Periodic Checkpoint
        # ═══════════════════════════════════════════════════════════════════
        save_interval = cfg["train"].get("save_interval", 100)
        if is_main() and (epoch + 1) % save_interval == 0:
            state = model.module.state_dict() if is_ddp() else model.state_dict()
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save({
                "model": state,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice,
            }, ckpt_path)
            logger.info(f"Checkpoint saved: {ckpt_path}")
        
        if is_ddp():
            dist.barrier()
    
    # ═══════════════════════════════════════════════════════════════════════
    # Finish
    # ═══════════════════════════════════════════════════════════════════════
    if is_main():
        logger.info(f"Done. Best dice: {best_dice:.4f}")
        if use_wandb:
            wandb.finish()
    
    if is_ddp():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
