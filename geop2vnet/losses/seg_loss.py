# Copyright (c) 2024
# Licensed under the Apache License, Version 2.0

"""
Segmentation Loss Functions using MONAI

Uses MONAI's battle-tested DiceCELoss implementation with:
    - Proper numerical stability
    - Automatic softmax handling
    - Flexible foreground/background configuration
    - Batch Dice computation (better for small targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional, List

from monai.losses import DiceCELoss as MonaiDiceCELoss


class DiceCELoss(nn.Module):
    """
    Combined Dice + CrossEntropy Loss using MONAI.
    
    Wrapper around MONAI's DiceCELoss with sensible defaults for
    aneurysm segmentation (small foreground targets).
    
    Args:
        dice_weight: Weight for Dice loss (lambda_dice)
        ce_weight: Weight for CE loss (lambda_ce)
        class_weights: Per-class weights for CE (optional)
        include_background: Whether to include background in Dice (default False)
        batch: Use batch Dice (default True, better for small targets)
    
    Input:
        logits: [B, C, D, H, W] raw logits (before softmax)
        targets: [B, D, H, W] ground truth labels (long indices)
    
    Output:
        dict with 'loss' key and scalar tensor value
    """
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 0.5,
        class_weights: Optional[List[float]] = None,
        include_background: bool = False,
        batch: bool = True,
    ) -> None:
        super().__init__()
        
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        
        # Convert class weights to tensor if provided
        ce_weight_tensor = None
        if class_weights is not None:
            ce_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        
        # Use MONAI's battle-tested implementation
        self.loss_fn = MonaiDiceCELoss(
            include_background=include_background,
            to_onehot_y=True,       # Target is [B, D, H, W] indices -> one-hot
            softmax=True,           # Apply softmax to logits
            lambda_dice=dice_weight,
            lambda_ce=ce_weight,
            weight=ce_weight_tensor,
            batch=batch,            # Batch Dice: more stable for small targets
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute combined Dice + CE loss.
        
        Args:
            logits: [B, C, D, H, W] raw logits
            targets: [B, D, H, W] ground truth labels
        
        Returns:
            dict with 'loss' key
        """
        # MONAI expects target shape [B, 1, D, H, W] for to_onehot_y=True
        if targets.dim() == 4:  # [B, D, H, W]
            targets = targets.unsqueeze(1)  # [B, 1, D, H, W]
        
        loss = self.loss_fn(logits, targets)
        
        return {"loss": loss}


def build_loss(cfg: dict | None = None) -> DiceCELoss:
    """
    Factory function for DiceCELoss.
    
    Args:
        cfg: Config dict with optional keys:
            - dice_weight: float (default 1.0)
            - ce_weight: float (default 0.5)
            - class_weights: list[float] (default None)
            - include_background: bool (default False)
            - batch: bool (default True)
    
    Returns:
        DiceCELoss instance
    """
    if cfg is None:
        cfg = {}
    
    return DiceCELoss(
        dice_weight=cfg.get("dice_weight", 1.0),
        ce_weight=cfg.get("ce_weight", 0.5),
        class_weights=cfg.get("class_weights"),
        include_background=cfg.get("include_background", False),
        batch=cfg.get("batch", True),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Unit Tests
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DiceCELoss (MONAI) Unit Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Build loss
    loss_fn = DiceCELoss(dice_weight=1.0, ce_weight=0.5).to(device)
    
    # Test input
    B, C, D, H, W = 2, 2, 32, 32, 32
    logits = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
    targets = torch.randint(0, C, (B, D, H, W), device=device)
    
    print(f"Logits: {logits.shape}")
    print(f"Targets: {targets.shape}")
    
    # Forward
    result = loss_fn(logits, targets)
    
    print(f"Loss: {result['loss'].item():.4f}")
    
    # Gradient check
    result["loss"].backward()
    print("Gradient flow: OK")
    
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)
