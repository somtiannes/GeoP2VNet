"""
Aneurysm Segmentation Evaluation Module.

Provides comprehensive evaluation metrics at both lesion-level and person-level:
- Lesion-level: Precision, Recall, Dice
- Person-level: Sensitivity, Specificity

Reference: Clinical evaluation standards for intracranial aneurysm detection.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib
from scipy import ndimage


# ============================================================================
# Data Classes for Structured Results
# ============================================================================

@dataclass
class LesionMetrics:
    """Lesion-level evaluation metrics."""
    recall_tp: int = 0
    recall_total: int = 0
    precision_tp: int = 0
    precision_total: int = 0
    dice_list: List[float] = field(default_factory=list)
    
    @property
    def recall(self) -> float:
        return self.recall_tp / max(self.recall_total, 1e-6)
    
    @property
    def precision(self) -> float:
        return self.precision_tp / max(self.precision_total, 1e-6)
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(p + r, 1e-6)
    
    @property
    def mean_dice(self) -> float:
        return np.mean(self.dice_list) if self.dice_list else 0.0


@dataclass
class PersonMetrics:
    """Person-level (case-level) evaluation metrics."""
    positive_tp: int = 0
    positive_total: int = 0
    negative_tp: int = 0
    negative_total: int = 0
    
    @property
    def sensitivity(self) -> float:
        """True positive rate for positive cases."""
        return self.positive_tp / max(self.positive_total, 1e-6)
    
    @property
    def specificity(self) -> float:
        """True negative rate for negative cases."""
        return self.negative_tp / max(self.negative_total, 1e-6)


@dataclass
class CaseResult:
    """Single case evaluation result."""
    subject: str
    dice: float = -1.0  # -1 for negative cases
    tp_dice: float = -1.0  # Dice only for TP lesions
    lesion_dice_list: List[float] = field(default_factory=list)
    lesion_recall_tp: int = 0
    lesion_recall_total: int = 0
    lesion_precision_tp: int = 0
    lesion_precision_total: int = 0
    
    @property
    def fp_count(self) -> int:
        """False positive lesion count."""
        return self.lesion_precision_total - self.lesion_precision_tp
    
    @property
    def is_positive(self) -> bool:
        """Whether this case has ground truth lesions."""
        return self.lesion_recall_total > 0


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def _get_connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Get connected components using scipy.ndimage.
    
    Uses 6-connectivity (face neighbors only) to match skimage.measure.label
    default behavior (connectivity=1 in 3D).
    
    Args:
        mask: Binary mask array.
        
    Returns:
        labeled: Labeled array where each connected component has unique ID.
        num_features: Number of connected components.
    """
    # 6-connectivity: only face neighbors (same as skimage connectivity=1)
    structure = ndimage.generate_binary_structure(3, 1)
    labeled, num_features = ndimage.label(mask, structure=structure)
    return labeled, num_features


def _get_component_properties(labeled: np.ndarray, num_features: int) -> List[Dict]:
    """
    Extract properties of each connected component.
    
    Args:
        labeled: Labeled array from ndimage.label.
        num_features: Number of components.
        
    Returns:
        List of component properties (bbox, coords, area).
    """
    props = []
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        if len(coords) == 0:
            continue
        
        # Bounding box
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0) + 1
        
        props.append({
            'label': i,
            'coords': coords,
            'area': len(coords),
            'bbox': (mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2]),
            'image': labeled[mins[0]:maxs[0], mins[1]:maxs[1], mins[2]:maxs[2]] == i
        })
    
    return props


def evaluate_case(
    subject: str,
    seg: np.ndarray,
    gt: np.ndarray,
    vessel: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    z_height_mm: float = 180.0,
    min_lesion_voxels: int = 10,
    affine: Optional[np.ndarray] = None,
    save_dir: Optional[str] = None,
) -> CaseResult:
    """
    Evaluate segmentation for a single case.
    
    Args:
        subject: Case ID.
        seg: Predicted segmentation mask.
        gt: Ground truth mask.
        vessel: Optional vessel mask for filtering FP outside vessels.
        spacing: Voxel spacing (z, y, x) in mm.
        z_height_mm: Height from top to consider (filter neck region).
        min_lesion_voxels: Minimum voxels for valid lesion.
        affine: NIfTI affine for saving results.
        save_dir: Directory to save processed segmentation.
        
    Returns:
        CaseResult with all evaluation metrics.
    """
    # Ensure binary masks
    seg = (seg > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)
    
    # Default vessel mask (all ones if not provided)
    if vessel is None:
        vessel = np.ones_like(seg, dtype=np.uint8)
    else:
        vessel = (vessel > 0).astype(np.uint8)
    
    # Filter by z-axis (remove neck region)
    z_spacing = spacing[0] if len(spacing) >= 1 else 1.0
    z_start = max(0, seg.shape[0] - int(z_height_mm / z_spacing))
    seg[:z_start, ...] = 0
    
    # Initialize result
    result = CaseResult(subject=subject)
    
    # Arrays for tracking TP lesions
    lesion_recall_seg = np.zeros_like(seg, dtype=np.uint8)
    lesion_recall_gt = np.zeros_like(seg, dtype=np.uint8)
    
    # ========================================
    # Precision: Evaluate predicted lesions
    # ========================================
    pred_labeled, pred_num = _get_connected_components(seg)
    pred_props = _get_component_properties(pred_labeled, pred_num)
    
    for prop in pred_props:
        x0, y0, z0, x1, y1, z1 = prop['bbox']
        roi_vessel = vessel[x0:x1, y0:y1, z0:z1]
        roi_gt = gt_binary[x0:x1, y0:y1, z0:z1]
        
        # Filter: must have enough voxels and overlap with vessel
        in_vessel = (prop['image'] * roi_vessel).sum() > 0
        if prop['area'] >= min_lesion_voxels and in_vessel:
            result.lesion_precision_total += 1
            
            # Check if overlaps with GT (true positive)
            if (prop['image'] * roi_gt).sum() > 0:
                result.lesion_precision_tp += 1
                # Mark as TP for recall dice calculation
                for coord in prop['coords']:
                    lesion_recall_seg[coord[0], coord[1], coord[2]] = prop['label']
        else:
            # Remove invalid lesions from seg
            for coord in prop['coords']:
                seg[coord[0], coord[1], coord[2]] = 0
    
    # ========================================
    # Recall: Evaluate ground truth lesions
    # ========================================
    gt_labeled, gt_num = _get_connected_components(gt_binary)
    gt_props = _get_component_properties(gt_labeled, gt_num)
    
    result.lesion_recall_total = len(gt_props)
    
    for prop in gt_props:
        x0, y0, z0, x1, y1, z1 = prop['bbox']
        roi_seg = seg[x0:x1, y0:y1, z0:z1]
        
        # Check if any predicted voxel overlaps
        intersect = (prop['image'] * roi_seg).sum()
        if intersect > 0:
            result.lesion_recall_tp += 1
            
            # Mark recalled GT lesion
            for coord in prop['coords']:
                lesion_recall_gt[coord[0], coord[1], coord[2]] = prop['label']
            
            # Calculate lesion-level Dice
            seg_labels = np.unique(lesion_recall_seg[prop['coords'][:, 0], 
                                                      prop['coords'][:, 1], 
                                                      prop['coords'][:, 2]])
            tp_voxels = sum((lesion_recall_seg == lbl).sum() 
                           for lbl in seg_labels if lbl > 0)
            gt_voxels = prop['area']
            lesion_dice = 2.0 * intersect / (tp_voxels + gt_voxels + 1e-6)
            result.lesion_dice_list.append(lesion_dice)
    
    # ========================================
    # Case-level Dice
    # ========================================
    if gt_binary.sum() > 0:
        overlap = (seg * gt_binary).sum()
        result.dice = 2.0 * overlap / (seg.sum() + gt_binary.sum() + 1e-6)
        
        # TP-only Dice
        lesion_recall_seg_binary = (lesion_recall_seg > 0).astype(np.uint8)
        tp_overlap = (lesion_recall_seg_binary * gt_binary).sum()
        result.tp_dice = 2.0 * tp_overlap / (lesion_recall_seg_binary.sum() + 
                                              gt_binary.sum() + 1e-6)
    
    # ========================================
    # Save processed segmentation
    # ========================================
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if affine is None:
            affine = np.eye(4)
        save_path = os.path.join(save_dir, f'{subject}_seg.nii.gz')
        nib.save(nib.Nifti1Image(seg, affine), save_path)
    
    return result


def _evaluate_case_wrapper(
    subject: str,
    seg_result: np.ndarray,
    other_infos: Dict[str, Any],
    mask_dir: str,
    vessel_dir: Optional[str] = None,
    use_vessel: bool = False,
    save_seg: bool = False,
    save_dir: Optional[str] = None,
) -> CaseResult:
    """
    Wrapper for evaluate_case that loads GT and vessel masks.
    
    Args:
        subject: Case ID.
        seg_result: Predicted segmentation.
        other_infos: Dict with 'spacing' or 'zxy_spacing', 'affine'.
        mask_dir: Directory containing GT masks.
        vessel_dir: Directory containing vessel masks.
        use_vessel: Whether to use vessel mask for filtering.
        save_seg: Whether to save processed segmentation.
        save_dir: Directory to save results.
        
    Returns:
        CaseResult.
    """
    # Load ground truth
    mask_path = os.path.join(mask_dir, f'{subject}_mask.nii.gz')
    alt_mask_path = os.path.join(mask_dir, f'{subject}_aneurysm.nii.gz')
    
    if os.path.exists(mask_path):
        gt = nib.load(mask_path).get_fdata()
    elif os.path.exists(alt_mask_path):
        gt = nib.load(alt_mask_path).get_fdata()
    else:
        gt = np.zeros_like(seg_result, dtype=np.uint8)
    
    # Load vessel mask
    vessel = None
    if use_vessel and vessel_dir:
        vessel_path = os.path.join(vessel_dir, f'{subject}_blood.nii.gz')
        if os.path.exists(vessel_path):
            vessel = nib.load(vessel_path).get_fdata()
    
    # Extract spacing
    spacing = other_infos.get('zxy_spacing', other_infos.get('spacing', (1.0, 1.0, 1.0)))
    affine = other_infos.get('affine', None)
    
    return evaluate_case(
        subject=subject,
        seg=seg_result,
        gt=gt,
        vessel=vessel,
        spacing=spacing,
        affine=affine,
        save_dir=save_dir if save_seg else None,
    )


# ============================================================================
# Batch Evaluation
# ============================================================================

@dataclass
class EvaluationSummary:
    """Summary of evaluation results."""
    lesion: LesionMetrics
    person: PersonMetrics
    case_dice_list: List[float]
    case_tp_dice_list: List[float]
    case_results: List[CaseResult]
    
    @property
    def mean_dice(self) -> float:
        return np.mean(self.case_dice_list) if self.case_dice_list else 0.0
    
    @property
    def mean_tp_dice(self) -> float:
        return np.mean(self.case_tp_dice_list) if self.case_tp_dice_list else 0.0
    
    def __str__(self) -> str:
        lines = [
            "=" * 70,
            "EVALUATION SUMMARY",
            "=" * 70,
            "",
            "Lesion-Level Metrics:",
            f"  Precision: {self.lesion.precision:.4f} "
            f"({self.lesion.precision_tp}/{self.lesion.precision_total})",
            f"  Recall:    {self.lesion.recall:.4f} "
            f"({self.lesion.recall_tp}/{self.lesion.recall_total})",
            f"  F1 Score:  {self.lesion.f1:.4f}",
            f"  Mean Dice: {self.lesion.mean_dice:.4f}",
            "",
            "Person-Level Metrics:",
            f"  Sensitivity: {self.person.sensitivity:.4f} "
            f"({self.person.positive_tp}/{self.person.positive_total})",
            f"  Specificity: {self.person.specificity:.4f} "
            f"({self.person.negative_tp}/{self.person.negative_total})",
            "",
            "Case Dice:",
            f"  Mean Dice (all):    {self.mean_dice:.4f}",
            f"  Mean Dice (TP only): {self.mean_tp_dice:.4f}",
            "=" * 70,
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            'lesion_precision': self.lesion.precision,
            'lesion_recall': self.lesion.recall,
            'lesion_f1': self.lesion.f1,
            'lesion_dice': self.lesion.mean_dice,
            'person_sensitivity': self.person.sensitivity,
            'person_specificity': self.person.specificity,
            'case_dice': self.mean_dice,
            'case_tp_dice': self.mean_tp_dice,
        }


def evaluate_batch(
    eval_results: Dict[str, Tuple[np.ndarray, Dict[str, Any]]],
    mask_dir: str,
    vessel_dir: Optional[str] = None,
    use_vessel: bool = False,
    save_seg: bool = False,
    save_dir: Optional[str] = None,
    num_workers: int = 16,
    verbose: bool = True,
) -> EvaluationSummary:
    """
    Evaluate multiple cases in parallel.
    
    Args:
        eval_results: Dict mapping subject -> (segmentation, other_infos).
        mask_dir: Directory containing GT masks.
        vessel_dir: Directory containing vessel masks.
        use_vessel: Whether to use vessel mask for filtering FP.
        save_seg: Whether to save processed segmentations.
        save_dir: Directory to save results.
        num_workers: Number of parallel workers.
        verbose: Print per-case results.
        
    Returns:
        EvaluationSummary with all metrics.
    """
    start_time = time.time()
    
    # Prepare evaluation function
    eval_fn = partial(
        _evaluate_case_wrapper,
        mask_dir=mask_dir,
        vessel_dir=vessel_dir,
        use_vessel=use_vessel,
        save_seg=save_seg,
        save_dir=save_dir,
    )
    
    # Run parallel evaluation
    subjects = list(eval_results.keys())
    results: List[CaseResult] = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for subject in subjects:
            seg_result, other_infos = eval_results[subject]
            future = executor.submit(eval_fn, subject, seg_result, other_infos)
            futures[future] = subject
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                subject = futures[future]
                print(f"Error evaluating {subject}: {e}")
    
    # Sort by subject for consistent output
    results.sort(key=lambda x: x.subject)
    
    # Aggregate metrics
    lesion = LesionMetrics()
    person = PersonMetrics()
    case_dice_list = []
    case_tp_dice_list = []
    
    for result in results:
        # Lesion-level
        lesion.recall_tp += result.lesion_recall_tp
        lesion.recall_total += result.lesion_recall_total
        lesion.precision_tp += result.lesion_precision_tp
        lesion.precision_total += result.lesion_precision_total
        lesion.dice_list.extend(result.lesion_dice_list)
        
        # Person-level
        if result.is_positive:
            person.positive_total += 1
            if result.lesion_recall_tp > 0:
                person.positive_tp += 1
            if result.dice >= 0:
                case_dice_list.append(result.dice)
            if result.tp_dice >= 0:
                case_tp_dice_list.append(result.tp_dice)
        else:
            person.negative_total += 1
            if result.lesion_precision_total == 0:
                person.negative_tp += 1
        
        # Print per-case results
        if verbose:
            lesion_dice_avg = np.mean(result.lesion_dice_list) if result.lesion_dice_list else 0.0
            print(f"{result.subject}: "
                  f"Recall={result.lesion_recall_tp}/{result.lesion_recall_total}, "
                  f"Precision={result.lesion_precision_tp}/{result.lesion_precision_total}, "
                  f"FP={result.fp_count}, "
                  f"Dice={result.dice:.4f}, "
                  f"TPDice={result.tp_dice:.4f}, "
                  f"LesionDice={lesion_dice_avg:.4f}")
    
    elapsed = time.time() - start_time
    if verbose:
        print(f"\nEvaluation completed in {elapsed:.2f}s")
    
    summary = EvaluationSummary(
        lesion=lesion,
        person=person,
        case_dice_list=case_dice_list,
        case_tp_dice_list=case_tp_dice_list,
        case_results=results,
    )
    
    if verbose:
        print(summary)
    
    return summary


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_evaluate(
    pred_dir: str,
    mask_dir: str,
    vessel_dir: Optional[str] = None,
    use_vessel: bool = False,
    num_workers: int = 16,
) -> EvaluationSummary:
    """
    Quick evaluation from directories.
    
    Args:
        pred_dir: Directory containing prediction NIfTI files (*_seg.nii.gz).
        mask_dir: Directory containing GT masks.
        vessel_dir: Directory containing vessel masks.
        use_vessel: Whether to use vessel mask.
        num_workers: Number of parallel workers.
        
    Returns:
        EvaluationSummary.
    """
    import glob
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(pred_dir, '*_seg.nii.gz'))
    
    eval_results = {}
    for pred_path in pred_files:
        filename = os.path.basename(pred_path)
        subject = filename.replace('_seg.nii.gz', '')
        
        nii = nib.load(pred_path)
        seg = nii.get_fdata()
        spacing = nii.header.get_zooms()[:3]
        
        eval_results[subject] = (seg, {
            'spacing': spacing,
            'affine': nii.affine,
        })
    
    return evaluate_batch(
        eval_results=eval_results,
        mask_dir=mask_dir,
        vessel_dir=vessel_dir,
        use_vessel=use_vessel,
        num_workers=num_workers,
    )


if __name__ == '__main__':
    # Example usage
    print("Evaluation module loaded successfully.")
    print("Use evaluate_batch() or quick_evaluate() for batch evaluation.")
