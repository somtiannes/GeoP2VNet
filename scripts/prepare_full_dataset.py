#!/usr/bin/env python
"""
Prepare a GeoP2VNet dataset from CTA images, aneurysm masks, and vessel masks.

Expected input naming by default:
  <case_id>_0000.nii.gz
  <case_id>_predicted_aneurysm.nii.gz
  <case_id>_predicted_vessel.nii.gz

The script creates:
  output_dir/images/<case_id>.nii.gz
  output_dir/masks/<case_id>_mask.nii.gz
  output_dir/pointclouds/<case_id>_points.npy
  output_dir/train.txt
  output_dir/val.txt
"""

from __future__ import annotations

import argparse
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare data for GeoP2VNet.")
    parser.add_argument("--source_dirs", nargs="+", required=True, help="Input case directories.")
    parser.add_argument("--output_dir", required=True, help="Prepared dataset output directory.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_surface_points", type=int, default=0, help="0 keeps all surface points.")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--image_suffix", default="_0000.nii.gz")
    parser.add_argument("--aneurysm_suffix", default="_predicted_aneurysm.nii.gz")
    parser.add_argument("--vessel_suffix", default="_predicted_vessel.nii.gz")
    parser.add_argument("--aneurysm_label", type=int, default=2)
    parser.add_argument("--copy_images", action="store_true", help="Copy images instead of creating symlinks.")
    return parser.parse_args()


def discover_samples(source_dirs: list[Path], image_suffix: str, aneurysm_suffix: str, vessel_suffix: str):
    samples = []
    for src_dir in source_dirs:
        if not src_dir.exists():
            print(f"Warning: {src_dir} does not exist, skipping.")
            continue
        for image_path in src_dir.glob(f"*{image_suffix}"):
            case_id = image_path.name.removesuffix(image_suffix)
            aneurysm_path = src_dir / f"{case_id}{aneurysm_suffix}"
            vessel_path = src_dir / f"{case_id}{vessel_suffix}"
            if aneurysm_path.exists() and vessel_path.exists():
                samples.append((case_id, image_path, aneurysm_path, vessel_path))
    return samples


def extract_surface_points(mask: np.ndarray, num_points: int | None = None) -> np.ndarray:
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    eroded = ndimage.binary_erosion(mask, iterations=1)
    surface = mask.astype(np.int8) - eroded.astype(np.int8)
    coords = np.argwhere(surface > 0).astype(np.float32)

    if len(coords) == 0:
        coords = np.argwhere(mask > 0).astype(np.float32)

    if num_points is not None and len(coords) > num_points:
        idx = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[idx]

    return coords


def process_sample(args):
    case_id, image_path, aneurysm_path, vessel_path, output_dir, aneurysm_label, num_surface_points, copy_images = args

    try:
        output_dir = Path(output_dir)
        aneurysm_nii = nib.load(str(aneurysm_path))
        aneurysm_data = aneurysm_nii.get_fdata()
        aneurysm_mask = (aneurysm_data == aneurysm_label).astype(np.uint8)

        vessel_nii = nib.load(str(vessel_path))
        vessel_mask = (vessel_nii.get_fdata() > 0).astype(np.uint8)
        points = extract_surface_points(vessel_mask, num_surface_points)

        dst_image = output_dir / "images" / f"{case_id}.nii.gz"
        if dst_image.exists() or dst_image.is_symlink():
            dst_image.unlink()
        if copy_images:
            shutil.copy2(image_path, dst_image)
        else:
            dst_image.symlink_to(image_path.resolve())

        dst_mask = output_dir / "masks" / f"{case_id}_mask.nii.gz"
        nib.save(nib.Nifti1Image(aneurysm_mask, aneurysm_nii.affine), str(dst_mask))

        dst_points = output_dir / "pointclouds" / f"{case_id}_points.npy"
        np.save(str(dst_points), points)

        info = f"aneurysm={int(aneurysm_mask.sum())}, vessel={int(vessel_mask.sum())}, points={len(points)}"
        return case_id, True, info
    except Exception as exc:
        return case_id, False, str(exc)


def write_split(output_dir: Path, case_ids: list[str], val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    case_ids = sorted(case_ids)
    rng.shuffle(case_ids)

    n_val = int(round(len(case_ids) * val_ratio))
    val_cases = sorted(case_ids[:n_val])
    train_cases = sorted(case_ids[n_val:])

    (output_dir / "train.txt").write_text("\n".join(train_cases) + "\n", encoding="utf-8")
    (output_dir / "val.txt").write_text("\n".join(val_cases) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_dirs = [Path(p) for p in args.source_dirs]
    output_dir = Path(args.output_dir)
    num_surface_points = args.num_surface_points if args.num_surface_points > 0 else None

    for subdir in ["images", "masks", "pointclouds"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    samples = discover_samples(source_dirs, args.image_suffix, args.aneurysm_suffix, args.vessel_suffix)
    print(f"Found {len(samples)} samples.")

    worker_args = [
        (case_id, image_path, aneurysm_path, vessel_path, output_dir, args.aneurysm_label, num_surface_points, args.copy_images)
        for case_id, image_path, aneurysm_path, vessel_path in samples
    ]

    processed = []
    failed = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_sample, item) for item in worker_args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            case_id, ok, info = future.result()
            if ok:
                processed.append(case_id)
            else:
                failed.append((case_id, info))

    write_split(output_dir, processed, args.val_ratio, args.seed)

    print(f"Processed {len(processed)} samples into {output_dir}.")
    if failed:
        print(f"Failed {len(failed)} samples:")
        for case_id, reason in failed:
            print(f"  {case_id}: {reason}")


if __name__ == "__main__":
    main()
