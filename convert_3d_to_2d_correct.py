#!/usr/bin/env python3
"""
convert_3d_to_2d_correct.py
============================
Correctly convert KiTS23 3D NIfTI volumes to 2D PNG slices.

Key design choices:
  - Canonical RAS+ orientation via nibabel
  - Axial slicing on axis 2 (z-axis in RAS+) → 512×512 square slices
  - CT windowing [-200, 300] HU → 16-bit PNG (0–65535)
  - Mask labels {0,1,2,3} preserved exactly as uint8
  - Optional visible masks (×85 scaling) for quick inspection
  - --no_resample flag to keep original 512×512 resolution

Usage:
  # With in-plane resampling to 1mm:
  python convert_3d_to_2d_correct.py

  # Keep original resolution (512×512) → recommended:
  python convert_3d_to_2d_correct.py --no_resample

  # Custom output directory:
  python convert_3d_to_2d_correct.py --no_resample --output_dir KiTS23_2D
"""

import os
import sys
import argparse
import glob
import time
import numpy as np
import nibabel as nib
import cv2
from scipy.ndimage import zoom


def parse_args():
    p = argparse.ArgumentParser(description="Convert KiTS23 3D NIfTI to 2D PNG")
    p.add_argument("--data_dir", default="kits23_repo/dataset",
                   help="Path to KiTS23 dataset directory")
    p.add_argument("--output_dir", default="KiTS23_2D",
                   help="Output directory for 2D slices")
    p.add_argument("--target_spacing", type=float, default=1.0,
                   help="Target in-plane spacing in mm (default: 1.0)")
    p.add_argument("--no_resample", action="store_true",
                   help="Skip in-plane resampling, keep original resolution")
    p.add_argument("--window_center", type=float, default=50,
                   help="CT window center (default: 50 HU)")
    p.add_argument("--window_width", type=float, default=500,
                   help="CT window width (default: 500 HU)")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cases that already have output")
    return p.parse_args()


def ct_window(volume, center, width):
    """Apply CT windowing and normalize to [0, 1]."""
    vmin = center - width / 2
    vmax = center + width / 2
    clipped = np.clip(volume, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return normalized


def process_case(case_dir, output_dir, args):
    """Process a single case: load 3D, slice axially, save 2D PNGs."""
    case_name = os.path.basename(case_dir)
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")

    if not os.path.exists(img_path) or not os.path.exists(seg_path):
        return 0, f"SKIP {case_name}: missing files"

    # Check if already done
    if args.skip_existing:
        existing = glob.glob(os.path.join(output_dir, "images",
                                          f"{case_name}_slice_*.png"))
        if len(existing) > 0:
            return len(existing), f"SKIP {case_name}: already exists ({len(existing)} slices)"

    # Load and canonicalize to RAS+
    img_nii = nib.as_closest_canonical(nib.load(img_path))
    seg_nii = nib.as_closest_canonical(nib.load(seg_path))

    img_data = img_nii.get_fdata().astype(np.float64)
    seg_data = seg_nii.get_fdata().astype(np.uint8)

    # Get voxel spacing
    spacing = img_nii.header.get_zooms()  # (x_spacing, y_spacing, z_spacing)

    # Optional: resample in-plane (axes 0,1) to target spacing
    if not args.no_resample:
        sx = spacing[0] / args.target_spacing
        sy = spacing[1] / args.target_spacing
        if abs(sx - 1.0) > 0.01 or abs(sy - 1.0) > 0.01:
            img_data = zoom(img_data, (sx, sy, 1.0), order=3)
            seg_data = zoom(seg_data, (sx, sy, 1.0), order=0)

    # Apply CT windowing
    img_windowed = ct_window(img_data, args.window_center, args.window_width)

    # Convert to 16-bit for lossless PNG storage
    img_16bit = (img_windowed * 65535).astype(np.uint16)

    # Create output directories
    img_out = os.path.join(output_dir, "images")
    mask_out = os.path.join(output_dir, "masks")
    vis_out = os.path.join(output_dir, "masks_visible")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)
    os.makedirs(vis_out, exist_ok=True)

    # Slice along axis 2 (axial in RAS+ canonical)
    n_slices = img_data.shape[2]
    for z in range(n_slices):
        slice_img = img_16bit[:, :, z]
        slice_seg = seg_data[:, :, z]

        fname = f"{case_name}_slice_{z:03d}.png"

        # Save 16-bit grayscale image
        cv2.imwrite(os.path.join(img_out, fname), slice_img)

        # Save raw mask (labels 0,1,2,3)
        cv2.imwrite(os.path.join(mask_out, fname), slice_seg)

        # Save visible mask (×85 scaling: 0→0, 1→85, 2→170, 3→255)
        vis_mask = (slice_seg.astype(np.uint16) * 85).astype(np.uint8)
        cv2.imwrite(os.path.join(vis_out, fname), vis_mask)

    return n_slices, f"OK {case_name}: {n_slices} slices, shape {img_data.shape[:2]}"


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Find all cases
    cases = sorted(glob.glob(os.path.join(data_dir, "case_*")))
    cases = [c for c in cases if os.path.isdir(c)]
    print(f"Found {len(cases)} cases in {data_dir}")
    print(f"Output → {output_dir}")
    print(f"Resampling: {'OFF (original resolution)' if args.no_resample else f'{args.target_spacing}mm'}")
    print(f"CT window: [{args.window_center - args.window_width/2}, "
          f"{args.window_center + args.window_width/2}] HU")
    print()

    os.makedirs(output_dir, exist_ok=True)

    total_slices = 0
    failures = 0
    t0 = time.time()

    # Write conversion log
    log_path = os.path.join(output_dir, "conversion.log")
    with open(log_path, "w") as log:
        log.write(f"KiTS23 3D→2D Conversion\n")
        log.write(f"{'='*60}\n")
        log.write(f"Source:      {os.path.abspath(data_dir)}\n")
        log.write(f"Output:      {os.path.abspath(output_dir)}\n")
        log.write(f"Resample:    {'OFF' if args.no_resample else f'{args.target_spacing}mm'}\n")
        log.write(f"CT window:   [{args.window_center - args.window_width/2}, "
                  f"{args.window_center + args.window_width/2}] HU\n")
        log.write(f"Cases:       {len(cases)}\n")
        log.write(f"{'='*60}\n\n")

        for i, case_dir in enumerate(cases):
            try:
                n, msg = process_case(case_dir, output_dir, args)
                total_slices += n
            except Exception as e:
                msg = f"FAIL {os.path.basename(case_dir)}: {e}"
                failures += 1

            log.write(msg + "\n")
            log.flush()

            if (i + 1) % 10 == 0 or (i + 1) == len(cases):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed * 60
                print(f"  [{i+1:>3d}/{len(cases)}] {total_slices:>6d} slices "
                      f"| {elapsed/60:.1f} min | {rate:.0f} cases/min | {msg}")

        log.write(f"\n{'='*60}\n")
        log.write(f"Total slices: {total_slices}\n")
        log.write(f"Failures:     {failures}\n")
        log.write(f"Time:         {(time.time()-t0)/60:.1f} min\n")

    print(f"\nDone: {total_slices} slices from {len(cases)} cases "
          f"({failures} failures) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
