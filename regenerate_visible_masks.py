#!/usr/bin/env python3
"""
regenerate_visible_masks.py
===========================
Regenerate masks_visible/ as COLOR-CODED RGB PNGs for clear visual inspection.

Color mapping:
  - Background (0) → Black   (0, 0, 0)
  - Kidney     (1) → Green   (0, 255, 0)
  - Tumor      (2) → Red     (255, 0, 0)
  - Cyst       (3) → Yellow  (255, 255, 0)

The raw masks/ folder (values 0,1,2,3) is left untouched — those are the
correct training labels.  masks_visible/ is purely for human viewing.

Usage:
  python regenerate_visible_masks.py
"""

import os
import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── colour look-up table (BGR for OpenCV) ───────────────────────────────
#                       B    G    R
COLOR_MAP = {
    0: np.array([  0,   0,   0], dtype=np.uint8),   # background → black
    1: np.array([  0, 255,   0], dtype=np.uint8),   # kidney     → green
    2: np.array([  0,   0, 255], dtype=np.uint8),   # tumor      → red
    3: np.array([  0, 255, 255], dtype=np.uint8),   # cyst       → yellow
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a single-channel label mask to a 3-channel BGR colour mask."""
    h, w = mask.shape[:2]
    colour = np.zeros((h, w, 3), dtype=np.uint8)
    for label, bgr in COLOR_MAP.items():
        colour[mask == label] = bgr
    return colour


def process_single_file(mask_path: str, vis_dir: str) -> str:
    """Read a raw mask, colorize it, and overwrite the visible version."""
    fname = os.path.basename(mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return f"SKIP {fname} (unreadable)"
    colour = colorize_mask(mask)
    out_path = os.path.join(vis_dir, fname)
    cv2.imwrite(out_path, colour)
    return fname


def main():
    base = "/mnt/raid/obed/Suvadip/KITS23_2D/KiTS23_2D"
    mask_dir = os.path.join(base, "masks")
    vis_dir  = os.path.join(base, "masks_visible")

    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    total = len(mask_files)
    print(f"Found {total} mask files to colorize")
    print(f"Output → {vis_dir}/")
    print(f"Colour legend: Green=kidney, Red=tumor, Yellow=cyst\n")

    os.makedirs(vis_dir, exist_ok=True)

    done = 0
    workers = min(32, os.cpu_count() or 8)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_single_file, mf, vis_dir): mf
            for mf in mask_files
        }
        for fut in as_completed(futures):
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"  [{done:>6d}/{total}] ({100*done/total:.1f}%)")

    print(f"\nDone — {done} colour masks written to {vis_dir}/")


if __name__ == "__main__":
    main()
