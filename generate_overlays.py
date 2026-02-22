#!/usr/bin/env python3
"""
generate_overlays.py
====================
Generate two visualization outputs for the 2D KiTS23 dataset:

1. masks_visible/  — Color masks on BLACK background with THICK CONTOURS
   around the structures so they're impossible to miss even at small sizes.

2. overlays/       — CT image with semi-transparent colour mask overlaid.
   This makes the anatomy AND the annotation visible at the same time.

Colour mapping (same for both):
  Background (0) → nothing
  Kidney     (1) → Green   (0, 255, 0)
  Tumor      (2) → Red     (255, 0, 0)
  Cyst       (3) → Yellow  (255, 255, 0)

Usage:
  python generate_overlays.py
"""

import os
import glob
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── BGR colours for OpenCV ──────────────────────────────────────────────
COLOR_MAP_BGR = {
    1: (0, 255, 0),     # kidney  → green
    2: (0, 0, 255),     # tumor   → red
    3: (0, 255, 255),   # cyst    → yellow
}

BASE = "/mnt/raid/obed/Suvadip/KITS23_2D/KiTS23_2D"
IMG_DIR  = os.path.join(BASE, "images")
MASK_DIR = os.path.join(BASE, "masks")
VIS_DIR  = os.path.join(BASE, "masks_visible")
OVL_DIR  = os.path.join(BASE, "overlays")


def process_file(fname: str) -> str:
    """Process one slice: regenerate visible mask + create overlay."""
    mask_path = os.path.join(MASK_DIR, fname)
    img_path  = os.path.join(IMG_DIR, fname)
    vis_path  = os.path.join(VIS_DIR, fname)
    ovl_path  = os.path.join(OVL_DIR, fname)

    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return f"SKIP {fname}"

    h, w = mask.shape[:2]
    has_content = mask.max() > 0

    # ── 1. Visible mask: colour fill + contours ─────────────────────────
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    if has_content:
        for label, bgr in COLOR_MAP_BGR.items():
            region = (mask == label).astype(np.uint8)
            if region.max() == 0:
                continue
            # Fill the region with colour
            vis[region == 1] = bgr
            # Draw thick contour for visibility at thumbnail size
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, bgr, thickness=3)

    cv2.imwrite(vis_path, vis)

    # ── 2. Overlay: CT image + semi-transparent mask ────────────────────
    img16 = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img16 is None:
        return fname

    # Convert 16-bit CT to 8-bit for display
    img8 = (img16.astype(np.float32) / 65535.0 * 255).astype(np.uint8)
    # Make it 3-channel grayscale
    overlay = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)

    if has_content:
        # Create a coloured mask layer
        colour_layer = np.zeros_like(overlay)
        for label, bgr in COLOR_MAP_BGR.items():
            colour_layer[mask == label] = bgr

        # Blend: where mask > 0, mix 60% image + 40% colour
        mask_bool = mask > 0
        alpha = 0.4
        overlay[mask_bool] = cv2.addWeighted(
            overlay, 1 - alpha, colour_layer, alpha, 0
        )[mask_bool]

        # Also draw bright contours on the overlay
        for label, bgr in COLOR_MAP_BGR.items():
            region = (mask == label).astype(np.uint8)
            if region.max() == 0:
                continue
            contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, bgr, thickness=2)

    cv2.imwrite(ovl_path, overlay)
    return fname


def main():
    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(OVL_DIR, exist_ok=True)

    mask_files = sorted(os.listdir(MASK_DIR))
    mask_files = [f for f in mask_files if f.endswith(".png")]
    total = len(mask_files)

    print(f"Processing {total} slices")
    print(f"  masks_visible/ → colour masks with contours")
    print(f"  overlays/      → CT + colour mask overlay")
    print(f"  Colours: Green=kidney, Red=tumor, Yellow=cyst\n")

    workers = min(32, os.cpu_count() or 8)
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_file, f): f for f in mask_files}
        for fut in as_completed(futures):
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"  [{done:>6d}/{total}] ({100*done/total:.1f}%)")

    print(f"\nDone — generated {done} visible masks + overlays")


if __name__ == "__main__":
    main()
