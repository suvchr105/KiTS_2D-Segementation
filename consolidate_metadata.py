#!/usr/bin/env python3
"""
consolidate_metadata.py
=======================
Rebuild consolidated metadata from on-disk 2D slice files.

Scans the output directory for all extracted slices and generates
a unified metadata CSV + JSON summary for the entire dataset.

Usage:
  python consolidate_metadata.py
  python consolidate_metadata.py --output_dir output
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description="Consolidate KiTS23 metadata")
    p.add_argument("--output_dir", default="output",
                   help="Pipeline output directory")
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)

    images_dir = os.path.join(output_dir, "images_2d")
    masks_dir = os.path.join(output_dir, "masks_semantic")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    if not os.path.exists(images_dir):
        print(f"ERROR: images directory not found: {images_dir}")
        sys.exit(1)

    # Find all slice files
    slice_files = sorted(glob.glob(os.path.join(images_dir, "*.npz")))
    print(f"Found {len(slice_files)} slice files in {images_dir}")

    records = []
    cases_seen = set()

    for i, sf in enumerate(slice_files):
        fname = os.path.basename(sf)
        # Parse filename: case_XXXXX_slice_ZZZ.npz
        parts = fname.replace(".npz", "").split("_slice_")
        if len(parts) != 2:
            continue

        case_name = parts[0]
        slice_idx = int(parts[1])
        cases_seen.add(case_name)

        # Load mask to compute stats
        mask_path = os.path.join(masks_dir, fname)
        if os.path.exists(mask_path):
            data = np.load(mask_path)
            mask = data["mask"] if "mask" in data else data[data.files[0]]
            total = mask.size
            kidney_px = int(np.sum(mask == 1))
            tumor_px = int(np.sum(mask == 2))
            cyst_px = int(np.sum(mask == 3))
            fg_px = kidney_px + tumor_px + cyst_px
        else:
            total = 0
            kidney_px = tumor_px = cyst_px = fg_px = 0

        records.append({
            "case": case_name,
            "slice_idx": slice_idx,
            "filename": fname,
            "total_pixels": total,
            "kidney_pixels": kidney_px,
            "tumor_pixels": tumor_px,
            "cyst_pixels": cyst_px,
            "foreground_pixels": fg_px,
            "foreground_ratio": fg_px / total if total > 0 else 0,
            "has_kidney": kidney_px > 0,
            "has_tumor": tumor_px > 0,
            "has_cyst": cyst_px > 0,
        })

        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1}/{len(slice_files)}...")

    # Save CSV
    df = pd.DataFrame(records)
    csv_path = os.path.join(metadata_dir, "all_slices_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metadata CSV: {csv_path}")

    # Save summary JSON
    summary = {
        "total_cases": len(cases_seen),
        "total_slices": len(records),
        "slices_with_kidney": int(df["has_kidney"].sum()),
        "slices_with_tumor": int(df["has_tumor"].sum()),
        "slices_with_cyst": int(df["has_cyst"].sum()),
        "avg_foreground_ratio": float(df["foreground_ratio"].mean()),
        "cases": sorted(list(cases_seen)),
    }

    json_path = os.path.join(metadata_dir, "dataset_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {json_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Dataset Summary")
    print(f"{'='*50}")
    print(f"  Cases:              {summary['total_cases']}")
    print(f"  Total slices:       {summary['total_slices']}")
    print(f"  With kidney:        {summary['slices_with_kidney']}")
    print(f"  With tumor:         {summary['slices_with_tumor']}")
    print(f"  With cyst:          {summary['slices_with_cyst']}")
    print(f"  Avg FG ratio:       {summary['avg_foreground_ratio']:.4f}")


if __name__ == "__main__":
    main()
