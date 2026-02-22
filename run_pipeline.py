#!/usr/bin/env python3
"""
run_pipeline.py
===============
Main orchestrator for the KiTS23 SOTA 3D→2D preprocessing pipeline.

This script:
  1. Loads/resamples 3D NIfTI volumes (nnU-Net style)
  2. Extracts 2.5D axial slices with context
  3. Generates detection bounding boxes (COCO format)
  4. Computes difficulty scores per slice
  5. Creates stratified train/val/test splits
  6. Extracts 3D patches for hybrid training
  7. Writes comprehensive metadata

Usage:
  python run_pipeline.py --gpu 3
  python run_pipeline.py --gpu 3 --skip_existing
  python run_pipeline.py --gpu 3 --start_case 264 --end_case 489
"""

import os
import sys
import time
import json
import argparse
import traceback
import numpy as np

# Ensure pipeline package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import PipelineConfig
from pipeline.preprocessing import Preprocessor
from pipeline.slice_extraction import SliceExtractor
from pipeline.detection import DetectionAnnotator
from pipeline.difficulty import DifficultyScorer
from pipeline.splits import DatasetSplitter
from pipeline.patches_3d import PatchExtractor3D
from pipeline.metadata import MetadataWriter


def parse_args():
    p = argparse.ArgumentParser(description="KiTS23 SOTA 3D→2D Pipeline")
    p.add_argument("--gpu", type=int, default=3, help="GPU ID")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip cases that already have output")
    p.add_argument("--start_case", type=int, default=0,
                   help="Start from this case index (for batch processing)")
    p.add_argument("--end_case", type=int, default=None,
                   help="End at this case index (exclusive)")
    p.add_argument("--kits23_repo", default="kits23_repo",
                   help="Path to kits23 repository")
    p.add_argument("--output_dir", default="output",
                   help="Output directory")
    return p.parse_args()


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Configure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = PipelineConfig(
        kits23_repo=os.path.join(base_dir, args.kits23_repo),
        output_dir=os.path.join(base_dir, args.output_dir),
        gpu_id=args.gpu,
    )
    config.raw_data_dir = os.path.join(config.kits23_repo, "dataset")
    config.create_output_dirs()

    # Find all cases
    cases = sorted([
        d for d in os.listdir(config.raw_data_dir)
        if d.startswith("case_") and os.path.isdir(
            os.path.join(config.raw_data_dir, d))
    ])

    # Apply range
    end = args.end_case if args.end_case else len(cases)
    cases = cases[args.start_case:end]

    print(f"{'='*60}")
    print(f"KiTS23 SOTA 3D→2D Pipeline")
    print(f"{'='*60}")
    print(f"Cases:    {len(cases)} ({args.start_case}..{end})")
    print(f"GPU:      {args.gpu}")
    print(f"Output:   {config.output_dir}")
    print(f"Spacing:  {config.target_spacing}")
    print(f"Context:  ±{config.context_slices} slices (2.5D)")
    print(f"{'='*60}\n")

    # Initialize modules
    preprocessor = Preprocessor(config)
    slicer = SliceExtractor(config)
    detector = DetectionAnnotator(config)
    scorer = DifficultyScorer(config)
    patcher = PatchExtractor3D(config)
    metadata_writer = MetadataWriter(config)

    t0 = time.time()
    total_slices = 0
    failures = []

    for i, case_name in enumerate(cases):
        case_dir = os.path.join(config.raw_data_dir, case_name)
        print(f"\n[{i+1}/{len(cases)}] Processing {case_name}...")

        # Skip existing?
        if args.skip_existing:
            existing = [
                f for f in os.listdir(config.images_dir)
                if f.startswith(case_name + "_") and f.endswith(".npz")
            ] if os.path.isdir(config.images_dir) else []
            if len(existing) > 0:
                total_slices += len(existing)
                print(f"  SKIP: already has {len(existing)} slices")
                continue

        try:
            # Step 1: Load, resample, normalize
            img_vol, seg_vol, spacing, orig_shape = preprocessor.process(
                case_dir)
            print(f"  Preprocessed: {orig_shape} → {img_vol.shape}, "
                  f"spacing={np.round(spacing, 2)}")

            # Step 2: Extract 2.5D slices
            slice_records = slicer.extract(case_name, img_vol, seg_vol)
            total_slices += len(slice_records)
            print(f"  Extracted: {len(slice_records)} slices")

            # Step 3: Detection annotations
            if config.generate_detection:
                det_records = detector.annotate(case_name, slice_records,
                                                seg_vol)
                print(f"  Detection: {sum(1 for r in det_records if r)} "
                      f"slices with boxes")

            # Step 4: Difficulty scoring
            if config.generate_difficulty:
                diff_records = scorer.score(case_name, slice_records,
                                            img_vol, seg_vol)
                print(f"  Difficulty: scored {len(diff_records)} slices")

            # Step 5: 3D patches
            if config.generate_3d_patches:
                n_patches = patcher.extract(case_name, img_vol, seg_vol)
                print(f"  3D Patches: {n_patches}")

            # Step 6: Metadata
            if config.generate_metadata:
                metadata_writer.write_case(case_name, img_vol, seg_vol,
                                           spacing, orig_shape,
                                           slice_records)

        except Exception as e:
            failures.append((case_name, str(e)))
            print(f"  FAILED: {e}")
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed * 60
        print(f"  Time: {elapsed/60:.1f} min total, {rate:.1f} cases/min")

    # Step 7: Dataset splits (once, after all cases)
    print(f"\n{'='*60}")
    print("Generating dataset splits...")
    splitter = DatasetSplitter(config)
    splitter.split()
    print(f"Splits saved to {config.splits_dir}/")

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Cases processed: {len(cases) - len(failures)}/{len(cases)}")
    print(f"Total slices:    {total_slices}")
    print(f"Failures:        {len(failures)}")
    print(f"Time:            {elapsed/60:.1f} min")
    if failures:
        print(f"\nFailed cases:")
        for case, err in failures:
            print(f"  {case}: {err}")


if __name__ == "__main__":
    main()
