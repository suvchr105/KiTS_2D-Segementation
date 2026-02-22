"""
KiTS23 Case-Wise Dataset Split (MICCAI Standard)
===================================================
CRITICAL: Never split by slice — always by case.
Slice-wise split causes information leakage (adjacent slices from same patient
in train/test).

Supports:
  - Random case-wise split
  - Stratified by tumor volume (ensures balanced tumor distribution)
  - Reproducible with seed
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_case_tumor_volume(case_dir: str) -> float:
    """Compute total tumor volume (voxel count) for stratification.
    
    Uses segmentation mask only (fast, no need to load imaging).
    """
    import nibabel as nib

    seg_path = os.path.join(case_dir, "segmentation.nii.gz")
    if not os.path.exists(seg_path):
        return 0.0

    seg = nib.load(seg_path)
    seg = nib.as_closest_canonical(seg)
    data = seg.get_fdata()

    tumor_voxels = float(np.sum(data == 2))
    return tumor_voxels


def stratified_case_split(
    case_ids: List[str],
    tumor_volumes: List[float],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    n_bins: int = 5,
) -> Dict[str, List[str]]:
    """Stratified case-wise split based on tumor volume.
    
    Strategy:
      1. Bin cases by tumor volume quintiles
      2. Within each bin, randomly assign to train/val/test
      3. Guarantees balanced tumor distribution across splits
    
    This is the MICCAI standard approach.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    rng = np.random.RandomState(seed)

    # Create bins based on tumor volume
    volumes = np.array(tumor_volumes)
    case_ids = np.array(case_ids)

    # Separate zero-tumor (no tumor) and tumor cases
    has_tumor = volumes > 0
    tumor_cases = case_ids[has_tumor]
    tumor_vols = volumes[has_tumor]
    no_tumor_cases = case_ids[~has_tumor]

    train, val, test = [], [], []

    # Stratify tumor cases by volume bins
    if len(tumor_cases) > 0:
        # Create quantile-based bins
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(tumor_vols, percentiles)
        bin_indices = np.digitize(tumor_vols, bin_edges[1:-1])

        for bin_id in range(n_bins):
            bin_mask = bin_indices == bin_id
            bin_cases = tumor_cases[bin_mask]
            rng.shuffle(bin_cases)

            n = len(bin_cases)
            n_train = max(1, int(n * train_ratio))
            n_val = max(0, int(n * val_ratio))
            # Remaining goes to test
            n_test = n - n_train - n_val

            train.extend(bin_cases[:n_train].tolist())
            val.extend(bin_cases[n_train:n_train + n_val].tolist())
            test.extend(bin_cases[n_train + n_val:].tolist())

    # Split no-tumor cases similarly
    if len(no_tumor_cases) > 0:
        rng.shuffle(no_tumor_cases)
        n = len(no_tumor_cases)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train.extend(no_tumor_cases[:n_train].tolist())
        val.extend(no_tumor_cases[n_train:n_train + n_val].tolist())
        test.extend(no_tumor_cases[n_train + n_val:].tolist())

    # Shuffle within each split
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    splits = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    logger.info(
        f"Split: train={len(train)}, val={len(val)}, test={len(test)} "
        f"(total={len(train)+len(val)+len(test)})"
    )

    return splits


def random_case_split(
    case_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Simple random case-wise split."""
    rng = np.random.RandomState(seed)
    ids = list(case_ids)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }

    logger.info(
        f"Split: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )

    return splits


def save_splits(
    splits: Dict[str, List[str]],
    output_dir: str,
    filename: str = "splits.json",
) -> str:
    """Save splits to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)

    save_data = {
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "stats": {
            "train_count": len(splits["train"]),
            "val_count": len(splits["val"]),
            "test_count": len(splits["test"]),
            "total": sum(len(v) for v in splits.values()),
        }
    }

    with open(path, 'w') as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Splits saved to {path}")
    return path


def load_splits(path: str) -> Dict[str, List[str]]:
    """Load splits from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: data[k] for k in ["train", "val", "test"]}


def generate_splits(
    raw_data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
    strategy: str = "tumor_volume",
) -> Dict[str, List[str]]:
    """Full split generation pipeline.
    
    Args:
        raw_data_dir: path to kits23_repo/dataset/
        output_dir: where to save splits.json
        strategy: "tumor_volume" (stratified) or "random"
    """
    # Discover all cases with imaging data
    case_dirs = sorted([
        d for d in os.listdir(raw_data_dir)
        if d.startswith("case_") and os.path.isdir(os.path.join(raw_data_dir, d))
    ])

    # Filter to cases that have imaging.nii.gz
    valid_cases = []
    for case_id in case_dirs:
        img_path = os.path.join(raw_data_dir, case_id, "imaging.nii.gz")
        seg_path = os.path.join(raw_data_dir, case_id, "segmentation.nii.gz")
        if os.path.exists(img_path) and os.path.exists(seg_path):
            valid_cases.append(case_id)

    logger.info(f"Found {len(valid_cases)} valid cases (with imaging + segmentation)")

    if strategy == "tumor_volume":
        logger.info("Computing tumor volumes for stratification...")
        volumes = []
        for case_id in valid_cases:
            vol = compute_case_tumor_volume(
                os.path.join(raw_data_dir, case_id)
            )
            volumes.append(vol)
            
        splits = stratified_case_split(
            valid_cases, volumes,
            train_ratio, val_ratio, test_ratio, seed
        )
    else:
        splits = random_case_split(
            valid_cases, train_ratio, val_ratio, test_ratio, seed
        )

    save_splits(splits, output_dir)
    return splits
