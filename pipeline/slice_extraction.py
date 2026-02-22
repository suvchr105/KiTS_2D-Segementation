"""
KiTS23 Slice Extraction Module
================================
2D and 2.5D slice extraction with tumor-aware sampling.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import cv2
import h5py

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Slice Category Classification
# ────────────────────────────────────────────────────────────────

def classify_slice(mask_slice: np.ndarray) -> str:
    """Classify a single mask slice into category.
    
    Returns: 'tumor' | 'kidney_only' | 'background'
    """
    unique = np.unique(mask_slice)
    if 2 in unique or 3 in unique:  # tumor or cyst
        return "tumor"
    elif 1 in unique:
        return "kidney_only"
    else:
        return "background"


# ────────────────────────────────────────────────────────────────
# 2.5D Context Stack
# ────────────────────────────────────────────────────────────────

def extract_25d_stack(
    volume: np.ndarray,
    k: int,
    context: int = 2,
    axis: int = 2,
) -> np.ndarray:
    """Extract 2.5D context stack centered at slice k.
    
    For context=2: returns slices [k-2, k-1, k, k+1, k+2] stacked.
    Boundary slices are replicated (reflect padding).
    
    Returns: (H, W, 2*context+1) array
    """
    n_slices = volume.shape[axis]
    indices = []
    for offset in range(-context, context + 1):
        idx = k + offset
        # Reflect padding at boundaries
        idx = max(0, min(n_slices - 1, idx))
        indices.append(idx)

    if axis == 2:
        stack = np.stack([volume[:, :, i] for i in indices], axis=-1)
    elif axis == 1:
        stack = np.stack([volume[:, i, :] for i in indices], axis=-1)
    else:
        stack = np.stack([volume[i, :, :] for i in indices], axis=-1)

    return stack


# ────────────────────────────────────────────────────────────────
# Single Slice Data Container
# ────────────────────────────────────────────────────────────────

def extract_slice_data(
    case_data: Dict[str, Any],
    k: int,
    case_id: str,
    context: int = 2,
    axis: int = 2,
) -> Dict[str, Any]:
    """Extract all data for a single slice.
    
    Returns dict with:
      - image_2d:    (H, W) single slice
      - image_25d:   (H, W, C) 2.5D context stack
      - mask:        (H, W) semantic mask (uint8)
      - category:    'tumor' | 'kidney_only' | 'background'
      - instance_masks: dict of (H, W) instance masks
      - metadata:    dict with stats
    """
    img = case_data["image"]
    seg = case_data["segmentation"]
    instances = case_data.get("instances", {})

    # 2D slice
    if axis == 2:
        img_2d = img[:, :, k]
        mask_2d = seg[:, :, k]
    elif axis == 1:
        img_2d = img[:, k, :]
        mask_2d = seg[:, k, :]
    else:
        img_2d = img[k, :, :]
        mask_2d = seg[k, :, :]

    # 2.5D stack
    img_25d = extract_25d_stack(img, k, context, axis)

    # Category
    category = classify_slice(mask_2d.astype(np.uint8))

    # Instance masks for this slice
    inst_2d = {}
    for name, inst_vol in instances.items():
        if axis == 2:
            inst_slice = inst_vol[:, :, k]
        elif axis == 1:
            inst_slice = inst_vol[:, k, :]
        else:
            inst_slice = inst_vol[k, :, :]
        if inst_slice.sum() > 0:
            inst_2d[name] = inst_slice.astype(np.uint8)

    # Compute metadata
    mask_u8 = mask_2d.astype(np.uint8)
    kidney_pixels = int(np.sum(mask_u8 == 1))
    tumor_pixels = int(np.sum(mask_u8 == 2))
    cyst_pixels = int(np.sum(mask_u8 == 3))
    foreground_pixels = kidney_pixels + tumor_pixels + cyst_pixels
    total_pixels = int(mask_u8.size)

    meta = {
        "case_id": case_id,
        "slice_idx": k,
        "category": category,
        "has_tumor": tumor_pixels > 0,
        "has_cyst": cyst_pixels > 0,
        "kidney_pixels": kidney_pixels,
        "tumor_pixels": tumor_pixels,
        "cyst_pixels": cyst_pixels,
        "foreground_pixels": foreground_pixels,
        "foreground_ratio": foreground_pixels / total_pixels if total_pixels > 0 else 0,
        "height": img_2d.shape[0],
        "width": img_2d.shape[1],
        "num_instances": len(inst_2d),
    }

    return {
        "image_2d": img_2d.astype(np.float32),
        "image_25d": img_25d.astype(np.float32),
        "mask": mask_u8,
        "category": category,
        "instance_masks": inst_2d,
        "metadata": meta,
    }


# ────────────────────────────────────────────────────────────────
# Save Functions
# ────────────────────────────────────────────────────────────────

def _save_npz(path: str, **arrays):
    np.savez_compressed(path, **arrays)


def _save_png_image(path: str, image: np.ndarray):
    """Save image as PNG (normalize to 0-255 for visualization)."""
    img = image.copy()
    if img.ndim == 3:
        # Take center channel for visualization
        img = img[:, :, img.shape[2] // 2]
    mn, mx = img.min(), img.max()
    img = ((img - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def _save_png_mask(path: str, mask: np.ndarray):
    """Save mask as PNG (integer labels, no normalization)."""
    cv2.imwrite(path, mask)


def save_slice(
    slice_data: Dict[str, Any],
    output_dir: str,
    case_id: str,
    slice_idx: int,
    save_format: str = "npz",
) -> Dict[str, str]:
    """Save extracted slice data to disk.
    
    Returns dict of saved file paths.
    """
    prefix = f"{case_id}_slice_{slice_idx:04d}"
    saved = {}

    img_dir = os.path.join(output_dir, "images_2d")
    mask_dir = os.path.join(output_dir, "masks_semantic")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    if save_format == "npz":
        # Save 2.5D image
        img_path = os.path.join(img_dir, f"{prefix}.npz")
        _save_npz(img_path, image_25d=slice_data["image_25d"])
        saved["image"] = img_path

        # Save mask
        mask_path = os.path.join(mask_dir, f"{prefix}.npz")
        _save_npz(mask_path, mask=slice_data["mask"])
        saved["mask"] = mask_path

    elif save_format == "png":
        img_path = os.path.join(img_dir, f"{prefix}.png")
        _save_png_image(img_path, slice_data["image_25d"])
        saved["image"] = img_path

        mask_path = os.path.join(mask_dir, f"{prefix}.png")
        _save_png_mask(mask_path, slice_data["mask"])
        saved["mask"] = mask_path

    elif save_format == "h5":
        h5_path = os.path.join(img_dir, f"{prefix}.h5")
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("image_25d", data=slice_data["image_25d"],
                             compression="gzip", compression_opts=4)
            f.create_dataset("mask", data=slice_data["mask"],
                             compression="gzip", compression_opts=4)
        saved["h5"] = h5_path

    # Save instance masks if present
    if slice_data.get("instance_masks"):
        inst_dir = os.path.join(output_dir, "masks_instance", case_id)
        os.makedirs(inst_dir, exist_ok=True)
        for name, inst_mask in slice_data["instance_masks"].items():
            inst_path = os.path.join(
                inst_dir, f"{prefix}_{name}.npz"
            )
            _save_npz(inst_path, mask=inst_mask)
            saved[f"instance_{name}"] = inst_path

    return saved


# ────────────────────────────────────────────────────────────────
# Extract All Slices from a Case
# ────────────────────────────────────────────────────────────────

def extract_all_slices(
    case_data: Dict[str, Any],
    case_id: str,
    output_dir: str,
    context: int = 2,
    axis: int = 2,
    save_format: str = "npz",
    save_all: bool = True,
    min_foreground_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Extract and save all 2D slices from a preprocessed case.
    
    Returns list of metadata dicts for each saved slice.
    """
    n_slices = case_data["image"].shape[axis]
    all_metadata = []

    for k in range(n_slices):
        slice_data = extract_slice_data(
            case_data, k, case_id, context, axis
        )

        # Filter logic
        if not save_all:
            if slice_data["category"] == "background":
                if slice_data["metadata"]["foreground_ratio"] < min_foreground_ratio:
                    continue

        # Save
        saved_paths = save_slice(
            slice_data, output_dir, case_id, k, save_format
        )

        meta = slice_data["metadata"].copy()
        meta.update(saved_paths)
        all_metadata.append(meta)

    logger.info(
        f"  {case_id}: extracted {len(all_metadata)}/{n_slices} slices "
        f"(tumor: {sum(1 for m in all_metadata if m.get('has_tumor'))}, "
        f"kidney: {sum(1 for m in all_metadata if m['category'] == 'kidney_only')})"
    )

    return all_metadata
