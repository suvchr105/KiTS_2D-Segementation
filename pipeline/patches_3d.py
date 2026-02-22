"""
KiTS23 Hybrid 2D+3D Patch Generation
=======================================
Generate 3D volumetric patches centered at tumors, kidneys, or random locations.
This creates a hybrid dataset enabling both 2D and 3D model training.

Novel research idea: combining 2D slices + 3D patches from the same dataset.
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import h5py

logger = logging.getLogger(__name__)


def find_tumor_centers(
    segmentation: np.ndarray,
    tumor_label: int = 2,
) -> List[Tuple[int, int, int]]:
    """Find center of mass of each tumor instance.
    
    Uses connected components to separate multiple tumors.
    """
    import cv2
    from scipy.ndimage import center_of_mass, label as scipy_label

    binary = (segmentation == tumor_label).astype(np.uint8)
    if binary.sum() == 0:
        return []

    labeled, num_features = scipy_label(binary)
    centers = []
    for i in range(1, num_features + 1):
        com = center_of_mass(binary, labeled, i)
        centers.append(tuple(int(c) for c in com))

    return centers


def find_kidney_centers(
    segmentation: np.ndarray,
    kidney_label: int = 1,
) -> List[Tuple[int, int, int]]:
    """Find center of mass of each kidney."""
    from scipy.ndimage import center_of_mass, label as scipy_label

    binary = (segmentation == kidney_label).astype(np.uint8)
    if binary.sum() == 0:
        return []

    labeled, num_features = scipy_label(binary)
    centers = []
    for i in range(1, num_features + 1):
        com = center_of_mass(binary, labeled, i)
        centers.append(tuple(int(c) for c in com))

    return centers


def extract_3d_patch(
    volume: np.ndarray,
    center: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = (64, 64, 64),
) -> Optional[np.ndarray]:
    """Extract a 3D patch centered at the given location.
    
    Uses zero-padding if patch extends beyond volume bounds.
    
    Returns: patch of shape patch_size, or None if center is invalid.
    """
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    cz, cy, cx = center

    # Compute start/end with bounds checking
    z_start = cz - pd // 2
    y_start = cy - ph // 2
    x_start = cx - pw // 2

    z_end = z_start + pd
    y_end = y_start + ph
    x_end = x_start + pw

    # Create output patch (zero-padded)
    patch = np.zeros(patch_size, dtype=volume.dtype)

    # Compute valid source and target ranges
    src_z_start = max(0, z_start)
    src_y_start = max(0, y_start)
    src_x_start = max(0, x_start)
    src_z_end = min(d, z_end)
    src_y_end = min(h, y_end)
    src_x_end = min(w, x_end)

    tgt_z_start = src_z_start - z_start
    tgt_y_start = src_y_start - y_start
    tgt_x_start = src_x_start - x_start
    tgt_z_end = tgt_z_start + (src_z_end - src_z_start)
    tgt_y_end = tgt_y_start + (src_y_end - src_y_start)
    tgt_x_end = tgt_x_start + (src_x_end - src_x_start)

    patch[tgt_z_start:tgt_z_end,
          tgt_y_start:tgt_y_end,
          tgt_x_start:tgt_x_end] = volume[src_z_start:src_z_end,
                                           src_y_start:src_y_end,
                                           src_x_start:src_x_end]

    return patch


def generate_patches_for_case(
    case_data: Dict[str, Any],
    case_id: str,
    output_dir: str,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    patches_per_case: int = 5,
    strategy: str = "tumor",
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate 3D patches for a single case.
    
    Strategy:
      - "tumor":  patches centered at tumor instances
      - "kidney": patches centered at kidney instances
      - "random": random patches within foreground
      - "mixed":  combination of all
    
    Returns list of patch metadata dicts.
    """
    rng = np.random.RandomState(seed)
    image = case_data["image"]
    seg = case_data["segmentation"]

    patch_dir = os.path.join(output_dir, "patches_3d", case_id)
    os.makedirs(patch_dir, exist_ok=True)

    centers = []

    if strategy in ("tumor", "mixed"):
        tumor_centers = find_tumor_centers(seg, tumor_label=2)
        centers.extend([(c, "tumor") for c in tumor_centers])

    if strategy in ("kidney", "mixed"):
        kidney_centers = find_kidney_centers(seg, kidney_label=1)
        centers.extend([(c, "kidney") for c in kidney_centers])

    if strategy in ("random", "mixed"):
        # Random centers within foreground
        fg_coords = np.argwhere(seg > 0)
        if len(fg_coords) > 0:
            n_random = max(1, patches_per_case - len(centers))
            indices = rng.choice(len(fg_coords), size=min(n_random, len(fg_coords)), replace=False)
            for idx in indices:
                c = tuple(int(x) for x in fg_coords[idx])
                centers.append((c, "random"))

    # Add jittered versions of tumor centers if we need more patches
    if len(centers) < patches_per_case:
        tumor_centers = find_tumor_centers(seg, tumor_label=2)
        for tc in tumor_centers:
            for _ in range(patches_per_case - len(centers)):
                jitter = rng.randint(-10, 10, size=3)
                jc = tuple(int(max(0, tc[i] + jitter[i])) for i in range(3))
                centers.append((jc, "tumor_jitter"))
                if len(centers) >= patches_per_case:
                    break
            if len(centers) >= patches_per_case:
                break

    # Limit to requested number
    centers = centers[:patches_per_case]

    patch_metadata = []
    for idx, (center, center_type) in enumerate(centers):
        img_patch = extract_3d_patch(image, center, patch_size)
        seg_patch = extract_3d_patch(seg, center, patch_size)

        if img_patch is None or seg_patch is None:
            continue

        # Save as HDF5
        patch_name = f"{case_id}_patch_{idx:03d}"
        h5_path = os.path.join(patch_dir, f"{patch_name}.h5")

        with h5py.File(h5_path, 'w') as f:
            f.create_dataset("image", data=img_patch.astype(np.float32),
                             compression="gzip", compression_opts=4)
            f.create_dataset("segmentation", data=seg_patch.astype(np.uint8),
                             compression="gzip", compression_opts=4)
            f.attrs["case_id"] = case_id
            f.attrs["center"] = list(center)
            f.attrs["center_type"] = center_type
            f.attrs["patch_size"] = list(patch_size)

        # Compute patch stats
        unique, counts = np.unique(seg_patch, return_counts=True)
        label_counts = dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))

        meta = {
            "case_id": case_id,
            "patch_idx": idx,
            "center": list(center),
            "center_type": center_type,
            "patch_size": list(patch_size),
            "h5_path": h5_path,
            "has_tumor": 2 in unique,
            "has_kidney": 1 in unique,
            "label_counts": label_counts,
            "tumor_ratio": float(label_counts.get(2, 0)) / np.prod(patch_size),
        }
        patch_metadata.append(meta)

    logger.info(
        f"  {case_id}: generated {len(patch_metadata)} patches "
        f"(tumor: {sum(1 for m in patch_metadata if m['has_tumor'])})"
    )

    return patch_metadata
