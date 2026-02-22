"""
KiTS23 Tumor Slice Difficulty Scoring
=======================================
Rare research idea: score each slice by difficulty for curriculum learning,
hard example mining, and fairness benchmarking.

Difficulty factors:
  - Small tumor area (harder to segment)
  - Low contrast between tumor and surroundings
  - Complex tumor boundary (irregular shape)
  - Multiple tumor instances (multi-instance challenge)
  - Tumor near kidney boundary (ambiguous region)
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

logger = logging.getLogger(__name__)


def compute_boundary_complexity(mask: np.ndarray, label: int = 2) -> float:
    """Compute boundary complexity as perimeter / sqrt(area).
    
    Higher = more irregular shape = harder segmentation.
    Circle has lowest complexity (~3.54).
    """
    binary = (mask == label).astype(np.float32)
    area = binary.sum()
    if area == 0:
        return 0.0

    # Perimeter via gradient magnitude
    gy, gx = np.gradient(binary)
    perimeter = np.sqrt(gy ** 2 + gx ** 2).sum()

    # Normalized complexity: perimeter / sqrt(area)
    complexity = perimeter / (np.sqrt(area) + 1e-8)
    return float(complexity)


def compute_contrast_score(
    image: np.ndarray, mask: np.ndarray, label: int = 2
) -> float:
    """Compute contrast between tumor and its immediate surroundings.
    
    Low contrast = harder to distinguish = more difficult.
    Returns inverse contrast (higher = harder).
    """
    tumor = mask == label
    if tumor.sum() == 0:
        return 0.0

    # Dilate tumor to get surrounding region
    dilated = binary_dilation(tumor, iterations=3)
    surround = dilated & (~tumor) & (mask != 0)  # non-background surroundings

    if surround.sum() < 10:
        # Not enough surrounding context
        return 0.5

    tumor_mean = image[tumor].mean()
    surround_mean = image[surround].mean()
    contrast = abs(tumor_mean - surround_mean)

    # Inverse: low contrast → high difficulty
    return float(1.0 / (contrast + 1e-6))


def compute_area_difficulty(
    mask: np.ndarray, label: int = 2, reference_area: float = 5000.0
) -> float:
    """Compute difficulty based on tumor area.
    
    Smaller tumors are harder to segment.
    Returns inverse area score (higher = harder).
    """
    area = float((mask == label).sum())
    if area == 0:
        return 0.0
    return float(reference_area / (area + 1e-6))


def compute_multi_instance_score(
    mask: np.ndarray, label: int = 2
) -> float:
    """Score based on number of separate tumor instances.
    
    Multiple instances = more complex scene = harder.
    """
    import cv2
    binary = (mask == label).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0

    num_labels, _ = cv2.connectedComponents(binary)
    num_instances = num_labels - 1  # subtract background

    # Score: 0 for single instance, increases with more
    return float(max(0, num_instances - 1))


def compute_boundary_proximity_score(
    mask: np.ndarray, tumor_label: int = 2, kidney_label: int = 1
) -> float:
    """Score based on how close tumor is to kidney boundary.
    
    Tumors near the kidney edge are harder to segment (ambiguous boundary).
    """
    tumor = mask == tumor_label
    kidney = mask == kidney_label

    if tumor.sum() == 0 or kidney.sum() == 0:
        return 0.0

    # Kidney boundary: dilated - eroded
    kidney_boundary = binary_dilation(kidney, iterations=2).astype(np.float32) - \
                      binary_erosion(kidney, iterations=2).astype(np.float32)
    kidney_boundary = np.clip(kidney_boundary, 0, 1)

    # Check overlap between tumor and kidney boundary
    overlap = (tumor.astype(np.float32) * kidney_boundary).sum()
    ratio = overlap / (tumor.sum() + 1e-8)

    return float(ratio)


def compute_difficulty_score(
    image_slice: np.ndarray,
    mask_slice: np.ndarray,
    weights: Optional[Dict[str, float]] = None,
    tumor_label: int = 2,
) -> Dict[str, float]:
    """Compute comprehensive difficulty score for a single slice.
    
    Args:
        image_slice: (H, W) preprocessed image
        mask_slice:  (H, W) semantic mask
        weights:     dict of component weights
        tumor_label: which label is tumor
    
    Returns dict with individual scores and total weighted difficulty.
    """
    if weights is None:
        weights = {
            "area_inv": 1.0,
            "contrast_inv": 0.5,
            "boundary_complexity": 0.2,
            "multi_instance": 0.3,
            "boundary_proximity": 0.2,
        }

    has_tumor = (mask_slice == tumor_label).sum() > 0

    if not has_tumor:
        return {
            "area_inv": 0.0,
            "contrast_inv": 0.0,
            "boundary_complexity": 0.0,
            "multi_instance": 0.0,
            "boundary_proximity": 0.0,
            "total_difficulty": 0.0,
            "difficulty_level": "none",
        }

    scores = {}
    scores["area_inv"] = compute_area_difficulty(mask_slice, tumor_label)
    scores["contrast_inv"] = compute_contrast_score(image_slice, mask_slice, tumor_label)
    scores["boundary_complexity"] = compute_boundary_complexity(mask_slice, tumor_label)
    scores["multi_instance"] = compute_multi_instance_score(mask_slice, tumor_label)
    scores["boundary_proximity"] = compute_boundary_proximity_score(mask_slice, tumor_label)

    # Weighted total
    total = sum(weights.get(k, 0) * v for k, v in scores.items())
    scores["total_difficulty"] = total

    # Categorize difficulty
    if total < 1.0:
        scores["difficulty_level"] = "easy"
    elif total < 3.0:
        scores["difficulty_level"] = "medium"
    elif total < 6.0:
        scores["difficulty_level"] = "hard"
    else:
        scores["difficulty_level"] = "very_hard"

    return scores


def compute_case_difficulty_summary(
    slice_difficulties: list,
) -> Dict[str, Any]:
    """Aggregate difficulty scores across all slices of a case.
    
    Returns summary statistics.
    """
    tumor_diffs = [s for s in slice_difficulties if s["total_difficulty"] > 0]

    if not tumor_diffs:
        return {
            "num_tumor_slices": 0,
            "mean_difficulty": 0.0,
            "max_difficulty": 0.0,
            "min_difficulty": 0.0,
            "std_difficulty": 0.0,
            "hardest_slice": -1,
            "difficulty_distribution": {"easy": 0, "medium": 0, "hard": 0, "very_hard": 0},
        }

    totals = [s["total_difficulty"] for s in tumor_diffs]
    levels = [s["difficulty_level"] for s in tumor_diffs]

    return {
        "num_tumor_slices": len(tumor_diffs),
        "mean_difficulty": float(np.mean(totals)),
        "max_difficulty": float(np.max(totals)),
        "min_difficulty": float(np.min(totals)),
        "std_difficulty": float(np.std(totals)),
        "hardest_slice": int(np.argmax(totals)),
        "difficulty_distribution": {
            "easy": levels.count("easy"),
            "medium": levels.count("medium"),
            "hard": levels.count("hard"),
            "very_hard": levels.count("very_hard"),
        },
    }
