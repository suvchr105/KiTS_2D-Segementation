"""
KiTS23 SOTA Preprocessing Core
================================
nnU-Net-style loading, orientation, resampling, normalization, body cropping.
"""

import os
import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import label as scipy_label
from scipy.ndimage import binary_fill_holes

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# 1. Loading & Canonical Orientation
# ────────────────────────────────────────────────────────────────

def load_nifti_canonical(path: str) -> nib.Nifti1Image:
    """Load NIfTI and convert to RAS+ canonical orientation.
    
    This is CRITICAL: different scanners store volumes in LPS/RAS etc.
    Without canonical conversion, masks can be flipped.
    """
    nii = nib.load(path)
    nii = nib.as_closest_canonical(nii)
    return nii


def load_case(case_dir: str) -> Dict[str, Any]:
    """Load a complete KiTS23 case (image + seg + instances).
    
    Returns dict with numpy arrays and spatial metadata.
    """
    img_path = os.path.join(case_dir, "imaging.nii.gz")
    seg_path = os.path.join(case_dir, "segmentation.nii.gz")
    inst_dir = os.path.join(case_dir, "instances")

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Missing imaging.nii.gz in {case_dir}")
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing segmentation.nii.gz in {case_dir}")

    img_nii = load_nifti_canonical(img_path)
    seg_nii = load_nifti_canonical(seg_path)

    img_data = img_nii.get_fdata().astype(np.float32)
    seg_data = seg_nii.get_fdata().astype(np.uint8)

    # Extract voxel spacing from affine
    spacing = np.abs(np.diag(img_nii.affine)[:3]).tolist()

    # Verify alignment
    assert img_data.shape == seg_data.shape, (
        f"Shape mismatch: img {img_data.shape} vs seg {seg_data.shape}"
    )

    result = {
        "image": img_data,
        "segmentation": seg_data,
        "spacing": spacing,
        "affine": img_nii.affine,
        "shape_original": img_data.shape,
    }

    # Load instance masks if available
    if os.path.isdir(inst_dir):
        instances = {}
        for fname in sorted(os.listdir(inst_dir)):
            if fname.endswith(".nii.gz"):
                inst_nii = load_nifti_canonical(os.path.join(inst_dir, fname))
                inst_data = inst_nii.get_fdata().astype(np.uint8)
                # Parse instance name
                name = fname.replace(".nii.gz", "")
                instances[name] = inst_data
        result["instances"] = instances
    else:
        result["instances"] = {}

    return result


# ────────────────────────────────────────────────────────────────
# 2. Spacing Harmonization (nnU-Net Core)
# ────────────────────────────────────────────────────────────────

def resample_volume(
    volume: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    is_mask: bool = False,
) -> np.ndarray:
    """Resample a 3D volume to target spacing using SimpleITK.
    
    Image: linear interpolation (preserves intensity)
    Mask:  nearest-neighbor (preserves integer labels)
    """
    sitk_img = sitk.GetImageFromArray(volume)
    sitk_img.SetSpacing(tuple(original_spacing))

    original_size = sitk_img.GetSize()
    new_size = [
        int(round(original_size[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(tuple(target_spacing))
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetTransform(sitk.Transform())

    if is_mask:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetDefaultPixelValue(0)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(volume.min()))

    resampled = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(resampled)


def resample_case(
    case_data: Dict[str, Any],
    target_spacing: Tuple[float, float, float],
) -> Dict[str, Any]:
    """Resample image, segmentation, and all instances to target spacing."""
    orig_spacing = case_data["spacing"]

    logger.debug(
        f"Resampling: {orig_spacing} -> {target_spacing}, "
        f"shape {case_data['image'].shape}"
    )

    case_data["image"] = resample_volume(
        case_data["image"], orig_spacing, target_spacing, is_mask=False
    )
    case_data["segmentation"] = resample_volume(
        case_data["segmentation"], orig_spacing, target_spacing, is_mask=True
    )

    # Resample instance masks
    if case_data.get("instances"):
        for name, inst in case_data["instances"].items():
            case_data["instances"][name] = resample_volume(
                inst, orig_spacing, target_spacing, is_mask=True
            )

    case_data["spacing_resampled"] = list(target_spacing)
    case_data["shape_resampled"] = case_data["image"].shape

    return case_data


# ────────────────────────────────────────────────────────────────
# 3. CT Intensity Normalization
# ────────────────────────────────────────────────────────────────

def normalize_ct(
    image: np.ndarray,
    window_low: float = -200.0,
    window_high: float = 300.0,
    method: str = "zscore_body",
    body_threshold: float = -500.0,
) -> np.ndarray:
    """CT-specific intensity normalization.
    
    Methods:
        zscore_body:   z-score computed inside body mask (nnU-Net standard)
        zscore_global: z-score on full volume
        minmax:        [0, 1] min-max after windowing
    """
    # Window clipping (kidney CT standard)
    image = np.clip(image, window_low, window_high)

    if method == "zscore_body":
        body_mask = image > body_threshold
        if body_mask.sum() > 0:
            mean = image[body_mask].mean()
            std = image[body_mask].std()
        else:
            mean = image.mean()
            std = image.std()
        image = (image - mean) / (std + 1e-8)

    elif method == "zscore_global":
        image = (image - image.mean()) / (image.std() + 1e-8)

    elif method == "minmax":
        mn, mx = image.min(), image.max()
        image = (image - mn) / (mx - mn + 1e-8)

    else:
        raise ValueError(f"Unknown normalization: {method}")

    return image


# ────────────────────────────────────────────────────────────────
# 4. Body Cropping (SOTA trick for memory & focus)
# ────────────────────────────────────────────────────────────────

def get_body_bbox(
    image: np.ndarray,
    threshold: float = -500.0,
    margin: int = 5,
) -> Tuple[Tuple[int, int], ...]:
    """Find tight bounding box around the body (air removal).
    
    Steps:
      1. Threshold to get body mask
      2. Fill holes
      3. Largest connected component
      4. Bounding box + margin
    """
    body = image > threshold
    body = binary_fill_holes(body)

    labeled, num_features = scipy_label(body)
    if num_features == 0:
        return tuple((0, s) for s in image.shape)

    # Find largest component
    component_sizes = np.bincount(labeled.ravel())[1:]  # skip background
    largest_component = np.argmax(component_sizes) + 1
    body = labeled == largest_component

    coords = np.argwhere(body)
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + margin + 1, image.shape)

    return tuple((int(mins[i]), int(maxs[i])) for i in range(3))


def crop_to_body(
    case_data: Dict[str, Any],
    threshold: float = -500.0,
    margin: int = 5,
) -> Dict[str, Any]:
    """Crop image, segmentation, and instances to body bounding box."""
    bbox = get_body_bbox(case_data["image"], threshold, margin)

    slices = tuple(slice(b[0], b[1]) for b in bbox)

    case_data["image"] = case_data["image"][slices].copy()
    case_data["segmentation"] = case_data["segmentation"][slices].copy()

    if case_data.get("instances"):
        for name in case_data["instances"]:
            case_data["instances"][name] = (
                case_data["instances"][name][slices].copy()
            )

    case_data["body_bbox"] = bbox
    case_data["shape_cropped"] = case_data["image"].shape

    return case_data


# ────────────────────────────────────────────────────────────────
# 5. Full Preprocessing Pipeline (single case)
# ────────────────────────────────────────────────────────────────

def preprocess_case(
    case_dir: str,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0),
    ct_window: Tuple[float, float] = (-200.0, 300.0),
    normalization: str = "zscore_body",
    body_threshold: float = -500.0,
    enable_body_crop: bool = True,
    crop_margin: int = 5,
) -> Dict[str, Any]:
    """Complete nnU-Net-style preprocessing for a single KiTS23 case.
    
    Pipeline:
      1. Load + canonical orientation
      2. Resample to target spacing
      3. Body cropping
      4. CT intensity normalization
    """
    case_id = os.path.basename(case_dir)
    logger.info(f"Preprocessing {case_id}...")

    # Step 1: Load + canonical
    case_data = load_case(case_dir)
    logger.debug(
        f"  Loaded: shape={case_data['image'].shape}, "
        f"spacing={case_data['spacing']}"
    )

    # Step 2: Resample
    case_data = resample_case(case_data, target_spacing)
    logger.debug(f"  Resampled: shape={case_data['image'].shape}")

    # Step 3: Body crop
    if enable_body_crop:
        case_data = crop_to_body(
            case_data, threshold=body_threshold, margin=crop_margin
        )
        logger.debug(f"  Cropped: shape={case_data['image'].shape}")

    # Step 4: Normalize
    case_data["image"] = normalize_ct(
        case_data["image"],
        window_low=ct_window[0],
        window_high=ct_window[1],
        method=normalization,
        body_threshold=body_threshold,
    )

    logger.info(
        f"  {case_id} done: final shape={case_data['image'].shape}, "
        f"labels={np.unique(case_data['segmentation']).tolist()}"
    )

    return case_data
