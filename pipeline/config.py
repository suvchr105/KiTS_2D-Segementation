"""
KiTS23 SOTA Pipeline Configuration
===================================
Central config for all preprocessing parameters.
Follows nnU-Net / MONAI best practices.
"""

import os
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PipelineConfig:
    """Master configuration for the KiTS23 preprocessing pipeline."""

    # ── Paths ──────────────────────────────────────────────────
    raw_data_dir: str = ""           # kits23_repo/dataset/
    output_dir: str = ""             # KITS23_2D/output/
    kits23_repo: str = ""            # kits23_repo root

    # ── GPU ────────────────────────────────────────────────────
    gpu_id: int = 3
    num_workers: int = 8

    # ── Spacing & Resampling (nnU-Net style) ───────────────────
    # 1x1x3 mm is optimal for axial 2D extraction from abdominal CT
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 3.0)
    image_interpolation: str = "linear"       # sitk.sitkLinear
    mask_interpolation: str = "nearest"        # sitk.sitkNearestNeighbor

    # ── CT Intensity (kidney CT window) ────────────────────────
    ct_window_low: float = -200.0
    ct_window_high: float = 300.0
    normalization: str = "zscore_body"  # "zscore_body" | "minmax" | "zscore_global"
    body_threshold: float = -500.0

    # ── Body Cropping ──────────────────────────────────────────
    enable_body_crop: bool = True
    crop_margin: int = 5  # voxels margin around body

    # ── Slice Extraction ───────────────────────────────────────
    slice_axis: int = 2                # 0=sagittal, 1=coronal, 2=axial
    # 2.5D context: number of adjacent slices on each side
    context_slices: int = 2            # k-2, k-1, k, k+1, k+2 → 5 channels
    output_channels: int = 5           # 2*context_slices + 1

    # ── Slice Filtering & Balancing ────────────────────────────
    save_all_slices: bool = True       # if False, only save tumor/kidney slices
    min_foreground_ratio: float = 0.0  # minimum kidney+tumor ratio to keep slice
    tumor_oversample_factor: int = 1   # how many extra copies of tumor slices

    # ── Output Sizes ───────────────────────────────────────────
    resize_to: Optional[Tuple[int, int]] = None  # None = keep original, e.g. (256, 256)

    # ── Output Formats ─────────────────────────────────────────
    save_images_as: str = "npz"        # "png" | "npz" | "h5"
    save_masks_as: str = "npz"         # "png" | "npz" | "h5"

    # ── Labels ─────────────────────────────────────────────────
    label_map: dict = field(default_factory=lambda: {
        0: "background",
        1: "kidney",
        2: "tumor",
        3: "cyst"
    })
    num_classes: int = 4

    # ── Dataset Split ──────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.20
    split_seed: int = 42
    stratify_by: str = "tumor_volume"  # "tumor_volume" | "random" | "case_id"

    # ── Detection Annotations ──────────────────────────────────
    generate_detection: bool = True
    detection_format: str = "coco"     # "coco" | "yolo" | "csv"
    min_tumor_area_px: int = 10        # minimum tumor area to generate bbox

    # ── Difficulty Scoring ─────────────────────────────────────
    generate_difficulty: bool = True
    difficulty_weights: dict = field(default_factory=lambda: {
        "area_inv": 1.0,
        "contrast_inv": 0.5,
        "boundary_complexity": 0.2,
        "multi_instance": 0.3
    })

    # ── Instance Masks ─────────────────────────────────────────
    generate_instances: bool = True

    # ── Hybrid 3D Patches ──────────────────────────────────────
    generate_3d_patches: bool = True
    patch_size_3d: Tuple[int, int, int] = (64, 64, 64)
    patches_per_case: int = 5
    patch_center_strategy: str = "tumor"  # "tumor" | "kidney" | "random"

    # ── Metadata ───────────────────────────────────────────────
    generate_metadata: bool = True

    # ── Logging ────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "pipeline.log"

    def __post_init__(self):
        if not self.raw_data_dir and self.kits23_repo:
            self.raw_data_dir = os.path.join(self.kits23_repo, "dataset")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.output_dir, "images_2d")

    @property
    def masks_semantic_dir(self) -> str:
        return os.path.join(self.output_dir, "masks_semantic")

    @property
    def masks_instance_dir(self) -> str:
        return os.path.join(self.output_dir, "masks_instance")

    @property
    def bboxes_dir(self) -> str:
        return os.path.join(self.output_dir, "bboxes")

    @property
    def patches_3d_dir(self) -> str:
        return os.path.join(self.output_dir, "patches_3d")

    @property
    def metadata_dir(self) -> str:
        return os.path.join(self.output_dir, "metadata")

    @property
    def splits_dir(self) -> str:
        return os.path.join(self.output_dir, "splits")

    def create_output_dirs(self):
        """Create all output directories."""
        dirs = [
            self.output_dir,
            self.images_dir,
            self.masks_semantic_dir,
            self.metadata_dir,
            self.splits_dir,
        ]
        if self.generate_instances:
            dirs.append(self.masks_instance_dir)
        if self.generate_detection:
            dirs.append(self.bboxes_dir)
        if self.generate_3d_patches:
            dirs.append(self.patches_3d_dir)
        for d in dirs:
            os.makedirs(d, exist_ok=True)
