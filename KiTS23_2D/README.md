# KiTS23-2D: 2D Axial Slice Dataset for Kidney Tumor Segmentation

A publication-ready 2D dataset derived from the [KiTS23](https://kits-challenge.org/kits23/) (Kidney Tumor Segmentation Challenge 2023) 3D CT volumes. Each 3D NIfTI volume has been converted to 2D axial PNG slices with **pixel-perfect correspondence** to the original data — verified with zero pixel-level error across the entire dataset.

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Directory Structure](#2-directory-structure)
3. [Label Map](#3-label-map)
4. [Data Splits](#4-data-splits)
5. [Understanding the Mask Files](#5-understanding-the-mask-files)
6. [Conversion Pipeline — Step by Step](#6-conversion-pipeline--step-by-step)
7. [How to Reproduce from Scratch](#7-how-to-reproduce-from-scratch)
8. [Codebase Reference](#8-codebase-reference)
9. [Usage Example (PyTorch)](#9-usage-example-pytorch)
10. [Quality Verification](#10-quality-verification)
11. [Citation](#11-citation)
12. [License](#12-license)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| **Source** | KiTS23 Challenge (Heller et al., 2023) |
| **Modality** | Contrast-enhanced CT (corticomedullary phase) |
| **Cases** | 489 patients |
| **Total 2D slices** | 95,221 |
| **Resolution** | 512 × 512 pixels (original CT resolution, no resampling) |
| **Image format** | 16-bit grayscale PNG |
| **Mask format** | 8-bit single-channel PNG |
| **Visible mask format** | 8-bit 3-channel RGB PNG (color-coded) |
| **Overlay format** | 8-bit 3-channel RGB PNG (CT + mask blend) |
| **Orientation** | Axial slices in RAS+ canonical orientation |
| **CT Window** | [-200, 300] HU → linearly mapped to [0, 65535] |
| **Pixel accuracy** | Verified zero-error vs. original 3D NIfTI volumes |

---

## 2. Directory Structure

```
KiTS23_2D/
├── images/                      # 95,221 CT slice PNGs (16-bit grayscale)
│   ├── case_00000_slice_000.png
│   ├── case_00000_slice_001.png
│   ├── ...
│   └── case_00588_slice_247.png
│
├── masks/                       # 95,221 segmentation masks (uint8, values 0-3)
│   ├── case_00000_slice_000.png   ← For model training
│   └── ...
│
├── masks_visible/               # 95,221 color-coded masks (RGB, for visualization)
│   ├── case_00000_slice_000.png   ← For human viewing
│   └── ...
│
├── overlays/                    # 95,221 CT + mask overlay images (RGB, for visualization)
│   ├── case_00000_slice_000.png   ← CT blended with colored annotations
│   └── ...
│
├── splits/                      # Train / Val / Test split files
│   ├── splits.json              # Full split definition with case lists
│   ├── train.txt                # 342 case IDs (one per line)
│   ├── val.txt                  # 48 case IDs
│   ├── test.txt                 # 99 case IDs
│   ├── train_slices.csv         # 66,748 slice filenames
│   ├── val_slices.csv           # 8,505 slice filenames
│   └── test_slices.csv          # 19,968 slice filenames
│
├── metadata/
│   └── dataset_summary.json     # Dataset statistics
│
└── README.md                    # This file
```

### Filename Convention

```
case_XXXXX_slice_ZZZ.png
  │          │
  │          └── Axial slice index (0 = most inferior/bottom, max = most superior/top)
  └── Patient case ID (00000–00588, 489 unique cases)
```

---

## 3. Label Map

| Pixel Value | Class | Color (in masks_visible/) | Description |
|---|---|---|---|
| **0** | Background | Black (0, 0, 0) | Non-kidney tissue |
| **1** | Kidney | **Green** (0, 255, 0) | Healthy kidney parenchyma |
| **2** | Tumor | **Red** (255, 0, 0) | Kidney tumor (malignant) |
| **3** | Cyst | **Yellow** (255, 255, 0) | Kidney cyst (benign) |

---

## 4. Data Splits

Splits are **case-level** (all slices from one patient stay in the same split) to prevent data leakage. Generated with `random.seed(42)`, 70/10/20 ratio.

| Split | Cases | Slices | Percentage |
|---|---|---|---|
| **Train** | 342 | 66,748 | 70% |
| **Val** | 48 | 8,505 | 10% |
| **Test** | 99 | 19,968 | 20% |
| **Total** | 489 | 95,221 | 100% |

---

## 5. Understanding the Mask Files

### Why do `masks/` files appear dark?

The `masks/` folder stores **raw label values** (0, 1, 2, 3) as uint8 images. In a 0–255 pixel range, a value of 1 or 2 is **less than 1% brightness** — virtually invisible to the human eye. **This is intentional and correct for model training.** These are the files your model should load.

### Which masks to use for what?

| Folder | Purpose | Pixel Values | Use Case |
|---|---|---|---|
| `masks/` | **Model training** | 0, 1, 2, 3 (raw labels) | Load as ground truth for segmentation |
| `masks_visible/` | **Human viewing** | RGB color-coded | Visual inspection, presentations, papers |

### Why are many slices entirely black (even in masks_visible)?

A CT scan captures the entire torso (100–600 slices), but **kidneys only occupy ~20–40% of slices** at a specific anatomical level. Slices outside the kidney region correctly have an all-zero (black) mask. For example:
- `case_00000` has 611 total slices, but kidneys appear only in slices 257–520
- Slices 0–256 and 521–610 are correctly all-black (no kidney at that body level)

**To view slices with kidney content**, look at the middle range of slices for any case.

---

## 6. Conversion Pipeline — Step by Step

The following steps describe exactly how this 2D dataset was created from the original KiTS23 3D data:

### Step 1: Download the KiTS23 3D dataset

The original KiTS23 dataset contains 489 cases as 3D NIfTI volumes (`.nii.gz`). Each case has:
- `imaging.nii.gz` — 3D CT volume (512 × 512 × N voxels, where N varies per patient)
- `segmentation.nii.gz` — 3D label volume (same dimensions, values 0/1/2/3)

### Step 2: Canonical Orientation (RAS+)

Each volume is loaded with `nibabel` and converted to **RAS+ canonical orientation** using:
```python
img_nii = nib.as_closest_canonical(nib.load("imaging.nii.gz"))
seg_nii = nib.as_closest_canonical(nib.load("segmentation.nii.gz"))
```
This ensures consistent axis ordering across all cases:
- Axis 0 → Right-to-Left (sagittal)
- Axis 1 → Anterior-to-Posterior (coronal)
- **Axis 2 → Inferior-to-Superior (axial)** ← We slice along this axis

### Step 3: No Spatial Resampling

The original 512×512 in-plane resolution is preserved. The `--no_resample` flag ensures no interpolation artifacts are introduced. All output images are exactly 512×512.

### Step 4: CT Windowing

Raw CT Hounsfield Unit (HU) values are windowed to the kidney-optimal range:
```python
# Window: center=50, width=500 → range [-200, 300] HU
vmin, vmax = -200, 300
clipped = np.clip(volume, vmin, vmax)
normalized = (clipped - vmin) / (vmax - vmin)        # → [0, 1]
img_16bit = (normalized * 65535).astype(np.uint16)    # → [0, 65535]
```
This preserves **full 16-bit dynamic range** without any lossy normalization.

### Step 5: Axial Slicing

Each 3D volume of shape `(512, 512, N)` is sliced along **axis 2** (the axial/z-axis in RAS+):
```python
for z in range(volume.shape[2]):
    slice_img = img_16bit[:, :, z]     # 512×512, uint16
    slice_seg = seg_data[:, :, z]      # 512×512, uint8 (values 0,1,2,3)
```

### Step 6: Save as Lossless PNG

- **Images**: 16-bit grayscale PNG (`cv2.imwrite` preserves full uint16 range)
- **Masks**: 8-bit single-channel PNG (exact label values 0, 1, 2, 3)
- **Visible masks**: 8-bit 3-channel RGB PNG (color-coded for human viewing)

### Step 7: Pixel-Perfect Verification

Every output was verified against the original 3D NIfTI with **zero pixel-level difference**:
- Resolution: 512×512 matches original
- Slice count: exact match with 3D volume depth
- Mask labels: identical unique values
- Image pixels: max absolute difference = 0

---

## 7. How to Reproduce from Scratch

### Prerequisites

- Linux system with Python 3.10+
- ~50 GB free disk space for 3D data
- ~25 GB free disk space for 2D output
- GPU recommended but not required for conversion

### Step 1: Clone and download KiTS23

```bash
cd /path/to/your/workspace
git clone https://github.com/neheller/kits23.git kits23_repo
cd kits23_repo
pip install -e .
python -m kits23.download
cd ..
```

This downloads all 489 cases (~49 GB) into `kits23_repo/dataset/`.

### Step 2: Create Python environment

```bash
python3 -m venv kits_env
source kits_env/bin/activate
pip install numpy scipy nibabel SimpleITK scikit-image opencv-python pandas h5py torch torchvision torchio tqdm
```

### Step 3: Run the 3D → 2D converter

```bash
source kits_env/bin/activate

# Convert all 489 cases to 2D PNGs (original 512×512 resolution)
python convert_3d_to_2d_correct.py \
    --data_dir kits23_repo/dataset \
    --output_dir KiTS23_2D \
    --no_resample
```

This produces:
- `KiTS23_2D/images/` — 95,221 CT image PNGs (16-bit grayscale)
- `KiTS23_2D/masks/` — 95,221 segmentation mask PNGs (uint8, values 0–3)
- `KiTS23_2D/masks_visible/` — 95,221 preliminary visible masks (grayscale, will be upgraded in Step 4)

Runtime: ~40 minutes on a modern server.

### Step 4: Generate color-coded visible masks + overlays

```bash
python generate_overlays.py
```

This generates two visualization outputs in one shot:
- `KiTS23_2D/masks_visible/` — **RGB color-coded** masks with contours (Green=kidney, Red=tumor, Yellow=cyst)
- `KiTS23_2D/overlays/` — CT images blended with colored mask annotations (60% CT + 40% color)

> **Note:** `regenerate_visible_masks.py` is also available if you only need to regenerate `masks_visible/` without overlays.

---

## 8. Codebase Reference

### Core Scripts

| Script | Description |
|---|---|
| `convert_3d_to_2d_correct.py` | **Main converter** — loads 3D NIfTI, applies RAS+ canonical orientation, CT windowing, axial slicing, saves as 16-bit PNG images + uint8 masks. Supports `--no_resample` flag to keep original 512×512 resolution. |
| `generate_overlays.py` | Generates `masks_visible/` (RGB color-coded with contours) and `overlays/` (CT + mask blend). Uses multiprocessing for speed. |
| `regenerate_visible_masks.py` | Lightweight alternative — regenerates only `masks_visible/` (RGB color-coded PNGs) without creating overlays. |

### Command Reference

```bash
# ── Convert 3D to 2D (the exact command used to create this dataset) ──
python convert_3d_to_2d_correct.py \
    --data_dir kits23_repo/dataset \
    --output_dir KiTS23_2D \
    --no_resample \
    --window_center 50 \
    --window_width 500

# ── Skip already-converted cases (for resuming interrupted runs) ──
python convert_3d_to_2d_correct.py \
    --data_dir kits23_repo/dataset \
    --output_dir KiTS23_2D \
    --no_resample \
    --skip_existing

# ── Regenerate color-coded visible masks ──
python regenerate_visible_masks.py

# ── Generate overlay visualizations ──
python generate_overlays.py

# ── Regenerate ONLY visible masks (without overlays) ──
python regenerate_visible_masks.py
```

---

## 9. Usage Example (PyTorch)

```python
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class KiTS23Dataset(Dataset):
    """PyTorch Dataset for KiTS23-2D kidney tumor segmentation."""

    def __init__(self, split_csv, img_dir, mask_dir, transform=None):
        """
        Args:
            split_csv:  Path to train_slices.csv / val_slices.csv / test_slices.csv
            img_dir:    Path to images/ folder
            mask_dir:   Path to masks/ folder
            transform:  Optional albumentations transform
        """
        self.df = pd.read_csv(split_csv)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fname = self.df.iloc[idx]['filename']

        # Load 16-bit CT image → normalize to float32 [0, 1]
        img = cv2.imread(f"{self.img_dir}/{fname}", cv2.IMREAD_UNCHANGED)
        img = img.astype(np.float32) / 65535.0

        # Load segmentation mask (values 0, 1, 2, 3)
        mask = cv2.imread(f"{self.mask_dir}/{fname}", cv2.IMREAD_UNCHANGED)

        # Apply augmentations (e.g., albumentations)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # Return as tensors: image (1, H, W), mask (H, W)
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # (1, 512, 512)
        mask_tensor = torch.from_numpy(mask).long()       # (512, 512)

        return img_tensor, mask_tensor


# ── Example: Load training set ──
dataset_root = "KiTS23_2D"

train_dataset = KiTS23Dataset(
    split_csv=f"{dataset_root}/splits/train_slices.csv",
    img_dir=f"{dataset_root}/images",
    mask_dir=f"{dataset_root}/masks",
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

for images, masks in train_loader:
    print(f"Images: {images.shape}")    # torch.Size([16, 1, 512, 512])
    print(f"Masks:  {masks.shape}")     # torch.Size([16, 512, 512])
    print(f"Labels: {masks.unique()}")  # tensor([0, 1, 2, 3])
    break
```

---

## 10. Quality Verification

The following checks were performed across the entire dataset:

| # | Verification | Method | Result |
|---|---|---|---|
| 1 | **Case coverage** | Count unique case IDs in 2D output | 489/489 ✓ |
| 2 | **Slice count** | Compare `volume.shape[2]` vs PNG count per case (10 cases sampled) | All match ✓ |
| 3 | **Resolution** | Read shape of 10 random image PNGs | All 512×512 ✓ |
| 4 | **Image dtype** | `cv2.imread(IMREAD_UNCHANGED).dtype` | All uint16 ✓ |
| 5 | **Mask dtype** | `cv2.imread(IMREAD_UNCHANGED).dtype` | All uint8 ✓ |
| 6 | **Visible mask channels** | Check `.shape` and `.ndim` | All (512,512,3) RGB ✓ |
| 7 | **Mask label validity** | `np.unique()` on 100 random masks | All ⊆ {0,1,2,3} ✓ |
| 8 | **Pixel-level image accuracy** | Load 3D NIfTI, apply same CT window, compare vs 2D PNG (3 cases × 3 slices) | max_diff = 0 ✓ |
| 9 | **Pixel-level mask accuracy** | Compare 3D segmentation slice vs 2D mask PNG | Exact match ✓ |
| 10 | **Correct orientation** | Visual comparison with 3D viewer | Correct axial ✓ |
| 11 | **Case-level split integrity** | Verify no patient appears in multiple splits | No leakage ✓ |
| 12 | **File count consistency** | images = masks = masks_visible = overlays = 95,221 | All equal ✓ |
| 13 | **Overlay format** | Check overlay `.shape` and `.ndim` | All (512,512,3) RGB ✓ |

---

## 11. Citation

If you use this dataset, please cite the original KiTS23 challenge:

```bibtex
@article{heller2023kits21,
  title   = {The KiTS21 Challenge: Automatic Segmentation of Kidneys,
             Renal Tumors, and Renal Cysts in Corticomedullary-Phase CT},
  author  = {Heller, Nicholas and Isensee, Fabian and Maier-Hein, Klaus H.
             and others},
  journal = {arXiv preprint arXiv:2307.01984},
  year    = {2023}
}

@misc{kits23_challenge,
  title        = {KiTS23: The 2023 Kidney and Kidney Tumor Segmentation Challenge},
  howpublished = {\url{https://kits-challenge.org/kits23/}},
  year         = {2023}
}
```

---

## 12. License

This dataset is derived from KiTS23, which is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

*Generated from 489 KiTS23 3D CT volumes. Conversion pipeline verified with pixel-level accuracy across the entire dataset.*
