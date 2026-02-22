"""
KiTS23 Detection Annotation Generator
=======================================
Convert instance segmentation masks to bounding box annotations.
Supports COCO, YOLO, and CSV formats.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# Bounding Box Extraction
# ────────────────────────────────────────────────────────────────

def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask to (x1, y1, x2, y2) bounding box.
    
    Returns None if mask is empty.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    y1, x1 = coords.min(axis=0)
    y2, x2 = coords.max(axis=0)
    return (int(x1), int(y1), int(x2), int(y2))


def mask_to_bbox_xywh(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Convert a binary mask to (x, y, w, h) bounding box (COCO format)."""
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def mask_to_yolo(
    mask: np.ndarray, img_h: int, img_w: int
) -> Optional[Tuple[float, float, float, float]]:
    """Convert a binary mask to YOLO format (cx, cy, w, h) normalized.
    
    Returns None if mask is empty.
    """
    bbox = mask_to_bbox(mask)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1 + 1) / img_w
    h = (y2 - y1 + 1) / img_h
    return (cx, cy, w, h)


# ────────────────────────────────────────────────────────────────
# Extract All Bboxes from a Semantic Mask
# ────────────────────────────────────────────────────────────────

def extract_bboxes_from_semantic(
    mask: np.ndarray,
    min_area: int = 10,
) -> List[Dict[str, Any]]:
    """Extract bounding boxes from a semantic segmentation mask.
    
    Uses connected components to find individual object instances.
    
    Returns list of dicts with:
      - class_id: int (1=kidney, 2=tumor, 3=cyst)
      - class_name: str
      - bbox_xyxy: (x1, y1, x2, y2)
      - bbox_xywh: (x, y, w, h)
      - area: int
      - mask: binary mask
    """
    label_names = {1: "kidney", 2: "tumor", 3: "cyst"}
    results = []

    for label_id, label_name in label_names.items():
        binary = (mask == label_id).astype(np.uint8)
        if binary.sum() == 0:
            continue

        # Connected components for individual instances
        num_labels, labels_cc = cv2.connectedComponents(binary)

        for cc_id in range(1, num_labels):
            cc_mask = (labels_cc == cc_id).astype(np.uint8)
            area = int(cc_mask.sum())

            if area < min_area:
                continue

            bbox = mask_to_bbox(cc_mask)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            results.append({
                "class_id": label_id,
                "class_name": label_name,
                "bbox_xyxy": bbox,
                "bbox_xywh": (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
                "area": area,
                "cc_mask": cc_mask,
            })

    return results


def extract_bboxes_from_instances(
    instance_masks: Dict[str, np.ndarray],
    min_area: int = 10,
) -> List[Dict[str, Any]]:
    """Extract bounding boxes from KiTS23 instance masks.
    
    Instance mask names follow pattern: 
      kidney_instance-X_annotation-Y
      tumor_instance-X_annotation-Y
    """
    results = []
    label_map = {"kidney": 1, "tumor": 2, "cyst": 3}

    for name, mask in instance_masks.items():
        if mask.sum() == 0:
            continue

        # Parse class from instance name
        class_name = name.split("_instance")[0]
        class_id = label_map.get(class_name, 0)
        if class_id == 0:
            continue

        area = int(mask.sum())
        if area < min_area:
            continue

        bbox = mask_to_bbox(mask)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "instance_name": name,
            "bbox_xyxy": bbox,
            "bbox_xywh": (x1, y1, x2 - x1 + 1, y2 - y1 + 1),
            "area": area,
        })

    return results


# ────────────────────────────────────────────────────────────────
# COCO Format Export
# ────────────────────────────────────────────────────────────────

class COCOAnnotationBuilder:
    """Build COCO-format detection annotations incrementally."""

    def __init__(self):
        self.images = []
        self.annotations = []
        self.categories = [
            {"id": 1, "name": "kidney", "supercategory": "organ"},
            {"id": 2, "name": "tumor", "supercategory": "lesion"},
            {"id": 3, "name": "cyst", "supercategory": "lesion"},
        ]
        self._ann_id = 1
        self._img_id = 1

    def add_image(
        self,
        file_name: str,
        height: int,
        width: int,
        case_id: str = "",
        slice_idx: int = 0,
    ) -> int:
        """Add an image entry. Returns image_id."""
        img_id = self._img_id
        self.images.append({
            "id": img_id,
            "file_name": file_name,
            "height": height,
            "width": width,
            "case_id": case_id,
            "slice_idx": slice_idx,
        })
        self._img_id += 1
        return img_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        bbox_xywh: Tuple[int, int, int, int],
        area: int,
        instance_name: str = "",
    ) -> int:
        """Add an annotation entry. Returns annotation_id."""
        ann_id = self._ann_id
        self.annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": list(bbox_xywh),
            "area": area,
            "iscrowd": 0,
            "instance_name": instance_name,
        })
        self._ann_id += 1
        return ann_id

    def to_dict(self) -> Dict:
        return {
            "images": self.images,
            "annotations": self.annotations,
            "categories": self.categories,
        }

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"COCO annotations saved: {path} "
                     f"({len(self.images)} images, {len(self.annotations)} annotations)")


# ────────────────────────────────────────────────────────────────
# YOLO Format Export
# ────────────────────────────────────────────────────────────────

def save_yolo_annotation(
    path: str,
    bboxes: List[Dict[str, Any]],
    img_h: int,
    img_w: int,
):
    """Save bounding boxes in YOLO format (class cx cy w h, normalized)."""
    lines = []
    for bb in bboxes:
        x1, y1, x2, y2 = bb["bbox_xyxy"]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1 + 1) / img_w
        h = (y2 - y1 + 1) / img_h
        # YOLO class is 0-indexed
        cls = bb["class_id"] - 1
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    with open(path, 'w') as f:
        f.write("\n".join(lines))


# ────────────────────────────────────────────────────────────────
# CSV Format Export
# ────────────────────────────────────────────────────────────────

def save_csv_annotation(
    path: str,
    bboxes: List[Dict[str, Any]],
    case_id: str,
    slice_idx: int,
):
    """Append bounding box annotations to CSV."""
    import csv
    file_exists = os.path.exists(path)

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "case_id", "slice_idx", "class_id", "class_name",
                "x1", "y1", "x2", "y2", "area"
            ])
        for bb in bboxes:
            x1, y1, x2, y2 = bb["bbox_xyxy"]
            writer.writerow([
                case_id, slice_idx, bb["class_id"], bb["class_name"],
                x1, y1, x2, y2, bb["area"]
            ])
