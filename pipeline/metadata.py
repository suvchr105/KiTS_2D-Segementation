"""
KiTS23 Metadata Intelligence Layer
=====================================
Generate comprehensive metadata CSVs for:
  - Slice-level statistics
  - Case-level summaries
  - Dataset-level overview
  - Difficulty scores
  - Detection annotations
  
Enables: curriculum learning, dynamic sampling, fairness evaluation.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MetadataCollector:
    """Collect and save metadata across the entire pipeline."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)

        self.slice_records: List[Dict[str, Any]] = []
        self.case_records: List[Dict[str, Any]] = []
        self.difficulty_records: List[Dict[str, Any]] = []
        self.detection_records: List[Dict[str, Any]] = []
        self.patch_records: List[Dict[str, Any]] = []

    def add_slice_metadata(self, meta: Dict[str, Any]):
        """Add metadata for one slice."""
        self.slice_records.append(meta)

    def add_case_metadata(self, meta: Dict[str, Any]):
        """Add metadata for one case."""
        self.case_records.append(meta)

    def add_difficulty_record(self, record: Dict[str, Any]):
        """Add difficulty score for one slice."""
        self.difficulty_records.append(record)

    def add_detection_record(self, record: Dict[str, Any]):
        """Add detection annotation for one slice."""
        self.detection_records.append(record)

    def add_patch_record(self, record: Dict[str, Any]):
        """Add 3D patch metadata."""
        self.patch_records.append(record)

    def save_all(self):
        """Save all collected metadata to disk."""
        self._save_slice_metadata()
        self._save_case_metadata()
        self._save_difficulty_metadata()
        self._save_detection_metadata()
        self._save_patch_metadata()
        self._save_dataset_summary()
        logger.info(f"All metadata saved to {self.metadata_dir}")

    def _save_slice_metadata(self):
        if not self.slice_records:
            return
        df = pd.DataFrame(self.slice_records)
        path = os.path.join(self.metadata_dir, "slice_metadata.csv")
        df.to_csv(path, index=False)
        logger.info(f"Slice metadata: {len(df)} records -> {path}")

    def _save_case_metadata(self):
        if not self.case_records:
            return
        df = pd.DataFrame(self.case_records)
        path = os.path.join(self.metadata_dir, "case_metadata.csv")
        df.to_csv(path, index=False)
        logger.info(f"Case metadata: {len(df)} records -> {path}")

    def _save_difficulty_metadata(self):
        if not self.difficulty_records:
            return
        df = pd.DataFrame(self.difficulty_records)
        path = os.path.join(self.metadata_dir, "difficulty_scores.csv")
        df.to_csv(path, index=False)
        logger.info(f"Difficulty scores: {len(df)} records -> {path}")

    def _save_detection_metadata(self):
        if not self.detection_records:
            return
        df = pd.DataFrame(self.detection_records)
        path = os.path.join(self.metadata_dir, "detection_annotations.csv")
        df.to_csv(path, index=False)
        logger.info(f"Detection annotations: {len(df)} records -> {path}")

    def _save_patch_metadata(self):
        if not self.patch_records:
            return
        # Clean up non-serializable fields
        clean_records = []
        for r in self.patch_records:
            cr = {k: v for k, v in r.items() if k != "label_counts"}
            if "label_counts" in r:
                cr["label_counts"] = json.dumps(r["label_counts"])
            clean_records.append(cr)
        df = pd.DataFrame(clean_records)
        path = os.path.join(self.metadata_dir, "patch_metadata.csv")
        df.to_csv(path, index=False)
        logger.info(f"Patch metadata: {len(df)} records -> {path}")

    def _save_dataset_summary(self):
        """Generate and save overall dataset summary."""
        if not self.slice_records:
            return

        df = pd.DataFrame(self.slice_records)

        summary = {
            "total_cases": int(df["case_id"].nunique()),
            "total_slices": len(df),
            "tumor_slices": int(df["has_tumor"].sum()) if "has_tumor" in df else 0,
            "kidney_only_slices": int((df["category"] == "kidney_only").sum()) if "category" in df else 0,
            "background_slices": int((df["category"] == "background").sum()) if "category" in df else 0,
            "avg_height": float(df["height"].mean()) if "height" in df else 0,
            "avg_width": float(df["width"].mean()) if "width" in df else 0,
            "total_tumor_pixels": int(df["tumor_pixels"].sum()) if "tumor_pixels" in df else 0,
            "total_kidney_pixels": int(df["kidney_pixels"].sum()) if "kidney_pixels" in df else 0,
        }

        if self.difficulty_records:
            diff_df = pd.DataFrame(self.difficulty_records)
            tumor_diff = diff_df[diff_df["total_difficulty"] > 0]
            if len(tumor_diff) > 0:
                summary["avg_difficulty"] = float(tumor_diff["total_difficulty"].mean())
                summary["max_difficulty"] = float(tumor_diff["total_difficulty"].max())
                for level in ["easy", "medium", "hard", "very_hard"]:
                    summary[f"difficulty_{level}"] = int(
                        (tumor_diff["difficulty_level"] == level).sum()
                    )

        if self.patch_records:
            summary["total_3d_patches"] = len(self.patch_records)
            summary["patches_with_tumor"] = sum(
                1 for p in self.patch_records if p.get("has_tumor")
            )

        path = os.path.join(self.metadata_dir, "dataset_summary.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Dataset summary -> {path}")

        # Also print to console
        logger.info("=" * 60)
        logger.info("DATASET SUMMARY")
        logger.info("=" * 60)
        for k, v in summary.items():
            logger.info(f"  {k}: {v}")
        logger.info("=" * 60)


def generate_train_val_test_csvs(
    metadata_dir: str,
    splits_dir: str,
):
    """Generate per-split CSV files for easy data loading.
    
    Creates: train_slices.csv, val_slices.csv, test_slices.csv
    Each contains slice-level metadata for that split only.
    """
    splits_path = os.path.join(splits_dir, "splits.json")
    slice_csv_path = os.path.join(metadata_dir, "slice_metadata.csv")

    if not os.path.exists(splits_path) or not os.path.exists(slice_csv_path):
        logger.warning("Cannot generate split CSVs: missing splits.json or slice_metadata.csv")
        return

    with open(splits_path, 'r') as f:
        splits = json.load(f)

    df = pd.read_csv(slice_csv_path)

    for split_name in ["train", "val", "test"]:
        case_ids = splits.get(split_name, [])
        split_df = df[df["case_id"].isin(case_ids)]

        out_path = os.path.join(metadata_dir, f"{split_name}_slices.csv")
        split_df.to_csv(out_path, index=False)

        logger.info(
            f"  {split_name}: {len(split_df)} slices from {len(case_ids)} cases -> {out_path}"
        )
