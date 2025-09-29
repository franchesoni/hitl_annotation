"""DB helpers used by ML training/inference scripts.

This module isolates functions that the ML scripts need, keeping the core
API database module (`db.py`) cleaner for the web server.

It reuses the same SQLite database and connection helper from db.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set

from .db import (  # re-exported
    DB_PATH,
    _get_conn,
    get_annotations,
    get_config,
    normalize_mask_path,
    to_ppm,
)

def get_all_samples() -> List[Dict[str, Any]]:
    """Return list of all samples with their IDs."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, sample_filepath FROM samples")
        return [
            {"id": row[0], "sample_filepath": row[1]}
            for row in cursor.fetchall()
        ]


def get_sample_ids_for_path_filter(pattern: Optional[str]) -> Optional[Set[int]]:
    """Return the set of sample IDs matching the provided glob pattern.

    When *pattern* is falsy, ``None`` is returned so callers can skip
    filtering without having to compute a set of all IDs.
    """

    if not pattern:
        return None

    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM samples WHERE sample_filepath GLOB ?",
            (pattern,),
        )
        return {int(row[0]) for row in cursor.fetchall()}


def set_predictions_batch(predictions_batch):
    """Efficiently set predictions for multiple samples in a single transaction.

    Accepts two input formats:
    - Legacy: List of tuples (sample_id, predictions_list)
    - New:    List of prediction dicts each containing at least
              {"sample_id", "class", "type", ...}

    Deletion is type-aware: existing predictions are removed only for the
    (sample_id, type) pairs present in the provided batch. This preserves
    other prediction types (e.g., keeps label predictions when inserting masks).
    """
    timestamp = int(time.time())

    if not predictions_batch:
        return

    # Normalize input into a flat list of prediction dicts with sample_id
    normalized_preds = []
    if isinstance(predictions_batch[0], (list, tuple)) and len(predictions_batch[0]) == 2:
        # Legacy format: [(sample_id, [pred_dict, ...]), ...]
        for sample_id, preds in predictions_batch:
            if not preds:
                continue
            for pred in preds:
                # Ensure required keys exist
                pred = dict(pred)
                pred["sample_id"] = sample_id
                normalized_preds.append(pred)
    else:
        # New format: [pred_dict_with_sample_id, ...]
        for pred in predictions_batch:
            if not isinstance(pred, dict) or "sample_id" not in pred:
                raise ValueError("Each prediction must be a dict with 'sample_id'.")
            normalized_preds.append(pred)

    if not normalized_preds:
        return

    with _get_conn() as conn:
        cursor = conn.cursor()

        # Resolve sample_id -> sample_filepath for all involved samples (used only for validation)
        sample_ids = sorted({p["sample_id"] for p in normalized_preds})
        placeholders = ",".join(["?"] * len(sample_ids))
        cursor.execute(
            f"SELECT id, sample_filepath FROM samples WHERE id IN ({placeholders})",
            sample_ids,
        )
        filepath_map = {row[0]: row[1] for row in cursor.fetchall()}

        # Build deletion specs: for 'label' delete by (sample_id, type),
        # for masks delete the full (sample_id, type) set so absent classes
        # are cleared, and for others (e.g., 'bbox') delete by
        # (sample_id, type, class)
        delete_specs = set()
        for p in normalized_preds:
            sid = p["sample_id"]
            ptype = p.get("type")
            pclass = p.get("class")
            if ptype is None:
                continue
            if ptype == "mask":
                delete_specs.add((sid, ptype, None))
            elif ptype == "label" or pclass is None:
                delete_specs.add((sid, ptype, None))
            else:
                delete_specs.add((sid, ptype, pclass))

        for sid, ptype, pclass in delete_specs:
            if pclass is None:
                cursor.execute(
                    "DELETE FROM predictions WHERE sample_id = ? AND type = ?",
                    (sid, ptype),
                )
            else:
                cursor.execute(
                    "DELETE FROM predictions WHERE sample_id = ? AND type = ? AND class = ?",
                    (sid, ptype, pclass),
                )

        # Prepare batch insert (convert coordinates to PPM and use col01, row01, width01, height01)
        insert_data = []
        for pred in normalized_preds:
            sid = pred["sample_id"]
            sfp = filepath_map.get(sid)
            if sfp is None:
                raise ValueError(f"Unknown sample_id {sid} in predictions batch")
            prob = pred.get("probability")
            prob_ppm = to_ppm(prob) if prob is not None else None
            # Convert coordinates to PPM
            col01 = to_ppm(pred.get("col")) if pred.get("col") is not None else None
            row01 = to_ppm(pred.get("row")) if pred.get("row") is not None else None
            width01 = to_ppm(pred.get("width")) if pred.get("width") is not None else None
            height01 = to_ppm(pred.get("height")) if pred.get("height") is not None else None
            base = [
                sid,
                pred.get("class"),
                pred.get("type"),
                prob_ppm,
                col01,
                row01,
                width01,
                height01,
                normalize_mask_path(pred.get("mask_path")),
            ]
            base.append(timestamp)
            insert_data.append(tuple(base))

        if insert_data:
            cursor.executemany(
                """
                INSERT INTO predictions (
                    sample_id, class, type,
                    probability, col01, row01, width01, height01, mask_path, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                insert_data,
            )


def store_training_stats(epoch, train_loss=None, valid_loss=None, accuracy=None):
    """Store training metrics for an epoch in the curves table.

    Compatibility: writes validation metrics using 'val_*' names
    (val_loss, val_accuracy) while keeping the function signature the
    same for callers providing 'valid_loss' and 'accuracy'.
    """
    timestamp = int(time.time())

    with _get_conn() as conn:
        cursor = conn.cursor()

        # Store each metric as a separate row
        if train_loss is not None:
            cursor.execute(
                """
                INSERT INTO curves (curve_name, value, epoch, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                ("train_loss", train_loss, epoch, timestamp),
            )

        if valid_loss is not None:
            cursor.execute(
                """
                INSERT INTO curves (curve_name, value, epoch, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                ("val_loss", valid_loss, epoch, timestamp),
            )

        if accuracy is not None:
            cursor.execute(
                """
                INSERT INTO curves (curve_name, value, epoch, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                ("val_accuracy", accuracy, epoch, timestamp),
            )

def reset_training_stats(curve_names: List[str] | None = None) -> None:
    """
    Delete all rows created by store_training_stats() from the 'curves' table.
    By default removes: train_loss, val_loss, val_accuracy.

    Note: there is no run_id, so this wipes history globally.
    """
    if curve_names is None:
        curve_names = ["train_loss", "val_loss", "val_accuracy"]

    if not curve_names:
        return

    placeholders = ",".join(["?"] * len(curve_names))
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM curves WHERE curve_name IN ({placeholders})", curve_names)
