"""Append additional samples, annotations, and predictions into the HITL database.

This helper mirrors :mod:`src.backend.db_init` but **does not** wipe or re-create
existing data.  Instead, it lets users provide a lightweight dictionary with new
entries that should be merged into the existing database.  Only the records that
are not already present will be inserted.

Usage
-----
1. Edit :func:`build_append_db_dict` below so that it returns a dictionary with
   the desired samples/annotations/predictions to import.  The schema is the same
   as :func:`src.backend.db_init.build_initial_db_dict`.
2. Run ``python -m src.backend.db_append`` once the database is already
   initialized.  Any new rows will be inserted while keeping the current data
   intact.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, Tuple

from src.backend.db import DB_PATH, SKIP_CLASS_SENTINEL, normalize_mask_path, to_ppm
from src.backend.db_init import validate_db_dict


def build_append_db_dict() -> Dict[str, Any]:
    """Return the entries that should be appended to the database.

    The dictionary must contain the keys ``"samples"``, ``"annotations"`` and
    ``"predictions"``.  Each key maps to a list following the same structure as
    the object returned by :func:`src.backend.db_init.build_initial_db_dict`.

    This function is intentionally a stub so that users can adapt it to their
    project.  Update the paths and metadata as needed before running the script.
    """

    # Example data; replace these with the actual items that should be appended.
    return {
        "samples": [
            # {"sample_filepath": "/absolute/path/to/image.jpg"},
        ],
        "annotations": [
            # {
            #     "sample_filepath": "/absolute/path/to/image.jpg",
            #     "type": "label",
            #     "class": "cat",
            # },
        ],
        "predictions": [
            # {
            #     "sample_filepath": "/absolute/path/to/image.jpg",
            #     "type": "label",
            #     "class": "cat",
            #     "probability": 0.9,
            # },
        ],
    }


def _ensure_database_exists() -> None:
    if not Path(DB_PATH).exists():
        raise RuntimeError(
            f"Database not found at {DB_PATH}. Initialise it before appending data."
        )


def _insert_samples(conn: sqlite3.Connection, samples: Iterable[Dict[str, Any]]) -> int:
    cursor = conn.cursor()
    inserted = 0
    for sample in samples:
        filepath = sample["sample_filepath"]
        cursor.execute(
            "INSERT OR IGNORE INTO samples (sample_filepath) VALUES (?);",
            (filepath,),
        )
        inserted += cursor.rowcount
    return inserted


def _get_sample_id(conn: sqlite3.Connection, sample_filepath: str) -> int:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id FROM samples WHERE sample_filepath = ?;",
        (sample_filepath,),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError(
            f"Sample '{sample_filepath}' was not found after insertion."
            " Ensure it exists on disk and was included in the samples list."
        )
    return row[0]


def _annotation_identity(values: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Return a tuple that uniquely identifies an annotation row."""

    return values


def _insert_annotations(
    conn: sqlite3.Connection, annotations: Iterable[Dict[str, Any]]
) -> int:
    cursor = conn.cursor()
    inserted = 0

    for ann in annotations:
        sample_id = _get_sample_id(conn, ann["sample_filepath"])
        ann_type = ann["type"]
        timestamp = ann.get("timestamp")

        if ann_type == "skip":
            stored_class = SKIP_CLASS_SENTINEL
        else:
            stored_class = ann["class"].strip()

        # Convert coordinates.  Accept either *_ppm fields or normalized floats.
        def _ppm(value: Any | None) -> Any | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return to_ppm(value)

        col01 = ann.get("col01")
        row01 = ann.get("row01")
        width01 = ann.get("width01")
        height01 = ann.get("height01")

        if col01 is None and "col" in ann:
            col01 = to_ppm(ann.get("col"))
        elif col01 is not None:
            col01 = _ppm(col01)

        if row01 is None and "row" in ann:
            row01 = to_ppm(ann.get("row"))
        elif row01 is not None:
            row01 = _ppm(row01)

        if width01 is None and "width" in ann:
            width01 = to_ppm(ann.get("width"))
        elif width01 is not None:
            width01 = _ppm(width01)

        if height01 is None and "height" in ann:
            height01 = to_ppm(ann.get("height"))
        elif height01 is not None:
            height01 = _ppm(height01)

        identity = _annotation_identity(
            (
                sample_id,
                stored_class,
                ann_type,
                col01,
                row01,
                width01,
                height01,
                timestamp,
            )
        )

        cursor.execute(
            """
            SELECT 1 FROM annotations
            WHERE sample_id = ? AND class = ? AND type = ?
              AND IFNULL(col01, -1) = IFNULL(?, -1)
              AND IFNULL(row01, -1) = IFNULL(?, -1)
              AND IFNULL(width01, -1) = IFNULL(?, -1)
              AND IFNULL(height01, -1) = IFNULL(?, -1)
              AND IFNULL(timestamp, -1) = IFNULL(?, -1)
            LIMIT 1;
            """,
            identity,
        )
        if cursor.fetchone():
            continue

        cursor.execute(
            """
            INSERT INTO annotations (
                sample_id, class, type,
                col01, row01, width01, height01, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            identity,
        )
        inserted += 1

    return inserted


def _prediction_identity(values: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Return a tuple that uniquely identifies a prediction row."""

    return values


def _insert_predictions(
    conn: sqlite3.Connection, predictions: Iterable[Dict[str, Any]]
) -> int:
    cursor = conn.cursor()
    inserted = 0

    for pred in predictions:
        sample_id = _get_sample_id(conn, pred["sample_filepath"])
        timestamp = pred.get("timestamp")

        probability = pred.get("probability")
        probability_ppm = to_ppm(probability) if probability is not None else None

        def _ppm_value(key: str) -> Any | None:
            if key + "01" in pred:
                value = pred.get(key + "01")
                if value is None:
                    return None
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return to_ppm(value)
            if key in pred:
                return to_ppm(pred.get(key))
            return None

        col01 = _ppm_value("col")
        row01 = _ppm_value("row")
        width01 = _ppm_value("width")
        height01 = _ppm_value("height")
        mask_path = normalize_mask_path(pred.get("mask_path"))

        identity = _prediction_identity(
            (
                sample_id,
                pred["class"],
                pred["type"],
                probability_ppm,
                col01,
                row01,
                width01,
                height01,
                mask_path,
                timestamp,
            )
        )

        cursor.execute(
            """
            SELECT 1 FROM predictions
            WHERE sample_id = ? AND class = ? AND type = ?
              AND IFNULL(probability, -1) = IFNULL(?, -1)
              AND IFNULL(col01, -1) = IFNULL(?, -1)
              AND IFNULL(row01, -1) = IFNULL(?, -1)
              AND IFNULL(width01, -1) = IFNULL(?, -1)
              AND IFNULL(height01, -1) = IFNULL(?, -1)
              AND IFNULL(mask_path, '') = IFNULL(?, '')
              AND IFNULL(timestamp, -1) = IFNULL(?, -1)
            LIMIT 1;
            """,
            identity,
        )
        if cursor.fetchone():
            continue

        cursor.execute(
            """
            INSERT INTO predictions (
                sample_id, class, type,
                probability, col01, row01, width01, height01, mask_path, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            identity,
        )
        inserted += 1

    return inserted


def main() -> None:
    db_dict = build_append_db_dict()
    validate_db_dict(db_dict)
    _ensure_database_exists()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        new_samples = _insert_samples(conn, db_dict["samples"])
        new_annotations = _insert_annotations(conn, db_dict["annotations"])
        new_predictions = _insert_predictions(conn, db_dict["predictions"])

    print(
        "Appended:"\
        f" {new_samples} samples,"\
        f" {new_annotations} annotations,"\
        f" {new_predictions} predictions."
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
