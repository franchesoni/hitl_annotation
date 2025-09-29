"""SQLite data access layer for the HITL annotation app.

This module centralizes all reads/writes to the database. Functions are
small, composable and return primitive Python types (dicts, lists, bools)
so that the Flask layer can remain thin. All write operations run within
`with _get_conn() as conn:` blocks to ensure transactions are committed or
rolled back atomically.

Tables used: samples, annotations, predictions, config, curves.
"""

import sqlite3
import os
import json
import time
from pathlib import Path


SKIP_ANNOTATION_TYPE = "skip"
SKIP_CLASS_SENTINEL = "__SKIP__"

DB_PATH = "session/app.db"
SESSION_DIR = Path(DB_PATH).parent
PREDS_DIR = SESSION_DIR / "preds"


def normalize_mask_path(mask_path: str | None) -> str | None:
    """Normalize a mask path so it lives under the session ``preds`` directory."""

    if not mask_path:
        return None

    try:
        PREDS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        # Directory creation failures are non-fatal for normalization
        pass

    try:
        path = Path(mask_path)
    except Exception:
        return None

    try:
        if path.is_absolute():
            resolved = path.resolve()
        else:
            parts = path.parts
            if parts and parts[0] == "preds":
                resolved = (SESSION_DIR / path).resolve()
            else:
                resolved = (PREDS_DIR / path).resolve()

        # Ensure the mask lives under the predictions directory
        resolved.relative_to(PREDS_DIR.resolve())
    except Exception:
        return None

    try:
        rel = resolved.relative_to(SESSION_DIR.resolve())
        return rel.as_posix()
    except Exception:
        try:
            rel = resolved.relative_to(PREDS_DIR.resolve())
            return f"preds/{rel.as_posix()}"
        except Exception:
            return None

def _get_conn():
    """Open a SQLite connection to the app database.

    Raises if the DB file is missing; initialization is handled elsewhere.

    Returns:
        sqlite3.Connection: Connection with a 30s timeout.
    """
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database not found at {DB_PATH}. Did you run db_init.py?")
    return sqlite3.connect(DB_PATH, timeout=30.0)


def to_ppm(value):
    """Convert a float in [0,1] to integer parts-per-million, clamped to [0, 1_000_000]."""
    v = float(value)
    ppm = int(round(v * 1_000_000))
    if ppm < 0:
        ppm = 0
    elif ppm > 1_000_000:
        ppm = 1_000_000
    return ppm


def from_ppm(ppm):
    """Convert stored coordinate to float in [0,1].

    Backward-compatible: if value already looks like a normalized float in [0,1],
    return as-is. Otherwise treat as integer PPM and scale down.
    """
    if ppm is None:
        return None
    try:
        f = float(ppm)
    except (TypeError, ValueError):
        return None
    # If it's already normalized, pass through
    if 0.0 <= f <= 1.0:
        return f
    # Otherwise interpret as integer PPM
    try:
        p = int(round(f))
    except (TypeError, ValueError):
        return None
    if p < 0:
        p = 0
    elif p > 1_000_000:
        p = 1_000_000
    return p / 1_000_000.0


def get_config():
    """Return the configuration as a dict or empty dict if not set.

    Includes the 'task' field ("classification" | "segmentation").
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT classes, ai_should_be_run, architecture, budget, resize,
                   last_claim_cleanup, task, sample_path_filter, mask_loss_weight
            FROM config
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        if row:
            (
                classes,
                ai_should_be_run,
                architecture,
                budget,
                resize,
                last_claim_cleanup,
                task,
                sample_path_filter,
                mask_loss_weight,
            ) = row
            sample_filter_normalized = _normalize_sample_path_filter(sample_path_filter)
            count_cursor = conn.cursor()
            sample_filter_count = _count_samples_matching_filter(count_cursor, sample_filter_normalized)
            return {
                "classes": json.loads(classes) if classes else [],
                "ai_should_be_run": bool(ai_should_be_run),
                "architecture": architecture,
                "budget": budget,
                "resize": resize,
                "last_claim_cleanup": last_claim_cleanup,
                "task": task,
                "sample_path_filter": sample_filter_normalized,
                "sample_path_filter_count": sample_filter_count,
                "mask_loss_weight": mask_loss_weight if mask_loss_weight is not None else 1.0,
            }
        return {}

def update_config(config):
    """Merge and persist the provided config dict with validation.

    Accepts keys: classes (list[str]), ai_should_be_run (bool), architecture (str),
    budget (int), resize (int), task ("classification"|"segmentation"),
    last_claim_cleanup (int|None). Unknown keys are ignored. Values are validated
    and coerced where reasonable; out-of-range or invalid types are dropped.
    """
    current = get_config() or {}
    # Remove computed fields before merging
    current.pop("sample_path_filter_count", None)
    if current.get("mask_loss_weight") is None:
        current["mask_loss_weight"] = 1.0

    # Helper validators
    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in {"1", "true", "yes", "on"}
        return None

    def _int(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    def _float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _clamp(v, lo, hi):
        if v is None:
            return None
        return max(lo, min(hi, v))

    def _classes_list(v):
        if not isinstance(v, list):
            return None
        out = []
        for it in v:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
        # dedupe while preserving order
        seen = set()
        uniq = []
        for s in out:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def _task(v):
        if isinstance(v, str) and v in {"classification", "segmentation"}:
            return v
        return None

    def _architecture(v):
        if v is None:
            return None
        if not isinstance(v, str) or not v.strip():
            return None
        arch = v.strip()
        # Optional allowlist using timm if available
        try:
            import timm  # type: ignore
            allowed = {"resnet18", "resnet34", "small", "small+", "base", "large"} | set(timm.list_models())
            if arch not in allowed:
                return None
        except Exception:
            # If timm not available here, accept any non-empty string
            pass
        return arch

    def _sample_path_filter(v):
        return _normalize_sample_path_filter(v, coerce_to_string=True)

    # Build a proposed update with validation/coercion
    proposed = {}
    if "classes" in config:
        cl = _classes_list(config.get("classes"))
        if cl is not None:
            proposed["classes"] = cl
    if "ai_should_be_run" in config:
        b = _bool(config.get("ai_should_be_run"))
        if b is not None:
            proposed["ai_should_be_run"] = b
    if "architecture" in config:
        a = _architecture(config.get("architecture"))
        if a is not None:
            proposed["architecture"] = a
    if "budget" in config:
        n = _int(config.get("budget"))
        if n is not None:
            proposed["budget"] = max(0, n)
    if "resize" in config:
        n = _int(config.get("resize"))
        if n is not None:
            proposed["resize"] = _clamp(n, 16, 4096)
    if "task" in config:
        t = _task(config.get("task"))
        if t is not None:
            proposed["task"] = t
    if "last_claim_cleanup" in config:
        n = _int(config.get("last_claim_cleanup"))
        proposed["last_claim_cleanup"] = n
    if "sample_path_filter" in config:
        proposed["sample_path_filter"] = _sample_path_filter(config.get("sample_path_filter"))

    if "mask_loss_weight" in config:
        f = _float(config.get("mask_loss_weight"))
        if f is not None and f >= 0.0:
            proposed["mask_loss_weight"] = f

    # tolerate camelCase from clients
    if "samplePathFilter" in config and "sample_path_filter" not in proposed:
        proposed["sample_path_filter"] = _sample_path_filter(config.get("samplePathFilter"))

    # Merge
    current.update(proposed)

    # Convert classes list to JSON string for storage
    classes_json = json.dumps(current.get("classes", []))
    sample_filter_value = current.get("sample_path_filter")

    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM config")
        count = cursor.fetchone()[0]

        if count > 0:
            cursor.execute(
                """
                UPDATE config
                   SET classes = ?,
                       ai_should_be_run = ?,
                       architecture = ?,
                       budget = ?,
                       resize = ?,
                       last_claim_cleanup = ?,
                       task = ?,
                       sample_path_filter = ?,
                       mask_loss_weight = ?
                """,
                (classes_json, int(current.get("ai_should_be_run", False)),
                 current.get("architecture"), current.get("budget"),
                 current.get("resize"), current.get("last_claim_cleanup"),
                 current.get("task"), sample_filter_value,
                 current.get("mask_loss_weight"))
            )
        else:
            cursor.execute(
                """
                INSERT INTO config (
                    classes, ai_should_be_run, architecture, budget, resize,
                    last_claim_cleanup, task, sample_path_filter, mask_loss_weight
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (classes_json, int(current.get("ai_should_be_run", False)),
                 current.get("architecture"), current.get("budget"),
                 current.get("resize"), current.get("last_claim_cleanup"),
                 current.get("task"), sample_filter_value,
                 current.get("mask_loss_weight"))
            )


def _normalize_sample_path_filter(value, *, coerce_to_string=False):
    """Normalize a user-provided sample filepath glob filter.

    Returns None when empty/invalid so the caller can treat it as disabled.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if coerce_to_string:
        stripped = str(value).strip()
        return stripped or None
    return None


def _count_samples_matching_filter(cursor, pattern):
    """Return the number of samples matching the provided SQLite GLOB pattern."""
    if not pattern:
        cursor.execute("SELECT COUNT(*) FROM samples")
    else:
        cursor.execute(
            "SELECT COUNT(*) FROM samples WHERE sample_filepath GLOB ?",
            (pattern,),
        )
    row = cursor.fetchone()
    return row[0] if row else 0


def count_samples_matching_filter(pattern):
    """Public helper to count samples matching a filepath glob pattern."""
    normalized = _normalize_sample_path_filter(pattern, coerce_to_string=True)
    with _get_conn() as conn:
        cursor = conn.cursor()
        return _count_samples_matching_filter(cursor, normalized)

def get_all_samples():
    """Return all samples. (Not implemented here; see src.backend.db_ml.get_all_samples)"""
    raise NotImplementedError("Moved to src.backend.db_ml.get_all_samples")


def _unlabeled_annotation_types_for_task():
    """Return annotation types marking a sample as completed for the current task."""
    task = (get_config() or {}).get("task", "classification")
    if task == "segmentation":
        return ("point",)
    return ("label", SKIP_ANNOTATION_TYPE)

def _get_next_unlabeled_sequential():
    """Returns the next unlabeled sample info without claiming it.

    "Unlabeled" depends on the current task:
      - classification → no label annotations
      - segmentation   → no point annotations
    """
    target_types = _unlabeled_annotation_types_for_task()
    sample_filter = (get_config() or {}).get("sample_path_filter")
    placeholders = ",".join(["?"] * len(target_types))
    with _get_conn() as conn:
        cursor = conn.cursor()
        filter_clause = "\n              AND s.sample_filepath GLOB ?" if sample_filter else ""
        params = list(target_types)
        if sample_filter:
            params.append(sample_filter)
        cursor.execute(
            f"""
            SELECT s.id, s.sample_filepath
            FROM samples s
            WHERE s.claimed = 0
              AND NOT EXISTS (
                SELECT 1 FROM annotations a
                WHERE a.sample_id = s.id AND a.type IN ({placeholders})
              )
            {filter_clause}
            ORDER BY s.id
            LIMIT 1
            """,
            params,
        )
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def _get_next_unlabeled_random():
    """Returns a random unlabeled sample info without claiming it (task-aware)."""
    target_types = _unlabeled_annotation_types_for_task()
    sample_filter = (get_config() or {}).get("sample_path_filter")
    placeholders = ",".join(["?"] * len(target_types))
    with _get_conn() as conn:
        cursor = conn.cursor()
        filter_clause = "\n              AND s.sample_filepath GLOB ?" if sample_filter else ""
        params = list(target_types)
        if sample_filter:
            params.append(sample_filter)
        cursor.execute(
            f"""
            SELECT s.id, s.sample_filepath
            FROM samples s
            WHERE s.claimed = 0
              AND NOT EXISTS (
                SELECT 1 FROM annotations a
                WHERE a.sample_id = s.id AND a.type IN ({placeholders})
              )
            {filter_clause}
            ORDER BY RANDOM()
            LIMIT 1
            """,
            params,
        )
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def _get_unlabeled_pick(pick, highest_probability=True):
    """Return unlabeled sample predicted as class `pick`, ordered by probability.

    Unlabeled criterion is task-aware; predictions are label-type.
    """
    target_types = _unlabeled_annotation_types_for_task()
    sample_filter = (get_config() or {}).get("sample_path_filter")
    placeholders = ",".join(["?"] * len(target_types))
    order_direction = "DESC" if highest_probability else "ASC"
    with _get_conn() as conn:
        cursor = conn.cursor()
        filter_clause = "\n              AND s.sample_filepath GLOB ?" if sample_filter else ""
        params = [pick, *target_types]
        if sample_filter:
            params.append(sample_filter)
        cursor.execute(
            f"""
            SELECT s.id, s.sample_filepath, p.probability
            FROM samples s
            JOIN predictions p ON s.id = p.sample_id AND p.type = 'label'
            WHERE s.claimed = 0
              AND p.class = ?
              AND NOT EXISTS (
                SELECT 1 FROM annotations a
                WHERE a.sample_id = s.id AND a.type IN ({placeholders})
              )
            {filter_clause}
            ORDER BY p.probability {order_direction}
            LIMIT 1
            """,
            params,
        )
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1], "probability": result[2]} if result else None

def _get_annotation_counts():
    """Return counts per class considering only label annotations.

    Used to determine the minority class for sampling strategies.
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT class, COUNT(*) as count
            FROM annotations
            WHERE type = 'label'
            GROUP BY class
            ORDER BY count ASC
            """
        )
        results = cursor.fetchall()
        return {row[0]: row[1] for row in results}

def _get_minority_unlabeled_frontier():
    """Returns the sample with the lowest probability in the minority class."""
    annotation_counts = _get_annotation_counts()
    # Get class with minimum annotations
    if len(annotation_counts) == 0:
        return None
    minority_class = min(annotation_counts.keys(), key=lambda x: annotation_counts[x])
    # Use get_unlabeled_pick with lowest probability
    return _get_unlabeled_pick(minority_class, highest_probability=False)

def _claim_sample(sample_id):
    """Atomically claim a sample by its ID. Returns True if successful."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE samples 
            SET claimed = 1
            WHERE id = ? AND claimed = 0
        """, (sample_id,))
        return cursor.rowcount > 0

def release_claim_by_id(sample_id):
    """Release the claim on a sample by ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE samples 
            SET claimed = 0
            WHERE id = ?
        """, (sample_id,))

def cleanup_claims_unconditionally():
    """Reset all claimed samples to unclaimed (claimed=0) without conditions.

    Intended as a safety/cleanup helper to clear stale claims across sessions.
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE samples
            SET claimed = 0
            WHERE claimed = 1
            """
        )

def get_annotations(sample_id):
    """Get all annotations for a specific sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.id, a.sample_id, s.sample_filepath, a.class, a.type,
                   a.col01, a.row01, a.width01, a.height01, a.mask_path, a.timestamp
            FROM annotations a
            JOIN samples s ON s.id = a.sample_id
            WHERE a.sample_id = ?
        """, (sample_id,))
        results = cursor.fetchall()
        annotations = []
        for row in results:
            ann_type = row[4]
            raw_class = row[3]
            class_value = None if ann_type == SKIP_ANNOTATION_TYPE or raw_class == SKIP_CLASS_SENTINEL else raw_class
            ann = {
                "id": row[0],
                "sample_id": row[1],
                "sample_filepath": row[2],
                "class": class_value,
                "type": ann_type,
                "timestamp": row[10]
            }
            # Add coordinates based on type
            if row[4] == "point":  # point type
                if row[5] is not None and row[6] is not None:
                    ann["col01"] = row[5]
                    ann["row01"] = row[6]
            elif row[4] == "bbox":  # bbox type
                if all(x is not None for x in row[5:9]):
                    ann["col01"] = row[5]
                    ann["row01"] = row[6]
                    ann["width01"] = row[7]
                    ann["height01"] = row[8]
            elif ann_type == "mask":
                mask_path = row[9]
                if mask_path:
                    ann["mask_path"] = mask_path
            elif ann_type == SKIP_ANNOTATION_TYPE:
                ann["skipped"] = True
            # label type has no coordinates
            annotations.append(ann)
        return annotations

def get_predictions(sample_id):
    """Get all predictions for a specific sample ID.

    Includes label, bbox, and mask predictions. For masks, returns
    the stored mask_path as provided by ML scripts.
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT p.id, p.sample_id, s.sample_filepath, p.class, p.type,
                   p.probability, p.col01, p.row01, p.width01, p.height01, p.mask_path, p.timestamp
            FROM predictions p
            JOIN samples s ON s.id = p.sample_id
            WHERE p.sample_id = ?
            ORDER BY p.timestamp ASC
            """,
            (sample_id,),
        )
        results = cursor.fetchall()
        predictions = []
        for row in results:
            pred = {
                "id": row[0],
                "sample_id": row[1],
                "sample_filepath": row[2],
                "class": row[3],
                "type": row[4],
                "timestamp": row[11],
            }
            ptype = row[4]
            if ptype == "label":
                pred["probability"] = from_ppm(row[5])
            elif ptype == "bbox":
                if all(x is not None for x in row[6:10]):
                    pred["col01"] = row[6]
                    pred["row01"] = row[7]
                    pred["width01"] = row[8]
                    pred["height01"] = row[9]
            elif ptype == "mask":
                pred["mask_path"] = row[10]
            predictions.append(pred)
        return predictions

def get_next_sample_by_strategy(strategy=None, pick=None):
    """
    Get the next sample to annotate based on the given strategy.
    Returns a dict with sample info or None if no samples available.
    Claims the sample atomically just before returning.
    """
    if strategy is None:  # default
        return get_next_sample_by_strategy("sequential")
    elif strategy == "sequential":
        sample_info = _get_next_unlabeled_sequential()
    elif strategy == "random":
        sample_info = _get_next_unlabeled_random()
    elif strategy == "pick_class" or strategy == "specific_class":
        assert pick is not None, "Pick must be provided for 'pick_class' or 'specific_class' strategy"
        sample_info = _get_unlabeled_pick(pick)
    elif strategy == "minority_frontier":
        sample_info = _get_minority_unlabeled_frontier()
    elif strategy == "minority_frontier_optimized":
        # Optimized version where frontend provides the minority class
        assert pick is not None, "Pick (minority class) must be provided for 'minority_frontier_optimized' strategy"
        sample_info = _get_unlabeled_pick(pick, highest_probability=False)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    if strategy != "sequential" and sample_info is None:
        # Per spec, fall back to random if no candidate found
        return get_next_sample_by_strategy("random", None)

    if not sample_info:
        return None
    if _claim_sample(sample_info["id"]):
        return sample_info
    else:
        # Sample was claimed by someone else, try again
        return get_next_sample_by_strategy(strategy, pick)

def get_sample_by_id(sample_id):
    """Get sample info by sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sample_filepath
            FROM samples
            WHERE id = ?
        """, (sample_id,))
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def get_sample_prev_by_id(sample_id):
    """Get the previous sample by ID (highest ID that is less than the given ID)."""
    sample_filter = (get_config() or {}).get("sample_path_filter")
    with _get_conn() as conn:
        cursor = conn.cursor()
        filter_clause = "\n              AND sample_filepath GLOB ?" if sample_filter else ""
        params = [sample_id]
        if sample_filter:
            params.append(sample_filter)
        cursor.execute(
            f"""
            SELECT id, sample_filepath
            FROM samples
            WHERE id < ?
            {filter_clause}
            ORDER BY id DESC
            LIMIT 1
        """,
            params,
        )
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def get_sample_next_by_id(sample_id):
    """Get the next sample by ID (lowest ID that is greater than the given ID)."""
    sample_filter = (get_config() or {}).get("sample_path_filter")
    with _get_conn() as conn:
        cursor = conn.cursor()
        filter_clause = "\n              AND sample_filepath GLOB ?" if sample_filter else ""
        params = [sample_id]
        if sample_filter:
            params.append(sample_filter)
        cursor.execute(
            f"""
            SELECT id, sample_filepath
            FROM samples
            WHERE id > ?
            {filter_clause}
            ORDER BY id ASC
            LIMIT 1
        """,
            params,
        )
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None

def upsert_annotation(sample_id, class_name, annotation_type="label", **kwargs):
    """
    Insert or replace an annotation for a sample.
    For label annotations: replaces existing label annotation for the sample
    For point/bbox annotations: adds new annotation (allows multiple per sample)
    
    Args:
        sample_id: The sample ID
        class_name: The annotation class
        annotation_type: Type of annotation ("label", "point", "bbox")
        **kwargs: Additional fields like row, col, width, height, timestamp
    """
    with _get_conn() as conn:
        cursor = conn.cursor()
        
        # Ensure sample exists
        cursor.execute("SELECT 1 FROM samples WHERE id = ?", (sample_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Sample with ID {sample_id} not found")
        
        if annotation_type in {"label", SKIP_ANNOTATION_TYPE}:
            # For label/skip, replace existing annotation for this sample and type
            cursor.execute(
                "DELETE FROM annotations WHERE sample_id = ? AND type = ?",
                (sample_id, annotation_type),
            )
        elif annotation_type == "mask" and class_name is not None:
            cursor.execute(
                "DELETE FROM annotations WHERE sample_id = ? AND type = ? AND class = ?",
                (sample_id, annotation_type, class_name),
            )

        if annotation_type == SKIP_ANNOTATION_TYPE:
            stored_class = SKIP_CLASS_SENTINEL
        else:
            if class_name is None or str(class_name).strip() == "":
                raise ValueError("Class name is required for annotation type '{}'".format(annotation_type))
            stored_class = class_name
        
        # Normalize coordinates to PPM integers if provided.
        # Accept either normalized floats (col,row,width,height) or ppm ints (col01,row01,width01,height01).
        def _clamp_ppm_int(v):
            try:
                iv = int(round(float(v)))
            except (TypeError, ValueError):
                return None
            if iv < 0:
                iv = 0
            elif iv > 1_000_000:
                iv = 1_000_000
            return iv

        # Prefer explicit ppm fields when present
        if any(k in kwargs for k in ("col01", "row01", "width01", "height01")):
            col01 = _clamp_ppm_int(kwargs.get("col01")) if kwargs.get("col01") is not None else None
            row01 = _clamp_ppm_int(kwargs.get("row01")) if kwargs.get("row01") is not None else None
            width01 = _clamp_ppm_int(kwargs.get("width01")) if kwargs.get("width01") is not None else None
            height01 = _clamp_ppm_int(kwargs.get("height01")) if kwargs.get("height01") is not None else None
        else:
            col01 = to_ppm(kwargs.get("col")) if kwargs.get("col") is not None else None
            row01 = to_ppm(kwargs.get("row")) if kwargs.get("row") is not None else None
            width01 = to_ppm(kwargs.get("width")) if kwargs.get("width") is not None else None
            height01 = to_ppm(kwargs.get("height")) if kwargs.get("height") is not None else None

        mask_path = kwargs.get("mask_path")
        if annotation_type == "mask":
            if not mask_path or not isinstance(mask_path, str):
                raise ValueError("mask annotations require a non-empty 'mask_path'")
            col01 = None
            row01 = None
            width01 = None
            height01 = None
        else:
            mask_path = None

        # Insert new annotation
        cursor.execute(
            """
            INSERT INTO annotations (
                sample_id, class, type,
                col01, row01, width01, height01, mask_path, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                stored_class,
                annotation_type,
                col01,
                row01,
                width01,
                height01,
                mask_path,
                kwargs.get("timestamp"),
            ),
        )

def add_point_annotation(sample_id, class_name, x, y, timestamp=None):
    """Add a single point annotation to a sample."""
    if timestamp is None:
        timestamp = int(time.time())
    # upsert_annotation handles conversion to PPM
    return upsert_annotation(sample_id, class_name, "point", col=x, row=y, timestamp=timestamp)

def delete_point_annotation(sample_id, x, y, tolerance=0.01):
    """Delete a point annotation near the specified coordinates for a sample."""
    with _get_conn() as conn:
            cursor = conn.cursor()
            # Convert provided normalized floats to PPM space for comparison
            col01 = to_ppm(x)
            row01 = to_ppm(y)
            tol_ppm = to_ppm(tolerance)
            cursor.execute(
                    """
                    DELETE FROM annotations 
                    WHERE sample_id = ? AND type = 'point'
                        AND ABS(col01 - ?) <= ? AND ABS(row01 - ?) <= ?
                    """,
                    (sample_id, col01, tol_ppm, row01, tol_ppm),
            )
            return cursor.rowcount > 0

def clear_point_annotations(sample_id):
    """Clear all point annotations for a sample."""
    """Delete all point annotations for a sample."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM annotations 
            WHERE sample_id = ? AND type = 'point'
        """, (sample_id,))
        return cursor.rowcount

def delete_annotation_by_sample_id(sample_id):
    """Delete all annotations for a specific sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM annotations
            WHERE sample_id = ?
            """,
            (sample_id,)
        )
        return cursor.rowcount > 0


def delete_annotations_by_type(sample_id, annotation_type):
    """Delete annotations for a sample limited to a specific type."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM annotations
            WHERE sample_id = ? AND type = ?
            """,
            (sample_id, annotation_type),
        )
        return cursor.rowcount


def delete_mask_annotation(sample_id, class_name):
    """Delete mask annotations for a specific class on a sample."""
    if not class_name:
        return 0
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM annotations
            WHERE sample_id = ? AND type = 'mask' AND class = ?
            """,
            (sample_id, class_name),
        )
        return cursor.rowcount

def get_annotation_stats():
    """Returns current annotation statistics including training stats and live accuracy."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        
        # Total samples
        cursor.execute("SELECT COUNT(*) FROM samples")
        total_samples = cursor.fetchone()[0]
        
        # Annotated samples
        cursor.execute("SELECT COUNT(DISTINCT sample_id) FROM annotations")
        annotated_samples = cursor.fetchone()[0]
        
        # Annotations by class
        cursor.execute("""
            SELECT class, COUNT(*) as count
            FROM annotations
            WHERE type = 'label' AND class != ?
            GROUP BY class
            ORDER BY count DESC
        """, (SKIP_CLASS_SENTINEL,))
        class_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get training stats from curves table
        cursor.execute("""
            SELECT e.epoch,
                   (
                       SELECT value FROM curves c
                       WHERE c.epoch = e.epoch AND c.curve_name = 'train_loss'
                       ORDER BY c.timestamp DESC LIMIT 1
                   ) AS train_loss,
                   (
                       SELECT value FROM curves c
                       WHERE c.epoch = e.epoch AND c.curve_name IN ('val_loss','valid_loss')
                       ORDER BY c.timestamp DESC LIMIT 1
                   ) AS valid_loss,
                   (
                       SELECT value FROM curves c
                       WHERE c.epoch = e.epoch AND c.curve_name IN ('val_accuracy','accuracy')
                       ORDER BY c.timestamp DESC LIMIT 1
                   ) AS accuracy,
                   (
                       SELECT MAX(timestamp) FROM curves c
                       WHERE c.epoch = e.epoch
                   ) AS timestamp
            FROM (
                SELECT DISTINCT epoch FROM curves WHERE epoch IS NOT NULL
            ) e
            ORDER BY e.epoch
        """)
        training_rows = cursor.fetchall()
        training_stats = [
            {
                "epoch": r[0],
                "train_loss": r[1],
                "valid_loss": r[2], 
                "accuracy": r[3],
                "timestamp": r[4],
            }
            for r in training_rows
        ]
        
        # Get all live accuracy points (let frontend handle windowing)
        cursor.execute("""
            SELECT value, timestamp FROM curves 
            WHERE curve_name = 'live_accuracy'
            ORDER BY timestamp ASC
        """)
        live_points = cursor.fetchall()
        live_accuracy_points = [{"value": p[0], "timestamp": p[1]} for p in live_points]
        
        return {
            "total": total_samples,
            "annotated": annotated_samples,
            "remaining": total_samples - annotated_samples,
            "annotation_counts": class_counts,
            "training_stats": training_stats,
            "live_accuracy_points": live_accuracy_points
        }

def export_annotations():
    """Export all annotations as a list of dictionaries."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.sample_id, s.sample_filepath, a.class, a.type,
                   a.col01, a.row01, a.width01, a.height01, a.mask_path, a.timestamp
            FROM annotations a
            JOIN samples s ON s.id = a.sample_id
            ORDER BY a.sample_id
        """)
        results = cursor.fetchall()
        
        annotations = []
        for row in results:
            ann_type = row[3]
            raw_class = row[2]
            class_value = None if ann_type == SKIP_ANNOTATION_TYPE or raw_class == SKIP_CLASS_SENTINEL else raw_class
            ann = {
                "sample_id": row[0],
                "sample_filepath": row[1],
                "class": class_value,
                "type": ann_type,
                "timestamp": row[9]
            }
            # Add coordinates based on type
            if row[3] == "point" and row[4] is not None and row[5] is not None:
                ann["col01"] = row[4]
                ann["row01"] = row[5]
            elif row[3] == "bbox" and all(x is not None for x in row[4:8]):
                ann["col01"] = row[4]
                ann["row01"] = row[5]
                ann["width01"] = row[6]
                ann["height01"] = row[7]
            elif ann_type == "mask" and row[8]:
                ann["mask_path"] = row[8]
            elif ann_type == SKIP_ANNOTATION_TYPE:
                ann["skipped"] = True

            annotations.append(ann)

        return annotations

def store_training_stats(*_, **__):
    raise NotImplementedError("Moved to src.backend.db_ml.store_training_stats")

def store_live_accuracy(sample_id, is_correct):
    """Store live accuracy measurement for a single annotation."""
    timestamp = int(time.time())
    
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO curves (curve_name, value, epoch, timestamp)
            VALUES (?, ?, ?, ?)
        """, ('live_accuracy', 1.0 if is_correct else 0.0, None, timestamp))

## Legacy API get_live_accuracy_stats removed. Use get_annotation_stats().

def get_most_recent_prediction(sample_id):
    """Get the most recent label prediction for a sample."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT class FROM predictions 
            WHERE sample_id = ? AND type = 'label'
            ORDER BY timestamp DESC
            LIMIT 1
        """, (sample_id,))
        result = cursor.fetchone()
        return result[0] if result else None
