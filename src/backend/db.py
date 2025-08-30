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

DB_PATH = "session/app.db"

def _get_conn():
    """Open a SQLite connection to the app database.

    Raises if the DB file is missing; initialization is handled elsewhere.

    Returns:
        sqlite3.Connection: Connection with a 30s timeout.
    """
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database not found at {DB_PATH}. Did you run db_init.py?")
    return sqlite3.connect(DB_PATH, timeout=30.0)


def get_config():
    """Return the configuration as a dict or empty dict if not set."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT classes, ai_should_be_run, architecture, budget, sleep, resize FROM config LIMIT 1")
        row = cursor.fetchone()
        if row:
            classes, ai_should_be_run, architecture, budget, sleep, resize = row
            return {
                "classes": json.loads(classes) if classes else [],
                "ai_should_be_run": bool(ai_should_be_run),
                "architecture": architecture,
                "budget": budget,
                "sleep": sleep,
                "resize": resize
            }
        return {}

def update_config(config):
    """Merge and persist the provided config dict.

    The config supports keys: "classes" (list[str]), "ai_should_be_run"
    (bool), "architecture" (str), "budget" (int), "sleep" (int),
    and "resize" (int). Missing keys retain their previous values.
    """
    current = get_config()
    current.update(config)
    
    # Convert classes list to JSON string for storage
    classes_json = json.dumps(current.get("classes", []))
    
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM config")
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute(
                "UPDATE config SET classes = ?, ai_should_be_run = ?, architecture = ?, budget = ?, sleep = ?, resize = ?",
                (classes_json, int(current.get("ai_should_be_run", False)), 
                 current.get("architecture"), current.get("budget"), 
                 current.get("sleep"), current.get("resize"))
            )
        else:
            cursor.execute(
                "INSERT INTO config (classes, ai_should_be_run, architecture, budget, sleep, resize) VALUES (?, ?, ?, ?, ?, ?)",
                (classes_json, int(current.get("ai_should_be_run", False)),
                 current.get("architecture"), current.get("budget"),
                 current.get("sleep"), current.get("resize"))
            )

def get_all_samples():
    raise NotImplementedError("Moved to src.backend.db_ml.get_all_samples")

def _get_next_unlabeled_sequential():
    """Returns the next unlabeled sample info without claiming it."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        # Find the first sample that has no annotations and is not claimed
        cursor.execute("""
            SELECT s.id, s.sample_filepath 
            FROM samples s 
            LEFT JOIN annotations a ON s.id = a.sample_id 
            WHERE a.sample_id IS NULL 
            AND s.claimed = 0
            ORDER BY s.id 
            LIMIT 1
        """)
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def _get_next_unlabeled_random():
    """Returns a random unlabeled sample info without claiming it."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        # Find a random sample that has no annotations and is not claimed
        cursor.execute("""
            SELECT s.id, s.sample_filepath 
            FROM samples s 
            LEFT JOIN annotations a ON s.id = a.sample_id 
            WHERE a.sample_id IS NULL 
            AND s.claimed = 0
            ORDER BY RANDOM() 
            LIMIT 1
        """)
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def _get_unlabeled_pick(pick, highest_probability=True):
    """Returns the sample of the provided class with the highest/lowest predicted probability.
    If not found returns None."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        # Find unlabeled, unclaimed samples with predictions for the given class,
        # ordered by prediction probability (highest or lowest first)
        order_direction = "DESC" if highest_probability else "ASC"
        cursor.execute(f"""
            SELECT s.id, s.sample_filepath, p.probability
            FROM samples s
            INNER JOIN predictions p ON s.id = p.sample_id
            LEFT JOIN annotations a ON s.id = a.sample_id
            WHERE a.sample_id IS NULL 
            AND s.claimed = 0
            AND p.class = ?
            AND p.type = 'label'
            ORDER BY p.probability {order_direction}
            LIMIT 1
        """, (pick,))
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1], "probability": result[2]} if result else None

def _get_annotation_counts():
    """Returns a dict with annotation counts per class, ordered by count (ascending)."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT class, COUNT(*) as count
            FROM annotations
            GROUP BY class
            ORDER BY count ASC
        """)
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

def get_annotations(sample_id):
    """Get all annotations for a specific sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sample_id, sample_filepath, class, type, x, y, width, height, timestamp
            FROM annotations
            WHERE sample_id = ?
        """, (sample_id,))
        results = cursor.fetchall()
        annotations = []
        for row in results:
            ann = {
                "id": row[0],
                "sample_id": row[1], 
                "sample_filepath": row[2],
                "class": row[3],
                "type": row[4],
                "timestamp": row[9]
            }
            # Add coordinates based on type
            if row[4] == "point":  # point type
                if row[5] is not None and row[6] is not None:
                    ann["col"] = row[5]  # x -> col
                    ann["row"] = row[6]  # y -> row
            elif row[4] == "bbox":  # bbox type
                if all(x is not None for x in row[5:9]):
                    ann["col"] = row[5]    # x -> col
                    ann["row"] = row[6]    # y -> row  
                    ann["width"] = row[7]
                    ann["height"] = row[8]
            # label type has no coordinates
            annotations.append(ann)
        return annotations

def get_predictions(sample_id):
    """Get all predictions for a specific sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sample_id, sample_filepath, class, type, probability, x, y, width, height, timestamp
            FROM predictions
            WHERE sample_id = ?
        """, (sample_id,))
        results = cursor.fetchall()
        predictions = []
        for row in results:
            pred = {
                "id": row[0],
                "sample_id": row[1],
                "sample_filepath": row[2],
                "class": row[3],
                "type": row[4],
                "timestamp": row[10]
            }
            # Add type-specific fields
            if row[4] == "label":  # label type
                pred["probability"] = row[5]
            elif row[4] == "bbox":  # bbox type
                if all(x is not None for x in row[6:10]):
                    pred["col"] = row[6]    # x -> col
                    pred["row"] = row[7]    # y -> row
                    pred["width"] = row[8]
                    pred["height"] = row[9]
            # mask type would need mask_path but it's not in current schema
            predictions.append(pred)
        return predictions

def set_predictions_batch(_):
    raise NotImplementedError("Moved to src.backend.db_ml.set_predictions_batch")


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
        # default to sequential if we got nothing
        return get_next_sample_by_strategy("sequential", None)

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
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sample_filepath
            FROM samples
            WHERE id < ?
            ORDER BY id DESC
            LIMIT 1
        """, (sample_id,))
        result = cursor.fetchone()
        return {"id": result[0], "sample_filepath": result[1]} if result else None


def get_sample_next_by_id(sample_id):
    """Get the next sample by ID (lowest ID that is greater than the given ID)."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, sample_filepath
            FROM samples
            WHERE id > ?
            ORDER BY id ASC
            LIMIT 1
        """, (sample_id,))
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
        
        # Get sample_filepath for the annotation
        cursor.execute("SELECT sample_filepath FROM samples WHERE id = ?", (sample_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"Sample with ID {sample_id} not found")
        sample_filepath = result[0]
        
        if annotation_type == "label":
            # For labels, replace existing label annotation for this sample
            cursor.execute("DELETE FROM annotations WHERE sample_id = ? AND type = 'label'", (sample_id,))
        
        # Insert new annotation 
        cursor.execute(
            """
            INSERT INTO annotations (
                sample_id, sample_filepath, class, type,
                x, y, width, height, timestamp
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample_id,
                sample_filepath,
                class_name,
                annotation_type,
                kwargs.get("col"),  # x
                kwargs.get("row"),  # y
                kwargs.get("width"),
                kwargs.get("height"),
                kwargs.get("timestamp"),
            ),
        )

def add_point_annotation(sample_id, class_name, x, y, timestamp=None):
    """Add a single point annotation to a sample."""
    if timestamp is None:
        timestamp = int(time.time())
    return upsert_annotation(sample_id, class_name, "point", col=x, row=y, timestamp=timestamp)

def delete_point_annotation(sample_id, x, y, tolerance=0.01):
    """Delete a point annotation near the specified coordinates."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM annotations 
            WHERE sample_id = ? AND type = 'point'
            AND ABS(x - ?) < ? AND ABS(y - ?) < ?
        """, (sample_id, x, tolerance, y, tolerance))
        return cursor.rowcount > 0

def clear_point_annotations(sample_id):
    """Clear all point annotations for a sample."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM annotations 
            WHERE sample_id = ? AND type = 'point'
        """, (sample_id,))
        return cursor.rowcount

def delete_annotation_by_sample_id(sample_id):
    """Delete annotation for a specific sample ID."""
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM annotations
            WHERE sample_id = ?
        """, (sample_id,))
        return cursor.rowcount > 0

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
            GROUP BY class
            ORDER BY count DESC
        """)
        class_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Get training stats from curves table
        cursor.execute("""
            SELECT epoch, 
                   MAX(CASE WHEN curve_name = 'train_loss' THEN value END) as train_loss,
                   MAX(CASE WHEN curve_name = 'valid_loss' THEN value END) as valid_loss,
                   MAX(CASE WHEN curve_name = 'accuracy' THEN value END) as accuracy,
                   MAX(timestamp) as timestamp
            FROM curves 
            WHERE epoch IS NOT NULL
            GROUP BY epoch
            ORDER BY epoch
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
            SELECT a.sample_id, a.sample_filepath, a.class, a.type, 
                   a.x, a.y, a.width, a.height, a.timestamp
            FROM annotations a
            ORDER BY a.sample_id
        """)
        results = cursor.fetchall()
        
        annotations = []
        for row in results:
            ann = {
                "sample_id": row[0],
                "sample_filepath": row[1],
                "class": row[2],
                "type": row[3],
                "timestamp": row[8]
            }
            # Add coordinates based on type
            if row[3] == "point" and row[4] is not None and row[5] is not None:
                ann["col"] = row[4]  # x -> col
                ann["row"] = row[5]  # y -> row
            elif row[3] == "bbox" and all(x is not None for x in row[4:8]):
                ann["col"] = row[4]    # x -> col
                ann["row"] = row[5]    # y -> row  
                ann["width"] = row[6]
                ann["height"] = row[7]
            
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
