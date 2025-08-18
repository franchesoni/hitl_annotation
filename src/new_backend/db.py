import sqlite3
import os
import json

DB_PATH = "session/app.db"

def get_conn():
    if not os.path.exists(DB_PATH):
        raise RuntimeError(f"Database not found at {DB_PATH}. Did you run init_db.py?")
    return sqlite3.connect(DB_PATH)

def put_config(config: dict):
    """
    Inserts or updates the configuration in the database.
    """
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (config['key'], config['value']))

def get_config():
    """Return the configuration as a dict or empty dict if not set."""
    with get_conn() as conn:
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
    """Merge and persist the provided config dict."""
    current = get_config()
    current.update(config)
    
    # Convert classes list to JSON string for storage
    classes_json = json.dumps(current.get("classes", []))
    
    with get_conn() as conn:
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

def get_next_unlabeled_sequential():
    """Returns the next unlabeled sample info without claiming it."""
    with get_conn() as conn:
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


def get_unlabeled_pick(pick, highest_probability=True):
    """Returns the sample of the provided class with the highest/lowest predicted probability.
    If not found returns None."""
    with get_conn() as conn:
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

def get_annotation_counts():
    """Returns a dict with annotation counts per class, ordered by count (ascending)."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT class, COUNT(*) as count
            FROM annotations
            GROUP BY class
            ORDER BY count ASC
        """)
        results = cursor.fetchall()
        return {row[0]: row[1] for row in results}

def get_minority_unlabeled_frontier():
    """Returns the sample with the lowest probability in the minority class."""
    annotation_counts = get_annotation_counts()
    # Get class with minimum annotations
    minority_class = min(annotation_counts.keys(), key=lambda x: annotation_counts[x])
    # Use get_unlabeled_pick with lowest probability
    return get_unlabeled_pick(minority_class, highest_probability=False)

def claim_sample(sample_id):
    """Atomically claim a sample by its ID. Returns True if successful."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE samples 
            SET claimed = 1
            WHERE id = ? AND claimed = 0
        """, (sample_id,))
        return cursor.rowcount > 0

def release_claim_by_id(sample_id):
    """Release the claim on a sample by ID."""
    with get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE samples 
            SET claimed = 0, claimed_at = NULL 
            WHERE id = ?
        """, (sample_id,))

def get_annotations(sample_id):
    """Get all annotations for a specific sample ID."""
    with get_conn() as conn:
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
    with get_conn() as conn:
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

def get_next_sample_by_strategy(strategy=None, pick=None):
    """
    Get the next sample to annotate based on the given strategy.
    Returns a dict with sample info or None if no samples available.
    Claims the sample atomically just before returning.
    """
    if strategy is None:  # default
        return get_next_sample_by_strategy("sequential")
    elif strategy == "sequential":
        sample_info = get_next_unlabeled_sequential()
    elif strategy == "pick_class":
        assert pick is not None, "Pick must be provided for 'pick_class' strategy"
        sample_info = get_unlabeled_pick(pick)
    elif strategy == "minority_frontier":
        sample_info = get_minority_unlabeled_frontier()
    
    if strategy != "sequential" and sample_info is None:
        # default to sequential if we got nothing
        return get_next_sample_by_strategy()

    if not sample_info:
        return None
    if claim_sample(sample_info["id"]):
        return sample_info
    else:
        # Sample was claimed by someone else, try again
        return get_next_sample_by_strategy(strategy, pick)
