def validate_db_dict(db):
    """
    Strictly validate that the db dict matches the required template:
    {
        "samples": [ {"filepath": str}, ... ],
        "annotations": [ { ... }, ... ],
        "predictions": [ { ... }, ... ]
    }
    For coordinates, uses 'row' and 'col' instead of 'x' and 'y'.
    Raises ValueError if any check fails.
    """
    # Top-level keys
    required_keys = {"samples", "annotations", "predictions"}
    if not isinstance(db, dict):
        raise ValueError("Database must be a dict.")
    if set(db.keys()) != required_keys:
        raise ValueError(
            f"Database must have keys {required_keys}, got {set(db.keys())}"
        )

    # Validate samples
    if not isinstance(db["samples"], list):
        raise ValueError("'samples' must be a list.")
    import os

    sample_filepaths = set()
    seen_filepaths = set()
    for s in db["samples"]:
        if not isinstance(s, dict):
            raise ValueError("Each sample must be a dict.")
        if set(s.keys()) != {"filepath"}:
            raise ValueError(
                f"Sample dict must have only 'filepath', got {set(s.keys())}"
            )
        if not isinstance(s["filepath"], str) or not s["filepath"].strip():
            raise ValueError("'filepath' must be a non-empty string.")
        if s["filepath"] in seen_filepaths:
            raise ValueError(f"Duplicate filepath in samples: {s['filepath']}")
        if not os.path.isfile(s["filepath"]):
            raise ValueError(f"Sample filepath does not exist: {s['filepath']}")
        seen_filepaths.add(s["filepath"])
        sample_filepaths.add(s["filepath"])

    # Validate annotations
    if not isinstance(db["annotations"], list):
        raise ValueError("'annotations' must be a list.")
    seen_annotations = set()
    for a in db["annotations"]:
        if not isinstance(a, dict):
            raise ValueError("Each annotation must be a dict.")
        required_ann_keys = {"sample_filepath", "type", "class"}
        allowed_types = {"label", "bbox", "point"}
        if not required_ann_keys.issubset(a.keys()):
            raise ValueError(
                f"Annotation missing required keys: {required_ann_keys - set(a.keys())}"
            )
        if a["type"] not in allowed_types:
            raise ValueError(
                f"Annotation type must be one of {allowed_types}, got {a['type']}"
            )
        if (
            not isinstance(a["sample_filepath"], str)
            or not a["sample_filepath"].strip()
        ):
            raise ValueError("'sample_filepath' must be a non-empty string.")
        if a["sample_filepath"] not in sample_filepaths:
            raise ValueError(
                f"Annotation sample_filepath '{a['sample_filepath']}' not found in samples."
            )
        if not isinstance(a["class"], str) or not a["class"].strip():
            raise ValueError("'class' must be a non-empty string.")
        # Deduplication
        ann_tuple = tuple(sorted(a.items()))
        if ann_tuple in seen_annotations:
            raise ValueError(f"Duplicate annotation: {a}")
        seen_annotations.add(ann_tuple)
        # Strictly check coordinates fields
        if a["type"] == "label":
            allowed_keys = {"sample_filepath", "type", "class", "timestamp"}
        elif a["type"] == "point":
            allowed_keys = {
                "sample_filepath",
                "type",
                "class",
                "row",
                "col",
                "timestamp",
            }
            if not ("row" in a and "col" in a):
                raise ValueError("Point annotation must have 'row' and 'col'.")
            if not (isinstance(a["row"], int) and isinstance(a["col"], int)):
                raise ValueError(
                    "'row' and 'col' must be integers for point annotation."
                )
            if a["row"] < 0 or a["col"] < 0:
                raise ValueError(
                    "'row' and 'col' must be non-negative for point annotation."
                )
        elif a["type"] == "bbox":
            allowed_keys = {
                "sample_filepath",
                "type",
                "class",
                "row",
                "col",
                "width",
                "height",
                "timestamp",
            }
            for k in ("row", "col", "width", "height"):
                if k not in a:
                    raise ValueError(f"Bbox annotation must have '{k}'.")
                if not isinstance(a[k], int):
                    raise ValueError(f"'{k}' must be integer for bbox annotation.")
                if a[k] < 0:
                    raise ValueError(f"'{k}' must be non-negative for bbox annotation.")
        if not set(a.keys()).issubset(allowed_keys):
            raise ValueError(
                f"Annotation keys for type {a['type']} must be subset of {allowed_keys}, got {set(a.keys())}"
            )
        if "timestamp" in a and not (
            isinstance(a["timestamp"], int) or a["timestamp"] is None
        ):
            raise ValueError("'timestamp' must be integer or None if present.")

    # Validate predictions
    if not isinstance(db["predictions"], list):
        raise ValueError("'predictions' must be a list.")
    seen_predictions = set()
    for p in db["predictions"]:
        if not isinstance(p, dict):
            raise ValueError("Each prediction must be a dict.")
        required_pred_keys = {"sample_filepath", "type", "class"}
        allowed_types = {"label", "bbox"}
        if not required_pred_keys.issubset(p.keys()):
            raise ValueError(
                f"Prediction missing required keys: {required_pred_keys - set(p.keys())}"
            )
        if p["type"] not in allowed_types:
            raise ValueError(
                f"Prediction type must be one of {allowed_types}, got {p['type']}"
            )
        if (
            not isinstance(p["sample_filepath"], str)
            or not p["sample_filepath"].strip()
        ):
            raise ValueError("'sample_filepath' must be a non-empty string.")
        if p["sample_filepath"] not in sample_filepaths:
            raise ValueError(
                f"Prediction sample_filepath '{p['sample_filepath']}' not found in samples."
            )
        if not isinstance(p["class"], str) or not p["class"].strip():
            raise ValueError("'class' must be a non-empty string.")
        # Deduplication
        pred_tuple = tuple(sorted(p.items()))
        if pred_tuple in seen_predictions:
            raise ValueError(f"Duplicate prediction: {p}")
        seen_predictions.add(pred_tuple)
        if p["type"] == "label":
            allowed_keys = {"sample_filepath", "type", "class", "probability"}
            if "probability" not in p:
                raise ValueError("Label prediction must have 'probability'.")
            if not (
                isinstance(p["probability"], float) or isinstance(p["probability"], int)
            ):
                raise ValueError(
                    "'probability' must be a float or int for label prediction."
                )
            if not (0.0 <= float(p["probability"]) <= 1.0):
                raise ValueError(
                    f"'probability' must be between 0 and 1 for label prediction, got {p['probability']}"
                )
        elif p["type"] == "bbox":
            allowed_keys = {
                "sample_filepath",
                "type",
                "class",
                "row",
                "col",
                "width",
                "height",
            }
            for k in ("row", "col", "width", "height"):
                if k not in p:
                    raise ValueError(f"Bbox prediction must have '{k}'.")
                if not isinstance(p[k], int):
                    raise ValueError(f"'{k}' must be integer for bbox prediction.")
                if p[k] < 0:
                    raise ValueError(f"'{k}' must be non-negative for bbox prediction.")
        if not set(p.keys()).issubset(allowed_keys):
            raise ValueError(
                f"Prediction keys for type {p['type']} must be subset of {allowed_keys}, got {set(p.keys())}"
            )


# Database connection and query functions
# the database is composed of three tables, samples, annotations and predictions
# samples contains the images
# annotations contains the image classes, bboxes (with class) and points (with class) and the associations to the images
# predictions contains (for the cls case) the probabilities for each class

# more specifically:
# samples: id, filepath
# annotations: id, sample_id, sample_filepath, type (label/bbox/point), class, coordinates (none, point or bbox)
# predictions: id, sample_id, sample_filepath, type (label/bbox), class, coordinates (probability or bbox)

"""
Database connection and query functions
The database is composed of three tables: samples, annotations, and predictions.
samples contains the images
annotations contains the image classes, bboxes (with class) and points (with class) and the associations to the images
predictions contains (for the cls case) the probabilities for each class

More specifically:
samples: id, filepath
annotations: id, sample_id, sample_filepath, type (label/bbox/point), class, coordinates (none, point or bbox)
predictions: id, sample_id, sample_filepath, type (label/bbox), class, coordinates (probability or bbox)
"""

import os
import sqlite3
import json
import time


class DatabaseAPI:

    def export_db_as_json(self, out_path):
        """
        Export the current database to a JSON file in the strict validated format.
        """
        # Gather all samples
        samples = [{"filepath": fp} for fp in self.get_samples()]
        # Gather all annotations
        annotations = []
        for fp in self.get_samples():
            anns = self.get_annotations(fp)
            for ann in anns:
                # Convert x/y to row/col if present
                ann_out = {k: v for k, v in ann.items() if k not in ("id", "sample_id")}
                if "x" in ann_out:
                    ann_out["col"] = ann_out.pop("x")
                if "y" in ann_out:
                    ann_out["row"] = ann_out.pop("y")
                # Remove None values
                ann_out = {k: v for k, v in ann_out.items() if v is not None}
                annotations.append(ann_out)
        # Gather all predictions
        predictions = []
        for fp in self.get_samples():
            preds = self.get_predictions(fp)
            for pred in preds:
                pred_out = {
                    k: v for k, v in pred.items() if k not in ("id", "sample_id")
                }
                if "x" in pred_out:
                    pred_out["col"] = pred_out.pop("x")
                if "y" in pred_out:
                    pred_out["row"] = pred_out.pop("y")
                # Remove None values
                pred_out = {k: v for k, v in pred_out.items() if v is not None}
                predictions.append(pred_out)
        db_dict = {
            "samples": samples,
            "annotations": annotations,
            "predictions": predictions,
        }
        validate_db_dict(db_dict)
        import json

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(db_dict, f, indent=2, ensure_ascii=False)

    def set_annotations(self, filepath, annotations):
        """
        Overwrite annotations for the given sample filepath.
        annotations: list of dicts with keys matching the annotations table columns (except id and sample_id, which are set automatically).
        Raises ValueError if the filepath is not in the samples table.
        Automatically adds a timestamp if not present in the annotation dict.
        """
        import time

        cursor = self.conn.cursor()
        # Get sample id for the filepath
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Sample filepath not found in database: {filepath}")
        sample_id = row[0]
        # Remove previous annotations for this sample
        cursor.execute("DELETE FROM annotations WHERE sample_id = ?", (sample_id,))
        # Insert new annotations
        if annotations:
            keys = [k for k in annotations[0] if k not in ("id", "sample_id")]
            if "timestamp" not in keys:
                keys.append("timestamp")
            columns = ["sample_id"] + keys
            values = []
            for ann in annotations:
                ann = dict(ann)  # copy
                if "timestamp" not in ann or ann["timestamp"] is None:
                    ann["timestamp"] = int(time.time())
                values.append(tuple([sample_id] + [ann.get(k) for k in keys]))
            placeholders = ",".join(["?"] * len(columns))
            sql = (
                f"INSERT INTO annotations ({','.join(columns)}) VALUES ({placeholders})"
            )
            cursor.executemany(sql, values)
        self.conn.commit()

    def set_predictions(self, filepath, predictions):
        """
        Overwrite predictions for the given sample filepath.
        predictions: list of dicts with keys matching the predictions table columns (except id and sample_id, which are set automatically).
        Raises ValueError if the filepath is not in the samples table.
        """
        cursor = self.conn.cursor()
        # Get sample id for the filepath
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Sample filepath not found in database: {filepath}")
        sample_id = row[0]
        # Remove previous predictions for this sample
        cursor.execute("DELETE FROM predictions WHERE sample_id = ?", (sample_id,))
        # Insert new predictions
        if predictions:
            keys = [k for k in predictions[0] if k not in ("id", "sample_id")]
            columns = ["sample_id"] + keys
            values = [
                tuple([sample_id] + [pred.get(k) for k in keys]) for pred in predictions
            ]
            placeholders = ",".join(["?"] * len(columns))
            sql = (
                f"INSERT INTO predictions ({','.join(columns)}) VALUES ({placeholders})"
            )
            cursor.executemany(sql, values)
        self.conn.commit()

    def get_predictions(self, filepath):
        """
        Return all prediction rows for the given sample filepath as a list of dicts.
        """
        cursor = self.conn.cursor()
        # Get sample id for the filepath
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            return []
        sample_id = row[0]
        # Get all predictions for this sample_id
        cursor.execute(
            "SELECT id, sample_id, sample_filepath, type, class, probability, x, y, width, height FROM predictions WHERE sample_id = ?",
            (sample_id,),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, r)) for r in cursor.fetchall()]

    def get_annotations(self, filepath):
        """
        Return all annotation rows for the given sample filepath as a list of dicts.
        """
        cursor = self.conn.cursor()
        # Get sample id for the filepath
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            return []
        sample_id = row[0]
        # Get all annotations for this sample_id (including timestamp)
        cursor.execute(
            "SELECT id, sample_id, sample_filepath, type, class, x, y, width, height, timestamp FROM annotations WHERE sample_id = ?",
            (sample_id,),
        )
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, r)) for r in cursor.fetchall()]

    def get_samples(self):
        """
        Return a list of all sample filepaths in the database.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT filepath FROM samples")
        return [row[0] for row in cursor.fetchall()]

    def set_samples(self, filepaths):
        """
        Overwrite the samples table to match the given list of filepaths.
        Removes samples not in the new list and adds new ones (only if the file exists).
        Also removes annotations and predictions associated with removed samples.
        Returns a dict with 'added', 'removed', and 'kept' filepaths.
        """
        if not isinstance(filepaths, list):
            raise ValueError("filepaths must be a list")
        non_existing = [fp for fp in filepaths if not os.path.isfile(fp)]
        if non_existing:
            raise ValueError(f"The following files do not exist: {non_existing}")
        filepaths_set = set(filepaths)
        cursor = self.conn.cursor()

        # Get current filepaths in DB
        cursor.execute("SELECT id, filepath FROM samples")
        id_to_fp = {row[0]: row[1] for row in cursor.fetchall()}
        current = set(id_to_fp.values())

        # Determine which to add and which to remove
        to_add = filepaths_set - current
        to_remove = current - filepaths_set
        kept = current & filepaths_set

        # Remove annotations and predictions associated with removed samples
        if to_remove:
            # Get sample_ids for the filepaths to remove
            remove_ids = [sid for sid, fp in id_to_fp.items() if fp in to_remove]
            if remove_ids:
                # Remove annotations
                cursor.executemany(
                    "DELETE FROM annotations WHERE sample_id = ?",
                    [(sid,) for sid in remove_ids],
                )
                # Remove predictions
                cursor.executemany(
                    "DELETE FROM predictions WHERE sample_id = ?",
                    [(sid,) for sid in remove_ids],
                )
                # Remove samples
                cursor.executemany(
                    "DELETE FROM samples WHERE id = ?", [(sid,) for sid in remove_ids]
                )

        # Add new samples
        if to_add:
            cursor.executemany(
                "INSERT INTO samples (filepath) VALUES (?)", [(fp,) for fp in to_add]
            )

        self.conn.commit()
        return {"added": list(to_add), "removed": list(to_remove), "kept": list(kept)}

    def save_label_annotation(self, filepath, class_name):
        """
        Save or update a 'label' annotation for the given image filepath with the given class.
        If a label annotation exists for this image, it is replaced. Otherwise, it is inserted.
        """
        import time
        cursor = self.conn.cursor()
        # Get sample id for the filepath
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Sample filepath not found in database: {filepath}")
        sample_id = row[0]
        # Remove previous label annotation for this sample
        cursor.execute(
            "DELETE FROM annotations WHERE sample_id = ? AND type = 'label'",
            (sample_id,)
        )
        # Insert new label annotation
        now = int(time.time())
        cursor.execute(
            "INSERT INTO annotations (sample_id, sample_filepath, type, class, timestamp) VALUES (?, ?, 'label', ?, ?)",
            (sample_id, filepath, class_name, now)
        )
        self.conn.commit()

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "annotation.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
			CREATE TABLE IF NOT EXISTS samples (
				id INTEGER PRIMARY KEY AUTOINCREMENT,
				filepath TEXT UNIQUE NOT NULL
			)
		"""
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL,
                sample_filepath TEXT,
                type TEXT,
                class TEXT,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                timestamp INTEGER,
                FOREIGN KEY(sample_id) REFERENCES samples(id)
            )
        """
        )
        cursor.execute(
            """
                        CREATE TABLE IF NOT EXISTS predictions (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                sample_id INTEGER NOT NULL,
                                sample_filepath TEXT,
                                type TEXT,
                                class TEXT,
                                probability REAL,
                                x INTEGER,
                                y INTEGER,
                                width INTEGER,
                                height INTEGER,
                                FOREIGN KEY(sample_id) REFERENCES samples(id)
                        )
                """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                architecture TEXT,
                classes TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS accuracy_stats (
                tries INTEGER NOT NULL DEFAULT 0,
                correct INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_stats (
                epoch INTEGER PRIMARY KEY,
                train_loss REAL,
                valid_loss REAL,
                accuracy REAL,
                timestamp INTEGER
            )
            """
        )
        cursor.execute("SELECT COUNT(*) FROM accuracy_stats")
        if cursor.fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO accuracy_stats (tries, correct) VALUES (0, 0)"
            )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def get_annotation_counts(self):
        """Return a dict mapping label class to number of annotations."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT class, COUNT(*) FROM annotations WHERE type='label' GROUP BY class"
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def delete_label_annotation(self, filepath):
        """
        Delete the 'label' annotation for the given image filepath, if it exists.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM samples WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Sample filepath not found in database: {filepath}")
        sample_id = row[0]
        cursor.execute(
            "DELETE FROM annotations WHERE sample_id = ? AND type = 'label'",
            (sample_id,)
        )
        self.conn.commit()

    def get_config(self):
        """Return the configuration as a dict or empty dict if not set."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT architecture, classes FROM config LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {}
        architecture, classes_json = row
        classes = json.loads(classes_json) if classes_json else []
        return {"architecture": architecture, "classes": classes}

    def update_config(self, config):
        """Merge and persist the provided config dict."""
        current = self.get_config()
        current.update(config)
        architecture = current.get("architecture")
        classes_json = json.dumps(current.get("classes", []))
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM config")
        exists = cursor.fetchone()[0] > 0
        if exists:
            cursor.execute(
                "UPDATE config SET architecture = ?, classes = ?",
                (architecture, classes_json),
            )
        else:
            cursor.execute(
                "INSERT INTO config (architecture, classes) VALUES (?, ?)",
                (architecture, classes_json),
            )
        self.conn.commit()
        return current

    def count_labeled_samples(self):
        """Return the number of unique samples that have a label annotation."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT COUNT(DISTINCT sample_id) FROM annotations WHERE type='label'"
        )
        row = cursor.fetchone()
        return row[0] if row else 0

    def count_total_samples(self):
        """Return the total number of samples in the database."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM samples")
        row = cursor.fetchone()
        return row[0] if row else 0

    def get_next_unlabeled_sequential(self, current_filepath=None):
        """Return the next unlabeled sample filepath in DB order."""
        cursor = self.conn.cursor()
        params = []
        sql = (
            "SELECT filepath FROM samples WHERE filepath NOT IN ("
            "SELECT sample_filepath FROM annotations WHERE type='label')"
        )
        if current_filepath:
            sql += " AND filepath <> ?"
            params.append(current_filepath)
        sql += " ORDER BY id LIMIT 1"
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None

    def get_next_unlabeled_default(self, current_filepath=None):
        """Return the next unlabeled sample filepath using default strategy."""
        cursor = self.conn.cursor()

        # Annotation counts per class
        cursor.execute(
            "SELECT class, COUNT(*) FROM annotations WHERE type='label' GROUP BY class"
        )
        ann_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Candidate predictions for unlabeled samples
        params = []
        sql = """
            SELECT p.sample_filepath, p.class, p.probability
            FROM predictions AS p
            JOIN samples AS s ON p.sample_id = s.id
            LEFT JOIN annotations AS a
                ON a.sample_id = s.id AND a.type='label'
            WHERE p.type='label' AND p.probability IS NOT NULL
              AND a.id IS NULL
        """
        if current_filepath:
            sql += " AND s.filepath <> ?"
            params.append(current_filepath)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        if not rows:
            return self.get_next_unlabeled_sequential(current_filepath)

        from collections import defaultdict

        candidate_by_class = defaultdict(list)
        for fp, cls, prob in rows:
            candidate_by_class[str(cls)].append((float(prob), fp))

        all_classes = set(candidate_by_class.keys()) | set(ann_counts.keys())
        for c in all_classes:
            ann_counts.setdefault(c, 0)

        minority = min(all_classes, key=lambda c: ann_counts[c])
        if minority in candidate_by_class:
            return max(candidate_by_class[minority], key=lambda x: x[0])[1]

        fallback = min(candidate_by_class.keys(), key=lambda c: ann_counts[c])
        return min(candidate_by_class[fallback], key=lambda x: x[0])[1]

    def get_next_unlabeled_for_class(self, class_name, current_filepath=None):
        """Return unlabeled sample with highest probability for the given class."""
        cursor = self.conn.cursor()
        params = [class_name]
        sql = """
            SELECT p.sample_filepath
            FROM predictions AS p
            JOIN samples AS s ON p.sample_id = s.id
            LEFT JOIN annotations AS a
                ON a.sample_id = s.id AND a.type='label'
            WHERE p.type='label' AND p.class = ? AND p.probability IS NOT NULL
              AND a.id IS NULL
        """
        if current_filepath:
            sql += " AND s.filepath <> ?"
            params.append(current_filepath)
        sql += " ORDER BY p.probability DESC LIMIT 1"
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None

    def get_accuracy_counts(self):
        """Return {'tries': int, 'correct': int} stored in the accuracy table."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT tries, correct FROM accuracy_stats LIMIT 1")
        row = cursor.fetchone()
        if not row:
            return {"tries": 0, "correct": 0}
        return {"tries": row[0], "correct": row[1]}

    def increment_accuracy(self, was_correct: bool) -> None:
        """Increment tries and correct counters depending on prediction result."""
        cursor = self.conn.cursor()
        if was_correct:
            cursor.execute(
                "UPDATE accuracy_stats SET tries = tries + 1, correct = correct + 1"
            )
        else:
            cursor.execute("UPDATE accuracy_stats SET tries = tries + 1")
        self.conn.commit()

    def add_training_stat(self, epoch: int, train_loss: float | None, valid_loss: float | None, accuracy: float | None) -> None:
        """Store training metrics for an epoch."""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO training_stats (epoch, train_loss, valid_loss, accuracy, timestamp) VALUES (?, ?, ?, ?, ?)",
            (epoch, train_loss, valid_loss, accuracy, int(time.time())),
        )
        self.conn.commit()

    def get_training_stats(self):
        """Return list of training stats ordered by epoch."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT epoch, train_loss, valid_loss, accuracy, timestamp FROM training_stats ORDER BY epoch"
        )
        rows = cursor.fetchall()
        return [
            {
                "epoch": r[0],
                "train_loss": r[1],
                "valid_loss": r[2],
                "accuracy": r[3],
                "timestamp": r[4],
            }
            for r in rows
        ]

