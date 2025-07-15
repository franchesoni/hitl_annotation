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


class DatabaseAPI:
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

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "annotation.db")
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
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
        self.conn.commit()

    def close(self):
        self.conn.close()
