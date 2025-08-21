from src.backend.db import DB_PATH
from pathlib import Path
import sqlite3


def build_initial_db_dict() -> dict:
    """
    Placeholder function to build the initial database dict for import/initialization.
    The user should edit this function to load or construct their data.

    The returned object must be a dict with the following structure:

    {
        "samples": [
            {"sample_filepath": str},
            ...
        ],
        "annotations": [
            {
                "sample_filepath": str,  # must match a filepath in samples
                "type": "label" | "bbox" | "point",
                "class": str,
                # For "point": "row", "col" (int, non-negative)
                # For "bbox": "row", "col", "width", "height" (int, non-negative)
                # For "label": no coordinates
                # Optional: "timestamp": int
            },
            ...
        ],
        "predictions": [
            {
                "sample_filepath": str,  # must match a filepath in samples
                "type": "label" | "bbox" | "mask",
                "class": str,
                # For "label": "probability" (float in [0,1])
                # For "bbox": "row", "col", "width", "height" (int, non-negative)
                # For "mask": "mask_path" (string)
            },
            ...
        ]
    }

    Returns:
        db_dict (dict): The database dictionary in the strict format.
    """
    # TODO: User should edit this function to load or build their data.
    ### EDIT START
    from pathlib import Path

    db_dict = {
        "samples": [
            {"sample_filepath": str(ppath)}
            for ppath in Path("/home/franchesoni/Downloads/mnist_png/testing").glob(
                "**/*.png"
            )
        ],
        "annotations": [],
        "predictions": [],
    }
    ### EDIT END
    return db_dict


def validate_db_dict(db):
    """
    Strictly validate that the db dict matches the required template:
    {
        "samples": [ {"sample_filepath": str}, ... ],
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
        if set(s.keys()) != {"sample_filepath"}:
            raise ValueError(
                f"Sample dict must have only 'sample_filepath', got {set(s.keys())}"
            )
        if (
            not isinstance(s["sample_filepath"], str)
            or not s["sample_filepath"].strip()
        ):
            raise ValueError("'sample_filepath' must be a non-empty string.")
        if s["sample_filepath"] in seen_filepaths:
            raise ValueError(f"Duplicate filepath in samples: {s['sample_filepath']}")
        if not os.path.isfile(s["sample_filepath"]):
            raise ValueError(f"Sample filepath does not exist: {s['sample_filepath']}")
        seen_filepaths.add(s["sample_filepath"])
        sample_filepaths.add(s["sample_filepath"])

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
        allowed_types = {"label", "bbox", "mask"}
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
        elif p["type"] == "mask":
            allowed_keys = {
                "sample_filepath",
                "type",
                "class",
                "mask_path",
            }
            if "mask_path" not in p:
                raise ValueError("Mask prediction must have 'mask_path'.")
            if not isinstance(p["mask_path"], str) or not p["mask_path"].strip():
                raise ValueError("'mask_path' must be a non-empty string.")
            if not os.path.isfile(p["mask_path"]):
                raise ValueError(f"Mask file does not exist: {p['mask_path']}")
        if not set(p.keys()).issubset(allowed_keys):
            raise ValueError(
                f"Prediction keys for type {p['type']} must be subset of {allowed_keys}, got {set(p.keys())}"
            )


def initialize_database_if_needed(db_path=DB_PATH):
    # create db dir if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # connect to db (creates file if needed)
    with sqlite3.connect(db_path) as conn:
        # config db
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        # set up tables if they don't exist
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_filepath TEXT UNIQUE NOT NULL,
                claimed INTEGER DEFAULT 0
            );
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL UNIQUE,
                sample_filepath TEXT NOT NULL,
                class TEXT NOT NULL,
                type TEXT NOT NULL,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                timestamp INTEGER,
                FOREIGN KEY (sample_id) REFERENCES samples (id),
                FOREIGN KEY (sample_filepath) REFERENCES samples (sample_filepath)
            );
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL,
                sample_filepath TEXT NOT NULL,
                class TEXT NOT NULL,
                type TEXT NOT NULL,
                probability REAL,
                x INTEGER,
                y INTEGER,
                width INTEGER,
                height INTEGER,
                timestamp INTEGER,
                FOREIGN KEY (sample_id) REFERENCES samples (id),
                FOREIGN KEY (sample_filepath) REFERENCES samples (sample_filepath)
            );
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS config (
                classes TEXT,
                ai_should_be_run INTEGER DEFAULT 0,
                architecture TEXT,
                budget INTEGER,
                sleep INTEGER,
                resize INTEGER
            );
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS curves (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                curve_name TEXT NOT NULL,
                value REAL NOT NULL,
                epoch INTEGER,
                timestamp INTEGER
            );
        """
        )

        # Create performance indexes to speed up common queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_claimed ON samples (claimed);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_annotations_sample_id ON annotations (sample_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_sample_id ON predictions (sample_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_class_type ON predictions (class, type);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_probability ON predictions (probability);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_curves_name_timestamp ON curves (curve_name, timestamp);")

        # now we can initialize the database
        initial_content = build_initial_db_dict()
        validate_db_dict(initial_content)

        # insert samples
        conn.executemany(
            "INSERT OR IGNORE INTO samples (sample_filepath) VALUES (?);",
            [(s["sample_filepath"],) for s in initial_content["samples"]],
        )

        # insert annotations
        for a in initial_content["annotations"]:
            conn.execute(
                """
                INSERT INTO annotations (
                    sample_id, sample_filepath, class, type,
                    x, y, width, height, timestamp
                )
                VALUES (
                    (SELECT id FROM samples WHERE sample_filepath = ?),
                    ?, ?, ?,
                    ?, ?, ?, ?, ?
                );
                """,
                (
                    a["sample_filepath"],
                    a["sample_filepath"],
                    a["class"],
                    a["type"],
                    a.get("col"),  # x
                    a.get("row"),  # y
                    a.get("width"),
                    a.get("height"),
                    a.get("timestamp"),
                ),
            )

        # insert predictions
        for p in initial_content["predictions"]:
            conn.execute(
                """
                INSERT INTO predictions (
                    sample_id, sample_filepath, class, type,
                    probability, x, y, width, height, timestamp
                )
                VALUES (
                    (SELECT id FROM samples WHERE sample_filepath = ?),
                    ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
                );
                """,
                (
                    p["sample_filepath"],
                    p["sample_filepath"],
                    p["class"],
                    p["type"],
                    p.get("probability"),
                    p.get("col"),  # x
                    p.get("row"),  # y
                    p.get("width"),
                    p.get("height"),
                    p.get("timestamp"),
                ),
            )

        # Initialize default configuration if not exists
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM config")
        config_count = cursor.fetchone()[0]
        if config_count == 0:
            import json
            default_config = {
                "classes": [],
                "ai_should_be_run": False,
                "architecture": "resnet18",
                "budget": 100,
                "sleep": 5,
                "resize": 224
            }
            cursor.execute(
                "INSERT INTO config (classes, ai_should_be_run, architecture, budget, sleep, resize) VALUES (?, ?, ?, ?, ?, ?)",
                (json.dumps(default_config["classes"]), int(default_config["ai_should_be_run"]), 
                 default_config["architecture"], default_config["budget"], 
                 default_config["sleep"], default_config["resize"])
            )
