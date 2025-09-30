import importlib
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.backend import db as db_module
from src.backend import db_init as db_init_module


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create an isolated database and sample for tests."""

    db_path = tmp_path / "session" / "app.db"
    session_dir = db_path.parent
    preds_dir = session_dir / "preds"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Patch db module globals to point to the temporary database
    monkeypatch.setattr(db_module, "DB_PATH", str(db_path))
    monkeypatch.setattr(db_module, "SESSION_DIR", session_dir)
    monkeypatch.setattr(db_module, "PREDS_DIR", preds_dir)

    # Ensure db_init uses the same database path and lightweight dataset
    monkeypatch.setattr(db_init_module, "DB_PATH", str(db_path))

    sample_file = tmp_path / "sample.png"
    sample_file.write_bytes(b"fake-image")

    def fake_build_initial_db_dict():
        return {
            "samples": [{"sample_filepath": str(sample_file)}],
            "annotations": [],
            "predictions": [],
        }

    monkeypatch.setattr(db_init_module, "build_initial_db_dict", fake_build_initial_db_dict)

    db_init_module.initialize_database_if_needed(str(db_path))

    with db_module._get_conn() as conn:  # pylint: disable=protected-access
        sample_id = conn.execute("SELECT id FROM samples").fetchone()[0]

    yield {
        "db": db_module,
        "db_init": db_init_module,
        "db_path": db_path,
        "session_dir": session_dir,
        "sample_id": sample_id,
        "sample_file": sample_file,
    }

    # Tests operate in isolated temp directories; no explicit cleanup required.


@pytest.fixture
def flask_client(test_db, monkeypatch):
    """Provide a Flask test client bound to the isolated database."""

    module_name = "src.backend.main"
    if module_name in sys.modules:
        del sys.modules[module_name]

    main = importlib.import_module(module_name)

    # Align main's session directories with the temporary environment
    monkeypatch.setattr(main, "SESSION_DIR", test_db["session_dir"])
    monkeypatch.setattr(main, "MASKS_DIR", test_db["session_dir"] / "masks")
    monkeypatch.setattr(main, "PREDS_DIR", test_db["session_dir"] / "preds")

    client = main.app.test_client()
    return client, main, test_db
