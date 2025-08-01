# Specification

A human-in-the-loop image annotation system with continuous training.

## Backend
- **Framework**: Starlette.
- **Endpoints**
  - `PUT /config` and `GET /config` manage architecture, class list and preprocessing.
  - `GET /next?current_id=ID` returns the next unlabeled image; headers describe prediction or existing annotation.
  - `GET /sample?id=ID` serves an image by ID with the same headers.
  - `POST /annotate` stores `{filepath, class}` (and optional bbox/point info).
  - `DELETE /annotate` removes the label annotation for a filepath.
  - `GET /stats` reports image counts, per-class annotation counts, and
    model performance metrics including tries, correct counts, accuracy
    and error rate.
- **DatabaseAPI** (`src/database/data.py`)
  - SQLite tables: `samples(filepath)`, `annotations(id, sample_id, sample_filepath, type, class, x, y, width, height, timestamp)`, `predictions(id, sample_id, sample_filepath, type, class, probability, x, y, width, height)`, `config(architecture, classes)`, `accuracy_stats(tries, correct)`.
  - Methods to set/get samples, annotations, predictions and export DB to JSON.
  - TODO: helper to fetch the next unlabeled sample.

## Frontend
- JS interface under `src/frontend2` with image viewer, zoom/pan/reset,
  keyboard shortcuts, class manager, undo, prediction overlay, and a developer
  checklist.

## Machine Learning
- `src/ml/fastai_training.py` loops forever:
  1. Gather latest label annotations and build `DataLoaders`.
  2. Train one epoch on ResNet (34 or `--arch small` for 18).
  3. Predict on remaining unlabeled images and write predictions to DB.
  4. Sleep for a configurable delay and repeat.

## Usage
1. Start backend: `uvicorn src.backend.main:app`.
2. Open the frontend in a browser.
3. Annotate using the UI (`/next` fetches an image; `/annotate` saves it).
4. Run `python -m src.ml.fastai_training` to train and update predictions.
5. Export the DB with `DatabaseAPI.export_db_as_json(path)` when done.

## Atomic TODO
1. Implement `DatabaseAPI.get_next_unlabeled` and update `/next`.
2. Extend `/annotate` DELETE to remove annotations via `DatabaseAPI`.
3. Add active learning (least confident of least frequent class).
4. Quick annotation mode with auto-advance.
5. Class removal and persistence in the config table.
6. User-editable preprocessing in config and training.
7. Zip image loader for bulk import.
8. Undo support to revert last annotation.
9. Button to export annotations.
