# Specification

A human-in-the-loop image annotation system with continuous training.

## Run

you must launch both

webapp:
```
uvicorn src.backend.main:app --port 8001 --reload
```
AI training:
```
python -m src.ml.fastai_training --arch vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k
```

## Backend
- **Framework**: Starlette.
- **Endpoints**
  - `PUT /config` and `GET /config` manage architecture, class list and preprocessing.
  - `GET /next?current_id=ID&strategy=STRAT` returns the next unlabeled image.
    `STRAT` may be `sequential` or `least_confident_minority` (default). Headers
    describe prediction or existing annotation.
  - `GET /sample?id=ID` serves an image by ID with the same headers.
  - `POST /annotate` stores `{filepath, class}` (and optional bbox/point info).
  - `DELETE /annotate` removes the label annotation for a filepath.
  - `GET /stats` reports image counts, annotations per class and model
    performance metrics such as accuracy and error rates.
  - `GET /export_db` downloads the current database as JSON.
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
  3. Save the model weights to `<db>.pth` so progress persists across runs.
  4. Predict on remaining unlabeled images and write predictions to DB.
  5. Sleep for a configurable delay and repeat.

## Usage
1. Start backend: `uvicorn src.backend.main:app`.
2. Open the frontend in a browser.
3. Annotate using the UI (`/next` fetches an image; `/annotate` saves it).
4. Run `python -m src.ml.fastai_training` to train and update predictions.
5. Export the DB via the "Export DB" button or `GET /export_db` when done.

# TO-DO

## next
- image loader (.zip)
- implement show -> save -> prefetch pattern
- user editable preprocessing
- better model export
- model inference example

