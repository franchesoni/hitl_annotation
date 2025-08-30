## Overview

This repository implements a lightweight human-in-the-loop image annotation app with an opinionated ML loop. The system is composed of a frontend UI, a backend HTTP API, a SQLite-backed data model, and task-specific ML workers. Detailed specs live in sibling documents under `system_arch/`.

## Main Parts

- Frontend UI: See `system_arch/frontend.md`.
  - Maintains a local mirror of config, syncing via `GET/PUT /api/config` with merge-only semantics.
  - Navigation workflows (Next/Prev/Undo) orchestrate config pushes, annotation saves, image fetches, and stats refresh in a fixed order.
  - Two views: classification (single-label save-on-click) and segmentation (point edits batched on leave). Live-accuracy slider derives a windowed accuracy from recent annotation events.

- Backend API: See `system_arch/api.md`.
  - Static app routes (`/`, `/classification`, `/segmentation`) and asset serving from `src/frontend`.
  - Health: `GET /api/health`.
  - Config: `GET/PUT /api/config` with merge-only writes; server also returns `available_architectures`.
  - Samples: `GET /api/samples/next|{id}|{id}/prev|{id}/next` return image bytes plus prediction headers (`X-Predictions-*`). Strategies include `sequential`, `random`, `minority_frontier`, `specific_class`.
  - Annotations: `GET /api/annotations/{id}`, `PUT /api/annotations/{id}` with overwrite-by-type semantics, and `DELETE /api/annotations/{id}` with optional `type`.
- Stats/Export: `GET /api/stats` (counts and available curves, including `live_accuracy`) and `GET /api/export` (annotations dump).

- Data Model: See `system_arch/data.md`.
  - Tables: `samples` (images, with a simple `claimed` flag), `annotations` (label/point/bbox using ppm integer coords), `predictions` (label/bbox/mask, including mask file paths), `curves` (training and live metrics), and `config` (single-row app config).
  - Conventions: coordinates and probabilities stored as integer parts-per-million; mask files live under `session/preds/`. DB resides in `session/app.db` and is initialized on startup from `src/backend/db_init.py`.

- ML Loop: See `system_arch/ml.md`.
  - Modules: fastai-based classification and DINOv3-based point-to-mask segmentation.
  - Gate and cycle: honor `ai_should_be_run`; per cycle perform conditional training, checkpointing, and capped prediction (`budget`).
  - Persistence: write predictions in batches with replacement per scope; log curves (e.g., `val_accuracy`, `live_accuracy`); save checkpoints and sidecar metadata.
  - Quality: maintain a split during process lifetime to prevent leakage; the split is lost if the ML process is killed (not persisted across restarts).

- User Flow: See `system_arch/flow.md`.
  - Entry via router to choose task. Users manage classes, annotate, navigate images, monitor training metrics, and export session data. Frontend hotkeys and non-blocking UX support efficient labeling.

## Minimal Tightening (Spec)

- Bold decisions are to keep v1 minimal and unambiguous. Code/schema updates should follow via to-do items.

- API paths: Use `DELETE /api/annotations/{id}` as the canonical delete route. Any `/clear` delete paths referenced elsewhere are legacy and should be removed.

- Image response headers: Define and implement only the essentials on image endpoints (`GET /api/samples/...`).
  - `X-Image-Id`: integer sample id.
  - `X-Image-Filepath`: absolute filesystem path to the served image.
  - Content-Type: `image/jpeg` or `image/png` per asset.

- Prediction headers (minimal): Frontend consumes only top-1 label or mask.
  - `X-Predictions-Type`: `label` or `mask`.
  - If `label`: `X-Predictions-Label` (string class) and `X-Predictions-Probability` (integer ppm, 0..1,000,000). Include the probability header whenever stored in DB; omit only for legacy/missing values.
  - If `mask`: `X-Predictions-Mask` as JSON `{class: url_path}` mapping; paths are served under `/preds/...` and must resolve inside `session/preds/` after normalization. The DB may store `mask_path` as an absolute or session-relative filesystem path; the server normalizes to safe relative URLs.
  - BBoxes are not exposed via headers in v1.

- “Unlabeled” definition (selection and stats):
  - Classification: a sample is unlabeled if it has no `label` annotation.
  - Segmentation: a sample is unlabeled if it has no `point` annotations.

- Sampling strategies (v1):
  - Supported: `sequential` (default), `random`, `minority_frontier`, `specific_class` (requires `class` query param). `specific_class` is both used for Last-Class Max-Prob and for a user-selected target class.
  - `minority_frontier` definition: compute labeled counts per class from `annotations(type='label')`; pick the class with the lowest labeled count (tie-break by lexicographic class name). Among samples whose current top-1 prediction equals that class, select the one with the lowest predicted `probability` (ppm). If none exist, fall back to `random`.

- Navigation edges: For `GET /api/samples/{id}/prev` and `{id}/next`, return 404 at boundaries (no wrap/clamp). Frontend should handle and disable the control as needed.

- Stats response (shape): `GET /api/stats` returns aggregate counts and available curves such as `live_accuracy` (an ordered list of `{value, timestamp}` points) and training curves like `val_accuracy`, `train_loss`, etc. Specific count keys are implementation-defined in v1.

- Export response (minimal): `GET /api/export` returns `{annotations: [...]}` only. Predictions are not included in v1 exports.

- Config PUT policy: Merge-only for known keys. Unknown keys are ignored. Timestamps are Unix seconds (UTC) wherever present.

- Task routes side-effect: `GET /classification` and `GET /segmentation` intentionally set `config.task` (idempotent). This is acceptable for v1.

- Segmentation mask artifacts (v1 simplification): mask PNGs do not have to match original image dimensions on disk. Frontend resizes masks to the displayed/original image dimensions for overlay. No separate alignment metadata is required.
