## Samples

Represents an input image stored in the `samples` table.

| Field | Type | Description | Required |
| --- | --- | --- | --- |
| `id` | integer | Unique sample ID (DB primary key) | Yes |
| `sample_filepath` | string | Absolute path of the image file (unique, not null) | Yes |
| `claimed` | integer | 0/1 flag for concurrency; 1 = currently claimed | Yes |

### Claiming Semantics (Current Behavior)
- Purpose: Prevent the same unlabeled sample from being handed out twice by `/api/samples/next` when multiple clients ask concurrently.
- Scope: Claims are only used during “next” selection of unlabeled samples. They do not affect direct ID navigation (`/api/samples/{id}`, `/api/samples/{id}/prev`, `/api/samples/{id}/next`).
- Set (claim): On a successful selection for `/api/samples/next`, the backend atomically executes `UPDATE samples SET claimed=1 WHERE id=? AND claimed=0`. If the row was already claimed, the selector retries and picks another sample according to the strategy.
- Released: The backend sets `claimed=0` when annotations are saved via `PUT /api/annotations/{id}` or when annotations are deleted via `DELETE /api/annotations/{id}`.
- Not released by reads: GET endpoints (including image fetches and stats) have no effect on the flag.
- No user/TTL: The flag has no user attribution and no timeout. If a client requests Next and then abandons the item without saving or deleting annotations, the sample stays claimed indefinitely and will be skipped by future Next selections.
- Selection filter: All “unlabeled” Next strategies (sequential, random, minority_frontier, specific_class) exclude rows with `claimed=1` from their candidate set.

Limitations and future direction
- Multi-user coordination is coarse (global 0/1) and can dead‑lock items if clients abandon without saving. There is no per‑user ownership, TTL, or explicit release API.
- See improvement items in `to-do` under “Sample claiming and leases” and “Single annotations write path” for a migration to explicit leasing (with expiry) and a single write path that reliably releases claims.

### Opportunistic claim cleanup
- Goal: Avoid forever-claimed samples when clients abandon work without saving or deleting annotations.
- Mechanism: Store the last cleanup timestamp in config as `last_claim_cleanup` (unix seconds). On access to a common API endpoint (preferred: `GET /api/config`), the backend checks whether the previous cleanup is older than a fixed interval (e.g., 60 minutes). If stale, it performs a cleanup and updates `last_claim_cleanup` to now.
- Cleanup action (minimal): `UPDATE samples SET claimed=0 WHERE claimed=1`. This may cause a one-time re-serve of an image still being worked on around the cleanup boundary, which is acceptable per tolerance.
- Interval: The interval is a backend constant (default 60 minutes). Future extensions may make it configurable; for now, keep it fixed to minimize schema/config churn.
- Interaction with manual release: Regular `PUT /api/annotations/{id}` and `DELETE /api/annotations/{id}` continue to release claims immediately after successful writes.

Example:
```json
{
  "id": 7,
  "sample_filepath": "/data/images/img_0007.jpg",
  "claimed": 0
}
```

## Annotation

Represents a user annotation stored in the `annotations` table.

| Field | Type | Description | Linked To | Required |
| --- | --- | --- | --- | --- |
| `id` | integer | Unique annotation ID (DB primary key) | - | Yes |
| `sample_id` | integer | Associated sample ID | `samples.id` | Yes |
| `class` | string | Annotation class label (`null` in API responses for `skip`) | - | Yes (ignored when `type=skip`) |
| `type` | string | One of: `label`, `point`, `bbox`, `skip` | - | Yes |
| `timestamp` | integer|null | Unix seconds when the annotation was created | - | No |
| `col01` | integer | X coordinate as parts-per-million of image width (0..1,000,000) | - | Yes if type ∈ {`point`,`bbox`} |
| `row01` | integer | Y coordinate as parts-per-million of image height (0..1,000,000) | - | Yes if type ∈ {`point`,`bbox`} |
| `width01` | integer | Box width as parts-per-million of image width (0..1,000,000) | - | Yes if type = `bbox` |
| `height01` | integer | Box height as parts-per-million of image height (0..1,000,000) | - | Yes if type = `bbox` |

Example (point):
```json
{
  "id": 42,
  "sample_id": 7,
  "class": "cat",
  "type": "point",
  "col01": 370000,
  "row01": 520000,
  "timestamp": 1716223450
}
```

## Prediction

Represents a model prediction stored in the `predictions` table.

| Field | Type | Description | Linked To | Required |
| --- | --- | --- | --- | --- |
| `id` | integer | Unique prediction ID (DB primary key) | - | Yes |
| `sample_id` | integer | Associated sample ID | `samples.id` | Yes |
| `class` | string | Predicted class label | - | Yes |
| `type` | string | One of: `label`, `bbox`, `mask` | - | Yes |
| `timestamp` | integer|null | Unix seconds when the prediction was written | - | No |
| `probability` | integer | Confidence as parts-per-million (0..1,000,000). Store as integer ppm (e.g., 0.735 → 735000). | - | Yes if type = `label` |
| `col01` | integer | X coordinate as parts-per-million of image width (0..1,000,000) | - | Yes if type = `bbox` |
| `row01` | integer | Y coordinate as parts-per-million of image height (0..1,000,000) | - | Yes if type = `bbox` |
| `width01` | integer | Box width as parts-per-million of image width (0..1,000,000) | - | Yes if type = `bbox` |
| `height01` | integer | Box height as parts-per-million of image height (0..1,000,000) | - | Yes if type = `bbox` |
| `mask_path` | string | Filesystem path to saved mask (typically under `session/preds/`) | - | Yes if type = `mask` |

Example (mask):
```json
{
  "id": 311,
  "sample_id": 7,
  "class": "cat",
  "type": "mask",
  "mask_path": "session/preds/img_0007_pred_class_cat.png",
  "timestamp": 1716223550
}
```

Example (label):
```json
{
  "id": 512,
  "sample_id": 7,
  "class": "dog",
  "type": "label",
  "probability": 823456
}
```

## Training Curves

Represents time series points stored in the `curves` table and consumed by the training curves view. No foreign keys to other tables. Although `live_accuracy` is produced by the backend (on label saves) rather than the ML worker, it is stored as a normal curve row with `curve_name='live_accuracy'` in this same table.

| Field | Type | Description | Required |
| --- | --- | --- | --- |
| `curve_name` | string | Metric name; e.g., `train_loss`, `val_loss`, `val_accuracy`, `live_accuracy` | Yes |
| `value` | number | Metric value | Yes |
| `epoch` | integer|null | Epoch index for epoch-scoped metrics; null for event-scoped points | No |
| `timestamp` | integer | Unix seconds when the point was recorded | Yes |

Row example (single point):
```json
{
  "curve_name": "train_loss",
  "value": 0.46,
  "epoch": 5,
  "timestamp": 1716224001
}
```

Aggregation semantics
- Epoch-scoped metrics: For each `(epoch, curve_name)`, the API returns the most recent point by `timestamp`. The aggregate’s `timestamp` is the max timestamp across metrics for that epoch.
- Event-scoped metrics: `live_accuracy` remains un-aggregated and is returned as a list of `{value, timestamp}` points; it is written by the backend to this table with `curve_name='live_accuracy'`.

Naming conventions
- Prefer `val_accuracy` for validation accuracy and `val_loss` for validation loss. Avoid using `accuracy` or `valid_loss` to keep keys consistent across `ml.md`, producers, and API consumers.

## Config

Represents the single-row app configuration stored in the `config` table and returned by `/api/config`. No foreign keys to other tables.

| Field | Type | Description | Required |
| --- | --- | --- | --- |
| `classes` | string[] | List of available class names | Yes |
| `task` | string | One of: `classification`, `segmentation` | Yes |
| `ai_should_be_run` | boolean | Whether the training/inference loop should run | Yes |
| `architecture` | string|null | Model architecture name (e.g., `small`, `large`, or a timm model name) | No |
| `budget` | integer|null | Training or processing budget (app-dependent meaning) | No |
| `resize` | integer|null | Image resize target (short side) for ML | No |
| `sample_path_filter` | string|null | Glob-style path filter applied to sample filenames (`null` disables) | No |
| `sample_path_filter_count` | integer | Number of samples matching the current filter; recomputed on read | No (computed) |
| `available_architectures` | string[] | Provided by API (computed), not stored in DB | No (computed) |
| `last_claim_cleanup` | integer|null | Unix seconds of the last opportunistic claim cleanup; backend updates this when running cleanup | No |

Example:
```json
{
  "classes": ["cat", "dog"],
  "ai_should_be_run": true,
  "architecture": "small",
  "budget": 1000,
  "resize": 1536,
  "sample_path_filter": "sessionA/*",
  "sample_path_filter_count": 128,
  "task": "classification",
  "available_architectures": ["resnet18", "resnet34", "...others..."],
  "last_claim_cleanup": 1716227000
}
```

Notes
- `sample_path_filter_count` currently rides along in the config payload for UI convenience; plan to migrate this counter to `/api/stats` in a future cleanup so config stays purely declarative.
- Coordinate semantics: All coordinates for points and bboxes are integers representing parts per million of the image dimension. `col01`/`width01` are ppm of image width; `row01`/`height01` are ppm of image height. Range: 0..1,000,000.
- Probabilities: For label predictions, `probability` is an integer ppm (0..1,000,000). ML writers MUST round floats to nearest ppm; readers MUST divide by 1,000,000 when float is needed. Range clamp: [0, 1_000_000].
- Single source of truth: `sample_filepath` exists only in `samples`. Child tables store `sample_id`. APIs expose the image filepath for user convenience via the `X-Image-Filepath` response header on image endpoints.
- Skip annotations: stored with `type='skip'` and an internal sentinel class; API and export responses surface `class=null` plus `skipped=true` to mark them explicitly.
- Mask safety and URL mapping:
  - Storage: `predictions.mask_path` stores a filesystem path to the mask artifact on disk, typically under `session/preds/`. Store either a full absolute path (preferred for unambiguous identification) or a path relative to the session root (e.g., `preds/<file>.png`).
  - API exposure (masks): Image responses expose mask predictions via headers as relative URL paths using `X-Predictions-Mask` (see `system_arch/api.md`). The backend maps a stored `mask_path` to a URL under a fixed, sandboxed route (e.g., `/preds/<file>.png`). Absolute filesystem paths for masks MUST NOT be sent to clients.
  - Mapping convention: If `mask_path` is absolute and under `session/preds/`, strip the `session/` prefix and serve it at `/<stripped_path>`; if `mask_path` is stored relative, serve it directly at `/<mask_path>`. Only allow requests whose normalized path resolves inside `session/preds/`.
  - Security: Enforce path normalization and directory whitelisting to prevent traversal; 404 any file outside `session/preds/`. Optionally validate existence before advertising URLs.
- Schema source of truth: See `src/backend/db_init.py` for table definitions and `src/backend/db.py` for serialization to API JSON.

## Initialization & Persistence

- Session dir: All runtime data lives under `session/`. The SQLite DB is `session/app.db` (with WAL files `app.db-wal`/`app.db-shm`). Prediction masks should be saved under `session/preds/`.
- DB creation: On startup, `src/backend/db_init.initialize_database_if_needed()` creates tables, indexes, and a default `config` row if missing. It also seeds initial data from `build_initial_db_dict()` after strict `validate_db_dict()` checks.
- Annotation/prediction seeding: The stock initializer leaves `annotations` and `predictions` empty (the stub `build_initial_db_dict()` returns empty lists). If you choose to preload annotations or predictions, you must extend both the validator and the insert logic to provide ppm-normalized geometry/probability fields; otherwise, expect to backfill them through the runtime APIs instead of at init time.
- Initial data source: `build_initial_db_dict()` defines where samples, annotations, and predictions are loaded from (paths must exist). Update this function to point at your dataset.
- Persistence model: All state changes (annotations, predictions, config, curves) are persisted to `session/app.db`. Mask files referenced by predictions remain on disk under `session/preds/`.
- Resetting: Stop the app, then delete `session/app.db` (and optionally `session/preds/` and other artifacts) to fully reset. On next start, the DB will be recreated and re-seeded via `db_init`.
