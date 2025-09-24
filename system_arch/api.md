## API Overview

### Static
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/` | — | — | `router.html` | Assets served from `src/frontend`. |
| GET | `/classification` | — | — | `classification/index.html` | Sets `config.task=classification` before responding (idempotent). |
| GET | `/segmentation` | — | — | `segmentation/index.html` | Sets `config.task=segmentation` before responding (idempotent). |
| GET | `/<asset>` | — | — | Static file | Examples: `/style.css`, `/shared/js/api.js`. |

### Health
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/api/health` | — | — | `{status: "ok"}` | — |

### Config
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/api/config` | — | — | Config dict incl. `available_architectures` | `available_architectures` is computed, not stored. |
| PUT | `/api/config` | — | JSON object | Updated config (merge semantics) | Merge-only for known keys: updates existing keys, adds new known keys; unknown keys are ignored. Response MUST return the full, updated config identical to a subsequent `GET /api/config`. |

Config fields include:
- `classes` (list[str])
- `ai_should_be_run` (bool)
- `architecture` (str)
- `budget` (int)
- `resize` (int)
- `task` (str | null)
- `sample_path_filter` (str | null): glob-style filepath filter applied client-side; stored as provided.
- `sample_path_filter_count` (int): number of samples matching the current filter; recomputed on each response (not persisted).
- `available_architectures` (list[str]): computed, not persisted.

### Samples
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/api/samples/next` | `strategy`, and if `strategy=specific_class` include `class` | — | Image bytes | Headers: `X-Image-*`, `X-Predictions-*`. |
| GET | `/api/samples/{id}` | — | — | Image bytes | Headers: `X-Image-*`, `X-Predictions-*`. |
| GET | `/api/samples/{id}/prev` | — | — | Image bytes | Sample previous to the given id. |
| GET | `/api/samples/{id}/next` | — | — | Image bytes | Sample next to the given id. |

Sampling strategies for `/api/samples/next`: `sequential` (default), `random`, `minority_frontier`, `specific_class`.

### Annotations
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/api/annotations/{id}` | — | — | `{annotations: [...]}` | All annotations for the sample. |
| PUT | `/api/annotations/{id}` | — | JSON list of annotations. Each item: `type` (`label` default), fields by type: `label` → `class` (string, required) and optional `timestamp`; `point` → `class` (string, required), `col01`,`row01` (integers, ppm of image width/height, 0..1,000,000); `bbox` → `class` (string, required), `col01`,`row01`,`width01`,`height01` (integers, ppm, 0..1,000,000); `skip` → optional `timestamp` (class ignored). | `{ok: true, count: <n>}` or updated annotations | Atomic per sample; overwrite-by-type: for each `type` present in the payload, existing annotations of that `type` for the sample are fully replaced by the provided list for that `type`. Releases claim. |
| DELETE | `/api/annotations/{id}` | — | — | `{ok: true}` | Delete all annotations for the sample. |

 

### Stats / Export
| Method | Path | Query Params | Request Body | Response | Notes |
|---|---|---|---|---|---|
| GET | `/api/stats` | — | — | Counts, curve metrics, live accuracy points | Counts are always included. |
| GET | `/api/export` | — | — | `{annotations: [...]}` | Full annotations dump. |

Notes
- Task side-effects: The server sets `config.task` when serving `/classification` and `/segmentation`. The current value is visible via `/api/config` and merge-updatable via `PUT /api/config`.
- JSON parsing uses `silent=False`; malformed JSON returns 400 with traceback (debug mode).
- Opportunistic claim cleanup: On `GET /api/config`, the backend may perform a claim cleanup if the last cleanup is older than a fixed interval (default 60 minutes). Cleanup resets `samples.claimed` to 0 for all claimed rows and updates `config.last_claim_cleanup` to the current time. This reduces the chance of forever-claimed samples.
- Image headers: Responses from `/api/samples/...` include:
  - `X-Image-Id` (int): Sample identifier.
  - `X-Image-Filepath` (string): Full/absolute filepath to the sample asset.
  - Content-Type reflects the asset (`image/jpeg` or `image/png`).
- Prediction headers: When available, predictions are exposed via headers. Schema:
  - `X-Predictions-Type`: `label` or `mask`.
  - If `label`:
    - `X-Predictions-Label` (string): Predicted class label.
    - `X-Predictions-Probability` (int ppm 0..1,000,000): Probability/confidence. Send this header whenever the DB row has a probability; omit only for legacy or missing values.
  - If `mask`:
    - `X-Predictions-Mask` (JSON object): `{class: url_path}` mapping, where each value is a relative URL to fetch the mask asset (e.g., `/preds/<file>.png`).
    - Class mapping semantics remain as specified; backend maps files under `session/preds/` to URLs under `/preds/` and never exposes absolute filesystem paths. The DB may store `mask_path` as an absolute path (preferred) or a session-relative path; the server normalizes either form to safe relative URLs.
  - BBoxes: Not exposed via headers in v1.
 - Stats: `/api/stats` returns available curves, including `live_accuracy`, and always includes counts.
- Write semantics summary:
  - Config PUT is merge-only: unspecified keys remain unchanged; new keys are added; no deletions.
  - Annotations PUT is overwrite-by-type: for any `type` present in the request list, the server replaces all existing annotations of that `type` for the sample with exactly those supplied; other `type`s not present remain untouched.
  - Transactionality: A `PUT /api/annotations/{id}` with multiple types is processed atomically per sample. Either all type replacements succeed and are committed, or none are applied.
- Skip semantics: Classification skip posts `{type:"skip"}` (no class needed). Responses surface skip entries with `class=null` and `skipped=true`. Skipped samples are excluded from future “Next” pulls and not counted as labels for training or stats.
