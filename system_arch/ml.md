## ML Components

### Modules
- `src/backend/db_ml.py`: ML-facing DB helpers
  - Re-exports: `DB_PATH`, `get_config`, `get_annotations`
  - Provides: `get_all_samples()`, `set_predictions_batch()`, `store_training_stats()`
- `src/ml/fastai_training.py`: classification trainer/predictor
  - Uses fastai ResNet18/34; cycles training and prediction on unlabeled samples
  - Stores label predictions via `set_predictions_batch()`
  - Logs training curves via `store_training_stats()` and saves checkpoint `session/checkpoint.pkl`
- `src/ml/dinov3_training.py`: segmentation with DINOv3 features + linear head
  - Loads local DINOv3 hub models; extracts 16×16 patch features
  - Trains a 1×1 Conv head on point and mask annotations, saves `session/dinov3_linear_classifier_seg.pkl`
  - Produces per-class masks per image → PNGs in `session/preds/` and DB `predictions(type='mask')`

### Configuration Fields
- `ai_should_be_run` (bool): enable ML loop
- `architecture` (str): `resnet18`/`resnet34` for fastai; `small`/`large` for DINOv3
- `budget` (int): max items to predict per cycle
- `resize` (int): image resize (fastai: shorter side; DINOv3: padded canvas size)
- `mask_loss_weight` (float): weighting applied to mask loss during segmentation training

### Data Exchange
- Reads: `get_all_samples()`, `get_annotations(sample_id)`, `get_config()`
- Writes: `set_predictions_batch([...])`, `store_training_stats(epoch, ...)`
- Predictions schema examples (field units and names):
  - Label: `{sample_id, class, type: 'label', probability}` where `probability` is an integer ppm (0..1,000,000).
  - BBox: `{sample_id, class, type: 'bbox', col01, row01, width01, height01}` where all coordinates are integers in ppm of image width/height (0..1,000,000).
  - Mask: `{sample_id, class, type: 'mask', mask_path}` where `mask_path` is a filesystem path (absolute preferred for unambiguous identification; session-relative also accepted). The API exposes mask URLs via headers per `system_arch/api.md` and never sends absolute paths to clients.

### Notes
- DINOv3 uses `torch.hub.load(..., source='local')` from `src/ml/dinov3`; weights optional.
- `set_predictions_batch` deletes previous predictions scoped by `(sample_id, type[, class])` before insert.

### Workflow
- Gate: read `ai_should_be_run` from config. If false, skip all ML actions for the cycle.
- Data check: query available samples and annotations; if there is no sufficient data for the current `task`, skip 
- Model lifecycle: lazily load the model on first use or when relevant config changes (e.g., `architecture`, `resize`, hyperparameters) require a reload. Persist checkpoints under `session/` for reuse across cycles or restarts.
- Train: when training is enabled and there are labeled samples, use a persistent split that prevents leakage (see Persistent Split below). Run one or more epochs per cycle as configured; after each epoch, compute and log validation accuracy via `store_training_stats()`.
- Checkpoint: after training, save or update the session-scoped checkpoint under `session/` so it can be reloaded in subsequent cycles or process restarts.
- Predict: select up to `budget` samples to predict per cycle.
  - Classification: choose unlabeled samples.
  - Segmentation: include both labeled and unlabeled samples (to refresh masks after additional point labels).
  - Write predictions to DB in batches via `set_predictions_batch()`; for segmentation, store per-class mask paths per sample.
- Looping: each cycle re-evaluates the gate, data availability, model freshness, performs training (if applicable), then prediction, and persists metrics and checkpoints. Pause 1 sec between cycles that are skipped.
- Masks are one png per binary mask (which is the lightest) and their size is such that when resized to orig image dimensions they align nicely. No separate alignment metadata is needed.

### Persistent Split (Leakage Prevention)
- Goal: avoid mixing newly annotated items into the validation set after they were used for training, which inflates validation accuracy.
- Strategy: maintain a session-scoped assignment of each labeled sample to one of `{train, val}` for a given task and class space.
  - Initialize the split once when enough labeled data exists (e.g., first time N>=min_samples_for_split) using a deterministic, optionally stratified method.
  - On subsequent cycles, keep existing assignments fixed; when new labeled samples appear, shuffle the new ones, assign the same percentages to the train / val splits, but simply appending, not remixing.
- If ml process is killed the split is lost.

### Validation Metrics
- Log: `epoch`, `train_loss` (if available), `val_accuracy`, `timestamp`.

### Checkpoints
- Save as fastai export or pickle under `session/`:
  - Classification: `checkpoint.pkl` (fastai Learner export)
  - Segmentation: `dinov3_linear_classifier_seg.pkl` (pickle with linear Conv head state)

### Segmentation Masks
- Filename scheme: write one PNG per binary mask to `session/preds/` named `<sample_id>_<class>.png`.
- Why: prevents collisions when different folders contain images with the same filename; do not base mask names on image stems or truncated paths.
- Serving: the backend should expose masks via `/preds/<file>.png` and never leak absolute paths to clients.
- Format: save as 1-bit binary PNG (lightest)
