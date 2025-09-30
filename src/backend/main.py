from pathlib import Path
from contextlib import suppress
from flask import Flask, request, jsonify, send_file, Response
from io import BytesIO
import base64
from functools import lru_cache
import json
import timm
import time
import mimetypes
import re
import shutil
from collections import defaultdict
from typing import List, Dict, Any

from src.backend.db import (
    get_config, update_config, get_next_sample_by_strategy,
    get_sample_by_id, upsert_annotation, delete_annotation_by_sample_id,
    get_annotation_stats, export_annotations, release_claim_by_id,
    get_most_recent_prediction, store_live_accuracy, get_annotations,
    cleanup_claims_unconditionally, delete_annotations_by_type, delete_mask_annotation,
    add_point_annotation, delete_point_annotation, clear_point_annotations,
    get_sample_prev_by_id, get_sample_next_by_id, get_predictions,
)
from src.backend.db_init import initialize_database_if_needed

initialize_database_if_needed()

####### INITIALIZE APP #######

# ...existing code...

# Place the endpoint after app is defined

# ...existing code...

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
# Runtime artifacts live at repo-root/session, not under src/
REPO_ROOT = BASE_DIR.parent
SESSION_DIR = REPO_ROOT / "session"
PREDS_DIR = SESSION_DIR / "preds"
MASKS_DIR = SESSION_DIR / "masks"

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# Fail loudly: enable development-style debugging and exception propagation
app.config.update(
    ENV="development",
    DEBUG=True,
    PROPAGATE_EXCEPTIONS=True,
    TRAP_HTTP_EXCEPTIONS=True,
)


def _task_conflict_response(current_task: str | None, requested_task: str) -> Response:
    """Return an HTML response that alerts the user about task mismatch."""
    message_task = current_task or "unknown"
    message = (
        f"The current task is {message_task}. You tried to open {requested_task}. "
        "You can't change tasks. To switch, reset or rename the session directory and restart the app."
    )
    html = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>Task Locked</title>"
        "</head><body><script>"
        f"alert({json.dumps(message)});"
        "window.location.replace('/');"
        "</script></body></html>"
    )
    response = app.make_response(html)
    response.status_code = 409
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response


@lru_cache(maxsize=1)
def _list_architectures():
    """Return all allowed model architectures."""
    resnets = ["small", "resnet18", "resnet34"]
    return resnets + [m for m in sorted(timm.list_models()) if m not in resnets]


def create_image_response(sample_info):
    """Create an image response with informative headers.

    Args:
        sample_info: Dict with 'id' and 'sample_filepath'.
    """
    sample_id = sample_info["id"]
    sample_filepath = sample_info["sample_filepath"]

    mime_type, _ = mimetypes.guess_type(sample_filepath)
    if mime_type is None:
        mime_type = "application/octet-stream"

    anns = get_annotations(sample_id)
    preds = get_predictions(sample_id)
    label_ann = next((a for a in anns if a.get("type") == "label"), None)
    mask_annotations = [a for a in anns if a.get("type") == "mask" and a.get("mask_path")]

    headers = {
        "X-Image-Id": str(sample_id),
        "X-Image-Filepath": str(sample_filepath),
    }
    # Only add X-Predictions-* headers for predictions
    pred_candidates = [p for p in preds if p.get("type") == "label"]
    for pred in pred_candidates:
        if pred.get("probability") is None:
            raise ValueError(
                f"Label prediction {pred.get('id', '<unknown>')} is missing a probability"
            )
    assert len(pred_candidates) <= 1, "Expected at most one prediction per image"
    if pred_candidates:
        pred_ann = pred_candidates[0]
        headers["X-Predictions-Type"] = "label"
        headers["X-Predictions-Label"] = str(pred_ann.get("class", ""))
        from src.backend.db import to_ppm
        prob = pred_ann.get("probability", None)
        headers["X-Predictions-Probability"] = str(to_ppm(prob))

    # If mask predictions exist, expose them as a JSON list of {class, url} objects
    mask_preds = [p for p in preds if p.get("type") == "mask" and p.get("mask_path")]
    mask_entries = []
    if not mask_annotations:
        for pred in mask_preds:
            cls = str(pred.get("class", ""))
            if not cls:
                continue
            mask_url = _mask_public_url(str(pred.get("mask_path")), PREDS_DIR, "preds")
            if not mask_url:
                continue
            mask_entries.append({
                "class": cls,
                "url": mask_url,
                "id": pred.get("id"),
                "timestamp": pred.get("timestamp"),
            })
    if mask_entries:
        headers["X-Predictions-Mask"] = json.dumps(mask_entries)

    response = send_file(sample_filepath, mimetype=mime_type)
    for key, value in headers.items():
        response.headers[key] = value
    return response


def _resolve_under_dir(path_str: str, storage_dir: Path) -> Path:
    """Resolve *path_str* to a file located under *storage_dir*.

    Raises ``ValueError`` when the provided path is invalid or escapes the
    storage directory and ``FileNotFoundError`` if the resolved path does not
    exist as a file.
    """

    if not path_str:
        raise ValueError("Empty path provided for resolution")

    raw_path = Path(path_str)

    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append((SESSION_DIR / raw_path).resolve())
        if raw_path.parts and raw_path.parts[0] == "session":
            candidates.append((REPO_ROOT / raw_path).resolve())
        if raw_path.parts and raw_path.parts[0] == storage_dir.name:
            suffix_parts = raw_path.parts[1:]
            suffix = Path(*suffix_parts) if suffix_parts else Path()
            candidates.append((storage_dir / suffix).resolve())
        else:
            candidates.append((storage_dir / raw_path).resolve())

    storage_root = storage_dir.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        if storage_root == resolved or storage_root in resolved.parents:
            if resolved.is_file():
                return resolved
            raise FileNotFoundError(f"Resolved path {resolved} is not a file")

    raise ValueError(f"Path {path_str} is outside of storage root {storage_root}")


def _mask_public_url(path_str: str, storage_dir: Path, mount_prefix: str) -> str | None:
    try:
        resolved = _resolve_under_dir(path_str, storage_dir)
    except (FileNotFoundError, ValueError):
        return None
    rel = resolved.relative_to(storage_dir.resolve())
    return f"/{mount_prefix}/{rel.as_posix()}"


def _serve_session_file(relpath: str, storage_dir: Path, not_found_error: str):
    try:
        safe_path = _resolve_under_dir(relpath, storage_dir)
    except (FileNotFoundError, ValueError):
        return jsonify({"error": not_found_error}), 404

    mime_type, _ = mimetypes.guess_type(str(safe_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    return send_file(safe_path, mimetype=mime_type)


@app.route("/")
def index():
    return app.send_static_file("router.html")


def _serve_task_page(task_name: str, template: str):
    """Return the static page for *task_name*, enforcing task consistency."""

    config = get_config() or {}
    current_task = config.get("task")

    if current_task is None:
        update_config({"task": task_name})
        return app.send_static_file(template)

    if current_task == task_name:
        return app.send_static_file(template)

    return _task_conflict_response(current_task, task_name)


@app.route("/classification")
def classification():
    return _serve_task_page("classification", "classification/index.html")


# Add segmentation route
@app.route("/segmentation")
def segmentation():
    return _serve_task_page("segmentation", "segmentation/index.html")


# Static files are served by Flask from FRONTEND_DIR via static_url_path=""


@app.route("/favicon.ico")
def favicon():
    """Serve a tiny transparent favicon to avoid 404s."""
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2l0XQAAAAASUVORK5CYII="
    )
    data = base64.b64decode(png_b64)
    return send_file(BytesIO(data), mimetype="image/png")


@app.get("/preds/<path:relpath>")
def get_pred_mask(relpath: str):
    """Serve prediction mask files from session/preds safely (read-only)."""
    return _serve_session_file(relpath, PREDS_DIR, "Prediction file not found")


@app.get("/masks/<path:relpath>")
def get_annotation_mask(relpath: str):
    """Serve accepted annotation mask files from session/masks safely."""
    return _serve_session_file(relpath, MASKS_DIR, "Mask file not found")


@app.route("/api/health", methods=["GET"])
def health():
    return {"status": "ok"}


####### API #########
@app.put("/api/config")
def put_config():
    """Updates the configuration in the db.

    Expects a JSON body with the configuration object.
    """
    config = request.get_json(silent=False) or {}
    update_config(config)
    # Return the full, updated config (same as GET /api/config)
    updated_config = get_config()
    updated_config["available_architectures"] = _list_architectures()
    return jsonify(updated_config)


@app.get("/api/config")  # available architectures should go here
def get_config_endpoint():
    """Gets the config from the db."""
    # Opportunistic in-memory cleanup: at most once per 60 minutes
    global _last_claim_cleanup_ts
    now = time.time()
    did_cleanup = False
    if _last_claim_cleanup_ts is None or (now - _last_claim_cleanup_ts) >= 60 * 60:
        try:
            cleanup_claims_unconditionally()
            did_cleanup = True
        finally:
            _last_claim_cleanup_ts = now
    config = get_config()
    # If cleanup was performed, update last_claim_cleanup in the response
    if did_cleanup:
        config["last_claim_cleanup"] = int(_last_claim_cleanup_ts)
    # Add available architectures to the config response
    config["available_architectures"] = _list_architectures()
    return jsonify(config)


@app.get("/api/samples/next")
def get_next_sample():
    """
    Returns the next sample to be annotated.
    Query param: `strategy` determines selection strategy.
    Query param: `pick` specifies class for pick_class strategy.
    """
    strategy = request.args.get("strategy")
    # Accept ?class=<name> for strategy=specific_class, fallback to legacy pick
    pick = None
    if strategy == "specific_class":
        pick = request.args.get("class") or request.args.get("pick")
    else:
        pick = request.args.get("pick")
    sample_info = get_next_sample_by_strategy(strategy, pick)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": "No more samples available for annotation"})


@app.get("/api/samples/<int:sample_id>")
def get_sample_by_id_endpoint(sample_id: int):
    """
    Returns a sample by its ID.
    Path param: `sample_id` identifies the resource.
    """
    sample_info = get_sample_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return jsonify({"error": f"Sample with ID {sample_id} not found"}), 404


@app.get("/api/samples/<int:sample_id>/prev")
def get_sample_prev(sample_id: int):
    """
    Returns the previous sample by ID (sample with ID < current_id).
    Path param: `sample_id` identifies the current sample.
    """
    sample_info = get_sample_prev_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return get_sample_by_id_endpoint(sample_id)


@app.get("/api/samples/<int:sample_id>/next")
def get_sample_next(sample_id: int):
    """
    Returns the next sample by ID (sample with ID > current_id).
    Path param: `sample_id` identifies the current sample.
    """
    sample_info = get_sample_next_by_id(sample_id)
    if sample_info:
        return create_image_response(sample_info)
    else:
        return get_sample_by_id_endpoint(sample_id)




# Implements DELETE /api/annotations/<int:sample_id> as described in api.md: no parameters, always deletes all annotations for the sample.
@app.delete("/api/annotations/<int:sample_id>")
def delete_annotations_endpoint(sample_id: int):
    """
    Deletes all annotations for the sample.
    Returns: {ok: true}
    """
    success = delete_annotation_by_sample_id(sample_id)
    return jsonify({"ok": True})


@app.get("/api/annotations/<int:sample_id>")
def get_annotations_endpoint(sample_id: int):
    """Get all annotations for a specific sample ID."""
    annotations = get_annotations(sample_id)
    for ann in annotations:
        if ann.get("type") == "mask" and ann.get("mask_path"):
            mask_url = _mask_public_url(str(ann.get("mask_path")), MASKS_DIR, "masks")
            if mask_url:
                ann["mask_url"] = mask_url
    return jsonify({"annotations": annotations})

@app.put("/api/annotations/<int:sample_id>")
def put_annotations_bulk(sample_id: int):
    """
    Replace all annotations for a sample, grouped by type, atomically per type.
    Accepts a JSON list of annotation dicts.
    On success, releases claim and returns {ok: true, count: <n>}.
    Implements overwrite-by-type semantics: for each type present in the payload, replaces all existing annotations of that type for the sample.
    Also stores live accuracy for label annotations as described in api.md and frontend.md.
    """
    items = request.get_json(silent=False) or []
    if not isinstance(items, list):
        return jsonify({"error": "Expected a JSON list of annotation objects"}), 400
    # Group by type
    allowed_types = {"label", "point", "bbox", "skip", "mask"}
    grouped = defaultdict(list)
    for ann in items:
        ann_type = ann.get("type", "label")
        if ann_type not in allowed_types:
            return jsonify({"error": f"Unsupported annotation type: {ann_type}"}), 400
        grouped[ann_type].append(ann)
    total = 0
    for ann_type, anns in grouped.items():
        # Remove existing annotations of this type first (overwrite-by-type semantics)
        delete_annotations_by_type(sample_id, ann_type)
        # Insert new ones
        for ann in anns:
            class_ = ann.get("class")
            timestamp = ann.get("timestamp")
            if ann_type == "skip":
                upsert_annotation(sample_id, None, ann_type, timestamp=timestamp)
                total += 1
                continue
            if not class_:
                continue
            # Prepare annotation data (supporting optional fields)
            annotation_data = {k: v for k, v in ann.items() if k not in ("class", "type")}
            upsert_annotation(sample_id, class_, ann_type, **annotation_data)
            total += 1
            # Store live accuracy for label annotations
            if ann_type == "label":
                predicted_class = get_most_recent_prediction(sample_id)
                if predicted_class is not None:
                    is_correct = (predicted_class == class_)
                    store_live_accuracy(sample_id, float(is_correct))
    release_claim_by_id(sample_id)
    return jsonify({"ok": True, "count": total})


@app.delete("/api/annotations/<int:sample_id>/points")
def delete_point_annotations_endpoint(sample_id: int):
    """Delete all point annotations for the given sample."""

    deleted_rows = clear_point_annotations(sample_id)
    release_claim_by_id(sample_id)
    return jsonify({"ok": True, "deleted": deleted_rows})


def _accept_single_mask_prediction(
    sample_id: int,
    prediction: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a single mask prediction as an annotation.

    Raises exceptions when invariants are not met so the caller can abort the workflow.
    """

    class_name = prediction.get("class")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class is required")
    class_name = class_name.strip()

    if "prediction_id" not in prediction:
        raise ValueError("prediction_id is required")
    try:
        prediction_id = int(prediction.get("prediction_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction_id must be an integer") from exc

    if "prediction_timestamp" not in prediction:
        raise ValueError("prediction_timestamp is required")
    try:
        prediction_timestamp = int(prediction.get("prediction_timestamp"))
    except (TypeError, ValueError) as exc:
        raise ValueError("prediction_timestamp must be an integer") from exc

    mask_predictions = [
        p
        for p in get_predictions(sample_id)
        if p.get("type") == "mask" and p.get("class") == class_name and p.get("mask_path")
    ]
    if not mask_predictions:
        raise LookupError(f"Mask prediction not found for class '{class_name}'")

    mask_predictions.sort(
        key=lambda p: ((p.get("timestamp") or 0), (p.get("id") or 0)),
        reverse=True,
    )
    latest = mask_predictions[0]
    latest_timestamp = latest.get("timestamp") or 0
    if latest.get("id") != prediction_id or latest_timestamp != prediction_timestamp:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "Mask prediction has changed",
                    "latest": {
                        "id": latest.get("id"),
                        "timestamp": latest_timestamp,
                    },
                }
            )
        )

    try:
        source_path = _resolve_under_dir(str(latest.get("mask_path")), PREDS_DIR)
    except (FileNotFoundError, ValueError) as exc:
        raise FileNotFoundError(f"Mask file is no longer available: {exc}") from exc

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    sanitized_class = re.sub(r"[^A-Za-z0-9_.-]", "_", class_name)
    suffix = source_path.suffix or ".png"
    dest_filename = (
        f"sample{sample_id}_{sanitized_class}_pred{prediction_id}_{int(time.time())}{suffix}"
    )
    dest_path = (MASKS_DIR / dest_filename).resolve()
    try:
        shutil.copy2(source_path, dest_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to store mask for class '{class_name}': {exc}") from exc

    relative_path = dest_path.relative_to(SESSION_DIR).as_posix()
    timestamp = int(time.time())

    previous_annotations = [
        ann
        for ann in get_annotations(sample_id)
        if ann.get("type") == "mask" and ann.get("class") == class_name and ann.get("mask_path")
    ]

    try:
        upsert_annotation(
            sample_id,
            class_name,
            "mask",
            mask_path=relative_path,
            timestamp=timestamp,
        )
    except Exception:
        with suppress(Exception):
            dest_path.unlink()
        raise

    for prev in previous_annotations:
        prev_path = _resolve_under_dir(str(prev.get("mask_path")), MASKS_DIR)
        if prev_path != dest_path:
            prev_path.unlink()

    mask_url = _mask_public_url(relative_path, MASKS_DIR, "masks")
    response_annotation = {
        "sample_id": sample_id,
        "class": class_name,
        "type": "mask",
        "mask_path": relative_path,
        "timestamp": timestamp,
    }
    if mask_url:
        response_annotation["mask_url"] = mask_url

    return response_annotation


@app.post("/api/annotations/<int:sample_id>/accept_mask")
def accept_mask_annotation(sample_id: int):
    """Promote ML mask predictions to persisted mask annotations."""

    payload = request.get_json(silent=True) or {}
    raw_predictions = payload.get("predictions")
    if raw_predictions is None:
        raw_predictions = [payload]

    if not isinstance(raw_predictions, list) or not raw_predictions:
        return jsonify({"error": "predictions must be a non-empty list"}), 400

    accepted: List[Dict[str, Any]] = []
    try:
        for raw in raw_predictions:
            if not isinstance(raw, dict):
                raise ValueError("Each prediction must be an object")
            annotation = _accept_single_mask_prediction(sample_id, raw)
            accepted.append(annotation)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except LookupError as exc:
        return jsonify({"error": str(exc)}), 404
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        # Preserve structured JSON errors when provided.
        try:
            data = json.loads(str(exc))
        except Exception:
            data = {"error": str(exc)}
        status = 409 if data.get("error") == "Mask prediction has changed" else 500
        return jsonify(data), status
    except Exception as exc:
        return jsonify({"error": f"Failed to accept mask prediction: {exc}"}), 500

    release_claim_by_id(sample_id)
    return jsonify({"ok": True, "annotations": accepted})


@app.delete("/api/annotations/<int:sample_id>/mask", defaults={"class_name": None})
@app.delete("/api/annotations/<int:sample_id>/mask/<class_name>")
def delete_mask_annotation_endpoint(sample_id: int, class_name: str | None):
    """Remove stored mask annotations for a sample.

    When ``class_name`` is provided only that class is removed; otherwise all
    mask annotations for the sample are deleted.
    """
    class_name = (class_name or "").strip() or None

    existing = [
        ann
        for ann in get_annotations(sample_id)
        if ann.get("type") == "mask"
        and ann.get("mask_path")
        and (class_name is None or ann.get("class") == class_name)
    ]
    if not existing:
        # Nothing to delete; return ok to keep frontend logic simple
        return jsonify({"ok": True, "deleted": 0, "deleted_files": 0})

    deleted_files = 0
    for ann in existing:
        prev_path = _resolve_under_dir(str(ann.get("mask_path")), MASKS_DIR)
        prev_path.unlink()
        deleted_files += 1

    deleted_rows = delete_mask_annotation(sample_id, class_name)
    release_claim_by_id(sample_id)

    scope = "all" if class_name is None else "class"
    return jsonify({"ok": True, "deleted": deleted_rows, "deleted_files": deleted_files, "scope": scope})


@app.get("/api/stats")
def get_stats():
    """Returns current statistics."""
    stats = get_annotation_stats()
    return jsonify(stats)


@app.get("/api/export")
def export_data():
    """Gets the annotations from the database as a file."""
    annotations = export_annotations()
    return jsonify({"annotations": annotations})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

# Module-level timestamp for opportunistic cleanup throttling
_last_claim_cleanup_ts = None
