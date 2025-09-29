from pathlib import Path
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
from typing import List

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
    pred_candidates = [
        p for p in preds if p.get("type") == "label" and p.get("probability") is not None
    ]
    assert len(pred_candidates) <= 1, "Expected at most one prediction per image"
    if pred_candidates:
        pred_ann = pred_candidates[0]
        headers["X-Predictions-Type"] = "label"
        headers["X-Predictions-Label"] = str(pred_ann.get("class", ""))
        from src.backend.db import to_ppm
        prob = pred_ann.get("probability", None)
        headers["X-Predictions-Probability"] = str(to_ppm(prob) if prob is not None else "")

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


def _safe_pred_path(relpath: str) -> Path | None:
    """Resolve a relative path under PREDS_DIR safely.

    Returns the resolved Path if it is a file strictly under PREDS_DIR.
    Otherwise returns None.
    """
    try:
        candidate = (PREDS_DIR / relpath).resolve()
    except Exception:
        return None

    try:
        # Ensure the resolved path is under PREDS_DIR
        candidate.relative_to(PREDS_DIR)
    except Exception:
        return None

    if candidate.is_file():
        return candidate
    return None


def _resolve_under_dir(path_str: str, storage_dir: Path) -> Path | None:
    """Resolve *path_str* to a file located under *storage_dir* if possible."""
    if not path_str:
        return None
    try:
        raw_path = Path(path_str)
    except Exception:
        return None

    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        try:
            candidates.append((SESSION_DIR / raw_path).resolve())
        except Exception:
            pass
        if raw_path.parts and raw_path.parts[0] == "session":
            try:
                candidates.append((REPO_ROOT / raw_path).resolve())
            except Exception:
                pass
        if raw_path.parts and raw_path.parts[0] == storage_dir.name:
            suffix_parts = raw_path.parts[1:]
            try:
                candidates.append((storage_dir / Path(*suffix_parts)).resolve())
            except Exception:
                pass
        else:
            try:
                candidates.append((storage_dir / raw_path).resolve())
            except Exception:
                pass

    storage_root = storage_dir.resolve()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        try:
            resolved.relative_to(storage_root)
        except Exception:
            continue
        if resolved.is_file():
            return resolved
    return None


def _mask_public_url(path_str: str, storage_dir: Path, mount_prefix: str) -> str | None:
    resolved = _resolve_under_dir(path_str, storage_dir)
    if resolved is None:
        return None
    rel = resolved.relative_to(storage_dir.resolve())
    return f"/{mount_prefix}/{rel.as_posix()}"


def _safe_mask_path(relpath: str) -> Path | None:
    """Resolve a relative path under MASKS_DIR safely."""
    try:
        candidate = (MASKS_DIR / relpath).resolve()
    except Exception:
        return None

    try:
        candidate.relative_to(MASKS_DIR)
    except Exception:
        return None

    if candidate.is_file():
        return candidate
    return None


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
    safe_path = _safe_pred_path(relpath)
    if safe_path is None:
        return jsonify({"error": "Prediction file not found"}), 404

    mime_type, _ = mimetypes.guess_type(str(safe_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    return send_file(safe_path, mimetype=mime_type)


@app.get("/masks/<path:relpath>")
def get_annotation_mask(relpath: str):
    """Serve accepted annotation mask files from session/masks safely."""
    safe_path = _safe_mask_path(relpath)
    if safe_path is None:
        return jsonify({"error": "Mask file not found"}), 404

    mime_type, _ = mimetypes.guess_type(str(safe_path))
    if mime_type is None:
        mime_type = "application/octet-stream"
    return send_file(safe_path, mimetype=mime_type)


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


@app.post("/api/annotations/<int:sample_id>/accept_mask")
def accept_mask_annotation(sample_id: int):
    """Promote an ML mask prediction to a persisted mask annotation."""
    payload = request.get_json(silent=True) or {}
    class_name = payload.get("class")
    if not isinstance(class_name, str) or not class_name.strip():
        return jsonify({"error": "class is required"}), 400
    class_name = class_name.strip()

    if "prediction_id" not in payload:
        return jsonify({"error": "prediction_id is required"}), 400
    try:
        prediction_id = int(payload.get("prediction_id"))
    except (TypeError, ValueError):
        return jsonify({"error": "prediction_id must be an integer"}), 400

    if "prediction_timestamp" not in payload:
        return jsonify({"error": "prediction_timestamp is required"}), 400
    try:
        prediction_timestamp = int(payload.get("prediction_timestamp"))
    except (TypeError, ValueError):
        return jsonify({"error": "prediction_timestamp must be an integer"}), 400

    mask_predictions = [
        p for p in get_predictions(sample_id)
        if p.get("type") == "mask" and p.get("class") == class_name and p.get("mask_path")
    ]
    if not mask_predictions:
        return jsonify({"error": "Mask prediction not found"}), 404

    mask_predictions.sort(
        key=lambda p: ((p.get("timestamp") or 0), (p.get("id") or 0)),
        reverse=True,
    )
    latest = mask_predictions[0]
    latest_timestamp = latest.get("timestamp") or 0
    if latest.get("id") != prediction_id or latest_timestamp != prediction_timestamp:
        return jsonify({
            "error": "Mask prediction has changed",
            "latest": {
                "id": latest.get("id"),
                "timestamp": latest_timestamp,
            },
        }), 409

    source_path = _resolve_under_dir(str(latest.get("mask_path")), PREDS_DIR)
    if source_path is None:
        return jsonify({"error": "Mask file is no longer available"}), 400

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
        return jsonify({"error": f"Failed to store mask: {exc}"}), 500

    relative_path = dest_path.relative_to(SESSION_DIR).as_posix()
    timestamp = int(time.time())

    previous_annotations = [
        ann for ann in get_annotations(sample_id)
        if ann.get("type") == "mask" and ann.get("class") == class_name and ann.get("mask_path")
    ]

    upsert_annotation(
        sample_id,
        class_name,
        "mask",
        mask_path=relative_path,
        timestamp=timestamp,
    )
    release_claim_by_id(sample_id)

    for prev in previous_annotations:
        prev_path = _resolve_under_dir(str(prev.get("mask_path")), MASKS_DIR)
        if prev_path and prev_path.exists() and prev_path != dest_path:
            try:
                prev_path.unlink()
            except Exception:
                pass

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

    return jsonify({"ok": True, "annotation": response_annotation})


@app.delete("/api/annotations/<int:sample_id>/mask/<class_name>")
def delete_mask_annotation_endpoint(sample_id: int, class_name: str):
    """Remove a stored mask annotation for a given class on a sample."""
    if class_name is None:
        return jsonify({"error": "class name is required"}), 400
    class_name = class_name.strip()
    if not class_name:
        return jsonify({"error": "class name is required"}), 400

    existing = [
        ann for ann in get_annotations(sample_id)
        if ann.get("type") == "mask" and ann.get("class") == class_name and ann.get("mask_path")
    ]
    if not existing:
        # Nothing to delete; return ok to keep frontend logic simple
        return jsonify({"ok": True, "deleted": 0})

    deleted_files = 0
    for ann in existing:
        prev_path = _resolve_under_dir(str(ann.get("mask_path")), MASKS_DIR)
        if prev_path and prev_path.exists():
            try:
                prev_path.unlink()
                deleted_files += 1
            except Exception:
                pass

    deleted_rows = delete_mask_annotation(sample_id, class_name)
    release_claim_by_id(sample_id)

    return jsonify({"ok": True, "deleted": deleted_rows, "deleted_files": deleted_files})


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
