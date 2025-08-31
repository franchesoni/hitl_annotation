from pathlib import Path
from flask import Flask, request, jsonify, send_file
from io import BytesIO
import base64
from functools import lru_cache
import timm
import time
import mimetypes
from collections import defaultdict

from src.backend.db import (
    get_config, update_config, get_next_sample_by_strategy,
    get_sample_by_id, upsert_annotation, delete_annotation_by_sample_id,
    get_annotation_stats, export_annotations, release_claim_by_id,
    get_most_recent_prediction, store_live_accuracy, get_annotations,
    cleanup_claims_unconditionally,
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

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
# Fail loudly: enable development-style debugging and exception propagation
app.config.update(
    ENV="development",
    DEBUG=True,
    PROPAGATE_EXCEPTIONS=True,
    TRAP_HTTP_EXCEPTIONS=True,
)


@lru_cache(maxsize=1)
def _list_architectures():
    """Return all allowed model architectures."""
    resnets = ["resnet18", "resnet34"]
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
    import json
    mask_preds = [p for p in preds if p.get("type") == "mask" and p.get("mask_path")]
    mask_map = {}
    for pred in mask_preds:
        mask_path_raw = str(pred.get("mask_path"))
        mask_url = None
        try:
            p = Path(mask_path_raw)
            # If absolute, ensure it's under PREDS_DIR and relativize
            if p.is_absolute():
                p_res = p.resolve()
                preds_root = PREDS_DIR.resolve()
                try:
                    rel = p_res.relative_to(preds_root)
                    mask_url = f"/preds/{rel.as_posix()}"
                except Exception:
                    mask_url = None
            else:
                parts = p.parts
                if len(parts) >= 2 and parts[0] == "session" and parts[1] == "preds":
                    rel = Path(*parts[2:])
                    mask_url = f"/preds/{rel.as_posix()}"
                elif len(parts) >= 1 and parts[0] == "preds":
                    rel = Path(*parts[1:])
                    mask_url = f"/preds/{rel.as_posix()}"
                else:
                    candidate = (PREDS_DIR / p).resolve()
                    try:
                        rel = candidate.relative_to(PREDS_DIR.resolve())
                        mask_url = f"/preds/{rel.as_posix()}"
                    except Exception:
                        mask_url = None
        except Exception:
            mask_url = None
        if mask_url:
            cls = str(pred.get("class", ""))
            if cls:
                mask_map[cls] = mask_url
    if mask_map:
        headers["X-Predictions-Mask"] = json.dumps(mask_map)

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


@app.route("/")
def index():
    return app.send_static_file("router.html")


@app.route("/classification")
def classification():
    # Set config.task to 'classification' (merge)
    update_config({"task": "classification"})
    return app.send_static_file("classification/index.html")

# Add segmentation route
@app.route("/segmentation")
def segmentation():
    # Set config.task to 'segmentation' (merge)
    update_config({"task": "segmentation"})
    return app.send_static_file("segmentation/index.html")


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
        return jsonify({"error": "No more samples available for annotation"}), 404


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
        return jsonify({"error": f"No previous sample found before ID {sample_id}"}), 404


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
        return jsonify({"error": f"No next sample found after ID {sample_id}"}), 404



# Implements DELETE /api/annotations/<int:sample_id> with optional 'type' query param
@app.delete("/api/annotations/<int:sample_id>")
def delete_annotations_endpoint(sample_id: int):
    """
    Deletes all annotations for the sample, or only the specified type if provided.
    Query param: 'type' (optional) to scope deletion (e.g., label, point, bbox).
    Returns: {ok: true}
    """
    ann_type = request.args.get("type")
    success = delete_annotation_by_sample_id(sample_id, ann_type) if ann_type else delete_annotation_by_sample_id(sample_id)
    if success:
        release_claim_by_id(sample_id)
        return jsonify({"ok": True})
    else:
        return jsonify({"error": f"No annotation(s) found for sample ID {sample_id}"}), 404


@app.get("/api/annotations/<int:sample_id>")
def get_annotations_endpoint(sample_id: int):
    """Get all annotations for a specific sample ID."""
    annotations = get_annotations(sample_id)
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
    grouped = defaultdict(list)
    for ann in items:
        ann_type = ann.get("type", "label")
        grouped[ann_type].append(ann)
    total = 0
    for ann_type, anns in grouped.items():
        # Remove existing annotations of this type
        delete_annotation_by_sample_id(sample_id, ann_type)
        # Insert new ones
        for ann in anns:
            class_ = ann.get("class")
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
